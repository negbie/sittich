// Package pipeline orchestrates the full speech-to-text pipeline: decode,
// resample, VAD, chunk, transcribe, and stitch.
package pipeline

import (
	"context"
	"fmt"
	"math"
	"os"
	"strings"
	"sync/atomic"
	"time"

	"github.com/negbie/sittich/internal/audio"
	"github.com/negbie/sittich/internal/types"
)

// PipelineConfig controls pipeline behaviour.
type PipelineConfig struct {
	// VADEnabled enables Silero VAD-based speech segmentation before
	// transcription. When false, audio is chunked at fixed intervals.
	VADEnabled bool

	// ChunkDuration is the maximum duration in seconds per chunk sent to the
	// engine. Callers are expected to provide this explicitly.
	ChunkDuration float64

	// ChunkMinTailDuration is the minimum tail duration used when balancing
	// oversized speech segments into smaller chunks.
	ChunkMinTailDuration float64

	// VADThreshold controls the speech probability threshold used by VAD.
	// If <= 0, the VAD implementation default is used.
	VADThreshold float32

	// VADMinSilenceDuration is the minimum silence duration in seconds needed
	// for VAD to split speech regions. If <= 0, the VAD implementation default
	// is used.
	VADMinSilenceDuration float32

	// VADMinSpeechDuration is the minimum speech duration in seconds required
	// for VAD to emit a speech region. If <= 0, the VAD implementation default
	// is used.
	VADMinSpeechDuration float32

	// VADSegmentPadding expands each detected speech segment by this many
	// seconds on both sides before chunking. If <= 0, no padding is applied.
	VADSegmentPadding float64

	// WordTimestamps requests word-level timing from the engine.
	WordTimestamps bool

	// Language is a BCP-47 hint passed to the engine (empty = auto-detect).
	Language string

	// VADModelPath is the filesystem path to the Silero VAD ONNX model.
	// Only required when VADEnabled is true.
	VADModelPath string

	// Debug enables detailed console logging.
	Debug bool
}

// Pipeline ties together audio decoding, optional VAD, chunking, engine
// inference, and result stitching.
type Pipeline struct {
	engine types.Engine
	vad    *VAD
	config PipelineConfig
	ready  atomic.Bool
}

// New creates a Pipeline that delegates transcription to the given engine.
func New(eng types.Engine) *Pipeline {
	p := &Pipeline{
		engine: eng,
	}
	if eng != nil {
		p.ready.Store(true)
	}
	return p
}

// NewPipeline creates a fully configured Pipeline.
func NewPipeline(eng types.Engine, cfg PipelineConfig) (*Pipeline, error) {
	p := &Pipeline{
		engine: eng,
		config: cfg,
	}

	if cfg.VADEnabled && cfg.VADModelPath != "" {
		v, err := NewVAD(
			cfg.VADModelPath,
			cfg.VADThreshold,
			cfg.VADMinSilenceDuration,
			cfg.VADMinSpeechDuration,
		)
		if err != nil {
			return nil, fmt.Errorf("pipeline: init VAD: %w", err)
		}
		p.vad = v
	}

	if eng != nil {
		p.ready.Store(true)
	}

	return p, nil
}

// Ready reports whether the pipeline has been fully initialised.
func (p *Pipeline) Ready() bool {
	return p.ready.Load()
}

// ModelName returns the name of the loaded model.
func (p *Pipeline) ModelName() string {
	if p.engine == nil {
		return ""
	}
	return p.engine.ModelName()
}

// Close releases resources held by the pipeline.
func (p *Pipeline) Close() error {
	p.ready.Store(false)
	if p.vad != nil {
		p.vad.Close()
	}
	return nil
}

// TranscribeFile decodes the audio file at path and runs the full pipeline.
func (p *Pipeline) TranscribeFile(ctx context.Context, path string, opts types.Options) (*types.Result, error) {
	if !p.ready.Load() {
		return nil, fmt.Errorf("pipeline: not ready, model is still loading")
	}

	samples, sampleRate, channels, err := decodeAudioFile(path)
	if err != nil {
		return nil, fmt.Errorf("pipeline: decode: %w", err)
	}

	merged := p.mergeOptions(opts)
	return p.processAudioInternal(ctx, samples, sampleRate, channels, merged, p.config.ChunkDuration)
}

// Process decodes the audio file at audioPath and runs the pipeline.
// If chunkDuration is > 0, it overrides the pipeline's default setting.
func (p *Pipeline) Process(ctx context.Context, audioPath string, chunkDuration float64) (*types.Result, error) {
	samples, sampleRate, channels, err := decodeAudioFile(audioPath)
	if err != nil {
		return nil, fmt.Errorf("pipeline: decode: %w", err)
	}

	opts := types.Options{
		Language:       p.config.Language,
		WordTimestamps: p.config.WordTimestamps,
		Debug:          p.config.Debug,
	}

	if chunkDuration <= 0 {
		chunkDuration = p.config.ChunkDuration
	}

	return p.processAudioInternal(ctx, samples, sampleRate, channels, opts, chunkDuration)
}

// ProcessAudio runs the pipeline on pre-decoded audio samples.
func (p *Pipeline) ProcessAudio(ctx context.Context, samples []float32, sampleRate int, chunkDuration float64) (*types.Result, error) {
	opts := types.Options{
		Language:       p.config.Language,
		WordTimestamps: p.config.WordTimestamps,
		Debug:          p.config.Debug,
	}

	if chunkDuration <= 0 {
		chunkDuration = p.config.ChunkDuration
	}

	return p.processAudioInternal(ctx, samples, sampleRate, 1, opts, chunkDuration)
}

// processAudioInternal is the core pipeline implementation.
func (p *Pipeline) processAudioInternal(ctx context.Context, samples []float32, sampleRate int, channels int, opts types.Options, chunkDuration float64) (*types.Result, error) {
	const targetRate = 16000

	totalStart := time.Now()

	resampled, err := p.decodeAndResample(samples, sampleRate, channels, targetRate)
	if err != nil {
		return nil, err
	}

	segments, err := p.prepareSpeechSegments(resampled, targetRate)
	if err != nil {
		return nil, err
	}

	chunks := p.buildChunks(segments, chunkDuration)
	chunkResults, err := p.transcribeChunks(ctx, resampled, targetRate, chunks, opts)
	if err != nil {
		return nil, err
	}

	combined := StitchResults(chunkResults)
	combined.Duration = float64(len(resampled)) / targetRate

	for i := range combined.Segments {
		combined.Segments[i].Text = strings.TrimSpace(combined.Segments[i].Text)
	}

	if p.config.Debug {
		fmt.Fprintf(os.Stderr, "   [Pipeline] Stage total=%s result_text_len=%d\n", time.Since(totalStart).Round(time.Millisecond), len(combined.FullText()))
	}

	return combined, nil
}

func (p *Pipeline) decodeAndResample(samples []float32, sampleRate int, channels int, targetRate int) ([]float32, error) {
	decodeStart := time.Now()
	if channels <= 0 {
		channels = 1
	}
	mono := audio.ToMono(samples, channels)
	resampled := audio.Resample(mono, sampleRate, targetRate)
	decodeElapsed := time.Since(decodeStart)

	if p.config.Debug {
		var peak float32
		for _, s := range resampled {
			abs := float32(math.Abs(float64(s)))
			if abs > peak {
				peak = abs
			}
		}
		fmt.Fprintf(os.Stderr, "   [Pipeline] Stage decode_resample=%s samples=%d duration=%.2fs peak=%.4f\n", decodeElapsed.Round(time.Millisecond), len(resampled), float64(len(resampled))/float64(targetRate), peak)
	}

	return resampled, nil
}

func (p *Pipeline) prepareSpeechSegments(samples []float32, sampleRate int) ([]SpeechSegment, error) {
	vadStart := time.Now()
	segments, err := p.detectSpeechSegments(samples, sampleRate)
	if err != nil {
		return nil, fmt.Errorf("pipeline: VAD: %w", err)
	}

	audioDuration := float64(len(samples)) / float64(sampleRate)
	rawSegments := make([]SpeechSegment, len(segments))
	copy(rawSegments, segments)

	padding := p.config.VADSegmentPadding
	if padding < 0 {
		padding = 0
	}

	for i := range segments {
		segments[i].Start -= padding
		segments[i].End += padding

		if segments[i].Start < 0 {
			segments[i].Start = 0
		}
		if segments[i].End < 0 {
			segments[i].End = 0
		}
		if segments[i].Start > audioDuration {
			segments[i].Start = audioDuration
		}
		if segments[i].End > audioDuration {
			segments[i].End = audioDuration
		}
		if segments[i].End < segments[i].Start {
			segments[i].End = segments[i].Start
		}
	}

	if p.config.Debug {
		fmt.Fprintf(os.Stderr, "   [Pipeline] Stage vad=%s segments=%d padding=%.2fs\n", time.Since(vadStart).Round(time.Millisecond), len(segments), padding)
		for i := range segments {
			fmt.Fprintf(
				os.Stderr,
				"   [Pipeline] VAD segment %d/%d raw_start=%.2f raw_end=%.2f padded_start=%.2f padded_end=%.2f audio_duration=%.2f\n",
				i+1,
				len(segments),
				rawSegments[i].Start,
				rawSegments[i].End,
				segments[i].Start,
				segments[i].End,
				audioDuration,
			)
		}
	}

	return segments, nil
}

func (p *Pipeline) buildChunks(segments []SpeechSegment, chunkDuration float64) []Chunk {
	chunkStart := time.Now()
	chunks := ChunkSpeechSegments(segments, chunkDuration, p.config.ChunkMinTailDuration)
	if p.config.Debug {
		fmt.Fprintf(os.Stderr, "   [Pipeline] Stage chunking=%s chunks=%d max_chunk=%.2fs min_tail=%.2fs\n", time.Since(chunkStart).Round(time.Millisecond), len(chunks), chunkDuration, p.config.ChunkMinTailDuration)
	}
	return chunks
}

func (p *Pipeline) transcribeChunks(ctx context.Context, resampled []float32, targetRate int, chunks []Chunk, opts types.Options) ([]ChunkResult, error) {
	asrTotalStart := time.Now()
	chunkResults := make([]ChunkResult, 0, len(chunks))

	for idx, c := range chunks {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		startSample := int(c.Start * float64(targetRate))
		endSample := int(c.End * float64(targetRate))
		if startSample < 0 {
			startSample = 0
		}
		if endSample > len(resampled) {
			endSample = len(resampled)
		}

		if p.config.Debug {
			fmt.Fprintf(os.Stderr, "   [Pipeline] Chunk %d/%d bounds start=%.2f end=%.2f start_sample=%d end_sample=%d\n", idx+1, len(chunks), c.Start, c.End, startSample, endSample)
		}
		if startSample >= endSample {
			if p.config.Debug {
				fmt.Fprintf(os.Stderr, "   [Pipeline] Chunk %d/%d skipped invalid_bounds start_sample=%d end_sample=%d\n", idx+1, len(chunks), startSample, endSample)
			}
			continue
		}

		chunkAudio := resampled[startSample:endSample]
		if len(chunkAudio) == 0 {
			if p.config.Debug {
				fmt.Fprintf(os.Stderr, "   [Pipeline] Chunk %d/%d skipped empty_audio start=%.2f end=%.2f\n", idx+1, len(chunks), c.Start, c.End)
			}
			continue
		}

		chunkASRStart := time.Now()
		if p.config.Debug {
			fmt.Fprintf(os.Stderr, "   [Pipeline] Chunk %d/%d start=%.2f end=%.2f samples=%d\n", idx+1, len(chunks), c.Start, c.End, len(chunkAudio))
		}

		res, err := p.engine.Transcribe(ctx, chunkAudio, targetRate, opts)
		if err != nil {
			return nil, fmt.Errorf("pipeline: transcribe chunk [%.2f-%.2f]: %w", c.Start, c.End, err)
		}

		if p.config.Debug {
			fmt.Fprintf(os.Stderr, "   [Pipeline] Chunk %d/%d asr=%s\n", idx+1, len(chunks), time.Since(chunkASRStart).Round(time.Millisecond))
			if res == nil {
				fmt.Fprintf(os.Stderr, "   [Pipeline] Chunk %d/%d empty_result=nil\n", idx+1, len(chunks))
			} else {
				textLen := len(strings.TrimSpace(res.FullText()))
				segmentCount := len(res.Segments)
				fmt.Fprintf(os.Stderr, "   [Pipeline] Chunk %d/%d result_text_len=%d segments=%d\n", idx+1, len(chunks), textLen, segmentCount)
				if textLen == 0 {
					fmt.Fprintf(os.Stderr, "   [Pipeline] Chunk %d/%d empty_result start=%.2f end=%.2f samples=%d\n", idx+1, len(chunks), c.Start, c.End, len(chunkAudio))
				}
			}
		}

		chunkResults = append(chunkResults, ChunkResult{
			Offset: c.Start,
			Result: res,
		})
	}

	if p.config.Debug {
		fmt.Fprintf(os.Stderr, "   [Pipeline] Stage asr_total=%s\n", time.Since(asrTotalStart).Round(time.Millisecond))
	}

	return chunkResults, nil
}

func (p *Pipeline) detectSpeechSegments(samples []float32, sampleRate int) ([]SpeechSegment, error) {
	if p.vad != nil && p.config.VADEnabled {
		segs, err := p.vad.DetectSpeech(samples, sampleRate)
		if err != nil {
			return nil, err
		}
		if len(segs) == 0 {
			dur := float64(len(samples)) / float64(sampleRate)
			return []SpeechSegment{{Start: 0, End: dur}}, nil
		}
		return segs, nil
	}

	dur := float64(len(samples)) / float64(sampleRate)
	return []SpeechSegment{{Start: 0, End: dur}}, nil
}

func (p *Pipeline) mergeOptions(opts types.Options) types.Options {
	if opts.Language == "" {
		opts.Language = p.config.Language
	}
	if !opts.WordTimestamps {
		opts.WordTimestamps = p.config.WordTimestamps
	}
	return opts
}

func decodeAudioFile(path string) ([]float32, int, int, error) {
	wv, err := audio.ReadWave(path)
	if err != nil {
		return nil, 0, 0, err
	}
	return wv.Samples, wv.SampleRate, wv.Channels, nil
}
