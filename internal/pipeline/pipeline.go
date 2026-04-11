// Package pipeline orchestrates the full speech-to-text pipeline: decode,
// resample, VAD, chunk, transcribe, and stitch.
package pipeline

import (
	"context"
	"fmt"
	"math"
	"os"
	"sort"
	"strings"
	"sync/atomic"
	"time"

	"github.com/negbie/sittich/internal/audio"
	"github.com/negbie/sittich/internal/config"
	"github.com/negbie/sittich/internal/speech"
)

// Pipeline ties together audio decoding, optional VAD, chunking, engine
// inference, and result stitching.
type Pipeline struct {
	engine speech.Engine
	config config.Pipeline
	ready  atomic.Bool
}

// New creates a Pipeline that delegates transcription to the given engine.
func New(eng speech.Engine) *Pipeline {
	p := &Pipeline{
		engine: eng,
	}
	if eng != nil {
		p.ready.Store(true)
	}
	return p
}

// NewPipeline creates a fully configured Pipeline.
func NewPipeline(eng speech.Engine, cfg config.Pipeline) (*Pipeline, error) {
	p := &Pipeline{
		engine: eng,
		config: cfg,
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
	return nil
}

// TranscribeFile decodes the audio file at path and runs the full pipeline.
func (p *Pipeline) TranscribeFile(ctx context.Context, path string, opts speech.Options) (*speech.Result, error) {
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
func (p *Pipeline) Process(ctx context.Context, audioPath string, chunkDuration float64) (*speech.Result, error) {
	samples, sampleRate, channels, err := decodeAudioFile(audioPath)
	if err != nil {
		return nil, fmt.Errorf("pipeline: decode: %w", err)
	}

	opts := speech.Options{
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
func (p *Pipeline) ProcessAudio(ctx context.Context, samples []float32, sampleRate int, chunkDuration float64) (*speech.Result, error) {
	opts := speech.Options{
		Language:       p.config.Language,
		WordTimestamps: p.config.WordTimestamps,
		Debug:          p.config.Debug,
	}

	if chunkDuration <= 0 {
		chunkDuration = p.config.ChunkDuration
	}

	return p.processAudioInternal(ctx, samples, sampleRate, 1, opts, chunkDuration)
}

func (p *Pipeline) processAudioInternal(ctx context.Context, samples []float32, sampleRate int, channels int, opts speech.Options, chunkDuration float64) (*speech.Result, error) {
	const (
		targetRate = 16000
		padding    = 0.30 // 300ms padding for context
	)

	totalStart := time.Now()

	// Audit signal quality
	p.verifySignal(samples, targetRate)
	resampled := samples

	// 1. Tiling & Chunking
	totalDur := float64(len(resampled)) / float64(targetRate)
	var segments []SpeechSegment
	for start := 0.0; start < totalDur; start += chunkDuration {
		end := start + chunkDuration
		if end > totalDur {
			end = totalDur
		}
		segments = append(segments, SpeechSegment{Start: start, End: end})
	}

	// Expand segments into chunks with context padding
	chunks := ChunkSpeechSegments(resampled, targetRate, segments, chunkDuration, p.config.ChunkOverlapDuration, p.config.ChunkMinTailDuration)
	for i := range chunks {
		chunks[i].Start -= padding
		if chunks[i].Start < 0 {
			chunks[i].Start = 0
		}
		chunks[i].End += padding
		if chunks[i].End > totalDur {
			chunks[i].End = totalDur
		}
	}

	// 2. Unitary Request Batch Submit
	// Instead of trickling chunks through workers, we submit the entire request
	// as one batch. This allows the Global Dispatcher and the underlying engine
	// to optimize the hardware utilization far more effectively.
	allChunkResults, err := p.transcribeChunks(ctx, resampled, targetRate, chunks, opts)
	if err != nil {
		return nil, err
	}

	if len(allChunkResults) == 0 {
		return &speech.Result{Duration: float64(len(resampled)) / targetRate}, nil
	}

	// Ensure results are sorted by offset for stitching
	sort.Slice(allChunkResults, func(i, j int) bool {
		return allChunkResults[i].Offset < allChunkResults[j].Offset
	})

	combined := StitchResults(allChunkResults)
	combined.Duration = float64(len(resampled)) / targetRate

	for i := range combined.Segments {
		combined.Segments[i].Text = strings.TrimSpace(combined.Segments[i].Text)
	}

	if p.config.Debug {
		fmt.Fprintf(os.Stderr, "   [Pipeline] Stage total=%s result_text_len=%d chunks=%d\n", time.Since(totalStart).Round(time.Millisecond), len(combined.FullText()), len(allChunkResults))
	}

	return combined, nil
}

func (p *Pipeline) decodeAndResample(_ []float32, _ int, _ int, targetRate int) ([]float32, error) {
	// The new DecodeWAV engine (via Sox) already returns 16kHz Mono samples.
	// We keep this stage only for debug peak logging and signal verification.
	// The incoming 'samples' are already 16kHz Mono.
	return nil, fmt.Errorf("pipeline: internal logic error, raw samples should not be passed to decodeAndResample anymore")
}

// verifySignal checks the signal amplitude and logs the peak for debugging.
func (p *Pipeline) verifySignal(samples []float32, targetRate int) {
	if !p.config.Debug {
		return
	}
	var peak float32
	for _, s := range samples {
		abs := float32(math.Abs(float64(s)))
		if abs > peak {
			peak = abs
		}
	}
	fmt.Fprintf(os.Stderr, "   [Pipeline] Stage decode_verified samples=%d duration=%.2fs peak=%.4f\n", len(samples), float64(len(samples))/float64(targetRate), peak)
}

func (p *Pipeline) transcribeChunks(ctx context.Context, resampled []float32, targetRate int, chunks []Chunk, opts speech.Options) ([]ChunkResult, error) {
	asrTotalStart := time.Now()

	batch := make([][]float32, 0, len(chunks))
	offsets := make([]float64, 0, len(chunks))

	for _, c := range chunks {
		startSample := int(c.Start * float64(targetRate))
		endSample := int(c.End * float64(targetRate))
		if startSample < 0 {
			startSample = 0
		}
		if endSample > len(resampled) {
			endSample = len(resampled)
		}

		if startSample >= endSample {
			continue
		}

		chunkAudio := resampled[startSample:endSample]
		if len(chunkAudio) == 0 {
			continue
		}

		batch = append(batch, chunkAudio)
		offsets = append(offsets, c.Start)
	}

	if len(batch) == 0 {
		return nil, nil
	}

	if p.config.Debug {
		fmt.Fprintf(os.Stderr, "   [Pipeline] Stage batch_asr_start chunks=%d\n", len(batch))
	}

	results, err := p.engine.TranscribeBatch(ctx, batch, targetRate, opts)
	if err != nil {
		return nil, err
	}

	chunkResults := make([]ChunkResult, len(results))
	for i, res := range results {
		chunkResults[i] = ChunkResult{
			Offset: offsets[i],
			Result: res,
		}
	}

	if p.config.Debug {
		fmt.Fprintf(os.Stderr, "   [Pipeline] Stage batch_asr_total=%s chunks=%d\n", time.Since(asrTotalStart).Round(time.Millisecond), len(chunks))
	}

	return chunkResults, nil
}

func (p *Pipeline) mergeOptions(opts speech.Options) speech.Options {
	if opts.Language == "" {
		opts.Language = p.config.Language
	}
	if !opts.WordTimestamps {
		opts.WordTimestamps = p.config.WordTimestamps
	}
	return opts
}

func decodeAudioFile(path string) ([]float32, int, int, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, 0, 0, err
	}
	defer f.Close()

	samples, err := audio.DecodeWAV(f)
	if err != nil {
		return nil, 0, 0, err
	}
	// DecodeWAV with Sox results in 16kHz Mono
	return samples, 16000, 1, nil
}
