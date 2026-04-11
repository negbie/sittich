package asr

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"sync"

	sherpa "github.com/k2-fsa/sherpa-onnx-go/sherpa_onnx"
	"github.com/negbie/sittich/internal/config"
	"github.com/negbie/sittich/internal/models"
	"github.com/negbie/sittich/internal/speech"
)

// Recognizer wraps Sherpa-ONNX recognizer with thread-safe access and proper cleanup.
type Recognizer struct {
	Config     *config.ASR
	recognizer *sherpa.OfflineRecognizer
	mu         sync.Mutex
	cond       *sync.Cond
	active     int
	closed     bool
}

// NewRecognizer creates a new ASR recognizer
func NewRecognizer(cfg *config.ASR) (*Recognizer, error) {
	sherpaConfig := sherpa.OfflineRecognizerConfig{
		FeatConfig: sherpa.FeatureConfig{
			SampleRate: 16000,
			FeatureDim: 128,
		},
		ModelConfig: sherpa.OfflineModelConfig{
			Transducer: sherpa.OfflineTransducerModelConfig{
				Encoder: filepath.Join(cfg.ModelPath, models.EncoderFile),
				Decoder: filepath.Join(cfg.ModelPath, models.DecoderFile),
				Joiner:  filepath.Join(cfg.ModelPath, models.JoinerFile),
			},
			Tokens:    filepath.Join(cfg.ModelPath, models.TokensFile),
			Provider:  "cpu",
			ModelType: "nemo_transducer",
		},
		DecodingMethod: cfg.DecodingMethod,
		MaxActivePaths: cfg.MaxActivePaths,
		BlankPenalty:   0.0,
	}

	sherpaConfig.ModelConfig.NumThreads = cfg.NumThreads

	recognizer := sherpa.NewOfflineRecognizer(&sherpaConfig)
	if recognizer == nil {
		return nil, fmt.Errorf("failed to create recognizer")
	}

	r := &Recognizer{
		Config:     cfg,
		recognizer: recognizer,
	}
	r.cond = sync.NewCond(&r.mu)

	return r, nil
}

// Transcribe handles raw PCM samples for the pipeline. It is thread-safe and
// allows concurrent Decode calls on independent streams while preventing Close
// from freeing the recognizer during in-flight work.
func (r *Recognizer) Transcribe(ctx context.Context, audio []float32, sampleRate int, opts speech.Options) (*speech.Result, error) {
	results, err := r.TranscribeBatch(ctx, [][]float32{audio}, sampleRate, opts)
	if err != nil {
		return nil, err
	}
	if len(results) == 0 {
		return &speech.Result{}, nil
	}
	return results[0], nil
}

// TranscribeBatch handles multiple raw PCM audio chunks in parallel.
func (r *Recognizer) TranscribeBatch(ctx context.Context, chunks [][]float32, sampleRate int, opts speech.Options) ([]*speech.Result, error) {
	if r == nil {
		return nil, fmt.Errorf("recognizer is nil")
	}

	if err := ctx.Err(); err != nil {
		return nil, err
	}

	r.mu.Lock()
	if r.closed || r.recognizer == nil {
		r.mu.Unlock()
		return nil, fmt.Errorf("recognizer already closed")
	}

	streams := make([]*sherpa.OfflineStream, 0, len(chunks))
	for _, audio := range chunks {
		s := sherpa.NewOfflineStream(r.recognizer)
		if s == nil {
			// Cleanup already created streams
			for _, st := range streams {
				sherpa.DeleteOfflineStream(st)
			}
			r.mu.Unlock()
			return nil, fmt.Errorf("failed to create offline stream")
		}

		// Process audio
		audio, calibration := r.calibrateAudio(audio)
		if opts.Debug {
			fmt.Fprintf(
				os.Stderr,
				"   [Recognizer] scale=%.3f clipped=%d skipped=%v\n",
				calibration.Scale,
				calibration.Clipped,
				calibration.Skipped,
			)
		}

		s.AcceptWaveform(sampleRate, audio)
		streams = append(streams, s)
	}

	r.active++
	r.mu.Unlock()

	defer func() {
		for _, s := range streams {
			sherpa.DeleteOfflineStream(s)
		}

		r.mu.Lock()
		r.active--
		if r.active == 0 {
			r.cond.Broadcast()
		}
		r.mu.Unlock()
	}()

	r.recognizer.DecodeStreams(streams)

	// Collect results
	results := make([]*speech.Result, len(streams))
	for i, s := range streams {
		sherpaResult := s.GetResult()
		if sherpaResult == nil {
			results[i] = &speech.Result{}
			continue
		}

		duration := float64(len(chunks[i])) / float64(sampleRate)
		res := &speech.Result{
			Duration: duration,
			Language: "", // Sherpa-ONNX offline doesn't always return language per result
		}

		res.Segments = []speech.Segment{
			{
				ID:    0,
				Start: 0,
				End:   duration,
				Text:  sherpaResult.Text,
			},
		}

		if len(sherpaResult.Timestamps) > 0 {
			res.Segments[0].Words = make([]speech.Word, len(sherpaResult.Tokens))
			for j, token := range sherpaResult.Tokens {
				if j < len(sherpaResult.Timestamps) {
					res.Segments[0].Words[j] = speech.Word{
						Word:  token,
						Start: float64(sherpaResult.Timestamps[j]),
						End:   float64(sherpaResult.Timestamps[j]) + 0.1,
					}
				}
			}
		}
		results[i] = res
	}

	return results, nil
}

// SupportedLanguages returns empty (auto-detect or as directed by model).
func (r *Recognizer) SupportedLanguages() []string {
	return nil
}

// ModelName returns the configured model path's basename.
func (r *Recognizer) ModelName() string {
	return filepath.Base(r.Config.ModelPath)
}

// Close cleans up the recognizer native resources. It is thread-safe,
// prevents double-deletion of native objects, and waits for in-flight
// transcriptions to finish before freeing the underlying recognizer.
func (r *Recognizer) Close() error {
	if r == nil {
		return nil
	}

	r.mu.Lock()
	if r.closed {
		r.mu.Unlock()
		return nil
	}

	r.closed = true
	for r.active > 0 {
		r.cond.Wait()
	}

	if r.recognizer != nil {
		sherpa.DeleteOfflineRecognizer(r.recognizer)
		r.recognizer = nil
	}
	r.mu.Unlock()

	return nil
}

// CalibrationStats holds details about signal processing.
type CalibrationStats struct {
	Scale   float32
	Clipped int
	Skipped bool
}

// calibrateAudio applies a fixed gain to the audio samples to bring them
// into the optimal range for the transuder model. It returns a new slice
// if scaling is applied, leaving the original audio untouched.
func (r *Recognizer) calibrateAudio(audio []float32) ([]float32, CalibrationStats) {
	stats := CalibrationStats{
		Scale: 1.0,
	}

	n := len(audio)
	if n == 0 {
		stats.Skipped = true
		return audio, stats
	}

	scale := float32(20.0)
	if r != nil && r.Config != nil {
		scale = r.Config.FixedScale
	}
	stats.Scale = scale

	if scale == 1.0 {
		return audio, stats
	}

	calibrated := make([]float32, n)
	for i := 0; i < n; i++ {
		v := audio[i] * scale
		if v > 1.0 {
			stats.Clipped++
			v = 1.0
		} else if v < -1.0 {
			stats.Clipped++
			v = -1.0
		}
		calibrated[i] = v
	}

	return calibrated, stats
}
