package asr

import (
	"context"
	"fmt"
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
	maxActive  int // max concurrent DecodeStreams calls (limits ONNX arena growth)
}

// NewRecognizer creates a new ASR recognizer
func NewRecognizer(cfg *config.ASR) (*Recognizer, error) {
	sherpaConfig := sherpa.OfflineRecognizerConfig{
		FeatConfig: sherpa.FeatureConfig{
			SampleRate: 16000,
			FeatureDim: 80,
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
		maxActive:  4, // Match dispatcher worker count for full parallelism
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

	streams := make([]*sherpa.OfflineStream, 0, len(chunks))
	for _, audio := range chunks {
		s := sherpa.NewOfflineStream(r.recognizer)
		if s == nil {
			for _, st := range streams {
				sherpa.DeleteOfflineStream(st)
			}
			return nil, fmt.Errorf("failed to create offline stream")
		}

		s.AcceptWaveform(sampleRate, audio)
		streams = append(streams, s)
	}

	// Acquire a concurrency slot. ONNX Runtime's arena allocator never releases
	// memory, so too many concurrent DecodeStreams calls cause unbounded growth.
	// Limiting to maxActive concurrent calls caps the arena to a predictable size.
	r.mu.Lock()
	for r.active >= r.maxActive {
		r.cond.Wait()
	}
	if r.closed || r.recognizer == nil {
		r.mu.Unlock()
		for _, s := range streams {
			sherpa.DeleteOfflineStream(s)
		}
		return nil, fmt.Errorf("recognizer already closed")
	}
	r.active++
	r.mu.Unlock()

	r.recognizer.DecodeStreams(streams)

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
