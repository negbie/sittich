package asr

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"

	sherpa "github.com/k2-fsa/sherpa-onnx-go/sherpa_onnx"
	"github.com/negbie/sittich/internal/models"
	"github.com/negbie/sittich/internal/types"
)

// Recognizer wraps Sherpa-ONNX recognizer with thread-safe access and proper cleanup.
type Recognizer struct {
	Config     *Config
	recognizer *sherpa.OfflineRecognizer
	mu         sync.Mutex
	cond       *sync.Cond
	active     int
	closed     bool
}

// NewRecognizer creates a new ASR recognizer
func NewRecognizer(cfg *Config) (*Recognizer, error) {
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
	}

	// If NumThreads is 0, we auto-detect based on available cores and worker count.
	// This prevents the "over-subscription" issue during concurrent transcription.
	if cfg.NumThreads <= 0 {
		availableCores := runtime.NumCPU()
		// If we don't know the worker count, we default to a conservative 1/4 of cores.
		// However, cfg should ideally handle this.
		sherpaConfig.ModelConfig.NumThreads = availableCores / 4
		if sherpaConfig.ModelConfig.NumThreads < 1 {
			sherpaConfig.ModelConfig.NumThreads = 1
		}
	} else {
		sherpaConfig.ModelConfig.NumThreads = 4
	}

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
func (r *Recognizer) Transcribe(ctx context.Context, audio []float32, sampleRate int, opts types.Options) (*types.Result, error) {
	if r == nil {
		return nil, fmt.Errorf("recognizer is nil")
	}

	// Check if context is already dead
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	r.mu.Lock()
	if r.closed || r.recognizer == nil {
		r.mu.Unlock()
		return nil, fmt.Errorf("recognizer already closed")
	}

	stream := sherpa.NewOfflineStream(r.recognizer)
	if stream == nil {
		r.mu.Unlock()
		return nil, fmt.Errorf("failed to create offline stream")
	}

	r.active++
	r.mu.Unlock()

	defer func() {
		sherpa.DeleteOfflineStream(stream)

		r.mu.Lock()
		r.active--
		if r.active == 0 {
			r.cond.Broadcast()
		}
		r.mu.Unlock()
	}()

	// Process audio
	calibrateAudio(audio)
	stream.AcceptWaveform(sampleRate, audio)
	r.recognizer.Decode(stream)

	// Get result
	sherpaResult := stream.GetResult()
	if sherpaResult == nil {
		if opts.Debug {
			fmt.Fprintf(os.Stderr, "   [Recognizer] empty_result=nil sample_rate=%d samples=%d\n", sampleRate, len(audio))
		}
		return &types.Result{}, nil
	}
	if opts.Debug {
		textLen := len(strings.TrimSpace(sherpaResult.Text))
		fmt.Fprintf(os.Stderr, "   [Recognizer] result_text_len=%d tokens=%d timestamps=%d\n", textLen, len(sherpaResult.Tokens), len(sherpaResult.Timestamps))
		if textLen == 0 {
			fmt.Fprintf(os.Stderr, "   [Recognizer] empty_result=text sample_rate=%d samples=%d\n", sampleRate, len(audio))
		}
	}

	// Convert to internal types
	result := &types.Result{
		Duration: float64(len(audio)) / float64(sampleRate),
	}

	result.Segments = []types.Segment{
		{
			ID:    0,
			Start: 0,
			End:   result.Duration,
			Text:  sherpaResult.Text,
		},
	}

	// Map timestamps if available
	if len(sherpaResult.Timestamps) > 0 {
		result.Segments[0].Words = make([]types.Word, len(sherpaResult.Tokens))
		for i, token := range sherpaResult.Tokens {
			if i < len(sherpaResult.Timestamps) {
				result.Segments[0].Words[i] = types.Word{
					Word:  token,
					Start: float64(sherpaResult.Timestamps[i]),
					End:   float64(sherpaResult.Timestamps[i]) + 0.1,
				}
			}
		}
	}

	return result, nil
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

// calibrateAudio adjusts the signal in-place for Parakeet-TDT INT8 calibration.
// It ensures the target intensity and acoustic floor requirements are met.
// This is a zero-allocation operation to reduce GC pressure on large files.
func calibrateAudio(audio []float32) {
	if len(audio) == 0 {
		return
	}

	maxVal := float32(0)
	for _, v := range audio {
		absV := float32(0)
		if v < 0 {
			absV = -v
		} else {
			absV = v
		}
		if absV > maxVal {
			maxVal = absV
		}
	}

	if maxVal < 1e-4 {
		return // Too quiet, avoid boosting noise
	}

	// Calculate scale to target ~200.0 peak.
	// This corresponds to approximately 5.4 log-units after feature extraction,
	// which is the required activation level for Parakeet-TDT INT8 calibration.
	scale := 200.0 / maxVal

	const floor = 1e-20
	for i, v := range audio {
		// Apply peak scaling in-place
		nv := v * scale

		// Ensure acoustic floor
		if nv > 0 && nv < floor {
			nv = floor
		} else if nv < 0 && nv > -floor {
			nv = -floor
		}
		audio[i] = nv
	}
}
