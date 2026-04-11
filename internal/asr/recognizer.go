package asr

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
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
	}
	r.cond = sync.NewCond(&r.mu)

	return r, nil
}

// Transcribe handles raw PCM samples for the pipeline. It is thread-safe and
// allows concurrent Decode calls on independent streams while preventing Close
// from freeing the recognizer during in-flight work.
func (r *Recognizer) Transcribe(ctx context.Context, audio []float32, sampleRate int, opts speech.Options) (*speech.Result, error) {
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
	calibration := r.calibrateAudio(audio)
	if opts.Debug {
		fmt.Fprintf(
			os.Stderr,
			"   [Recognizer] calibration samples=%d active_ratio=%.4f p99_peak=%.6f scale=%.3f clipped=%d skipped=%v reason=%s\n",
			len(audio),
			calibration.ActiveRatio,
			calibration.Peak,
			calibration.Scale,
			calibration.ClippedSamples,
			calibration.Skipped,
			calibration.SkipReason,
		)
	}
	stream.AcceptWaveform(sampleRate, audio)
	r.recognizer.Decode(stream)

	// Get result
	sherpaResult := stream.GetResult()
	if sherpaResult == nil {
		if opts.Debug {
			fmt.Fprintf(os.Stderr, "   [Recognizer] empty_result=nil sample_rate=%d samples=%d\n", sampleRate, len(audio))
		}
		return &speech.Result{}, nil
	}
	if opts.Debug {
		textLen := len(strings.TrimSpace(sherpaResult.Text))
		fmt.Fprintf(os.Stderr, "   [Recognizer] result_text_len=%d tokens=%d timestamps=%d\n", textLen, len(sherpaResult.Tokens), len(sherpaResult.Timestamps))
		if textLen == 0 {
			fmt.Fprintf(os.Stderr, "   [Recognizer] empty_result=text sample_rate=%d samples=%d\n", sampleRate, len(audio))
		}
	}

	// Convert to internal types
	result := &speech.Result{
		Duration: float64(len(audio)) / float64(sampleRate),
	}

	result.Segments = []speech.Segment{
		{
			ID:    0,
			Start: 0,
			End:   result.Duration,
			Text:  sherpaResult.Text,
		},
	}

	// Map timestamps if available
	if len(sherpaResult.Timestamps) > 0 {
		result.Segments[0].Words = make([]speech.Word, len(sherpaResult.Tokens))
		for i, token := range sherpaResult.Tokens {
			if i < len(sherpaResult.Timestamps) {
				result.Segments[0].Words[i] = speech.Word{
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

// calibrationStats captures chunk-level calibration behavior for debug logging.
type calibrationStats struct {
	ActiveRatio    float64
	Peak           float32
	Scale          float32
	ClippedSamples int
	Skipped        bool
	SkipReason     string
}

// calibrateAudio adjusts the signal in-place for Parakeet-TDT INT8 calibration.
// It keeps the original German-friendly robust-peak strategy, but adds light
// gating and gain caps so silence/noise is less likely to be over-amplified.
func (r *Recognizer) calibrateAudio(audio []float32) calibrationStats {
	stats := calibrationStats{
		Scale:      1.0,
		Skipped:    true,
		SkipReason: "empty_audio",
	}

	n := len(audio)
	if n == 0 {
		return stats
	}

	const (
		gateThreshold  = 0.003
		minActiveRatio = 0.01
		minPeak        = 1e-4
		minGain        = 1.0
		clipLimit      = 220.0
	)

	maxMSamples := 8192
	targetPeak := float32(180.0)
	maxGain := float32(200.0)

	if r != nil && r.Config != nil {
		if r.Config.CalibrationMaxMSamples > 0 {
			maxMSamples = r.Config.CalibrationMaxMSamples
		}
		if r.Config.CalibrationTargetPeak > 0 {
			targetPeak = r.Config.CalibrationTargetPeak
		}
		if r.Config.CalibrationMaxGain > 0 {
			maxGain = r.Config.CalibrationMaxGain
		}
	}

	m := n
	if m > maxMSamples {
		m = maxMSamples
	}

	tmp := make([]float32, 0, m)
	stride := float64(n) / float64(m)

	for i := 0; i < m; i++ {
		v := audio[int(float64(i)*stride)]
		if v < 0 {
			v = -v
		}
		if v >= gateThreshold {
			tmp = append(tmp, v)
		}
	}

	if len(tmp) == 0 {
		stats.SkipReason = "no_active_samples"
		return stats
	}

	stats.ActiveRatio = float64(len(tmp)) / float64(m)
	if stats.ActiveRatio < minActiveRatio {
		stats.SkipReason = "active_ratio_below_threshold"
		return stats
	}

	slices.Sort(tmp)

	idx := int(float64(len(tmp)) * 0.99)
	if idx >= len(tmp) {
		idx = len(tmp) - 1
	}
	peak := tmp[idx]
	stats.Peak = peak

	if peak < minPeak {
		stats.SkipReason = "peak_below_threshold"
		return stats
	}

	scale := targetPeak / peak
	if scale < minGain {
		scale = minGain
	} else if scale > maxGain {
		scale = maxGain
	}
	stats.Scale = scale
	stats.Skipped = false
	stats.SkipReason = "none"

	for i, v := range audio {
		nv := v * scale
		if nv > clipLimit {
			nv = clipLimit
			stats.ClippedSamples++
		} else if nv < -clipLimit {
			nv = -clipLimit
			stats.ClippedSamples++
		}
		audio[i] = nv
	}

	return stats
}
