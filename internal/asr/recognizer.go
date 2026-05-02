package asr

import (
	"context"
	"fmt"
	"path/filepath"
	"runtime/debug"
	"strings"
	"sync"
	"time"

	sherpa "github.com/k2-fsa/sherpa-onnx-go/sherpa_onnx"
	"github.com/negbie/sittich/internal/config"
	"github.com/negbie/sittich/internal/models"
)

// Recognizer wraps the Sherpa-ONNX engine with concurrency control.
type Recognizer struct {
	Config     *config.ASR
	recognizer *sherpa.OfflineRecognizer
	mu         sync.Mutex
	cond       *sync.Cond
	active     int
	inFlight   int
	closed     bool
	idleTimer  *time.Timer
}

func NewRecognizer(cfg *config.ASR) (*Recognizer, error) {
	r := &Recognizer{
		Config: cfg,
	}
	r.cond = sync.NewCond(&r.mu)
	if !cfg.Lazy {
		if err := r.ensureLoaded(); err != nil {
			return nil, err
		}
		r.release()
	}

	return r, nil
}

func (r *Recognizer) ensureLoaded() error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.closed {
		return fmt.Errorf("asr: recognizer closed")
	}

	if r.idleTimer != nil {
		r.idleTimer.Stop()
	}

	r.inFlight++

	if r.recognizer != nil {
		return nil
	}

	sherpaConfig := sherpa.OfflineRecognizerConfig{
		FeatConfig: sherpa.FeatureConfig{
			SampleRate: 16000,
			FeatureDim: 80,
		},
		ModelConfig: sherpa.OfflineModelConfig{
			Transducer: sherpa.OfflineTransducerModelConfig{
				Encoder: filepath.Join(r.Config.ModelPath, models.EncoderFile),
				Decoder: filepath.Join(r.Config.ModelPath, models.DecoderFile),
				Joiner:  filepath.Join(r.Config.ModelPath, models.JoinerFile),
			},
			Tokens:     filepath.Join(r.Config.ModelPath, models.TokensFile),
			Provider:   "cpu",
			ModelType:  "nemo_transducer",
			NumThreads: r.Config.NumThreads,
		},
		DecodingMethod: r.Config.DecodingMethod,
		MaxActivePaths: r.Config.MaxActivePaths,
	}

	recognizer := sherpa.NewOfflineRecognizer(&sherpaConfig)
	if recognizer == nil {
		r.inFlight--
		r.cond.Broadcast()
		return fmt.Errorf("asr: failed to initialize sherpa-onnx")
	}
	r.recognizer = recognizer
	return nil
}

// Transcribe handles a single audio chunk.
func (r *Recognizer) Transcribe(ctx context.Context, audio []float32, sampleRate int, opts Options) (*Result, error) {
	results, err := r.TranscribeBatch(ctx, [][]float32{audio}, sampleRate, opts)
	if err != nil {
		return nil, err
	}
	if len(results) == 0 {
		return &Result{}, nil
	}
	return results[0], nil
}

// TranscribeBatch handles multiple chunks. It uses a semaphore to limit concurrent
// ONNX decodes, preventing unbounded memory growth in the runtime arena.
func (r *Recognizer) TranscribeBatch(ctx context.Context, chunks [][]float32, sampleRate int, opts Options) ([]*Result, error) {
	if r == nil {
		return nil, fmt.Errorf("asr: recognizer is nil")
	}

	if err := r.ensureLoaded(); err != nil {
		return nil, err
	}
	defer r.release()

	// Ensure cond.Wait() unblocks when context is cancelled.
	ctxDone := make(chan struct{})
	go func() {
		select {
		case <-ctx.Done():
			r.cond.Broadcast()
		case <-ctxDone:
		}
	}()
	defer close(ctxDone)

	results := make([]*Result, len(chunks))

	for i, chunk := range chunks {
		if err := ctx.Err(); err != nil {
			return nil, err
		}

		r.mu.Lock()
		maxActive := r.Config.MaxConcurrency
		if maxActive <= 0 {
			maxActive = 1
		}
		for r.active >= maxActive && !r.closed && ctx.Err() == nil {
			r.cond.Wait()
		}
		if r.closed {
			r.mu.Unlock()
			return nil, fmt.Errorf("asr: recognizer closed")
		}
		if err := ctx.Err(); err != nil {
			r.mu.Unlock()
			return nil, err
		}
		r.active++
		r.mu.Unlock()

		err := func() error {
			s := sherpa.NewOfflineStream(r.recognizer)
			if s == nil {
				return fmt.Errorf("asr: failed to create stream")
			}
			defer sherpa.DeleteOfflineStream(s)

			s.AcceptWaveform(sampleRate, chunk)
			r.recognizer.DecodeStreams([]*sherpa.OfflineStream{s})

			res := s.GetResult()
			duration := float64(len(chunk)) / float64(sampleRate)

			out := &Result{Duration: duration}
			if res != nil {
				if len(res.YsLogProbs) > 0 {
					var sum float64
					for _, p := range res.YsLogProbs {
						sum += float64(p)
					}
					out.Confidence = sum / float64(len(res.YsLogProbs))
				}
				cleanText := strings.ReplaceAll(res.Text, "\u2581", " ")
				seg := Segment{
					End:  duration,
					Text: strings.TrimSpace(cleanText),
				}
				if len(res.Timestamps) > 0 {
					n := len(res.Timestamps)
					if len(res.Tokens) < n {
						n = len(res.Tokens)
					}
					seg.Words = make([]Word, n)
					for k := 0; k < n; k++ {
						start := float64(res.Timestamps[k])
						wend := start + 0.1
						if k+1 < n {
							if next := float64(res.Timestamps[k+1]); next > start {
								wend = next
							}
						}
						seg.Words[k] = Word{Word: res.Tokens[k], Start: start, End: wend}
					}
				}
				out.Segments = []Segment{seg}
			}
			results[i] = out
			return nil
		}()

		r.mu.Lock()
		r.active--
		r.cond.Broadcast()
		r.mu.Unlock()

		if err != nil {
			return nil, err
		}
	}

	return results, nil
}

func (r *Recognizer) release() {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.inFlight--
	r.cond.Broadcast()

	if r.inFlight == 0 && r.recognizer != nil {
		if r.Config.Lazy {
			r.unload()
		} else if r.Config.IdleTimeout > 0 {
			if r.idleTimer != nil {
				r.idleTimer.Stop()
			}
			r.idleTimer = time.AfterFunc(r.Config.IdleTimeout, r.onIdleTimeout)
		}
	}
}

func (r *Recognizer) unload() {
	if r.recognizer != nil {
		sherpa.DeleteOfflineRecognizer(r.recognizer)
		r.recognizer = nil
		// Return both Go and C++ heap memory to the OS immediately.
		debug.FreeOSMemory()
		trimCHeap()
	}
}

func (r *Recognizer) onIdleTimeout() {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.inFlight == 0 && !r.closed {
		r.unload()
	}
}

func (r *Recognizer) SupportedLanguages() []string { return nil }
func (r *Recognizer) ModelName() string            { return filepath.Base(r.Config.ModelPath) }
func (r *Recognizer) VADPath() string              { return r.Config.VADPath }

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
	if r.idleTimer != nil {
		r.idleTimer.Stop()
	}
	r.cond.Broadcast()
	for r.inFlight > 0 {
		r.cond.Wait()
	}
	if r.recognizer != nil {
		sherpa.DeleteOfflineRecognizer(r.recognizer)
		r.recognizer = nil
	}
	r.mu.Unlock()
	return nil
}
