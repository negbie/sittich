package asr

import (
	"context"
	"fmt"
	"path/filepath"
	"runtime/debug"
	"strings"
	"sync"

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
	maxActive  int // limits concurrent DecodeStreams to cap ONNX arena memory
}

func NewRecognizer(cfg *config.ASR) (*Recognizer, error) {
	r := &Recognizer{
		Config:    cfg,
		maxActive: cfg.MaxActive,
	}
	r.cond = sync.NewCond(&r.mu)

	if !cfg.Lazy {
		if err := r.ensureLoaded(); err != nil {
			return nil, err
		}
	}

	return r, nil
}

func (r *Recognizer) ensureLoaded() error {
	r.mu.Lock()
	defer r.mu.Unlock()

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

	limit := r.maxActive
	if limit < 1 {
		limit = 1
	}

	results := make([]*Result, len(chunks))

	for i := 0; i < len(chunks); i += limit {
		end := i + limit
		if end > len(chunks) {
			end = len(chunks)
		}
		currentBatch := chunks[i:end]
		batchLen := len(currentBatch)

		r.mu.Lock()
		for r.active+batchLen > limit {
			r.cond.Wait()
		}
		if r.closed {
			r.mu.Unlock()
			return nil, fmt.Errorf("asr: recognizer closed")
		}
		r.active += batchLen
		r.mu.Unlock()

		streams := make([]*sherpa.OfflineStream, batchLen)
		for j, chunk := range currentBatch {
			s := sherpa.NewOfflineStream(r.recognizer)
			if s == nil {
				for _, st := range streams[:j] {
					sherpa.DeleteOfflineStream(st)
				}
				r.mu.Lock()
				r.active -= batchLen
				r.cond.Broadcast()
				r.mu.Unlock()
				return nil, fmt.Errorf("asr: failed to create stream")
			}
			s.AcceptWaveform(sampleRate, chunk)
			streams[j] = s
		}

		r.recognizer.DecodeStreams(streams)

		for j, s := range streams {
			res := s.GetResult()
			duration := float64(len(currentBatch[j])) / float64(sampleRate)

			out := &Result{Duration: duration}
			if res != nil {
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
			results[i+j] = out
			sherpa.DeleteOfflineStream(s)
		}

		r.mu.Lock()
		r.active -= batchLen
		r.cond.Broadcast()
		r.mu.Unlock()
	}

	return results, nil
}

func (r *Recognizer) release() {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.inFlight--

	if !r.Config.Lazy {
		return
	}

	if r.inFlight == 0 && r.recognizer != nil {
		sherpa.DeleteOfflineRecognizer(r.recognizer)
		r.recognizer = nil
		// Return both Go and C++ heap memory to the OS immediately.
		debug.FreeOSMemory()
		trimCHeap()
	}
}

func (r *Recognizer) SupportedLanguages() []string { return nil }
func (r *Recognizer) ModelName() string            { return filepath.Base(r.Config.ModelPath) }

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
