package asr

import (
	"context"
	"fmt"
	"path/filepath"
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
	closed     bool
	maxActive  int // limits concurrent DecodeStreams to cap ONNX arena memory
}

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
			Tokens:     filepath.Join(cfg.ModelPath, models.TokensFile),
			Provider:   "cpu",
			ModelType:  "nemo_transducer",
			NumThreads: cfg.NumThreads,
		},
		DecodingMethod: cfg.DecodingMethod,
		MaxActivePaths: cfg.MaxActivePaths,
	}

	recognizer := sherpa.NewOfflineRecognizer(&sherpaConfig)
	if recognizer == nil {
		return nil, fmt.Errorf("asr: failed to initialize sherpa-onnx")
	}

	r := &Recognizer{
		Config:     cfg,
		recognizer: recognizer,
		maxActive:  cfg.MaxActive,
	}
	r.cond = sync.NewCond(&r.mu)

	return r, nil
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
	if r == nil || r.recognizer == nil {
		return nil, fmt.Errorf("asr: recognizer is nil or closed")
	}

	streams := make([]*sherpa.OfflineStream, len(chunks))
	for i, chunk := range chunks {
		s := sherpa.NewOfflineStream(r.recognizer)
		if s == nil {
			for _, st := range streams[:i] {
				sherpa.DeleteOfflineStream(st)
			}
			return nil, fmt.Errorf("asr: failed to create stream")
		}
		s.AcceptWaveform(sampleRate, chunk)
		streams[i] = s
	}

	r.mu.Lock()
	for r.active >= r.maxActive {
		r.cond.Wait()
	}
	if r.closed {
		r.mu.Unlock()
		for _, s := range streams {
			sherpa.DeleteOfflineStream(s)
		}
		return nil, fmt.Errorf("asr: recognizer closed")
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
		r.cond.Signal()
		r.mu.Unlock()
	}()

	results := make([]*Result, len(streams))
	for i, s := range streams {
		res := s.GetResult()
		duration := float64(len(chunks[i])) / float64(sampleRate)
		
		out := &Result{Duration: duration}
		if res != nil {
			cleanText := strings.ReplaceAll(res.Text, "\u2581", " ")
			seg := Segment{
				End:  duration,
				Text: strings.TrimSpace(cleanText),
			}
			if len(res.Timestamps) > 0 {
				n := min(len(res.Tokens), len(res.Timestamps))
				seg.Words = make([]Word, n)
				for j := range n {
					start := float64(res.Timestamps[j])
					end := start + 0.1
					if j+1 < n {
						if next := float64(res.Timestamps[j+1]); next > start {
							end = next
						}
					}
					seg.Words[j] = Word{Word: res.Tokens[j], Start: start, End: end}
				}
			}
			out.Segments = []Segment{seg}
		}
		results[i] = out
	}
	return results, nil
}

func (r *Recognizer) SupportedLanguages() []string { return nil }
func (r *Recognizer) ModelName() string            { return filepath.Base(r.Config.ModelPath) }

func (r *Recognizer) Close() error {
	if r == nil { return nil }
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
