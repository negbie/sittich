package asr

import (
	"context"
)

// Engine is the central abstraction for speech-to-text backends.
type Engine interface {
	// Transcribe processes raw PCM audio samples and returns a structured
	// transcription result. The audio slice must contain float32 samples
	// normalised to [-1, 1]. sampleRate is in Hz (e.g. 16000).
	Transcribe(ctx context.Context, audio []float32, sampleRate int, opts Options) (*Result, error)

	// TranscribeBatch processes multiple raw PCM audio chunks in parallel.
	TranscribeBatch(ctx context.Context, chunks [][]float32, sampleRate int, opts Options) ([]*Result, error)

	// SupportedLanguages returns the BCP-47 language codes the engine can handle.
	SupportedLanguages() []string

	// ModelName returns a human-readable identifier for the loaded model.
	ModelName() string

	// VADPath returns the path to the VAD model if configured.
	VADPath() string

	// Close releases all resources held by the engine.
	Close() error
}

// Options controls transcription behaviour on a per-request basis.
type Options struct {
	Language       string
	WordTimestamps bool
	Debug          bool
}

// Result holds the complete output of a transcription run.
type Result struct {
	Language   string
	Duration   float64
	Confidence float64
	Segments   []Segment
}

// FullText concatenates all segment texts separated by spaces.
func (r *Result) FullText() string {
	if r == nil || len(r.Segments) == 0 {
		return ""
	}
	var total int
	for i := range r.Segments {
		total += len(r.Segments[i].Text) + 1
	}
	buf := make([]byte, 0, total)
	for i := range r.Segments {
		if i > 0 {
			buf = append(buf, ' ')
		}
		buf = append(buf, r.Segments[i].Text...)
	}
	return string(buf)
}

// Segment represents a contiguous chunk of transcribed speech.
type Segment struct {
	ID    int
	Start float64
	End   float64
	Text  string
	Words []Word
}

// Word represents a single word with timing information.
type Word struct {
	Word  string
	Start float64
	End   float64
}
