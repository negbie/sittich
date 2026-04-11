//go:build !cgo

package pipeline

import "fmt"

// SpeechSegment represents a contiguous region of detected speech.
type SpeechSegment struct {
	Start float64 // Start time in seconds.
	End   float64 // End time in seconds.
}

// VAD is a stub that always returns an error when CGo is disabled.
type VAD struct{}

// NewVAD returns an error indicating that VAD requires a CGo build.
func NewVAD(_ string, _, _, _ float32, _ int) (*VAD, error) {
	return nil, fmt.Errorf("vad: not available (built without CGo)")
}

// DetectSpeech is a stub that always returns an error.
func (v *VAD) DetectSpeech(_ []float32, _ int) ([]SpeechSegment, error) {
	return nil, fmt.Errorf("vad: not available (built without CGo)")
}

// Close is a no-op in the stub.
func (v *VAD) Close() {}
