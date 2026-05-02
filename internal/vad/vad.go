package vad

import (
	"fmt"

	sherpa "github.com/k2-fsa/sherpa-onnx-go/sherpa_onnx"
)

// Segment represents a detected speech region.
type Segment struct {
	// Start is the start time in seconds.
	Start float64
	// Samples contains the mono PCM data for this segment.
	Samples []float32
}

// Detector wraps the sherpa-onnx Voice Activity Detector.
type Detector struct {
	vad *sherpa.VoiceActivityDetector
}

// NewDetector creates a new VAD detector using the Silero model at the given path.
func NewDetector(modelPath string, sampleRate int) (*Detector, error) {
	config := &sherpa.VadModelConfig{
		SileroVad: sherpa.SileroVadModelConfig{
			Model:              modelPath,
			Threshold:          0.5,
			MinSilenceDuration: 0.5,
			MinSpeechDuration:  0.25,
			WindowSize:         512,
			MaxSpeechDuration:  20.0,
		},
		SampleRate: sampleRate,
		NumThreads: 1,
		Provider:   "cpu",
		Debug:      0,
	}

	vad := sherpa.NewVoiceActivityDetector(config, 300.0)
	if vad == nil {
		return nil, fmt.Errorf("failed to create VAD")
	}

	return &Detector{vad: vad}, nil
}

// Close releases the underlying VAD resources.
func (d *Detector) Close() {
	if d.vad != nil {
		sherpa.DeleteVoiceActivityDetector(d.vad)
		d.vad = nil
	}
}

// Segment extracts speech segments from the provided audio samples.
func (d *Detector) Segment(samples []float32, sampleRate int) ([]Segment, error) {
	if d.vad == nil {
		return nil, fmt.Errorf("detector closed")
	}

	var segments []Segment
	// Push in exactly 512-sample frames, which is optimal for the Silero VAD state machine.
	frameSize := 512
	for i := 0; i < len(samples); i += frameSize {
		end := i + frameSize
		if end > len(samples) {
			end = len(samples)
		}
		d.vad.AcceptWaveform(samples[i:end])
		for !d.vad.IsEmpty() {
			s := d.vad.Front()
			if s != nil {
				segments = append(segments, Segment{
					Start:   float64(s.Start) / float64(sampleRate),
					Samples: s.Samples,
				})
			}
			d.vad.Pop()
		}
	}

	d.vad.Flush()
	for !d.vad.IsEmpty() {
		s := d.vad.Front()
		if s != nil {
			segments = append(segments, Segment{
				Start:   float64(s.Start) / float64(sampleRate),
				Samples: s.Samples,
			})
		}
		d.vad.Pop()
	}

	return segments, nil
}
