//go:build cgo

package pipeline

import (
	"fmt"
	"sync"

	sherpa "github.com/k2-fsa/sherpa-onnx-go/sherpa_onnx"
)

// VAD wraps a Silero voice-activity detector configuration and explicitly owns
// a bounded set of reusable detector instances.
type VAD struct {
	config    sherpa.VadModelConfig
	detectors chan *sherpa.VoiceActivityDetector

	mu     sync.Mutex
	all    []*sherpa.VoiceActivityDetector
	closed bool
}

// SpeechSegment represents a contiguous region of detected speech.
type SpeechSegment struct {
	Start float64 // Start time in seconds.
	End   float64 // End time in seconds.
}

// NewVAD creates a VAD configuration wrapper and initializes detector ownership.
func NewVAD(modelPath string) (*VAD, error) {
	config := sherpa.VadModelConfig{
		SileroVad: sherpa.SileroVadModelConfig{
			Model:              modelPath,
			Threshold:          0.5,
			MinSilenceDuration: 0.5,
			MinSpeechDuration:  0.25,
			WindowSize:         512,
		},
		SampleRate: 16000,
		NumThreads: 1,
		Debug:      0,
		Provider:   "cpu",
	}

	return &VAD{
		config:    config,
		detectors: make(chan *sherpa.VoiceActivityDetector, 8),
	}, nil
}

func (v *VAD) acquireDetector() (*sherpa.VoiceActivityDetector, error) {
	v.mu.Lock()
	if v.closed {
		v.mu.Unlock()
		return nil, fmt.Errorf("vad: detector is closed")
	}
	v.mu.Unlock()

	select {
	case det := <-v.detectors:
		if det == nil {
			return nil, fmt.Errorf("vad: acquired nil detector")
		}
		return det, nil
	default:
		det := sherpa.NewVoiceActivityDetector(&v.config, 30.0)
		if det == nil {
			return nil, fmt.Errorf("vad: failed to create detector")
		}

		v.mu.Lock()
		defer v.mu.Unlock()
		if v.closed {
			sherpa.DeleteVoiceActivityDetector(det)
			return nil, fmt.Errorf("vad: detector is closed")
		}
		v.all = append(v.all, det)
		return det, nil
	}
}

func (v *VAD) releaseDetector(detector *sherpa.VoiceActivityDetector) {
	if detector == nil {
		return
	}

	detector.Reset()

	v.mu.Lock()
	closed := v.closed
	v.mu.Unlock()

	if closed {
		return
	}

	select {
	case v.detectors <- detector:
	default:
	}
}

// DetectSpeech runs a Silero VAD detector over the provided 16 kHz mono audio.
func (v *VAD) DetectSpeech(audioSamples []float32, sampleRate int) ([]SpeechSegment, error) {
	targetRate := int(v.config.SampleRate)
	if sampleRate != targetRate {
		return nil, fmt.Errorf("vad: expected %d Hz audio, got %d Hz", targetRate, sampleRate)
	}

	detector, err := v.acquireDetector()
	if err != nil {
		return nil, err
	}
	defer v.releaseDetector(detector)

	windowSize := int(v.config.SileroVad.WindowSize)
	if windowSize <= 0 {
		windowSize = 512
	}

	// 2. Feed audio in windows
	for i := 0; i+windowSize <= len(audioSamples); i += windowSize {
		window := audioSamples[i : i+windowSize]
		detector.AcceptWaveform(window)
	}

	remainder := len(audioSamples) % windowSize
	if remainder > 0 {
		detector.AcceptWaveform(audioSamples[len(audioSamples)-remainder:])
	}

	detector.Flush()

	// 3. Extract segments
	var segments []SpeechSegment
	totalSamples := len(audioSamples)
	for !detector.IsEmpty() {
		seg := detector.Front()
		detector.Pop()

		startSample := seg.Start
		if startSample < 0 {
			startSample = 0
		}
		if startSample > totalSamples {
			startSample = totalSamples
		}

		endSample := seg.Start + len(seg.Samples)
		if endSample < startSample {
			endSample = startSample
		}
		if endSample > totalSamples {
			endSample = totalSamples
		}

		startSec := float64(startSample) / float64(sampleRate)
		endSec := float64(endSample) / float64(sampleRate)

		segments = append(segments, SpeechSegment{
			Start: startSec,
			End:   endSec,
		})
	}

	return segments, nil
}

// Close explicitly destroys all owned native detectors.
func (v *VAD) Close() {
	v.mu.Lock()
	if v.closed {
		v.mu.Unlock()
		return
	}
	v.closed = true
	all := v.all
	v.all = nil
	close(v.detectors)
	v.mu.Unlock()

	for range v.detectors {
	}

	for _, det := range all {
		if det != nil {
			sherpa.DeleteVoiceActivityDetector(det)
		}
	}
}
