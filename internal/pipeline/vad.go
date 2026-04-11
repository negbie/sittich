//go:build cgo

package pipeline

import (
	"fmt"
	"sync"

	sherpa "github.com/k2-fsa/sherpa-onnx-go/sherpa_onnx"
)

// VAD wraps a Silero voice-activity detector configuration.
type VAD struct {
	config sherpa.VadModelConfig

	mu     sync.Mutex
	closed bool
}

// SpeechSegment represents a contiguous region of detected speech.
type SpeechSegment struct {
	Start float64 // Start time in seconds.
	End   float64 // End time in seconds.
}

// NewVAD creates a VAD configuration wrapper and initializes detector ownership.
func NewVAD(modelPath string, threshold, minSilenceDuration, minSpeechDuration float32, numThreads int) (*VAD, error) {
	config := sherpa.VadModelConfig{
		SileroVad: sherpa.SileroVadModelConfig{
			Model:              modelPath,
			Threshold:          threshold,
			MinSilenceDuration: minSilenceDuration,
			MinSpeechDuration:  minSpeechDuration,
			WindowSize:         512,
		},
		SampleRate: 16000,
		NumThreads: numThreads,
		Debug:      0,
		Provider:   "cpu",
	}

	return &VAD{
		config: config,
	}, nil
}

// DetectSpeech runs a Silero VAD detector over the provided 16 kHz mono audio.
func (v *VAD) DetectSpeech(audioSamples []float32, sampleRate int) ([]SpeechSegment, error) {
	targetRate := int(v.config.SampleRate)
	if sampleRate != targetRate {
		return nil, fmt.Errorf("vad: expected %d Hz audio, got %d Hz", targetRate, sampleRate)
	}

	v.mu.Lock()
	if v.closed {
		v.mu.Unlock()
		return nil, fmt.Errorf("vad: detector is closed")
	}
	v.mu.Unlock()

	detector := sherpa.NewVoiceActivityDetector(&v.config, 30.0)
	if detector == nil {
		return nil, fmt.Errorf("vad: failed to create detector")
	}
	defer sherpa.DeleteVoiceActivityDetector(detector)

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

// Close marks the VAD as closed.
func (v *VAD) Close() {
	v.mu.Lock()
	defer v.mu.Unlock()
	if v.closed {
		return
	}
	v.closed = true
}
