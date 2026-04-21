package audio

import (
	"math"
)

// ConditionAudioSignal prepares audio for ASR with an optimized, single-pass pipeline.
func ConditionAudioSignal(samples []float32, sampleRate int) []float32 {
	n := len(samples)
	if n == 0 {
		return samples
	}

	// Pass 1: Calculate DC offset (sum) and Peak amplitude simultaneously.
	// This reduces memory bandwidth usage by half compared to separate passes.
	var sum float64
	var peak float32

	for _, s := range samples {
		sum += float64(s)
		abs := s
		if abs < 0 {
			abs = -s
		}
		if abs > peak {
			peak = abs
		}
	}

	mean := float32(sum / float64(n))

	// Check for digital silence / underflow.
	// -60 dBFS = 0.001. If the peak (relative to mean) is below this, boost is pointless.
	const silenceThreshold = 0.001
	if peak <= silenceThreshold {
		for i := range samples {
			samples[i] = 0
		}
		return samples
	}

	// Calculate normalization scale if the signal exceeds unit range.
	// We only scale DOWN to prevent clipping; we never scale UP (boost).
	scale := float32(1.0)
	if peak > 1.0 {
		scale = 1.0 / peak
	}

	// Pass 2: Apply DC offset removal, scaling, and soft-clipping in one go.
	// We use math.Tanh for transparent, analog-style saturation.
	const drive = 1.5
	for i := range samples {
		// 1. Remove DC
		s := (samples[i] - mean)
		// 2. Scale down if necessary
		s *= scale
		// 3. Soft-clip
		if s > 1.0 {
			s = float32(math.Tanh(float64(s-1.0))) + 1.0
		} else if s < -1.0 {
			s = float32(math.Tanh(float64(s+1.0))) - 1.0
		}
	}

	return samples
}
