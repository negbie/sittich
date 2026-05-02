package audio

import (
	"math"
)

func ConditionAudioSignal(samples []float32, sampleRate int) {
	if len(samples) == 0 || sampleRate <= 0 {
		return
	}

	if !validateSamples(samples) {
		return
	}

	normalizePeak(samples, 0.90)
}

func validateSamples(samples []float32) bool {
	for _, s := range samples {
		if math.IsNaN(float64(s)) || math.IsInf(float64(s), 0) {
			return false
		}
	}
	return true
}

func normalizePeak(samples []float32, targetPeak float32) {
	if len(samples) == 0 {
		return
	}

	var maxPeak float32
	for _, s := range samples {
		absS := float32(math.Abs(float64(s)))
		if absS > maxPeak {
			maxPeak = absS
		}
	}

	if maxPeak < 1e-6 {
		return
	}

	multiplier := targetPeak / maxPeak
	for i := range samples {
		samples[i] *= multiplier
	}
}
