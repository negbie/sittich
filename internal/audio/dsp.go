package audio

import (
	"math"
)

// ConditionAudioSignal applies DSP conditioning to audio samples before ASR inference
func ConditionAudioSignal(samples []float32, targetPeak float32, sampleRate int, mode string) {
	if len(samples) == 0 {
		return
	}

	removeDCOffset(samples)

	switch mode {
	case "minimal":
		normalizePeak(samples, targetPeak)
	case "gentle":
		normalizePeak(samples, targetPeak)
		applySoftDRC(samples, sampleRate)
		normalizeLoudness(samples, -18.0)
		applySoftLimiter(samples)
	case "aggressive":
		applyPreEmphasis(samples, 0.20)
		normalizePeak(samples, targetPeak)
		applyDRC(samples, sampleRate)
		applyNoiseGate(samples, sampleRate, 0.001)
		normalizeLoudness(samples, -16)
		applySoftLimiter(samples)
	default:
	}

}
func removeDCOffset(samples []float32) {
	var sum float64
	for _, s := range samples {
		sum += float64(s)
	}
	mean := float32(sum / float64(len(samples)))
	for i := range samples {
		samples[i] -= mean
	}
}

func applyPreEmphasis(samples []float32, alpha float32) {
	for i := len(samples) - 1; i > 0; i-- {
		samples[i] = samples[i] - alpha*samples[i-1]
	}
}

// applySoftDRC applies a gentle 2:1 compressor at -30 dBFS for leveling
// varied input without squashing natural speech dynamics.
func applySoftDRC(samples []float32, sampleRate int) {
	const (
		thresholdDB = -30.0
		ratio       = 2.0
		kneeDB      = 6.0 // 6dB soft knee
		attackSec   = 0.005
		releaseSec  = 0.1
	)
	attackCoeff := float32(math.Exp(-1.0 / (attackSec * float64(sampleRate))))
	releaseCoeff := float32(math.Exp(-1.0 / (releaseSec * float64(sampleRate))))

	var envelope float32
	for i, s := range samples {
		input := float32(math.Abs(float64(s)))
		if input > envelope {
			envelope = attackCoeff*envelope + (1-attackCoeff)*input
		} else {
			envelope = releaseCoeff*envelope + (1-releaseCoeff)*input
		}

		if envelope < 1e-6 {
			continue
		}

		envDB := 20.0 * math.Log10(float64(envelope))
		var gainReductionDB float32
		if envDB > thresholdDB+(float64(kneeDB)/2.0) {
			gainReductionDB = float32((thresholdDB - envDB) * (1.0 - 1.0/ratio))
		} else if envDB > thresholdDB-(float64(kneeDB)/2.0) {
			// Soft-knee region
			diff := envDB - (thresholdDB - float64(kneeDB)/2.0)
			gainReductionDB = float32(-0.5 * (1.0 - 1.0/ratio) * diff * diff / float64(kneeDB))
		}

		gainReduction := float32(math.Pow(10, float64(gainReductionDB)/20.0))
		samples[i] = s * gainReduction
	}
}

// applyDRC applies an aggressive compressor with soft-knee.
func applyDRC(samples []float32, sampleRate int) {
	const (
		thresholdDB = -24.0
		ratio       = 1.8
		kneeDB      = 6.0
		attackSec   = 0.010
		releaseSec  = 0.075
	)
	attackCoeff := float32(math.Exp(-1.0 / (attackSec * float64(sampleRate))))
	releaseCoeff := float32(math.Exp(-1.0 / (releaseSec * float64(sampleRate))))

	var envelope float32
	for i, s := range samples {
		input := float32(math.Abs(float64(s)))
		if input > envelope {
			envelope = attackCoeff*envelope + (1-attackCoeff)*input
		} else {
			envelope = releaseCoeff*envelope + (1-releaseCoeff)*input
		}

		if envelope < 1e-6 {
			continue
		}

		envDB := 20.0 * math.Log10(float64(envelope))
		var gainReductionDB float32
		if envDB > thresholdDB+(float64(kneeDB)/2.0) {
			gainReductionDB = float32((thresholdDB - envDB) * (1.0 - 1.0/ratio))
		} else if envDB > thresholdDB-(float64(kneeDB)/2.0) {
			diff := envDB - (thresholdDB - float64(kneeDB)/2.0)
			gainReductionDB = float32(-0.5 * (1.0 - 1.0/ratio) * diff * diff / float64(kneeDB))
		}

		gainReduction := float32(math.Pow(10, float64(gainReductionDB)/20.0))
		samples[i] = s * gainReduction
	}
}

func normalizePeak(samples []float32, targetPeak float32) {
	var peak float32
	for _, s := range samples {
		abs := float32(math.Abs(float64(s)))
		if abs > peak {
			peak = abs
		}
	}

	if peak == 0 || math.Abs(float64(peak-targetPeak)) < 0.01 {
		return
	}

	multiplier := targetPeak / peak
	for i := range samples {
		samples[i] *= multiplier
	}
}

// normalizeLoudness normalizes the audio to a target RMS loudness (in dBFS),
// keeping typical ASR models in their sweet spot (e.g. -16 to -20 dBFS).
func normalizeLoudness(samples []float32, targetDB float32) {
	if len(samples) == 0 {
		return
	}

	var sumSquares float64
	for _, s := range samples {
		sumSquares += float64(s) * float64(s)
	}
	rms := float32(math.Sqrt(sumSquares / float64(len(samples))))

	// Silence protection: if signal is below -60 dBFS RMS, assume silence.
	if rms < 0.001 {
		return
	}

	targetRMS := float32(math.Pow(10, float64(targetDB)/20.0))
	multiplier := targetRMS / rms

	// Gain limits: Never boost more than 10x (+20dB) to avoid blasting noise.
	// Never reduce more than 10x (-20dB) to prevent complete silencing.
	if multiplier > 10.0 {
		multiplier = 10.0
	} else if multiplier < 0.1 {
		multiplier = 0.1
	}

	for i := range samples {
		val := samples[i] * multiplier
		samples[i] = val
	}
}

// applySoftLimiter applies a suave, continuous soft-knee clipper to keep signal <= 0.98.
func applySoftLimiter(samples []float32) {
	const threshold = 0.90
	const margin = 0.99 - threshold // Leave a tiny bit of headroom (-0.08 dB)

	for i, s := range samples {
		sign := float32(1.0)
		absS := s
		if s < 0 {
			sign = -1.0
			absS = -s
		}

		if absS <= threshold {
			continue
		}

		// Continuous soft-knee formula: y = T + (M-T) * tanh((x-T)/(M-T))
		// This is perfectly smooth (C-infinity) at s=threshold and asymptotes to 1.0.
		res := threshold + margin*float32(math.Tanh(float64((absS-threshold)/margin)))
		samples[i] = sign * res
	}
}

// applyNoiseGate applies a smooth noise gate using an RMS-based envelope.
func applyNoiseGate(samples []float32, sampleRate int, threshold float32) {
	const (
		attackSec  = 0.01 // 10ms
		releaseSec = 0.1  // 100ms
	)
	attackCoeff := float32(math.Exp(-1.0 / (attackSec * float64(sampleRate))))
	releaseCoeff := float32(math.Exp(-1.0 / (releaseSec * float64(sampleRate))))

	var gateGain float32 = 1.0
	var envelope float32

	for i, s := range samples {
		input := float32(math.Abs(float64(s)))
		if input > envelope {
			envelope = attackCoeff*envelope + (1-attackCoeff)*input
		} else {
			envelope = releaseCoeff*envelope + (1-releaseCoeff)*input
		}

		targetGain := float32(1.0)
		if envelope < threshold {
			targetGain = 0.0
		}

		if targetGain > gateGain {
			gateGain = attackCoeff*gateGain + (1-attackCoeff)*targetGain
		} else {
			gateGain = releaseCoeff*gateGain + (1-releaseCoeff)*targetGain
		}

		samples[i] *= gateGain
	}
}
