package audio

import (
	"math"
)

// ConditionAudioSignal applies DSP conditioning to audio samples before ASR inference
func ConditionAudioSignal(samples []float32, targetPeak float32, sampleRate int, mode string) {
	if len(samples) == 0 {
		return
	}

	RemoveDCOffset(samples)

	switch mode {
	case "minimal":
		NormalizePeak(samples, targetPeak)
	case "gentle":
		ApplySlowAGC(samples, sampleRate)
		NormalizePeak(samples, targetPeak) // Pre-Gain Staging for consistent compressor reaction
		ApplySoftDRC(samples, sampleRate)
		NormalizeLoudness(samples, -18.0, targetPeak)
	case "aggressive":
		ApplySlowAGC(samples, sampleRate)
		NormalizePeak(samples, targetPeak) // Pre-Gain Staging for consistent compressor reaction
		ApplySoftDRC(samples, sampleRate)  // SoftDRC (2:1) for better phonetic integrity
		NormalizeLoudness(samples, -16.0, targetPeak)  // Final RMS loudness target
		ApplySafetyLimiter(samples, targetPeak)        // Brick-wall safety check
	default:
		NormalizePeak(samples, targetPeak)
	}
}

// RemoveDCOffset removes the mean from the signal to eliminate DC bias.
func RemoveDCOffset(samples []float32) {
	if len(samples) == 0 {
		return
	}
	var sum float64
	for _, s := range samples {
		sum += float64(s)
	}
	mean := float32(sum / float64(len(samples)))
	for i := range samples {
		samples[i] -= mean
	}
}

// ApplySlowAGC levels out long-term volume variations using a Dual-Loop Gated Leveler.
func ApplySlowAGC(samples []float32, sampleRate int) {
	const (
		targetRMS    = 0.10  // -20 dBFS logical target
		maxGain      = 8.0   // Maximum +18dB boost
		gateRMS      = 0.005 // -46 dBFS. Below this, assume silence.
		
		// Fast envelope for gate detection
		fastAttackSec  = 0.01 // 10ms
		fastReleaseSec = 0.05 // 50ms
		
		// Slow envelope for speech leveling
		slowAttackSec  = 0.2  // 200ms
		slowReleaseSec = 3.0  // 3s
	)

	fastAttCoeff := float64(math.Exp(-1.0 / (float64(fastAttackSec) * float64(sampleRate))))
	fastRelCoeff := float64(math.Exp(-1.0 / (float64(fastReleaseSec) * float64(sampleRate))))
	slowAttCoeff := float64(math.Exp(-1.0 / (float64(slowAttackSec) * float64(sampleRate))))
	slowRelCoeff := float64(math.Exp(-1.0 / (float64(slowReleaseSec) * float64(sampleRate))))

	var fastEnvSq float64
	var slowEnvSq float64 = targetRMS * targetRMS

	// Seed envelope to avoid harsh onset calculations
	seedLen := sampleRate / 10
	if seedLen > len(samples) {
		seedLen = len(samples)
	}
	var seedSum float64
	for i := 0; i < seedLen; i++ {
		seedSum += float64(samples[i]) * float64(samples[i])
	}
	if seedLen > 0 {
		fastEnvSq = seedSum / float64(seedLen)
		if fastEnvSq > slowEnvSq {
			slowEnvSq = fastEnvSq
		}
	}

	var smoothedGain float64 = 1.0
	gainSmoothCoeff := float64(math.Exp(-1.0 / (0.05 * float64(sampleRate)))) // Smooth gain over 50ms

	for i, s := range samples {
		inputSq := float64(s) * float64(s)

		// 1. Fast Envelope (Voice Activity Detector)
		if inputSq > fastEnvSq {
			fastEnvSq = fastAttCoeff*fastEnvSq + (1-fastAttCoeff)*inputSq
		} else {
			fastEnvSq = fastRelCoeff*fastEnvSq + (1-fastRelCoeff)*inputSq
		}

		// 2. Slow Envelope (Speech Level Tracker)
		if math.Sqrt(fastEnvSq) > gateRMS {
			// Speech active: update the slow envelope
			if fastEnvSq > slowEnvSq {
				slowEnvSq = slowAttCoeff*slowEnvSq + (1-slowAttCoeff)*fastEnvSq
			} else {
				slowEnvSq = slowRelCoeff*slowEnvSq + (1-slowRelCoeff)*fastEnvSq
			}
		}
		// If below gateRMS, we do nothing to slowEnvSq. It freezes cleanly.

		slowEnv := math.Sqrt(slowEnvSq)
		if slowEnv < 0.0001 {
			slowEnv = 0.0001
		}

		// 3. Target Gain
		targetGain := targetRMS / slowEnv
		if targetGain > maxGain {
			targetGain = maxGain
		}

		// 4. Gain Application
		smoothedGain = gainSmoothCoeff*smoothedGain + (1-gainSmoothCoeff)*targetGain
		samples[i] = float32(float64(s) * smoothedGain)
	}
}

// ApplySoftDRC applies a gentle 2:1 compressor.
func ApplySoftDRC(samples []float32, sampleRate int) {
	const (
		thresholdLin = 0.0316 // -30 dBFS
		ratio        = 2.0
		attackSec    = 0.005
		releaseSec   = 0.1
	)

	attackCoeff := float32(math.Exp(-1.0 / (attackSec * float64(sampleRate))))
	releaseCoeff := float32(math.Exp(-1.0 / (releaseSec * float64(sampleRate))))

	var envelope float32
	if len(samples) > 0 {
		envelope = float32(math.Abs(float64(samples[0])))
	}
	for i, s := range samples {
		input := float32(math.Abs(float64(s)))

		if input > envelope {
			envelope = attackCoeff*envelope + (1-attackCoeff)*input
		} else {
			envelope = releaseCoeff*envelope + (1-releaseCoeff)*input
		}

		var gainReduction float32 = 1.0
		if envelope > thresholdLin {
			gainReduction = (thresholdLin + (envelope-thresholdLin)/ratio) / envelope
		}

		samples[i] = s * gainReduction
	}
}

// NormalizePeak scales the signal so that the absolute peak is targetPeak.
func NormalizePeak(samples []float32, targetPeak float32) {
	var peak float32
	for _, s := range samples {
		abs := float32(math.Abs(float64(s)))
		if abs > peak {
			peak = abs
		}
	}

	if peak < 1e-7 || math.Abs(float64(peak-targetPeak)) < 1e-5 {
		return
	}

	multiplier := targetPeak / peak
	for i := range samples {
		samples[i] *= multiplier
	}
}

// NormalizeLoudness normalizes the audio to a target RMS loudness (in dBFS).
func NormalizeLoudness(samples []float32, targetDB float32, targetPeak float32) {
	if len(samples) == 0 {
		return
	}

	var sumSquares float64
	for _, s := range samples {
		sumSquares += float64(s) * float64(s)
	}
	rms := float32(math.Sqrt(sumSquares / float64(len(samples))))

	if rms < 0.001 {
		return
	}

	targetRMS := float32(math.Pow(10, float64(targetDB)/20.0))
	multiplier := targetRMS / rms

	if multiplier > 8.0 {
		multiplier = 8.0
	} else if multiplier < 0.1 {
		multiplier = 0.1
	}

	for i := range samples {
		val := samples[i] * multiplier
		if val > targetPeak {
			val = targetPeak
		} else if val < -targetPeak {
			val = -targetPeak
		}
		samples[i] = val
	}
}

// ApplySafetyLimiter hard-clips samples to [-targetPeak, targetPeak].
func ApplySafetyLimiter(samples []float32, targetPeak float32) {
	for i, s := range samples {
		if s > targetPeak {
			samples[i] = targetPeak
		} else if s < -targetPeak {
			samples[i] = -targetPeak
		}
	}
}
