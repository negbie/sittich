package audio

import "math"

// Resample converts audio samples from one sample rate to another.
// It applies a lightweight low-pass filter before downsampling and uses
// cubic interpolation for smoother reconstruction than linear interpolation.
//
// If fromRate == toRate the original slice is returned without copying.
func Resample(samples []float32, fromRate, toRate int) []float32 {
	if fromRate == toRate || len(samples) == 0 {
		return samples
	}
	if fromRate <= 0 || toRate <= 0 {
		return nil
	}

	filtered := samples
	if toRate < fromRate {
		filtered = lowPassFilter(samples, fromRate, toRate)
	}

	ratio := float64(toRate) / float64(fromRate)
	outLen := int(math.Round(float64(len(filtered)) * ratio))
	if outLen <= 0 {
		return nil
	}

	out := make([]float32, outLen)
	srcLast := len(filtered) - 1

	for i := range out {
		srcIdx := float64(i) / ratio
		if srcIdx <= 0 {
			out[i] = filtered[0]
			continue
		}
		if srcIdx >= float64(srcLast) {
			out[i] = filtered[srcLast]
			continue
		}

		idx1 := int(srcIdx)
		frac := float32(srcIdx - float64(idx1))

		idx0 := idx1 - 1
		if idx0 < 0 {
			idx0 = 0
		}
		idx2 := idx1 + 1
		if idx2 > srcLast {
			idx2 = srcLast
		}
		idx3 := idx1 + 2
		if idx3 > srcLast {
			idx3 = srcLast
		}

		out[i] = cubicInterpolate(
			filtered[idx0],
			filtered[idx1],
			filtered[idx2],
			filtered[idx3],
			frac,
		)
	}

	return out
}

func lowPassFilter(samples []float32, fromRate, toRate int) []float32 {
	if len(samples) == 0 {
		return nil
	}

	const taps = 15
	out := make([]float32, len(samples))

	cutoff := 0.5 * float64(toRate) / float64(fromRate)
	if cutoff > 0.5 {
		cutoff = 0.5
	}
	if cutoff <= 0 {
		copy(out, samples)
		return out
	}

	center := taps / 2
	coeffs := make([]float64, taps)
	var sum float64

	for i := 0; i < taps; i++ {
		n := float64(i - center)
		var sinc float64
		if n == 0 {
			sinc = 2 * cutoff
		} else {
			sinc = math.Sin(2*math.Pi*cutoff*n) / (math.Pi * n)
		}

		window := 0.54 - 0.46*math.Cos(2*math.Pi*float64(i)/float64(taps-1))
		coeffs[i] = sinc * window
		sum += coeffs[i]
	}

	if sum != 0 {
		for i := range coeffs {
			coeffs[i] /= sum
		}
	}

	for i := range samples {
		var acc float64
		for j := 0; j < taps; j++ {
			idx := i + j - center
			if idx < 0 {
				idx = 0
			} else if idx >= len(samples) {
				idx = len(samples) - 1
			}
			acc += float64(samples[idx]) * coeffs[j]
		}
		out[i] = float32(acc)
	}

	return out
}

func cubicInterpolate(y0, y1, y2, y3, t float32) float32 {
	a0 := -0.5*y0 + 1.5*y1 - 1.5*y2 + 0.5*y3
	a1 := y0 - 2.5*y1 + 2*y2 - 0.5*y3
	a2 := -0.5*y0 + 0.5*y2
	a3 := y1

	return ((a0*t+a1)*t+a2)*t + a3
}

// ToMono mixes multi-channel interleaved audio down to a single channel by
// averaging all channels for each sample frame.
//
// If channels <= 1 the original slice is returned unchanged.
// If len(samples) is not evenly divisible by channels, trailing samples that
// do not form a complete frame are discarded.
func ToMono(samples []float32, channels int) []float32 {
	if channels <= 1 || len(samples) == 0 {
		return samples
	}

	frames := len(samples) / channels
	mono := make([]float32, frames)
	inv := 1.0 / float32(channels)

	for i := 0; i < frames; i++ {
		var sum float32
		base := i * channels
		for ch := 0; ch < channels; ch++ {
			sum += samples[base+ch]
		}
		mono[i] = sum * inv
	}

	return mono
}
