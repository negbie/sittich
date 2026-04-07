package audio

// Resample converts audio samples from one sample rate to another using
// linear interpolation. This is adequate for speech signals where the
// target rate is typically 16 kHz. For music or high-fidelity audio a
// windowed-sinc resampler would be preferable.
//
// If fromRate == toRate the original slice is returned without copying.
func Resample(samples []float32, fromRate, toRate int) []float32 {
	if fromRate == toRate || len(samples) == 0 {
		return samples
	}

	ratio := float64(toRate) / float64(fromRate)
	outLen := int(float64(len(samples)) * ratio)
	if outLen == 0 {
		return nil
	}

	out := make([]float32, outLen)
	srcLast := float64(len(samples) - 1)

	for i := range out {
		// Map the output index back to the source timeline.
		srcIdx := float64(i) / ratio
		if srcIdx >= srcLast {
			out[i] = samples[len(samples)-1]
			continue
		}

		// Integer and fractional parts for linear interpolation.
		idx0 := int(srcIdx)
		frac := float32(srcIdx - float64(idx0))
		out[i] = samples[idx0]*(1.0-frac) + samples[idx0+1]*frac
	}

	return out
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
