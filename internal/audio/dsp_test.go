package audio

import (
	"math"
	"testing"
)

func TestRemoveDCOffset(t *testing.T) {
	samples := []float32{0.5, 0.7, 0.9} // Mean is 0.7
	RemoveDCOffset(samples)

	var sum float64
	for _, s := range samples {
		sum += float64(s)
	}
	if math.Abs(sum) > 1e-6 {
		t.Errorf("expected sum to be 0, got %f", sum)
	}
}

func TestNormalizePeak(t *testing.T) {
	samples := []float32{0.1, -0.5, 0.2}
	NormalizePeak(samples, 1.0)

	var peak float32
	for _, s := range samples {
		abs := float32(math.Abs(float64(s)))
		if abs > peak {
			peak = abs
		}
	}

	if math.Abs(float64(peak-1.0)) > 1e-6 {
		t.Errorf("expected peak to be 1.0, got %f", peak)
	}
}

func TestApplySafetyLimiter(t *testing.T) {
	samples := []float32{1.5, -2.0, 0.5}
	ApplySafetyLimiter(samples, 1.0)

	for _, s := range samples {
		if s > 1.0 || s < -1.0 {
			t.Errorf("sample %f outside [-1, 1]", s)
		}
	}
}

func TestNormalizeLoudness(t *testing.T) {
	samples := make([]float32, 16000)
	for i := range samples {
		samples[i] = 0.05 // low constant level
	}
	
	NormalizeLoudness(samples, -16.0, 1.0)
	
	var sumSquares float64
	for _, s := range samples {
		sumSquares += float64(s) * float64(s)
	}
	rms := math.Sqrt(sumSquares / float64(len(samples)))
	targetRMS := math.Pow(10, -16.0/20.0)
	
	if math.Abs(rms-targetRMS) > 0.01 {
		t.Errorf("expected RMS around %f, got %f", targetRMS, rms)
	}
}
