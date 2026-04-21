package asr

import (
	"fmt"
	"runtime/debug"
	"sync"

	sherpa "github.com/k2-fsa/sherpa-onnx-go/sherpa_onnx"
)

// Denoiser wraps the Sherpa-ONNX speech enhancement engine.
type Denoiser struct {
	denoiser *sherpa.OfflineSpeechDenoiser
	mu       sync.Mutex

	modelPath  string
	numThreads int
	lazy       bool
	inFlight   int
}

func NewDenoiser(modelPath string, numThreads int, lazy bool) (*Denoiser, error) {
	d := &Denoiser{
		modelPath:  modelPath,
		numThreads: numThreads,
		lazy:       lazy,
	}

	if !lazy {
		if err := d.ensureLoaded(); err != nil {
			return nil, err
		}
	}

	return d, nil
}

func (d *Denoiser) ensureLoaded() error {
	d.mu.Lock()
	defer d.mu.Unlock()

	d.inFlight++

	if d.denoiser != nil {
		return nil
	}

	config := sherpa.OfflineSpeechDenoiserConfig{
		Model: sherpa.OfflineSpeechDenoiserModelConfig{
			Gtcrn: sherpa.OfflineSpeechDenoiserGtcrnModelConfig{
				Model: d.modelPath,
			},
			Provider:   "cpu",
			NumThreads: int32(d.numThreads),
		},
	}

	denoiser := sherpa.NewOfflineSpeechDenoiser(&config)
	if denoiser == nil {
		return fmt.Errorf("asr: failed to initialize speech denoiser")
	}
	d.denoiser = denoiser
	return nil
}

// Run applies denoising to the given audio samples.
func (d *Denoiser) Run(samples []float32, sampleRate int) ([]float32, error) {
	if d == nil {
		return samples, nil
	}

	if err := d.ensureLoaded(); err != nil {
		return samples, err
	}
	defer d.release()

	d.mu.Lock()
	denoiser := d.denoiser
	d.mu.Unlock()

	if denoiser == nil {
		return samples, nil
	}

	result := denoiser.Run(samples, sampleRate)
	if result.Samples == nil {
		return nil, fmt.Errorf("asr: denoising failed")
	}

	return result.Samples, nil
}

func (d *Denoiser) release() {
	d.mu.Lock()
	defer d.mu.Unlock()

	d.inFlight--

	if !d.lazy {
		return
	}

	if d.inFlight == 0 && d.denoiser != nil {
		sherpa.DeleteOfflineSpeechDenoiser(d.denoiser)
		d.denoiser = nil
		debug.FreeOSMemory()
		trimCHeap()
	}
}

func (d *Denoiser) Close() {
	if d == nil {
		return
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.denoiser != nil {
		sherpa.DeleteOfflineSpeechDenoiser(d.denoiser)
		d.denoiser = nil
	}
}
