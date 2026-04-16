package asr

import (
	"context"
	"fmt"
	"os"
	"sync"

	"github.com/negbie/sittich/internal/speech"
)

// Dispatcher implements a bounded parallel ASR execution planner.
// It satisfies the speech.Engine interface by delegating to an underlying
// recognizer while managing concurrent execution of size-1 micro-batches.
type Dispatcher struct {
	engine speech.Engine
	sem    chan struct{}
	ctx    context.Context
	cancel context.CancelFunc
	debug  bool
}

// NewDispatcher creates a new ASR dispatcher that implements bounded 
// parallel execution of size-1 micro-batches.
func NewDispatcher(engine speech.Engine, workers int, debug bool) *Dispatcher {
	// Architecture recommends max 4 parallel decodes per request/instance
	// to avoid ONNX runtime penalties.
	maxParallel := workers
	if maxParallel > 4 {
		maxParallel = 4
	}
	if maxParallel < 1 {
		maxParallel = 1
	}

	ctx, cancel := context.WithCancel(context.Background())
	return &Dispatcher{
		engine: engine,
		sem:    make(chan struct{}, maxParallel),
		ctx:    ctx,
		cancel: cancel,
		debug:  debug,
	}
}

// Transcribe satisfies the speech.Engine interface.
func (d *Dispatcher) Transcribe(ctx context.Context, audio []float32, sampleRate int, opts speech.Options) (*speech.Result, error) {
	results, err := d.TranscribeBatch(ctx, [][]float32{audio}, sampleRate, opts)
	if err != nil {
		return nil, err
	}
	if len(results) == 0 {
		return nil, fmt.Errorf("dispatcher: no results returned")
	}
	return results[0], nil
}

// TranscribeBatch satisfies the speech.Engine interface. It implements
// bounded internal parallelism with micro-batch size 1.
func (d *Dispatcher) TranscribeBatch(ctx context.Context, chunks [][]float32, sampleRate int, opts speech.Options) ([]*speech.Result, error) {
	n := len(chunks)
	if n == 0 {
		return nil, nil
	}

	if d.debug {
		fmt.Fprintf(os.Stderr, "   [Dispatcher] dispatching_batch size=%d chunks=%d\n", n, n)
	}

	results := make([]*speech.Result, n)
	errs := make([]error, n)

	var wg sync.WaitGroup
	for i := range chunks {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()

			// Acquire concurrency slot
			select {
			case d.sem <- struct{}{}:
				defer func() { <-d.sem }()
			case <-ctx.Done():
				errs[idx] = ctx.Err()
				return
			case <-d.ctx.Done():
				errs[idx] = fmt.Errorf("dispatcher closed")
				return
			}

			// Execute size-1 micro-batch
			res, err := d.engine.Transcribe(ctx, chunks[idx], sampleRate, opts)
			results[idx] = res
			errs[idx] = err
		}(i)
	}

	wg.Wait()

	for _, err := range errs {
		if err != nil {
			return nil, err
		}
	}

	return results, nil
}

func (d *Dispatcher) SupportedLanguages() []string { return d.engine.SupportedLanguages() }
func (d *Dispatcher) ModelName() string            { return d.engine.ModelName() }
func (d *Dispatcher) Close() error {
	d.cancel()
	return d.engine.Close()
}
