package asr

import (
	"context"
	"fmt"
	"os"
	"sync"
)

// Dispatcher implements a bounded parallel ASR execution planner.
type Dispatcher struct {
	engine Engine
	sem    chan struct{}
	ctx    context.Context
	cancel context.CancelFunc
}

// NewDispatcher creates a new ASR dispatcher that implements bounded 
// parallel execution of size-1 micro-batches.
func NewDispatcher(engine Engine, workers int, debug bool) *Dispatcher {
	maxParallel := workers
	if maxParallel < 1 {
		maxParallel = 1
	}

	ctx, cancel := context.WithCancel(context.Background())
	return &Dispatcher{
		engine: engine,
		sem:    make(chan struct{}, maxParallel),
		ctx:    ctx,
		cancel: cancel,
	}
}

// Transcribe satisfies the Engine interface.
func (d *Dispatcher) Transcribe(ctx context.Context, audio []float32, sampleRate int, opts Options) (*Result, error) {
	results, err := d.TranscribeBatch(ctx, [][]float32{audio}, sampleRate, opts)
	if err != nil {
		return nil, err
	}
	if len(results) == 0 {
		return &Result{}, nil
	}
	return results[0], nil
}

// TranscribeBatch satisfies the Engine interface. It implements
// bounded internal parallelism with micro-batch size 1.
func (d *Dispatcher) TranscribeBatch(ctx context.Context, chunks [][]float32, sampleRate int, opts Options) ([]*Result, error) {
	n := len(chunks)
	if n == 0 {
		return nil, nil
	}

	results := make([]*Result, n)
	errs := make([]error, n)
	var wg sync.WaitGroup

	for i := range chunks {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()

			select {
			case d.sem <- struct{}{}:
				defer func() { <-d.sem }()
			case <-ctx.Done():
				errs[idx] = ctx.Err()
				return
			}

			if opts.Debug {
				fmt.Fprintf(os.Stderr, "   [Dispatcher] chunk_idx=%d/%d start_transcribe\n", idx+1, n)
			}

			res, err := d.engine.Transcribe(ctx, chunks[idx], sampleRate, opts)
			if err != nil {
				errs[idx] = err
				return
			}
			results[idx] = res

			if opts.Debug {
				fmt.Fprintf(os.Stderr, "   [Dispatcher] chunk_idx=%d/%d done_transcribe\n", idx+1, n)
			}
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
func (d *Dispatcher) Close() error                 { d.cancel(); return d.engine.Close() }
