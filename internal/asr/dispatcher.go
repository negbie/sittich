package asr

import (
	"context"
	"fmt"
	"os"
	"sync"
	"time"

	"github.com/negbie/sittich/internal/speech"
)

// Dispatcher implements a global ASR job queue with cross-request batching.
// It satisfies the speech.Engine interface by delegating to an underlying
// recognizer while aggregating small transcription calls into larger batches.
type Dispatcher struct {
	engine    speech.Engine
	jobChan   chan *asrJob
	batchSize int
	window    time.Duration
	wg        sync.WaitGroup
	ctx       context.Context
	cancel    context.CancelFunc
	closeOnce sync.Once
	debug     bool
}

type asrJob struct {
	audio      []float32
	sampleRate int
	opts       speech.Options
	resultChan chan batchResult
	ctx        context.Context
}

type batchResult struct {
	res *speech.Result
	err error
}

// NewDispatcher creates a new global ASR dispatcher.
func NewDispatcher(engine speech.Engine, workers int, batchSize int, window time.Duration, debug bool) *Dispatcher {
	ctx, cancel := context.WithCancel(context.Background())
	d := &Dispatcher{
		engine:    engine,
		jobChan:   make(chan *asrJob, 128),
		batchSize: batchSize,
		window:    window,
		ctx:       ctx,
		cancel:    cancel,
		debug:     debug,
	}

	for i := 0; i < workers; i++ {
		d.wg.Add(1)
		go d.worker(i)
	}

	return d
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

// TranscribeBatch satisfies the speech.Engine interface. It decomposes the
// batch into individual jobs and collects them back after the dispatcher
// processes them (potentially alongside jobs from other requests).
func (d *Dispatcher) TranscribeBatch(ctx context.Context, chunks [][]float32, sampleRate int, opts speech.Options) ([]*speech.Result, error) {
	n := len(chunks)
	if n == 0 {
		return nil, nil
	}

	type pendingResult struct {
		ch  chan batchResult
		idx int
	}
	pending := make([]pendingResult, n)

	for i, chunk := range chunks {
		ch := make(chan batchResult, 1)
		job := &asrJob{
			audio:      chunk,
			sampleRate: sampleRate,
			opts:       opts,
			resultChan: ch,
			ctx:        ctx,
		}
		pending[i] = pendingResult{ch: ch, idx: i}

		select {
		case d.jobChan <- job:
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-d.ctx.Done():
			return nil, fmt.Errorf("dispatcher closed")
		}
	}

	results := make([]*speech.Result, n)
	for _, p := range pending {
		select {
		case res := <-p.ch:
			if res.err != nil {
				return nil, res.err
			}
			results[p.idx] = res.res
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-d.ctx.Done():
			return nil, fmt.Errorf("dispatcher closed")
		}
	}

	return results, nil
}

func (d *Dispatcher) SupportedLanguages() []string { return d.engine.SupportedLanguages() }
func (d *Dispatcher) ModelName() string          { return d.engine.ModelName() }
func (d *Dispatcher) Close() error {
	d.closeOnce.Do(func() {
		d.cancel()
		close(d.jobChan)
		d.wg.Wait()
	})
	return d.engine.Close()
}

func (d *Dispatcher) worker(id int) {
	defer d.wg.Done()

	for {
		var batch []*asrJob

		// 1. Wait for first job (blocking)
		select {
		case <-d.ctx.Done():
			return
		case job, ok := <-d.jobChan:
			if !ok {
				return
			}
			batch = append(batch, job)
		}

		// 2. Greedy Drain & Wait-and-Batch
		// First, greedily pull any jobs already waiting in the channel (zero latency)
	GreedyLoop:
		for len(batch) < d.batchSize {
			select {
			case nextJob, ok := <-d.jobChan:
				if !ok {
					break GreedyLoop
				}
				batch = append(batch, nextJob)
			default:
				break GreedyLoop
			}
		}

		// Second, if we still have room, wait for a short window
		if len(batch) < d.batchSize {
			timer := time.NewTimer(d.window)

		BatchLoop:
			for len(batch) < d.batchSize {
				select {
				case <-d.ctx.Done():
					timer.Stop()
					return
				case nextJob, ok := <-d.jobChan:
					if !ok {
						break BatchLoop
					}
					batch = append(batch, nextJob)
				case <-timer.C:
					break BatchLoop
				}
			}
			timer.Stop()
		}

		// 3. Execute Batch
		if d.debug {
			fmt.Fprintf(os.Stderr, "   [Dispatcher] worker=%d dispatching_batch size=%d\n", id, len(batch))
		}

		audioChunks := make([][]float32, len(batch))
		for i, j := range batch {
			audioChunks[i] = j.audio
		}

		// We assume all chunks in the batch have the same sample rate and options
		// as the first one. In sittich this is currently always true (16kHz).
		results, err := d.engine.TranscribeBatch(d.ctx, audioChunks, batch[0].sampleRate, batch[0].opts)

		// 4. Distribute Results
		for i, j := range batch {
			if err != nil {
				j.resultChan <- batchResult{err: err}
				continue
			}
			j.resultChan <- batchResult{res: results[i]}
		}
	}
}
