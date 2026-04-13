package worker

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"time"

	"github.com/negbie/sittich/internal/config"
	"github.com/negbie/sittich/internal/pipeline"
	"github.com/negbie/sittich/internal/speech"
)

// Pool manages a pool of transcription workers. Each worker has its own
// private pipeline instance to process audio in isolation.
type Pool struct {
	workers      int
	queue        chan *Job
	engine       speech.Engine
	pipelineCfg  config.Pipeline
	ctx          context.Context
	cancel       context.CancelFunc
	wg           sync.WaitGroup
	busyCount    atomic.Int32
	debug        bool
	shutdownOnce sync.Once
}

// NewPool creates a new worker pool
func NewPool(workers int, queueSize int, engine speech.Engine, cfg config.Pipeline, debug bool, dataDir string) *Pool {
	ctx, cancel := context.WithCancel(context.Background())

	p := &Pool{
		workers:     workers,
		queue:       make(chan *Job, queueSize),
		engine:      engine,
		pipelineCfg: cfg,
		ctx:         ctx,
		cancel:      cancel,
		debug:       debug,
	}

	// Start workers
	for i := 0; i < workers; i++ {
		p.wg.Add(1)
		go p.worker(i)
	}

	return p
}

func (p *Pool) Submit(job *Job) error {
	// 1. Immediate non-blocking attempt for high-throughput
	select {
	case p.queue <- job:
		return nil
	default:
	}

	// 2. Queue is full: block while waiting for a worker to free up a slot OR 
	// until the request is cancelled by the HTTP client (timeout/disconnect).
	select {
	case p.queue <- job:
		return nil
	case <-job.Ctx.Done():
		return fmt.Errorf("queue full, request cancelled (max %d jobs)", cap(p.queue))
	}
}

func (p *Pool) QueueSize() int {
	return len(p.queue)
}

func (p *Pool) BusyWorkers() int {
	return int(p.busyCount.Load())
}

func (p *Pool) TotalWorkers() int {
	return p.workers
}

func (p *Pool) Shutdown() {
	p.shutdownOnce.Do(func() {
		p.cancel()     // Signal workers to stop
		close(p.queue) // Close job queue
		p.wg.Wait()    // Wait for all workers to finish

		// Now safe to close the shared engine
		if p.engine != nil {
			p.engine.Close()
		}
	})
}

func (p *Pool) worker(id int) {
	defer p.wg.Done()

	// 1. Initialise a private Pipeline for this worker.
	pipe, err := pipeline.NewPipeline(p.engine, p.pipelineCfg)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Worker %d: failed to initialise pipeline: %v\n", id, err)
		return
	}
	defer pipe.Close()

	for {
		select {
		case <-p.ctx.Done():
			return
		case job, ok := <-p.queue:
			if !ok {
				return
			}
			p.processJobWithPipeline(id, job, pipe)
		}
	}
}

func (p *Pool) processJobWithPipeline(workerID int, job *Job, pipe *pipeline.Pipeline) {
	p.busyCount.Add(1)
	defer p.busyCount.Add(-1)

	if p.debug {
		queueWait := time.Since(job.StartTime).Round(time.Millisecond)
		fileName := filepath.Base(job.FilePath)
		if fileName == "." || fileName == string(filepath.Separator) {
			fileName = "<memory>"
		}
		fmt.Fprintf(os.Stderr, "[Worker %d] Job %s start file=%s queue_wait=%s\n", workerID, job.ID, fileName, queueWait)
	}

	if job.FilePath != "" {
		defer os.Remove(job.FilePath)
	}

	startTime := time.Now()

	result, err := pipe.Process(job.Ctx, job.FilePath, float64(job.ChunkSize), job.SoxFlags...)
	if err != nil {
		if p.debug {
			fmt.Fprintf(os.Stderr, "[Worker %d] Job %s failed processing_time=%s err=%v\n", workerID, job.ID, time.Since(startTime).Round(time.Millisecond), err)
		}
		p.sendDone(job, JobDone{Err: fmt.Errorf("pipeline processing failed: %w", err)})
		return
	}

	processingTime := time.Since(startTime).Seconds()
	rtFactor := result.Duration / processingTime
	if rtFactor < 0 {
		rtFactor = 0
	}

	if p.debug {
		processingElapsed := time.Since(startTime).Round(time.Millisecond)
		totalElapsed := time.Since(job.StartTime).Round(time.Millisecond)
		fmt.Fprintf(os.Stderr, "[Worker %d] Job %s done processing_time=%s total_time=%s rtf=%.2f\n", workerID, job.ID, processingElapsed, totalElapsed, rtFactor)
	}

	p.sendDone(job, JobDone{
		Result: &JobResult{
			Duration:       result.Duration,
			ProcessingTime: processingTime,
			RealtimeFactor: rtFactor,
			Text:           result.FullText(),
			Segments:       result.Segments,
		},
	})
}

// sendDone delivers the job result or error back to the caller.
// It is thread-safe and ensures we don't block indefinitely if the 
// original request context was cancelled while the worker was busy.
func (p *Pool) sendDone(job *Job, done JobDone) {
	select {
	case job.Done <- done:
	case <-job.Ctx.Done():
		// Request was cancelled/timed out, caller is no longer listening.
	case <-p.ctx.Done():
		// Global pool shutdown.
	}
}
