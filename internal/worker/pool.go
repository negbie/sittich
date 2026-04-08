package worker

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"time"

	"github.com/negbie/sittich/internal/asr"
	"github.com/negbie/sittich/internal/models"
	"github.com/negbie/sittich/internal/pipeline"
	"github.com/negbie/sittich/internal/server"
)

// Pool manages a pool of transcription workers. Each worker has its own
// private Pipeline/VAD instance to process audio in isolation.
type Pool struct {
	workers      int
	queue        chan *server.Job
	recognizer   *asr.Recognizer
	pipelineCfg  pipeline.PipelineConfig
	ctx          context.Context
	cancel       context.CancelFunc
	wg           sync.WaitGroup
	busyCount    atomic.Int32
	debug        bool
	shutdownOnce sync.Once
}

// NewPool creates a new worker pool
func NewPool(workers int, queueSize int, recognizer *asr.Recognizer, cfg pipeline.PipelineConfig, debug bool, dataDir string) *Pool {
	ctx, cancel := context.WithCancel(context.Background())

	vadPath, _ := models.GetVADPath(dataDir)
	cfg.VADModelPath = vadPath
	cfg.VADEnabled = cfg.VADEnabled && vadPath != ""

	p := &Pool{
		workers:    workers,
		queue:      make(chan *server.Job, queueSize),
		recognizer: recognizer,
		pipelineCfg: cfg,
		ctx:    ctx,
		cancel: cancel,
		debug:  debug,
	}

	// Start workers
	for i := 0; i < workers; i++ {
		p.wg.Add(1)
		go p.worker(i)
	}

	return p
}

func (p *Pool) Submit(job *server.Job) error {
	select {
	case p.queue <- job:
		return nil
	default:
		return fmt.Errorf("queue full (max %d jobs)", cap(p.queue))
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
		p.cancel()
		close(p.queue)
		p.wg.Wait()

		// Explicitly close the shared recognizer once all workers are done
		if p.recognizer != nil {
			p.recognizer.Close()
		}
	})
}

func (p *Pool) worker(id int) {
	defer p.wg.Done()

	// 1. Initialise a private Pipeline for this worker.
	pipe, err := pipeline.NewPipeline(p.recognizer, p.pipelineCfg)
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

func (p *Pool) processJobWithPipeline(workerID int, job *server.Job, pipe *pipeline.Pipeline) {
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

	result, err := pipe.Process(job.Ctx, job.FilePath, float64(job.ChunkSize))
	if err != nil {
		if p.debug {
			fmt.Fprintf(os.Stderr, "[Worker %d] Job %s failed processing_time=%s err=%v\n", workerID, job.ID, time.Since(startTime).Round(time.Millisecond), err)
		}
		select {
		case job.Error <- fmt.Errorf("pipeline processing failed: %w", err):
		case <-job.Ctx.Done():
		case <-p.ctx.Done():
		}
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

	select {
	case job.Result <- server.JobResult{
		Duration:       result.Duration,
		ProcessingTime: processingTime,
		RealtimeFactor: rtFactor,
		Text:           result.FullText(),
		Segments:       result.Segments,
	}:
	case <-job.Ctx.Done():
	case <-p.ctx.Done():
	}
}
