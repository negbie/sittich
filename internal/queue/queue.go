package queue

import (
	"context"
	"fmt"
	"sync"

	"github.com/negbie/sittich/internal/types"
)

// Queue provides bounded concurrency for transcription jobs. It uses a
// semaphore (buffered channel) to limit the number of goroutines that can
// execute transcription work simultaneously.
type Queue struct {
	// sem is a counting semaphore implemented as a buffered channel.
	sem chan struct{}

	// activeJobs tracks the number of in-flight jobs.
	activeJobs sync.WaitGroup
}

// NewQueue creates a queue that allows at most maxConcurrent jobs to run
// in parallel. maxConcurrent must be >= 1.
func NewQueue(maxConcurrent int) *Queue {
	if maxConcurrent < 1 {
		maxConcurrent = 1
	}
	return &Queue{
		sem: make(chan struct{}, maxConcurrent),
	}
}

// Submit enqueues a transcription function for execution and blocks until
// the function completes or the context is cancelled.
func (q *Queue) Submit(ctx context.Context, fn func() (*types.Result, error)) (*types.Result, error) {
	select {
	case q.sem <- struct{}{}:
	case <-ctx.Done():
		return nil, fmt.Errorf("queue: context cancelled while waiting for worker slot: %w", ctx.Err())
	}

	q.activeJobs.Add(1)

	defer func() {
		<-q.sem
		q.activeJobs.Done()
	}()

	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("queue: context cancelled after acquiring slot: %w", err)
	}

	return q.execute(fn)
}

func (q *Queue) execute(fn func() (*types.Result, error)) (result *types.Result, err error) {
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("queue: panic recovered during job execution: %v", r)
			result = nil
		}
	}()

	return fn()
}

// Drain blocks until all in-flight jobs complete.
func (q *Queue) Drain(ctx context.Context) error {
	done := make(chan struct{})
	go func() {
		q.activeJobs.Wait()
		close(done)
	}()

	select {
	case <-done:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (q *Queue) Len() int {
	return len(q.sem)
}

func (q *Queue) Cap() int {
	return cap(q.sem)
}
