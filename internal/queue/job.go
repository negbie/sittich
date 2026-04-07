package queue

import (
	"sync/atomic"

	"github.com/negbie/sittich/internal/types"
)

// Status represents the lifecycle state of a queued job.
type Status int32

const (
	StatusPending Status = iota
	StatusRunning
	StatusDone
)

func (s Status) String() string {
	switch s {
	case StatusPending:
		return "pending"
	case StatusRunning:
		return "running"
	case StatusDone:
		return "done"
	default:
		return "unknown"
	}
}

// Job represents a single transcription unit of work.
type Job struct {
	ID     string
	status atomic.Int32
	Result *types.Result
	Error  error
	done   chan struct{}
}

func NewJob(id string) *Job {
	j := &Job{
		ID:   id,
		done: make(chan struct{}),
	}
	j.status.Store(int32(StatusPending))
	return j
}

func (j *Job) Status() Status {
	return Status(j.status.Load())
}

func (j *Job) Done() <-chan struct{} {
	return j.done
}

func (j *Job) setRunning() {
	j.status.Store(int32(StatusRunning))
}

func (j *Job) finish(result *types.Result, err error) {
	j.Result = result
	j.Error = err
	j.status.Store(int32(StatusDone))
	close(j.done)
}
