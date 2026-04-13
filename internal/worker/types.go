package worker

import (
	"context"
	"time"

	"github.com/negbie/sittich/internal/speech"
)

// Job represents a transcription job
type Job struct {
	ID        string
	FilePath  string
	Format    string
	ChunkSize int
	SoxFlags  []string
	Done      chan JobDone
	StartTime time.Time
	Ctx       context.Context
}

// JobDone holds either a successful result or an error.
type JobDone struct {
	Result *JobResult
	Err    error
}

// JobResult holds the result of a transcription job
type JobResult struct {
	Duration       float64
	ProcessingTime float64
	RealtimeFactor float64
	Text           string
	Segments       []speech.Segment
}
