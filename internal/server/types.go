package server

import (
	"context"

	"github.com/negbie/sittich/internal/speech"
)

// TranscribeRequest represents a transcription request
type TranscribeRequest struct {
	URL       string   `json:"url,omitempty"`
	Base64    string   `json:"base64,omitempty"`
	Format    string   `json:"format"`     // text, json, vtt
	ChunkSize int      `json:"chunk_size"` // seconds
	SoxFlags  []string `json:"sox_flags"`  // optional additional sox effects
}

// TranscribeResponse represents a transcription response
type TranscribeResponse struct {
	Success        bool            `json:"success"`
	Error          string          `json:"error,omitempty"`
	Duration       float64         `json:"duration_seconds"`
	ProcessingTime float64         `json:"processing_time_seconds"`
	RealtimeFactor float64         `json:"realtime_factor"`
	Text           string          `json:"text"`
	Segments       []speech.Segment `json:"segments,omitempty"`
}

// HealthResponse represents a health check response
type HealthResponse struct {
	Status      string `json:"status"`
	ModelLoaded bool   `json:"model_loaded"`
	Version     string `json:"version"`
	Uptime      string `json:"uptime"`
	Workers     int    `json:"workers"`
	BusyWorkers int    `json:"busy_workers"`
	Proxy       string `json:"proxy,omitempty"`
}

// Recognizer is the interface for ASR recognizer
type Recognizer interface {
	Transcribe(ctx context.Context, audio []float32, sampleRate int, opts speech.Options) (*speech.Result, error)
	Close() error
}
