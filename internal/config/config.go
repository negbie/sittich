package config

// Default values for various configuration options.
const (
	DefaultListenAddr     = ":5092"
	DefaultChunkSize      = 40
	DefaultChunkOverlap   = 1.0
	DefaultChunkMinTail   = 1.5
	DefaultFormat         = "text"
	DefaultMaxActivePaths = 2
	DefaultDecodingMethod = "greedy_search"
	DefaultWorkers        = 4
	DefaultMaxUploadMB    = 32
	DefaultFixedScale     = 1.5
	DefaultNumThreads     = 2
	DefaultMaxQueueSize   = 10
)

// Server holds server configuration.
type Server struct {
	ListenAddr       string
	MaxUploadMB      int64
	Workers          int
	MaxQueueSize     int
	Debug            bool
	DefaultFormat    string
	DefaultChunkSize int
}

// ASR holds ASR configuration.
type ASR struct {
	ModelPath      string
	NumThreads     int
	DecodingMethod string
	MaxActivePaths int
	FixedScale     float32
}

// Pipeline holds pipeline behaviour.
type Pipeline struct {
	// ChunkDuration is the maximum duration in seconds per chunk sent to the
	// engine. Callers are expected to provide this explicitly.
	ChunkDuration float64

	// ChunkOverlapDuration is the duration in seconds of overlap between
	// adjacent chunks to ensure continuity at boundaries.
	ChunkOverlapDuration float64

	// ChunkMinTailDuration is the minimum tail duration used when balancing
	// oversized speech segments into smaller chunks.
	ChunkMinTailDuration float64

	// WordTimestamps requests word-level timing from the engine.
	WordTimestamps bool

	// Language is a BCP-47 hint passed to the engine (empty = auto-detect).
	Language string

	// Debug enables detailed console logging.
	Debug bool
}
