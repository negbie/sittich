package config

// Server holds server configuration.
type Server struct {
	ListenAddr       string
	MaxUploadMB      int64
	Workers          int
	Debug            bool
	DefaultFormat    string
	DefaultChunkSize int
	Proxy            string
}

// ASR holds ASR configuration.
type ASR struct {
	ModelPath      string
	NumThreads     int
	MaxActive      int
	DecodingMethod string
	MaxActivePaths int
}

// Pipeline holds pipeline behaviour.
type Pipeline struct {
	// ChunkDuration is the maximum duration in seconds per chunk sent to the
	// engine. Callers are expected to provide this explicitly.
	ChunkDuration float64

	// ChunkOverlapDuration is the duration in seconds of overlap between
	// adjacent chunks to ensure continuity at boundaries.
	ChunkOverlapDuration float64

	// WordTimestamps requests word-level timing from the engine.
	WordTimestamps bool

	// Language is a BCP-47 hint passed to the engine (empty = auto-detect).
	Language string

	// Debug enables detailed console logging.
	Debug bool
}
