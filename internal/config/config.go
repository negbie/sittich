package config

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

	// UseVAD enables the voice activity detector (Silero VAD) for chunking.
	UseVAD bool

	// VADModelPath is the absolute path to the silero_vad.onnx model.
	VADModelPath string

	// VADThreshold is the probability threshold for speech detection (0.0 to 1.0, default 0.5).
	VADThreshold float32

	// VADMinSilenceDuration is the minimum silence duration in seconds to separate segments (default 0.2).
	VADMinSilenceDuration float32

	// VADMinSpeechDuration is the minimum speech duration in seconds to keep a segment (default 0.2).
	VADMinSpeechDuration float32

	// Debug enables detailed console logging.
	Debug bool
}
