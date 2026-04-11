package config

// Default values for various configuration options.
const (
	DefaultListenAddr             = ":5092"
	DefaultChunkSize              = 30
	DefaultChunkMinTail           = 1.5
	DefaultFormat                 = "text"
	DefaultMaxActivePaths         = 2
	DefaultDecodingMethod         = "modified_beam_search"
	DefaultWorkers                = 4
	DefaultMaxUploadMB            = 32
	DefaultVADThreshold           = 0.5
	DefaultVADMinSilence          = 0.45
	DefaultVADMinSpeech           = 0.25
	DefaultCalibrationTargetPeak  = 10.0
	DefaultCalibrationMaxGain     = 20.0
	DefaultCalibrationMaxMSamples = 2048
	DefaultNumThreads             = 2
	DefaultMaxQueueSize           = 10
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
	ModelPath              string
	NumThreads             int
	DecodingMethod         string
	MaxActivePaths         int
	CalibrationTargetPeak  float32
	CalibrationMaxGain     float32
	CalibrationMaxMSamples int
}

// Pipeline holds pipeline behaviour.
type Pipeline struct {
	// VADEnabled enables Silero VAD-based speech segmentation before
	// transcription. When false, audio is chunked at fixed intervals.
	VADEnabled bool

	// ChunkDuration is the maximum duration in seconds per chunk sent to the
	// engine. Callers are expected to provide this explicitly.
	ChunkDuration float64

	// ChunkMinTailDuration is the minimum tail duration used when balancing
	// oversized speech segments into smaller chunks.
	ChunkMinTailDuration float64

	// VADThreshold controls the speech probability threshold used by VAD.
	VADThreshold float32

	// VADMinSilenceDuration is the minimum silence duration in seconds needed
	// for VAD to split speech regions.
	VADMinSilenceDuration float32

	// VADMinSpeechDuration is the minimum speech duration in seconds required
	// for VAD to emit a speech region.
	VADMinSpeechDuration float32

	// WordTimestamps requests word-level timing from the engine.
	WordTimestamps bool

	// Language is a BCP-47 hint passed to the engine (empty = auto-detect).
	Language string

	// VADModelPath is the filesystem path to the Silero VAD ONNX model.
	VADModelPath string

	// NumThreads is the number of threads for VAD.
	NumThreads int

	// Debug enables detailed console logging.
	Debug bool
}
