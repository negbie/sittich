package asr

// Config holds ASR configuration.
type Config struct {
	ModelPath      string
	NumThreads     int
	SampleRate     int
	FeatureDim     int
	Provider       string
	DecodingMethod string
	MaxActivePaths int
}
