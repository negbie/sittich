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

func DefaultConfig(modelPath string) *Config {
	return &Config{
		ModelPath:      "./data",
		NumThreads:     4,
		SampleRate:     16000,
		FeatureDim:     80,
		Provider:       "cpu",
		DecodingMethod: "modified_beam_search",
		MaxActivePaths: 2,
	}
}
