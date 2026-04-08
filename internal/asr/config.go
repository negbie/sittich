package asr

// Config holds ASR configuration.
type Config struct {
	ModelPath              string
	NumThreads             int
	SampleRate             int
	FeatureDim             int
	Provider               string
	DecodingMethod         string
	MaxActivePaths         int
	CalibrationTargetPeak  float32
	CalibrationMaxGain     float32
	CalibrationMaxMSamples int
}
