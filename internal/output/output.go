package output

import (
	"github.com/negbie/sittich/internal/asr"
)

// Output writes the final transcription result to a specific format.
type Output func(result *asr.Result)
