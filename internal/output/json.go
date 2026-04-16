package output

import (
	"encoding/json"
	"io"

	"github.com/negbie/sittich/internal/asr"
)

// WriteJSON writes the transcription in JSON format
func WriteJSON(out io.Writer, result *asr.Result) error {
	return json.NewEncoder(out).Encode(result)
}
