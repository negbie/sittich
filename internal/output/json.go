package output

import (
	"encoding/json"
	"io"

	"github.com/negbie/sittich/internal/speech"
)

// WriteJSON writes the transcription in JSON format
func WriteJSON(out io.Writer, result *speech.Result) error {
	return json.NewEncoder(out).Encode(result)
}
