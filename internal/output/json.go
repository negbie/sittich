package output

import (
	"encoding/json"
	"io"

	"github.com/negbie/sittich/internal/types"
)

type JSONOutput struct {
	Version  string          `json:"version"`
	Duration float64         `json:"duration"`
	Language string          `json:"language,omitempty"`
	Text     string          `json:"text"`
	Segments []types.Segment `json:"segments"`
}

// WriteJSON writes the transcription result in JSON format
func WriteJSON(out io.Writer, result *types.Result) error {
	data := JSONOutput{
		Version:  "1.0",
		Duration: result.Duration,
		Language: result.Language,
		Text:     result.FullText(),
		Segments: result.Segments,
	}

	enc := json.NewEncoder(out)
	enc.SetIndent("", "  ")
	return enc.Encode(data)
}
