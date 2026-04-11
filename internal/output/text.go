package output

import (
	"fmt"
	"io"
	"strings"

	"github.com/negbie/sittich/internal/speech"
)

// WriteText writes the transcription in plain text format
func WriteText(out io.Writer, result *speech.Result) error {
	for _, seg := range result.Segments {
		fmt.Fprintln(out, strings.TrimSpace(seg.Text))
	}
	return nil
}
