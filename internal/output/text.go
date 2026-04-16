package output

import (
	"fmt"
	"io"

	"github.com/negbie/sittich/internal/asr"
)

// WriteText writes the transcription result as plain text.
func WriteText(w io.Writer, result *asr.Result) {
	fmt.Fprintln(w, result.FullText())
}
