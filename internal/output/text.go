package output

import (
	"fmt"
	"io"

	"github.com/negbie/sittich/internal/types"
)

// WriteText writes the transcription in plain text format.
func WriteText(out io.Writer, result *types.Result) error {
	for i, seg := range result.Segments {
		if i > 0 {
			fmt.Fprint(out, " ")
		}
		fmt.Fprint(out, seg.Text)
	}
	fmt.Fprintln(out)
	return nil
}
