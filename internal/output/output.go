package output

import (
	"io"

	"github.com/negbie/sittich/internal/types"
)

// Write writes formatted output to the given writer.
func Write(out io.Writer, format string, result *types.Result) error {
	switch format {
	case "json":
		return WriteJSON(out, result)
	case "vtt":
		return WriteVTT(out, result)
	default:
		return WriteText(out, result)
	}
}
