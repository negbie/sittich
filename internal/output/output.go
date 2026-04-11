package output

import (
	"io"

	"github.com/negbie/sittich/internal/speech"
)

// Writer is the interface for different output formats
type Writer interface {
	Write(out io.Writer, result *speech.Result) error
}
