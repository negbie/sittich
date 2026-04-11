package output

import (
	"fmt"
	"io"
	"strings"
	"time"

	"github.com/negbie/sittich/internal/speech"
)

// WriteVTT writes the transcription in WebVTT format
func WriteVTT(out io.Writer, result *speech.Result) error {
	fmt.Fprintln(out, "WEBVTT")
	fmt.Fprintln(out)

	for i, seg := range result.Segments {
		fmt.Fprintf(out, "%d\n", i+1)
		fmt.Fprintf(out, "%s --> %s\n", formatVTTTime(seg.Start), formatVTTTime(seg.End))
		fmt.Fprintln(out, strings.TrimSpace(seg.Text))
		fmt.Fprintln(out)
	}

	return nil
}

func formatVTTTime(seconds float64) string {
	d := time.Duration(seconds * float64(time.Second))
	h := d / time.Hour
	d -= h * time.Hour
	m := d / time.Minute
	d -= m * time.Minute
	s := d / time.Second
	ms := (d - s*time.Second) / time.Millisecond

	return fmt.Sprintf("%02d:%02d:%02d.%03d", h, m, s, ms)
}
