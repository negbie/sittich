package output

import (
	"fmt"
	"io"
	"time"

	"github.com/negbie/sittich/internal/asr"
)

// WriteVTT writes the result in WebVTT format.
func WriteVTT(w io.Writer, result *asr.Result) {
	fmt.Fprintln(w, "WEBVTT")
	fmt.Fprintln(w)

	for i, seg := range result.Segments {
		fmt.Fprintf(w, "%d\n", i+1)
		fmt.Fprintf(w, "%s --> %s\n", formatVTTTime(seg.Start), formatVTTTime(seg.End))
		fmt.Fprintf(w, "%s\n\n", seg.Text)
	}
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
