package audio

import (
	"bufio"
	"context"
	"io"
)

// DecodeWAV decodes audio using Sox. It handles any format Sox supports
// (WAV, FLAC, MP3, OGG, etc.), resamples to 16kHz mono, and optionally
// applies signal processing (high-pass, low-pass, gain) via Sox effects.
func DecodeWAV(ctx context.Context, r io.Reader, soxFlags ...string) ([]float32, error) {
	br := bufio.NewReader(r)
	return decodeWithSox(ctx, br, soxFlags...)
}
