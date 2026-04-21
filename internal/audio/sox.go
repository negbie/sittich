package audio

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"os/exec"
	"unsafe"
)

// decodeWithSox shells out to the 'sox' utility to decode and process audio in-memory.
// It converts any input format supported by Sox into 32-bit little-endian raw floats at 16kHz Mono.
// Thanks to exec.CommandContext, the process is safely terminated if context cancels.
func decodeWithSox(ctx context.Context, r io.Reader, extraFlags ...string) ([]float32, error) {
	args := []string{
		"-D", "-",
		"-t", "raw", "-c", "1", "-r", "16000", "-e", "floating-point", "-b", "32", "-",
	}

	if len(extraFlags) > 0 {
		args = append(args, extraFlags...)
	} else {
		args = append(args, "gain", "-n", "-3")
		args = append(args, "vad")
	}

	cmd := exec.CommandContext(ctx, "sox", args...)
	cmd.Stdin = r
	var stderr bytes.Buffer
	cmd.Stderr = &stderr

	rawBytes, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("audio: sox execution failed: %v (stderr: %s)", err, stderr.String())
	}

	if len(rawBytes) == 0 {
		return nil, nil
	}

	// Optimization: Return a view of the byte slice as float32s to avoid doubling
	// memory usage for large audio files. Go's heap-allocated byte slices are
	// typically 8 or 16-byte aligned, which is sufficient for float32 (4-byte).
	// The underlying rawBytes will be kept alive by the returned slice.
	count := len(rawBytes) / 4
	return unsafe.Slice((*float32)(unsafe.Pointer(&rawBytes[0])), count), nil
}
