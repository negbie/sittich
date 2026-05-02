package audio

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"os/exec"
	"unsafe"
)

// Decode shells out to the 'sox' utility to decode and process audio in-memory.
// It converts any input format supported by Sox into 32-bit little-endian raw floats at 16kHz Mono.
// Thanks to exec.CommandContext, the process is safely terminated if context cancels.
func Decode(ctx context.Context, r io.Reader, extraFlags ...string) ([]float32, error) {
	args := []string{
		"-D", "-",
		"-t", "raw", "-c", "1", "-e", "floating-point", "-b", "32", "-",
	}

	if len(extraFlags) > 0 {
		args = append(args, extraFlags...)
	} else {
		args = append(args, "rate", "16000")
		args = append(args, "gain", "-n", "-1")
		args = append(args, "vad")
	}

	cmd := exec.CommandContext(ctx, "sox", args...)
	cmd.Stdin = r
	var stderr bytes.Buffer
	cmd.Stderr = &stderr

	// Use a limited buffer to prevent OOM from decompression bombs.
	const maxAudioBytes = 50 * 1024 * 1024
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, fmt.Errorf("audio: failed to create stdout pipe: %w", err)
	}

	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("audio: sox start failed: %w", err)
	}

	// We read maxAudioBytes + 1 so we can detect if the limit was exceeded.
	rawBytes, err := io.ReadAll(io.LimitReader(stdout, maxAudioBytes+1))
	if err != nil {
		return nil, fmt.Errorf("audio: failed to read sox output: %w", err)
	}

	if len(rawBytes) > maxAudioBytes {
		cmd.Process.Kill() // Force kill so Sox doesn't block on a full stdout pipe
		cmd.Wait()         // Clean up the zombie process
		return nil, fmt.Errorf("audio: file exceeds maximum allowed duration (approx 13 minutes)")
	}

	if err := cmd.Wait(); err != nil {
		return nil, fmt.Errorf("audio: sox failed: %v (stderr: %s)", err, stderr.String())
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

// EncodeToFile saves the given samples to a file using Sox.
func EncodeToFile(samples []float32, path string, sampleRate int) error {
	if len(samples) == 0 {
		return fmt.Errorf("audio: cannot encode empty samples")
	}

	// Convert float32 slice to raw byte slice for writing to Sox stdin
	rawBytes := unsafe.Slice((*byte)(unsafe.Pointer(&samples[0])), len(samples)*4)

	cmd := exec.Command("sox",
		"-t", "raw", "-c", "1", "-r", fmt.Sprintf("%d", sampleRate), "-e", "floating-point", "-b", "32", "-",
		path,
	)

	cmd.Stdin = bytes.NewReader(rawBytes)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr

	if err := cmd.Run(); err != nil {
		return fmt.Errorf("audio: sox encode failed: %v (stderr: %s)", err, stderr.String())
	}

	return nil
}
