package audio

import (
	"bytes"
	"context"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os/exec"
)

// decodeWithSox shells out to the 'sox' utility to decode and process audio in-memory.
// It leverages pipes for zero-disk streaming, converting any input format supported
// by Sox into 32-bit little-endian raw floats at 16kHz Mono.
func decodeWithSox(ctx context.Context, r io.Reader, extraFlags ...string) ([]float32, error) {
	initialCapacity := 1024 * 1024 // ~1 min of 16kHz float32

	args := []string{
		"-",                                                             // Infile: read from stdin
		"-t", "raw", "-c", "1", "-e", "floating-point", "-b", "32", "-", // Outfile: write raw float32 to stdout
		"rate", "-v", "16k",
	}

	if len(extraFlags) > 0 {
		args = append(args, extraFlags...)
	} else {
		args = append(args, "silence", "1", "0.05", "0.1%")
		args = append(args, "gain", "-h")
	}

	cmd := exec.CommandContext(ctx, "sox", args...)
	cmd.Stdin = r
	var stderr bytes.Buffer
	cmd.Stderr = &stderr

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, fmt.Errorf("audio: stdout pipe: %w", err)
	}
	defer stdout.Close()

	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("audio: failed to start sox: %w", err)
	}

	done := make(chan struct{})
	var readErr error
	var samples []float32

	go func() {
		defer close(done)
		samples, readErr = readSoxOutput(stdout, initialCapacity)
	}()

	select {
	case <-done:
		// Sox process finished its work or failed.
		if readErr != nil {
			return nil, readErr
		}
		if err := cmd.Wait(); err != nil {
			return nil, fmt.Errorf("sox error: %s", stderr.String())
		}
		return samples, nil
	case <-ctx.Done():
		// HTTP request timeout or cancellation: immediately terminate the subprocess.
		if cmd.Process != nil {
			cmd.Process.Kill()
		}
		cmd.Wait() // Reclaim process resources
		return nil, ctx.Err()
	}
}

// readSoxOutput reads 32-bit float samples from the Sox process output.
func readSoxOutput(stdout io.Reader, initialCapacity int) ([]float32, error) {
	samples := make([]float32, 0, initialCapacity)
	// Use a 32KB buffer for raw reads
	raw := make([]byte, 32*1024)

	for {
		n, err := io.ReadFull(stdout, raw)
		if n > 0 {
			// Process whole samples (4 bytes each)
			count := n / 4
			for i := 0; i < count; i++ {
				bits := binary.LittleEndian.Uint32(raw[i*4 : (i+1)*4])
				samples = append(samples, math.Float32frombits(bits))
			}
		}
		if err == io.EOF || err == io.ErrUnexpectedEOF {
			return samples, nil
		}
		if err != nil {
			return nil, fmt.Errorf("audio: read sox pipe: %w", err)
		}
	}
}
