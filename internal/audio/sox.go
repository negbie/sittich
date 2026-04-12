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

// decodeWithSox shells out to 'sox' to decode and preprocess audio.
// Sox handles decoding, resampling to 16kHz, mono mixdown, filtering,
// and gain normalization — producing 32-bit float output for the ASR model.
func decodeWithSox(ctx context.Context, r io.Reader) ([]float32, error) {
	initialCapacity := 1024 * 1024 // ~1 min of 16kHz float32

	// Build Sox command: input from stdin, output raw float32 to stdout
	// sox [input-options] - [output-options] - [effects]
	args := []string{
		"-",                                                                            // input from stdin
		"-t", "raw", "-r", "16000", "-c", "1", "-e", "floating-point", "-b", "32", "-", // output options + stdout
		"highpass", "100",
		"gain", "-n",
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
		if readErr != nil {
			return nil, readErr
		}
		if err := cmd.Wait(); err != nil {
			return nil, fmt.Errorf("sox error: %s", stderr.String())
		}
		return samples, nil
	case <-ctx.Done():
		if cmd.Process != nil {
			cmd.Process.Kill()
		}
		cmd.Wait()
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
