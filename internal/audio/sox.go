package audio

import (
	"bufio"
	"context"
	"encoding/binary"
	"fmt"
	"io"
	"log/slog"
	"os/exec"
)

// DecodeWAV decodes any audio file. It uses native Go decoding for standard
// PCM/Float WAV files and falls back to 'sox' for compressed formats
// like WAV49 (MS-GSM).
func DecodeWAV(ctx context.Context, r io.Reader) ([]float32, error) {
	br := bufio.NewReader(r)

	// Peek at the first 64 bytes to detect the format
	header, err := br.Peek(64)
	if err == nil {
		isNative := false
		if len(header) >= 12 && string(header[0:4]) == "RIFF" && string(header[8:12]) == "WAVE" {
			// Find fmt chunk to check format
			offset := 12
			for offset < len(header)-8 {
				chunkID := string(header[offset : offset+4])
				chunkSize := binary.LittleEndian.Uint32(header[offset+4 : offset+8])
				if chunkID == "fmt " && len(header) >= offset+10 {
					audioFormat := binary.LittleEndian.Uint16(header[offset+8 : offset+10])
					if audioFormat == 1 || audioFormat == 3 {
						isNative = true
					}
					break
				}
				offset += 8 + int(chunkSize)
			}
		}

		if isNative {
			slog.Debug("audio: using native decoder")
			return DecodeNative(br)
		}
	}

	slog.Debug("audio: falling back to sox decoder")
	return decodeWithSox(ctx, br)
}

// decodeWithSox shells out to 'sox' to decode audio.
func decodeWithSox(ctx context.Context, r io.Reader) ([]float32, error) {
	// 1. Probe the file length if it's a file on disk to pre-allocate memory
	var initialCapacity int
	// Note: r might be a bufio.Reader wrapping an os.File
	// We check the underlying reader if possible, but it's not strictly necessary.
	initialCapacity = 1024 * 1024 // Default to 1min of audio
	if initialCapacity < 1024*1024 {
		initialCapacity = 1024 * 1024 // Default to 1min of audio
	}

	args := []string{
		"-",
		"-t", "raw", "-r", "16000", "-c", "1", "-e", "signed-integer", "-b", "16", "-",
	}

	cmd := exec.CommandContext(ctx, "sox", args...)
	cmd.Stdin = r
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, fmt.Errorf("audio: stdout pipe: %w", err)
	}
	defer stdout.Close()

	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("audio: failed to start sox: %w", err)
	}

	// Add context-aware processing with timeout
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
			if exitErr, ok := err.(*exec.ExitError); ok {
				return nil, fmt.Errorf("sox error: %s", string(exitErr.Stderr))
			}
			return nil, fmt.Errorf("sox wait: %w", err)
		}
		return samples, nil
	case <-ctx.Done():
		// Context cancelled or timed out - clean up the hung process
		if cmd.Process != nil {
			cmd.Process.Kill() // Force kill the Sox process
		}
		cmd.Wait() // Reap the zombie process
		return nil, ctx.Err()
	}
}

// readSoxOutput reads and converts the Sox process output in a separate goroutine
func readSoxOutput(stdout io.Reader, initialCapacity int) ([]float32, error) {
	samples := make([]float32, 0, initialCapacity)
	buf := make([]byte, 32*1024)

	for {
		n, err := io.ReadFull(stdout, buf)
		if n > 0 {
			// Convert every 2 bytes (int16) to a float32
			for i := 0; i < n; i += 2 {
				v := int16(binary.LittleEndian.Uint16(buf[i:]))
				samples = append(samples, float32(v)/32768.0)
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
