package audio

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"os/exec"
)

// DecodeWAV decodes any audio file by shelling out to 'sox'. It streams the
// output directly into a float32 slice to minimize memory allocations.
func DecodeWAV(r io.Reader) ([]float32, error) {
	// 1. Probe the file length if it's a file on disk to pre-allocate memory
	var initialCapacity int
	if f, ok := r.(*os.File); ok {
		if st, err := f.Stat(); err == nil && st.Size() > 0 {
			// Rough estimate: raw 16kHz Mono is ~32KB/sec.
			// Input file size is a good upper bound for capacity.
			initialCapacity = int(st.Size() / 2)
		}
	}
	if initialCapacity < 1024*1024 {
		initialCapacity = 1024 * 1024 // Default to 1min of audio
	}

	args := []string{
		"-",
		"-t", "raw", "-r", "16000", "-c", "1", "-e", "signed-integer", "-b", "16", "-",
	}

	cmd := exec.Command("sox", args...)
	cmd.Stdin = r
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, fmt.Errorf("audio: stdout pipe: %w", err)
	}
	defer stdout.Close()

	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("audio: failed to start sox: %w", err)
	}

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
			break
		}
		if err != nil {
			return nil, fmt.Errorf("audio: read sox pipe: %w", err)
		}
	}

	if err := cmd.Wait(); err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok {
			return nil, fmt.Errorf("sox error: %s", string(exitErr.Stderr))
		}
		return nil, fmt.Errorf("sox wait: %w", err)
	}

	return samples, nil
}
