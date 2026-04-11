package audio

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"os/exec"
)

// DecodeWAV decodes any WAV file by shelling out to 'sox'.
func DecodeWAV(r io.Reader) ([]float32, error) {
	tempFile, err := os.CreateTemp("", "sittich-*.wav")
	if err != nil {
		return nil, fmt.Errorf("audio: create temp: %w", err)
	}
	defer os.Remove(tempFile.Name())
	defer tempFile.Close()

	if _, err := io.Copy(tempFile, r); err != nil {
		return nil, fmt.Errorf("audio: buffer wav: %w", err)
	}

	// Minimal peeker: Peek sample rate to decide filter strategy.
	rate := uint32(16000)
	var buf [32]byte
	if n, _ := tempFile.ReadAt(buf[:], 0); n == 32 {
		for i := 0; i < 24; i++ {
			if string(buf[i:i+4]) == "fmt " {
				rate = binary.LittleEndian.Uint32(buf[i+12 : i+16])
				break
			}
		}
	}

	args := []string{tempFile.Name(), "-t", "raw", "-r", "16000", "-c", "1", "-e", "signed-integer", "-b", "16", "-"}
	
	// Apply accuracy filters ONLY for low-bandwidth audio (<= 8kHz).
	if rate <= 8000 {
		args = append(args, "highpass", "80", "lowpass", "3500")
	}
	
	// args = append(args, "norm", "-1")

	cmd := exec.Command("sox", args...)
	out, err := cmd.Output()
	if err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok {
			return nil, fmt.Errorf("sox error: %s", string(exitErr.Stderr))
		}
		return nil, fmt.Errorf("failed to run sox: %w (is sox installed?)", err)
	}

	samples := make([]float32, len(out)/2)
	for i := 0; i < len(samples); i++ {
		v := int16(binary.LittleEndian.Uint16(out[i*2:]))
		samples[i] = float32(v) / 32768.0
	}
	return samples, nil
}
