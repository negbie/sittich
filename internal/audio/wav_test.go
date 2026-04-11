package audio

import (
	"bytes"
	"encoding/binary"
	"testing"
)

func TestDecodePCM(t *testing.T) {
	// Create a dummy PCM WAV (16kHz, Mono)
	buf := &bytes.Buffer{}
	buf.Write([]byte("RIFF"))
	binary.Write(buf, binary.LittleEndian, uint32(0)) // size
	buf.Write([]byte("WAVE"))
	buf.Write([]byte("fmt "))
	binary.Write(buf, binary.LittleEndian, uint32(16)) // chunk size
	binary.Write(buf, binary.LittleEndian, uint16(1))  // format 1 (PCM)
	binary.Write(buf, binary.LittleEndian, uint16(1))  // channels
	binary.Write(buf, binary.LittleEndian, uint32(16000)) // sample rate
	binary.Write(buf, binary.LittleEndian, uint32(32000)) // byte rate
	binary.Write(buf, binary.LittleEndian, uint16(2))     // block align
	binary.Write(buf, binary.LittleEndian, uint16(16))    // bits per sample
	buf.Write([]byte("data"))
	binary.Write(buf, binary.LittleEndian, uint32(4)) // data size
	binary.Write(buf, binary.LittleEndian, int16(32767))
	binary.Write(buf, binary.LittleEndian, int16(-32768))

	samples, err := DecodeWAV(buf)
	if err != nil {
		t.Fatalf("failed to decode PCM WAV: %v", err)
	}

	if len(samples) != 2 {
		t.Errorf("expected 2 samples, got %d", len(samples))
	}
	// With normalization, 32767 results in approx 1.0 (actually -1dB ≈ 0.89)
	if samples[0] < 0.8 {
		t.Errorf("expected normalized sample, got %f", samples[0])
	}
}
