package audio

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
)

// ErrInvalidWAV is returned when a WAV file is malformed.
var ErrInvalidWAV = errors.New("audio: invalid WAV file")

// Wave represents audio data
type Wave struct {
	Samples    []float32
	SampleRate int
	Channels   int
}

type wavHeader struct {
	AudioFormat   uint16
	NumChannels   uint16
	SampleRate    uint32
	ByteRate      uint32
	BlockAlign    uint16
	BitsPerSample uint16
}

const (
	wavFormatPCM   = 1
	wavFormatFloat = 3
)

// ReadWave reads a WAV file and returns the audio data.
// It supports 8/16-bit PCM and 32-bit float formats.
func ReadWave(filename string) (*Wave, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, fmt.Errorf("failed to open file: %w", err)
	}
	defer file.Close()

	return DecodeWAV(file)
}

// DecodeWAV decodes WAV data from an io.Reader.
func DecodeWAV(r io.Reader) (*Wave, error) {
	var riffID [4]byte
	if err := binary.Read(r, binary.LittleEndian, &riffID); err != nil {
		return nil, fmt.Errorf("%w: read RIFF: %v", ErrInvalidWAV, err)
	}
	if string(riffID[:]) != "RIFF" {
		return nil, fmt.Errorf("%w: missing RIFF", ErrInvalidWAV)
	}

	var fileSize uint32
	binary.Read(r, binary.LittleEndian, &fileSize) // Skip

	var waveID [4]byte
	binary.Read(r, binary.LittleEndian, &waveID)
	if string(waveID[:]) != "WAVE" {
		return nil, fmt.Errorf("%w: missing WAVE", ErrInvalidWAV)
	}

	var hdr wavHeader
	var hdrFound bool

	for {
		var chunkID [4]byte
		if err := binary.Read(r, binary.LittleEndian, &chunkID); err != nil {
			if errors.Is(err, io.EOF) {
				break
			}
			return nil, err
		}

		var chunkSize uint32
		if err := binary.Read(r, binary.LittleEndian, &chunkSize); err != nil {
			return nil, err
		}

		id := string(chunkID[:])
		switch id {
		case "fmt ":
			if err := parseFmtChunk(r, chunkSize, &hdr); err != nil {
				return nil, err
			}
			hdrFound = true
		case "data":
			if !hdrFound {
				return nil, fmt.Errorf("%w: data before fmt", ErrInvalidWAV)
			}
			samples, err := streamDataChunk(r, chunkSize, &hdr)
			if err != nil {
				return nil, err
			}
			return &Wave{
				Samples:    samples,
				SampleRate: int(hdr.SampleRate),
				Channels:   int(hdr.NumChannels),
			}, nil
		default:
			// Skip unknown chunks
			if _, err := io.CopyN(io.Discard, r, int64(chunkSize)); err != nil {
				return nil, err
			}
		}
	}

	return nil, fmt.Errorf("%w: no data", ErrInvalidWAV)
}

func parseFmtChunk(r io.Reader, size uint32, hdr *wavHeader) error {
	if err := binary.Read(r, binary.LittleEndian, &hdr.AudioFormat); err != nil {
		return err
	}
	if err := binary.Read(r, binary.LittleEndian, &hdr.NumChannels); err != nil {
		return err
	}
	if err := binary.Read(r, binary.LittleEndian, &hdr.SampleRate); err != nil {
		return err
	}
	if err := binary.Read(r, binary.LittleEndian, &hdr.ByteRate); err != nil {
		return err
	}
	if err := binary.Read(r, binary.LittleEndian, &hdr.BlockAlign); err != nil {
		return err
	}
	if err := binary.Read(r, binary.LittleEndian, &hdr.BitsPerSample); err != nil {
		return err
	}
	if size > 16 {
		if _, err := io.CopyN(io.Discard, r, int64(size-16)); err != nil {
			return err
		}
	}
	return nil
}

func streamDataChunk(r io.Reader, size uint32, hdr *wavHeader) ([]float32, error) {
	bytesPerSample := int(hdr.BitsPerSample) / 8
	if bytesPerSample == 0 {
		return nil, fmt.Errorf("%w: invalid bits per sample: %d", ErrInvalidWAV, hdr.BitsPerSample)
	}

	supportedFormat := (hdr.AudioFormat == wavFormatPCM && (hdr.BitsPerSample == 8 || hdr.BitsPerSample == 16)) ||
		(hdr.AudioFormat == wavFormatFloat && hdr.BitsPerSample == 32)
	if !supportedFormat {
		return nil, fmt.Errorf("%w: unsupported WAV format audio_format=%d bits_per_sample=%d", ErrInvalidWAV, hdr.AudioFormat, hdr.BitsPerSample)
	}

	numSamples := int(size) / bytesPerSample
	samples := make([]float32, numSamples)

	// Use a small buffer to read and convert in chunks
	const bufSamples = 4096
	buf := make([]byte, bufSamples*bytesPerSample)

	for i := 0; i < numSamples; {
		toRead := bufSamples
		if i+toRead > numSamples {
			toRead = numSamples - i
		}

		n, err := io.ReadFull(r, buf[:toRead*bytesPerSample])
		if err != nil && !errors.Is(err, io.EOF) {
			return nil, err
		}
		if n == 0 {
			break
		}

		// Convert the buffer to float32
		actualRead := n / bytesPerSample
		for j := 0; j < actualRead; j++ {
			off := j * bytesPerSample
			idx := i + j

			switch {
			case hdr.AudioFormat == wavFormatPCM && hdr.BitsPerSample == 16:
				v := int16(binary.LittleEndian.Uint16(buf[off : off+2]))
				samples[idx] = float32(v) / 32768.0
			case hdr.AudioFormat == wavFormatPCM && hdr.BitsPerSample == 8:
				samples[idx] = (float32(buf[off]) - 128.0) / 128.0
			case hdr.AudioFormat == wavFormatFloat && hdr.BitsPerSample == 32:
				bits := binary.LittleEndian.Uint32(buf[off : off+4])
				samples[idx] = math.Float32frombits(bits)
			}
		}
		i += actualRead
	}

	return samples, nil
}
