package pipeline

import (
	"context"
	"fmt"
	"math"
	"os"
	"time"

	"github.com/negbie/sittich/internal/asr"
	"github.com/negbie/sittich/internal/audio"
	"github.com/negbie/sittich/internal/config"
	"github.com/negbie/sittich/internal/vad"
)

// Pipeline orchestrates audio processing and transcription.
type Pipeline struct {
	Engine asr.Engine
	Config config.Pipeline
}

// Process runs the transcription pipeline.
func (p *Pipeline) Process(ctx context.Context, path string, chunkDuration float64, soxFlags ...string) (*asr.Result, error) {
	const targetRate = 16000
	start := time.Now()

	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("pipeline: open: %w", err)
	}
	defer f.Close()

	samples, err := audio.Decode(ctx, f, soxFlags...)
	if err != nil {
		return nil, fmt.Errorf("pipeline: decode: %w", err)
	}

	if p.Config.Debug {
		audio.DebugPlotWaveform(samples, "Input")
	}

	totalDur := float64(len(samples)) / float64(targetRate)
	if chunkDuration <= 0 {
		chunkDuration = p.Config.ChunkDuration
	}

	audio.ConditionAudioSignal(samples, targetRate)

	if p.Config.Debug {
		audio.DebugPlotWaveform(samples, "Processed")
		audio.EncodeToFile(samples, "debug.wav", targetRate)
		p.verifySignal(samples, targetRate)
	}

	overlap := p.Config.ChunkOverlapDuration
	if overlap < 0.4 {
		overlap = 0.4 // Safety floor for robust stitching
	}
	padding := 1.2 // Increased to 1.2s for safer model context (multiple of 40ms)
	if overlap/2 > padding {
		padding = overlap / 2
	}

	var chunks []Chunk
	if p.Config.VADEnabled && p.Engine.VADPath() != "" {
		detector, err := vad.NewDetector(p.Engine.VADPath(), targetRate)
		if err != nil {
			fmt.Fprintf(os.Stderr, "   [Pipeline] VAD init failed: %v, falling back to energy-based chunking\n", err)
		} else {
			defer detector.Close()
			speechSegments, err := detector.Segment(samples, targetRate)
			if err != nil {
				fmt.Fprintf(os.Stderr, "   [Pipeline] VAD segment failed: %v, falling back to energy-based chunking\n", err)
			} else if len(speechSegments) > 0 {
				if p.Config.Debug {
					fmt.Fprintf(os.Stderr, "   [Pipeline] VAD detected %d speech segments\n", len(speechSegments))
				}
				chunks = p.chunkVAD(speechSegments, targetRate, chunkDuration, overlap, padding, len(samples))
			} else if p.Config.Debug {
				fmt.Fprintf(os.Stderr, "   [Pipeline] VAD detected NO speech segments\n")
			}
		}
	}

	if len(chunks) == 0 {
		chunks, err = ChunkAudioEnergyAware(samples, targetRate, chunkDuration, 5.0, overlap, padding)
		if err != nil {
			return nil, fmt.Errorf("pipeline: chunking failed: %w", err)
		}
	}
	asrResults, err := p.transcribeChunks(ctx, samples, targetRate, chunks)
	if err != nil {
		return nil, err
	}

	if len(asrResults) == 0 {
		return &asr.Result{Duration: totalDur}, nil
	}

	combined := StitchResults(asrResults, targetRate)
	combined.Duration = totalDur

	if p.Config.Debug {
		fmt.Fprintf(os.Stderr, "   [Pipeline] Processed %d chunks in %s\n",
			len(asrResults), time.Since(start).Round(time.Millisecond))
	}

	return combined, nil
}

func (p *Pipeline) transcribeChunks(ctx context.Context, samples []float32, targetRate int, chunks []Chunk) ([]ChunkResult, error) {
	batch := make([][]float32, 0, len(chunks))
	validIndices := make([]int, 0, len(chunks))

	for i, c := range chunks {
		start := c.Start
		end := c.End
		if start < 0 {
			start = 0
		}
		if end > len(samples) {
			end = len(samples)
		}
		if start >= end {
			continue
		}

		batch = append(batch, samples[start:end])
		validIndices = append(validIndices, i)
	}

	if len(batch) == 0 {
		return nil, nil
	}

	results, err := p.Engine.TranscribeBatch(ctx, batch, targetRate, asr.Options{
		Language:       p.Config.Language,
		WordTimestamps: p.Config.WordTimestamps,
		Debug:          p.Config.Debug,
	})
	if err != nil {
		return nil, fmt.Errorf("pipeline: batch transcribe: %w", err)
	}

	if len(results) == 0 && len(batch) > 0 {
		return nil, nil
	}

	chunkResults := make([]ChunkResult, 0, len(results))
	// NOTE: TranscribeBatch must return results in the same order as the input batch.
	// If the engine reorders results, the validIndices mapping below breaks silently.
	for i, res := range results {
		if res == nil {
			continue
		}

		idx := validIndices[i]
		chunkResults = append(chunkResults, ChunkResult{
			Offset:    chunks[idx].Start,
			OrigStart: chunks[idx].OrigStart,
			OrigEnd:   chunks[idx].OrigEnd,
			Result:    res,
		})
	}
	return chunkResults, nil
}

func (p *Pipeline) verifySignal(samples []float32, targetRate int) {
	var peak float32
	for _, s := range samples {
		abs := float32(math.Abs(float64(s)))
		if abs > peak {
			peak = abs
		}
	}
	fmt.Fprintf(os.Stderr, "   [Pipeline] signal samples=%d duration=%.2fs peak=%.4f\n",
		len(samples), float64(len(samples))/float64(targetRate), peak)
}

func (p *Pipeline) chunkVAD(segments []vad.Segment, sampleRate int, targetDuration, overlap, padding float64, totalSamples int) []Chunk {
	var chunks []Chunk
	paddingSamples := int(padding * float64(sampleRate))

	for i, s := range segments {
		startIdx := int(math.Round(s.Start * float64(sampleRate)))
		// Snap to frame boundary so sub-chunk absolute positions stay aligned.
		startIdx = ((startIdx + frameSize/2) / frameSize) * frameSize
		numSamples := len(s.Samples)
		endIdx := startIdx + numSamples

		origStart := startIdx
		origEnd := endIdx

		if i == 0 {
			origStart = 0
		} else {
			// Split the gap between the previous segment and this segment exactly in half.
			// This perfectly prevents overlapping ownerships and dropped words.
			prevStartIdx := int(math.Round(segments[i-1].Start * float64(sampleRate)))
			prevEndIdx := prevStartIdx + len(segments[i-1].Samples)

			if startIdx > prevEndIdx {
				mid := prevEndIdx + (startIdx-prevEndIdx)/2
				origStart = mid
				chunks[len(chunks)-1].OrigEnd = mid
			} else {
				// Segments overlap (should not happen with Silero, but just in case)
				origStart = prevEndIdx
				chunks[len(chunks)-1].OrigEnd = prevEndIdx
			}
		}

		if i == len(segments)-1 {
			origEnd = totalSamples
		}

		if float64(numSamples)/float64(sampleRate) <= targetDuration+overlap {
			pStart := startIdx - paddingSamples
			if pStart < 0 {
				pStart = 0
			}
			pEnd := endIdx + paddingSamples
			if pEnd > totalSamples {
				pEnd = totalSamples
			}

			chunks = append(chunks, Chunk{
				Start:     pStart,
				End:       pEnd,
				OrigStart: origStart,
				OrigEnd:   origEnd,
			})
		} else {
			// Long segment, split it using energy-aware chunker
			subChunks, err := ChunkAudioEnergyAware(s.Samples, sampleRate, targetDuration, 5.0, overlap, padding)
			if err == nil {
				for j, sc := range subChunks {
					scOrigStart := startIdx + sc.OrigStart
					scOrigEnd := startIdx + sc.OrigEnd

					// Adjust first and last sub-chunk to respect bridged boundaries
					if j == 0 {
						scOrigStart = origStart
					}
					if j == len(subChunks)-1 {
						scOrigEnd = origEnd
					}

					chunks = append(chunks, Chunk{
						Start:     startIdx + sc.Start,
						End:       startIdx + sc.End,
						OrigStart: scOrigStart,
						OrigEnd:   scOrigEnd,
					})
				}
			}
		}
	}
	return chunks
}
