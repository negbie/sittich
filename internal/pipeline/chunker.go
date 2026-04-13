package pipeline

import (
	"math"
)

// Chunk represents a slice of audio destined for a single engine inference
// call. Start and End are in seconds relative to the original audio.
type Chunk struct {
	Start     float64
	End       float64
	OrigStart float64
	OrigEnd   float64
}

// ChunkAudioEnergyAware slices the audio into chunks by finding the quietest
// feasible moment (lowest RMS energy) near the target chunk duration boundary.
// This prevents splitting words by ensuring cuts happen during natural pauses
// or at least at low-amplitude signal points.
func ChunkAudioEnergyAware(samples []float32, sampleRate int, targetDuration, searchWindow, overlap float64, padding float64) []Chunk {
	totalDur := float64(len(samples)) / float64(sampleRate)
	if totalDur <= targetDuration {
		return []Chunk{{Start: 0, End: totalDur, OrigStart: 0, OrigEnd: totalDur}}
	}

	var chunks []Chunk
	start := 0.0

	for start < totalDur {
		// 1. Determine intended bounds
		origStart := start
		origEnd := start + targetDuration

		if origEnd >= totalDur {
			origEnd = totalDur
		} else {
			// Search for the lowest energy point within the trailing 'searchWindow'
			// of the current target chunk duration.
			searchEnd := origEnd
			searchStart := searchEnd - searchWindow
			if searchStart <= origStart+overlap {
				// Ensure we don't look back into the previous chunk's overlap region.
				searchStart = origStart + overlap
			}

			if searchStart < searchEnd {
				splitPoint := findQuietestSplitPoint(samples, sampleRate, searchStart, searchEnd)
				origEnd = splitPoint
			}
		}

		// 3. Apply padding for Engine Context
		chunkStart := origStart - padding
		if chunkStart < 0 {
			chunkStart = 0
		}
		chunkEnd := origEnd + padding
		if chunkEnd > totalDur {
			chunkEnd = totalDur
		}

		chunks = append(chunks, Chunk{
			Start:     chunkStart,
			End:       chunkEnd,
			OrigStart: origStart,
			OrigEnd:   origEnd,
		})

		// 4. Advance
		if origEnd >= totalDur {
			break
		}

		start = origEnd - overlap
		if start <= origStart {
			// failsafe if overlap >= chunk duration
			start = origEnd
		}
	}

	return chunks
}

// findQuietestSplitPoint slides a 100ms window across the search region
// and returns the exact time index of the lowest RMS energy.
func findQuietestSplitPoint(samples []float32, sampleRate int, searchStart, searchEnd float64) float64 {
	startIdx := int(searchStart * float64(sampleRate))
	endIdx := int(searchEnd * float64(sampleRate))

	if startIdx < 0 {
		startIdx = 0
	}
	if endIdx > len(samples) {
		endIdx = len(samples)
	}
	if startIdx >= endIdx {
		return searchEnd
	}

	windowSamples := int(0.1 * float64(sampleRate)) // 100ms window
	stepSamples := int(0.05 * float64(sampleRate))  // 50ms step

	if endIdx-startIdx <= windowSamples {
		return searchEnd // Region too small, just return end
	}

	minRMS := float64(math.MaxFloat64)
	bestCenterIdx := endIdx

	for i := startIdx; i+windowSamples <= endIdx; i += stepSamples {
		window := samples[i : i+windowSamples]

		var sumSq float64
		for _, s := range window {
			sumSq += float64(s) * float64(s)
		}
		rms := math.Sqrt(sumSq / float64(len(window)))

		if rms < minRMS {
			minRMS = rms
			bestCenterIdx = i + (windowSamples / 2)
		}
	}

	return float64(bestCenterIdx) / float64(sampleRate)
}
