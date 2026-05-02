package pipeline

import "fmt"

const frameSize = 640 // 40ms frame at 16kHz

// Chunk represents a slice of audio for engine inference.
type Chunk struct {
	Start     int
	End       int
	OrigStart int
	OrigEnd   int
}

// ChunkAudioEnergyAware slices audio by finding the quietest moments (lowest RMS)
// near target boundaries. Split points are snapped to 40ms frame boundaries
// to prevent artifacts in model encoder states.
func ChunkAudioEnergyAware(samples []float32, sampleRate int, targetDuration, searchWindow, overlap, padding float64) ([]Chunk, error) {
	totalSamples := len(samples)
	if totalSamples == 0 {
		return nil, nil
	}

	// Comprehensive input validation with descriptive errors
	if sampleRate <= 0 {
		return nil, fmt.Errorf("chunker: invalid sample rate %d", sampleRate)
	}
	if targetDuration < 1.0 {
		targetDuration = 1.0 // Safety floor to prevent nano-chunking OOM
	}
	if searchWindow < 1.0 {
		searchWindow = 1.0
	}
	if overlap < 0 || padding < 0 {
		return nil, fmt.Errorf("chunker: overlap (%.2f) and padding (%.2f) must be >= 0", overlap, padding)
	}
	if overlap >= targetDuration {
		return nil, fmt.Errorf("chunker: overlap (%.2f) must be less than target duration (%.2f)", overlap, targetDuration)
	}

	targetSamples := int(targetDuration * float64(sampleRate))
	searchSamples := int(searchWindow * float64(sampleRate))
	overlapSamples := ((int(overlap*float64(sampleRate)) + frameSize/2) / frameSize) * frameSize
	paddingSamples := int(padding * float64(sampleRate))

	var chunks []Chunk
	currentStart := 0

	for currentStart < totalSamples {
		currentEnd := currentStart + targetSamples

		if currentEnd >= totalSamples {
			currentEnd = totalSamples
		} else {
			sStart := currentEnd - searchSamples
			if sStart < currentStart+overlapSamples {
				sStart = currentStart + overlapSamples
			}

			if sStart < currentEnd {
				idx, err := findQuietestSampleIndex(samples, sampleRate, sStart, currentEnd)
				if err != nil {
					return nil, fmt.Errorf("chunker: failed to find split point: %w", err)
				}
				currentEnd = idx
			}
		}

		pStart := currentStart - paddingSamples
		if pStart < 0 {
			pStart = 0
		}
		pEnd := currentEnd + paddingSamples
		if pEnd > totalSamples {
			pEnd = totalSamples
		}

		chunks = append(chunks, Chunk{
			Start:     pStart,
			End:       pEnd,
			OrigStart: currentStart,
			OrigEnd:   currentEnd,
		})

		if currentEnd >= totalSamples {
			break
		}
		
		nextStart := currentEnd - overlapSamples
		if nextStart <= currentStart {
			// Ensured nextStart always advances past currentStart to prevent infinite loops
			currentStart = currentEnd
		} else {
			currentStart = nextStart
		}
	}

	// Eliminate ownership overlaps: split any overlapping OrigStart/OrigEnd
	// regions at the midpoint so the stitcher never accepts a word from two chunks.
	for i := 1; i < len(chunks); i++ {
		if chunks[i].OrigStart < chunks[i-1].OrigEnd {
			mid := chunks[i].OrigStart + (chunks[i-1].OrigEnd-chunks[i].OrigStart)/2
			chunks[i-1].OrigEnd = mid
			chunks[i].OrigStart = mid
		}
	}

	return chunks, nil
}

// findQuietestSampleIndex returns the index of lowest RMS energy in the search
// region using a sliding window. Result is snapped to a 40ms frame boundary (640 samples).
func findQuietestSampleIndex(samples []float32, sampleRate int, startIdx, endIdx int) (int, error) {
	// Bounds clamping before array access
	if startIdx < 0 {
		startIdx = 0
	}
	if endIdx > len(samples) {
		endIdx = len(samples)
	}
	if startIdx >= endIdx {
		return startIdx, fmt.Errorf("invalid search region: start (%d) >= end (%d)", startIdx, endIdx)
	}

	windowSize := int(0.1 * float64(sampleRate)) // 100ms
	if windowSize <= 0 {
		windowSize = frameSize
	}

	// Invalid search region: check for minimum region size before searching
	if (endIdx - startIdx) < windowSize {
		return endIdx, nil // If region is too small, default to endIdx to avoid cutting words in half
	}

	var currentSumSq float64
	for i := startIdx; i < startIdx+windowSize; i++ {
		currentSumSq += float64(samples[i] * samples[i])
	}

	minSumSq := currentSumSq
	bestIndex := startIdx + (windowSize / 2)

	// O(n) sliding window approach.
	for i := startIdx + 1; i+windowSize <= endIdx; i++ {
		currentSumSq -= float64(samples[i-1] * samples[i-1])
		currentSumSq += float64(samples[i+windowSize-1] * samples[i+windowSize-1])

		if currentSumSq < minSumSq {
			minSumSq = currentSumSq
			bestIndex = i + (windowSize / 2)
		}
	}

	// Snap split point to nearest frame boundary.
	snapped := ((bestIndex + frameSize/2) / frameSize) * frameSize
	
	// Failsafe: absolute bounds check.
	if snapped <= startIdx {
		return startIdx + frameSize, nil
	}
	if snapped >= endIdx {
		return endIdx, nil
	}

	return snapped, nil
}
