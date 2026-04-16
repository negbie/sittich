package pipeline

const frameSize = 640 // 40ms frame at 16kHz

// Chunk represents a slice of audio for engine inference.
type Chunk struct {
	Start     float64
	End       float64
	OrigStart float64
	OrigEnd   float64
}

// ChunkAudioEnergyAware slices audio by finding the quietest moments (lowest RMS)
// near target boundaries. Split points are snapped to 40ms frame boundaries
// to prevent artifacts in model encoder states.
func ChunkAudioEnergyAware(samples []float32, sampleRate int, targetDuration, searchWindow, overlap, padding float64) []Chunk {
	totalSamples := len(samples)
	if totalSamples == 0 {
		return nil
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
				currentEnd = findQuietestSampleIndex(samples, sampleRate, sStart, currentEnd)
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
			Start:     float64(pStart) / float64(sampleRate),
			End:       float64(pEnd) / float64(sampleRate),
			OrigStart: float64(currentStart) / float64(sampleRate),
			OrigEnd:   float64(currentEnd) / float64(sampleRate),
		})

		if currentEnd >= totalSamples {
			break
		}
		
		nextStart := currentEnd - overlapSamples
		if nextStart <= currentStart {
			currentStart = currentEnd
		} else {
			currentStart = nextStart
		}
	}

	return chunks
}

// findQuietestSampleIndex returns the index of lowest RMS energy in the search
// region using a sliding window. Result is snapped to a 40ms frame boundary (640 samples).
func findQuietestSampleIndex(samples []float32, sampleRate int, startIdx, endIdx int) int {
	windowSize := int(0.1 * float64(sampleRate)) // 100ms
	if (endIdx - startIdx) <= windowSize {
		return endIdx
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
		return startIdx + frameSize
	}
	if snapped >= endIdx {
		return endIdx
	}

	return snapped
}
