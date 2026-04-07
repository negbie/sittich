package pipeline

import (
	"sort"
)

// Chunk represents a slice of audio destined for a single engine inference
// call. Start and End are in seconds relative to the original audio.
type Chunk struct {
	Start float64
	End   float64
}

// ChunkSpeechSegments groups consecutive SpeechSegments into Chunks whose
// total duration does not exceed maxDuration seconds. It never splits a
// single SpeechSegment -- cuts are always placed at silence boundaries
// between segments.
func ChunkSpeechSegments(segments []SpeechSegment, maxDuration float64) []Chunk {
	if len(segments) == 0 {
		return nil
	}
	if maxDuration <= 0 {
		maxDuration = DefaultChunkDuration
	}

	// 1. Sort segments by Start time to ensure chronological processing
	// and prevent "backwards" chunks if segments are out of order.
	sort.Slice(segments, func(i, j int) bool {
		return segments[i].Start < segments[j].Start
	})

	var chunks []Chunk

	// Initialise with the first valid segment
	var firstIdx int
	for firstIdx < len(segments) && segments[firstIdx].Start >= segments[firstIdx].End {
		firstIdx++
	}
	if firstIdx >= len(segments) {
		return nil
	}

	chunkStart := segments[firstIdx].Start
	chunkEnd := segments[firstIdx].End

	for i := firstIdx + 1; i < len(segments); i++ {
		seg := segments[i]

		// Skip invalid or zero-duration segments
		if seg.Start >= seg.End {
			continue
		}

		proposedEnd := seg.End
		proposedDuration := proposedEnd - chunkStart

		if proposedDuration > maxDuration {
			// Close the current chunk at the end of the previous segment.
			chunks = append(chunks, Chunk{
				Start: chunkStart,
				End:   chunkEnd,
			})
			// Start a new chunk from this segment.
			chunkStart = seg.Start
			chunkEnd = seg.End
		} else {
			// Extend current chunk to include this segment.
			// Note: We use max(chunkEnd, seg.End) to handle slightly overlapping
			// segments from the VAD more robustly.
			if seg.End > chunkEnd {
				chunkEnd = seg.End
			}
		}
	}

	// Emit the final chunk.
	if chunkEnd > chunkStart {
		chunks = append(chunks, Chunk{
			Start: chunkStart,
			End:   chunkEnd,
		})
	}

	return chunks
}
