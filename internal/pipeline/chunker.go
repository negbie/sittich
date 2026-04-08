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
// total duration does not exceed maxDuration seconds. It prefers cuts at
// silence boundaries, but if a single speech segment exceeds maxDuration it
// is split into smaller balanced sub-segments first so ASR never receives an
// oversized chunk or a tiny tail chunk solely because VAD under-segmented
// the audio.
func ChunkSpeechSegments(segments []SpeechSegment, maxDuration float64, minTailDuration float64) []Chunk {
	if len(segments) == 0 {
		return nil
	}

	const minBalancedChunks = 2

	// 1. Sort segments by Start time to ensure chronological processing
	// and prevent "backwards" chunks if segments are out of order.
	sort.Slice(segments, func(i, j int) bool {
		return segments[i].Start < segments[j].Start
	})

	expanded := make([]SpeechSegment, 0, len(segments))
	for _, seg := range segments {
		if seg.Start >= seg.End {
			continue
		}

		duration := seg.End - seg.Start
		if duration <= maxDuration {
			expanded = append(expanded, seg)
			continue
		}

		pieces := int(duration / maxDuration)
		if pieces < minBalancedChunks {
			pieces = minBalancedChunks
		}
		if duration-float64(pieces)*maxDuration > minTailDuration {
			pieces++
		}

		step := duration / float64(pieces)
		if step > maxDuration {
			step = maxDuration
		}

		start := seg.Start
		for i := 0; i < pieces; i++ {
			end := start + step
			if i == pieces-1 || end > seg.End {
				end = seg.End
			}
			if end > start {
				expanded = append(expanded, SpeechSegment{
					Start: start,
					End:   end,
				})
			}
			start = end
		}
	}

	if len(expanded) == 0 {
		return nil
	}

	var chunks []Chunk

	chunkStart := expanded[0].Start
	chunkEnd := expanded[0].End

	for i := 1; i < len(expanded); i++ {
		seg := expanded[i]

		proposedEnd := seg.End
		proposedDuration := proposedEnd - chunkStart

		if proposedDuration > maxDuration {
			chunks = append(chunks, Chunk{
				Start: chunkStart,
				End:   chunkEnd,
			})
			chunkStart = seg.Start
			chunkEnd = seg.End
			continue
		}

		if seg.End > chunkEnd {
			chunkEnd = seg.End
		}
	}

	if chunkEnd > chunkStart {
		chunks = append(chunks, Chunk{
			Start: chunkStart,
			End:   chunkEnd,
		})
	}

	return chunks
}
