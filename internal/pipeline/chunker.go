package pipeline

import (
	"math"
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
// silence boundaries. If a single speech segment exceeds maxDuration, it
// is split into smaller balanced sub-segments with overlapDuration.
// It uses energy-aware splitting for oversized segments if samples are provided.
func ChunkSpeechSegments(samples []float32, sampleRate int, segments []SpeechSegment, maxDuration float64, overlapDuration float64, minTailDuration float64) []Chunk {
	if len(segments) == 0 {
		return nil
	}

	// 1. Sort segments by Start time to ensure chronological processing
	sort.Slice(segments, func(i, j int) bool {
		return segments[i].Start < segments[j].Start
	})

	expanded := make([]SpeechSegment, 0, len(segments))
	for _, seg := range segments {
		if seg.Start >= seg.End {
			continue
		}

		duration := seg.End - seg.Start
		// Use a slightly lower threshold for "max" to leave room for balancing
		if duration <= maxDuration {
			expanded = append(expanded, seg)
			continue
		}

		// Split oversized segment into balanced chunks with overlap
		targetL := maxDuration * 0.95
		if targetL <= overlapDuration {
			targetL = maxDuration
		}

		n := int(math.Ceil((duration - overlapDuration) / (targetL - overlapDuration)))
		if n < 2 {
			n = 2
		}

		l := (duration-overlapDuration)/float64(n) + overlapDuration
		for l > maxDuration && n < 100 {
			n++
			l = (duration-overlapDuration)/float64(n) + overlapDuration
		}

		start := seg.Start
		for i := 0; i < n; i++ {
			end := start + l
			if i == n-1 || end > seg.End {
				end = seg.End
			} else if len(samples) > 0 {
				// Search for a better split point (energy minimum) in a 2s window
				end = findBestSplitPoint(samples, sampleRate, start, end, 2.0)
				// Ensure we don't exceed maxDuration after adjustment
				if end-start > maxDuration {
					end = start + maxDuration
				}
				// Ensure we don't go backwards or past the segment end
				if end <= start {
					end = start + l
				}
				if end > seg.End {
					end = seg.End
				}
			}

			if end > start {
				expanded = append(expanded, SpeechSegment{
					Start: start,
					End:   end,
				})
			}
			// Next chunk starts at end - overlap
			start = end - overlapDuration
		}
	}

	if len(expanded) == 0 {
		return nil
	}

	var chunks []Chunk
	if len(expanded) > 0 {
		chunkStart := expanded[0].Start
		chunkEnd := expanded[0].End

		for i := 1; i < len(expanded); i++ {
			seg := expanded[i]
			proposedEnd := seg.End
			proposedDuration := proposedEnd - chunkStart

			// If the segment itself is already an overlapping "piece" from a large split,
			// or if adding it exceeds maxDuration, we close the current chunk.
			// However, since we want overlaps, closing a chunk means the next one
			// should start 'overlapDuration' before the previous one ended if they were contiguous.
			
			if proposedDuration > maxDuration {
				chunks = append(chunks, Chunk{
					Start: chunkStart,
					End:   chunkEnd,
				})
				// Start new chunk. To maintain continuity if this was a VAD-based cut,
				// we could ideally add overlap, but VAD cuts are usually at silence.
				// For now, we just start at the next segment.
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
	}

	return chunks
}

// findBestSplitPoint scans a window around targetEndSec and returns the time
// in seconds of the sample with the lowest absolute amplitude.
func findBestSplitPoint(samples []float32, sampleRate int, startSec, targetEndSec, windowSec float64) float64 {
	winStart := targetEndSec - windowSec/2
	winEnd := targetEndSec + windowSec/2

	// Don't search before the segment start
	if winStart < startSec {
		winStart = startSec
	}

	startIdx := int(winStart * float64(sampleRate))
	endIdx := int(winEnd * float64(sampleRate))

	if startIdx < 0 {
		startIdx = 0
	}
	if endIdx > len(samples) {
		endIdx = len(samples)
	}
	if startIdx >= endIdx {
		return targetEndSec
	}

	minEnergy := float32(math.MaxFloat32)
	bestIdx := startIdx

	// We use a small sliding window average for robustness against single-sample spikes
	const smoothing = 16
	for i := startIdx; i+smoothing < endIdx; i++ {
		var energy float32
		for j := 0; j < smoothing; j++ {
			v := samples[i+j]
			if v < 0 {
				v = -v
			}
			energy += v
		}
		if energy < minEnergy {
			minEnergy = energy
			bestIdx = i + smoothing/2
		}
	}

	return float64(bestIdx) / float64(sampleRate)
}
