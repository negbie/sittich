package pipeline

import (
	"strings"

	"github.com/negbie/sittich/internal/asr"
)

// ChunkResult pairs an engine transcription result with its original offset.
type ChunkResult struct {
	Offset    float64
	OrigStart float64
	OrigEnd   float64
	Result    *asr.Result
}

// StitchResults merges multiple ChunkResults into a single asr.Result.
func StitchResults(chunks []ChunkResult, padding float64, debug bool) *asr.Result {
	if len(chunks) == 0 {
		return &asr.Result{}
	}

	combined := &asr.Result{}
	if combined.Language == "" {
		for _, cr := range chunks {
			if cr.Result != nil && cr.Result.Language != "" {
				combined.Language = cr.Result.Language
				break
			}
		}
	}

	// 1. Sort chunks by offset for sequential stitching
	// (Already sorted in pipeline.go, but we do 2. merge next)

	// 2. Merge chunks sequentially using temporal ownership
	allWords := make([]asr.Word, 0)
	var prevOrigEnd float64

	for _, cr := range chunks {
		if cr.Result == nil || len(cr.Result.Segments) == 0 {
			continue
		}

		rawTokens := make([]asr.Word, 0)
		for _, seg := range cr.Result.Segments {
			rawTokens = append(rawTokens, seg.Words...)
		}

		if len(allWords) == 0 {
			for _, w := range rawTokens {
				w.Start += cr.Offset
				w.End += cr.Offset
				if w.Start < cr.OrigEnd {
					allWords = append(allWords, w)
				}
			}
			prevOrigEnd = cr.OrigEnd
			continue
		}

		chunkGroups := make([]asr.Word, 0, len(rawTokens))
		for _, g := range rawTokens {
			g.Start += cr.Offset
			g.End += cr.Offset
			chunkGroups = append(chunkGroups, g)
		}

		splitPoint := (cr.OrigStart + prevOrigEnd) / 2

		trimmedA := make([]asr.Word, 0, len(allWords))
		for _, w := range allWords {
			mid := w.Start + (w.End-w.Start)/2
			if mid < splitPoint {
				trimmedA = append(trimmedA, w)
			}
		}

		trimmedB := make([]asr.Word, 0, len(rawTokens))
		for _, g := range chunkGroups {
			mid := g.Start + (g.End-g.Start)/2
			if mid >= splitPoint && mid < cr.OrigEnd {
				trimmedB = append(trimmedB, g)
			}
		}

		allWords = append(trimmedA, trimmedB...)
		prevOrigEnd = cr.OrigEnd
	}

	// 3. Rebuild text from stitched words with SmartJoin for BPE/SentencePiece tokens.
	var builder strings.Builder
	for i, w := range allWords {
		t := w.Word
		// SentencePiece uses U+2581 (lower one eighth block ' ') as a space prefix.
		if strings.HasPrefix(t, "\u2581") {
			if i > 0 {
				builder.WriteString(" ")
			}
			builder.WriteString(strings.TrimPrefix(t, "\u2581"))
		} else {
			// If it doesn't have the space marker, it's a subtoken fragment.
			// Join it directly to the previous word.
			builder.WriteString(t)
		}
	}
	resText := strings.TrimSpace(builder.String())

	// Clean up markers from individual word tokens for JSON/metadata consumers.
	for i := range allWords {
		allWords[i].Word = strings.TrimPrefix(allWords[i].Word, "\u2581")
	}

	combined.Segments = []asr.Segment{
		{
			ID:    0,
			Text:  resText,
			Start: 0,
			End:   0, // Final duration set by Pipeline.Process
			Words: allWords,
		},
	}

	return combined
}
