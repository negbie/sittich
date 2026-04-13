package pipeline

import (
	"strings"

	"github.com/negbie/sittich/internal/speech"
)

// ChunkResult pairs an engine transcription result with the time offset (in
// seconds) of the chunk relative to the original audio.
type ChunkResult struct {
	Offset    float64        // Start time of this chunk in the original audio (including padding).
	OrigStart float64        // Original start time before padding.
	OrigEnd   float64        // Original end time before padding.
	Result    *speech.Result // Transcription result for the chunk.
}

// StitchResults merges multiple ChunkResults into a single speech.Result by
// offsetting chunk-relative timestamps into the original audio timeline and
// merging overlapping regions using a robust midpoint-temporal split.
func StitchResults(chunks []ChunkResult, padding float64, debug bool) *speech.Result {
	if len(chunks) == 0 {
		return &speech.Result{}
	}

	combined := &speech.Result{}
	if combined.Language == "" {
		for _, cr := range chunks {
			if cr.Result != nil && cr.Result.Language != "" {
				combined.Language = cr.Result.Language
				combined.LanguageProb = cr.Result.LanguageProb
				break
			}
		}
	}

	// 1. Shift all timestamps to global timeline
	for i := range chunks {
		cr := &chunks[i]
		if cr.Result == nil {
			continue
		}
		for j := range cr.Result.Segments {
			seg := &cr.Result.Segments[j]
			seg.Start += cr.Offset
			seg.End += cr.Offset
			for k := range seg.Words {
				w := &seg.Words[k]
				w.Start += cr.Offset
				w.End += cr.Offset
			}
		}
	}

	// 2. Merge chunks sequentially using temporal ownership
	allWords := make([]speech.Word, 0)
	var prevOrigEnd float64

	for _, cr := range chunks {
		if cr.Result == nil {
			continue
		}

		// Group raw tokens into logical words
		rawTokens := make([]speech.Word, 0)
		for _, seg := range cr.Result.Segments {
			rawTokens = append(rawTokens, seg.Words...)
		}
		chunkGroups := groupTokensToWords(rawTokens)
		if len(chunkGroups) == 0 {
			continue
		}

		if len(allWords) == 0 {
			// First chunk: keep everything within its logical range [OrigStart, OrigEnd]
			for _, g := range chunkGroups {
				mid := g.Start + (g.End-g.Start)/2
				if mid < cr.OrigEnd {
					allWords = append(allWords, g.Tokens...)
				}
			}
			prevOrigEnd = cr.OrigEnd
			continue
		}

		// Successive chunks: Split at the midpoint of the overlap
		splitPoint := (cr.OrigStart + prevOrigEnd) / 2

		// A. Trim existing words to the split point
		trimmedA := make([]speech.Word, 0, len(allWords))
		for _, w := range allWords {
			mid := w.Start + (w.End-w.Start)/2
			if mid < splitPoint {
				trimmedA = append(trimmedA, w)
			}
		}

		// B. Collect words from the new chunk starting from the split point
		trimmedB := make([]speech.Word, 0, len(rawTokens))
		for _, g := range chunkGroups {
			mid := g.Start + (g.End-g.Start)/2
			if mid >= splitPoint && mid < cr.OrigEnd {
				trimmedB = append(trimmedB, g.Tokens...)
			}
		}

		// C. Correct boundary artifacts (Dumb-but-Correct De-duplication)
		if len(trimmedA) > 0 && len(trimmedB) > 0 {
			// Get the last word of A and first word of B for comparison
			lastWordA := groupTokensToWords(trimmedA)
			firstWordB := groupTokensToWords(trimmedB)

			if len(lastWordA) > 0 && len(firstWordB) > 0 {
				wa := lastWordA[len(lastWordA)-1]
				wb := firstWordB[0]

				if normalizeForMatch(wa.Text()) == normalizeForMatch(wb.Text()) {
					// Drop the redundant word from the beginning of B
					trimmedB = trimmedB[len(wb.Tokens):]
				}
			}
		}

		allWords = append(trimmedA, trimmedB...)
		prevOrigEnd = cr.OrigEnd
	}

	// 3. Rebuild monolithic segment text from merged tokens
	var builder strings.Builder
	builder.Grow(len(allWords) * 6)
	for i, w := range allWords {
		t := w.Word
		if strings.HasPrefix(t, "\u2581") {
			if i > 0 {
				builder.WriteByte(' ')
			}
			t = strings.TrimPrefix(t, "\u2581")
		} else if strings.HasPrefix(t, " ") {
			if i > 0 {
				builder.WriteByte(' ')
			}
			t = strings.TrimPrefix(t, " ")
		}
		builder.WriteString(t)
	}
	resText := strings.TrimSpace(builder.String())

	combined.Segments = []speech.Segment{
		{
			ID:    0,
			Text:  resText,
			Words: allWords,
		},
	}
	if len(allWords) > 0 {
		combined.Segments[0].Start = allWords[0].Start
		combined.Segments[0].End = allWords[len(allWords)-1].End
		combined.Duration = combined.Segments[0].End
	}

	return combined
}

type wordGroup struct {
	Tokens []speech.Word
	Start  float64
	End    float64
}

func (g wordGroup) Text() string {
	var s string
	for _, t := range g.Tokens {
		s += t.Word
	}
	return s
}

// groupTokensToWords consolidates subword tokens into logical words.
func groupTokensToWords(tokens []speech.Word) []wordGroup {
	if len(tokens) == 0 {
		return nil
	}
	var groups []wordGroup
	var current wordGroup
	for _, t := range tokens {
		isNew := strings.HasPrefix(t.Word, "\u2581") || strings.HasPrefix(t.Word, " ")
		if isNew || len(current.Tokens) == 0 {
			if len(current.Tokens) > 0 {
				groups = append(groups, current)
			}
			current = wordGroup{Tokens: []speech.Word{t}, Start: t.Start, End: t.End}
		} else {
			current.Tokens = append(current.Tokens, t)
			current.End = t.End
		}
	}
	if len(current.Tokens) > 0 {
		groups = append(groups, current)
	}
	return groups
}

func normalizeForMatch(s string) string {
	s = strings.ReplaceAll(s, "\u2581", "")
	s = strings.ToLower(strings.TrimSpace(s))
	s = strings.Trim(s, ".,;!?: ")
	return s
}
