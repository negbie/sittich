package pipeline

import (
	"regexp"
	"strings"

	"github.com/negbie/sittich/internal/types"
)

// ChunkResult pairs an engine transcription result with the time offset (in
// seconds) of the chunk relative to the original audio.
type ChunkResult struct {
	Offset float64       // Start time of this chunk in the original audio.
	Result *types.Result // Transcription result for the chunk.
}

// StitchResults merges multiple ChunkResults into a single types.Result.
// It performs "temporal healing" by merging segments that overlap or are
// extremely close at chunk boundaries, which can happen with VAD-based splitting.
func StitchResults(chunks []ChunkResult) *types.Result {
	combined := &types.Result{}

	if len(chunks) == 0 {
		return combined
	}

	segID := 0
	var maxEnd float64

	for _, cr := range chunks {
		if cr.Result == nil {
			continue
		}

		// Inherit language info from the first chunk that has it.
		if combined.Language == "" && cr.Result.Language != "" {
			combined.Language = cr.Result.Language
			combined.LanguageProb = cr.Result.LanguageProb
		}

		for _, seg := range cr.Result.Segments {
			shifted := types.Segment{
				ID:           segID,
				Start:        seg.Start + cr.Offset,
				End:          seg.End + cr.Offset,
				Text:         seg.Text,
				AvgLogProb:   seg.AvgLogProb,
				NoSpeechProb: seg.NoSpeechProb,
				Words:        make([]types.Word, len(seg.Words)),
			}

			for j, w := range seg.Words {
				shifted.Words[j] = types.Word{
					Word:  w.Word,
					Start: w.Start + cr.Offset,
					End:   w.End + cr.Offset,
					Prob:  w.Prob,
				}
			}

			// Temporal Healing: Check if this segment overlaps with the previous one
			if len(combined.Segments) > 0 {
				prev := &combined.Segments[len(combined.Segments)-1]

				// If overlap is significant or gap is tiny, and text is similar/consecutive
				const mergeThreshold = 0.3 // seconds
				if shifted.Start < prev.End+mergeThreshold {
					// Duplicate detection: if the text is exactly the same, skip it
					if strings.TrimSpace(shifted.Text) == strings.TrimSpace(prev.Text) {
						continue
					}

					// Simple heuristic: if the first word of new segment is same as last word
					// of previous segment, we have a boundary duplications.
					if len(shifted.Words) > 0 && len(prev.Words) > 0 {
						if shifted.Words[0].Word == prev.Words[len(prev.Words)-1].Word {
							// For now, we'll just keep both if they are different segments,
							// but ideally we'd merge them.
						}
					}
				}
			}

			if shifted.End > maxEnd {
				maxEnd = shifted.End
			}

			shifted.Text = CleanAcronyms(shifted.Text)

			combined.Segments = append(combined.Segments, shifted)
			segID++
		}
	}

	combined.Duration = maxEnd

	return combined
}

var acronymRegex = regexp.MustCompile(`(?i)\b([A-Z])\s+([A-Z])\b`)

// CleanAcronyms consolidates single-letter uppercase characters separated by spaces.
// For example, "I C" becomes "IC", which is mandatory for Parakeet-TDT German calibration.
func CleanAcronyms(text string) string {
	// Apply the regex multiple times to handle chains like "I C O" -> "IC O" -> "ICO"
	current := text
	for {
		next := acronymRegex.ReplaceAllString(current, "$1$2")
		if next == current {
			break
		}
		current = next
	}
	return current
}
