package pipeline

import (
	"github.com/negbie/sittich/internal/types"
)

// ChunkResult pairs an engine transcription result with the time offset (in
// seconds) of the chunk relative to the original audio.
type ChunkResult struct {
	Offset float64       // Start time of this chunk in the original audio.
	Result *types.Result // Transcription result for the chunk.
}

// StitchResults merges multiple ChunkResults into a single types.Result by
// offsetting chunk-relative timestamps into the original audio timeline and
// appending segments in order.
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

			if shifted.End > maxEnd {
				maxEnd = shifted.End
			}

			combined.Segments = append(combined.Segments, shifted)
			segID++
		}
	}

	combined.Duration = maxEnd

	return combined
}
