package pipeline

import (
	"sort"
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

func StitchResults(chunks []ChunkResult, padding float64, debug bool) *asr.Result {
	if len(chunks) == 0 {
		return &asr.Result{}
	}

	var language string
	for _, cr := range chunks {
		if cr.Result != nil && cr.Result.Language != "" {
			language = cr.Result.Language
			break
		}
	}

	allWords := make([]asr.Word, 0)

	for i, cr := range chunks {
		if cr.Result == nil {
			continue
		}

		rawWords := make([]asr.Word, 0)
		for _, seg := range cr.Result.Segments {
			rawWords = append(rawWords, seg.Words...)
		}

		for _, w := range rawWords {
			absStart := w.Start + cr.Offset
			absEnd := w.End + cr.Offset
			mid := absStart + (absEnd-absStart)/2

			isFirst := i == 0
			isLast := i == len(chunks)-1

			inWindow := mid >= cr.OrigStart && mid < cr.OrigEnd

			// Defensive guards for absolute file boundaries (first and last chunks).
			if isFirst && mid < cr.OrigEnd {
				inWindow = true
			}
			if isLast && mid >= cr.OrigStart {
				inWindow = true
			}

			if inWindow {
				w.Start = absStart
				w.End = absEnd
				allWords = append(allWords, w)
			}
		}
	}

	// Safety sort in case chunks or words arrived out of order.
	sort.Slice(allWords, func(i, j int) bool {
		return allWords[i].Start < allWords[j].Start
	})

	var builder strings.Builder
	for i := range allWords {
		t := allWords[i].Word
		hasSpace := strings.HasPrefix(t, "\u2581")
		cleanToken := strings.TrimPrefix(t, "\u2581")

		if hasSpace && i > 0 {
			builder.WriteString(" ")
		}
		builder.WriteString(cleanToken)

		// Update the word in the list for the final result metadata.
		allWords[i].Word = cleanToken
	}

	var start, end float64
	if len(allWords) > 0 {
		start = allWords[0].Start
		end = allWords[len(allWords)-1].End
	}

	return &asr.Result{
		Language: language,
		Segments: []asr.Segment{
			{
				ID:    0,
				Text:  strings.TrimSpace(builder.String()),
				Start: start,
				End:   end,
				Words: allWords,
			},
		},
	}
}
