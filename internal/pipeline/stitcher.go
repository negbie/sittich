package pipeline

import (
	"math"
	"sort"
	"strings"

	"github.com/negbie/sittich/internal/asr"
)

type ChunkResult struct {
	Offset    int
	OrigStart int
	OrigEnd   int
	Result    *asr.Result
}

func StitchResults(chunks []ChunkResult, sampleRate int) *asr.Result {
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

	// Collect words with absolute timestamps, filtered by ownership windows.
	allWords := make([]asr.Word, 0)
	for i, cr := range chunks {
		if cr.Result == nil {
			continue
		}

		for _, seg := range cr.Result.Segments {
			for _, w := range seg.Words {
				offsetSec := float64(cr.Offset) / float64(sampleRate)
				absStart := w.Start + offsetSec
				absEnd := w.End + offsetSec

				wStartSamples := int(math.Round(w.Start * float64(sampleRate)))
				wEndSamples := int(math.Round(w.End * float64(sampleRate)))
				midSamples := cr.Offset + wStartSamples + (wEndSamples-wStartSamples)/2

				isFirst := i == 0
				isLast := i == len(chunks) - 1

				inWindow := midSamples >= cr.OrigStart && midSamples < cr.OrigEnd
				if isFirst && midSamples < cr.OrigEnd {
					inWindow = true
				}
				if isLast && midSamples >= cr.OrigStart {
					inWindow = true
				}

				if inWindow {
					w.Start = absStart
					w.End = absEnd
					allWords = append(allWords, w)
				}
			}
		}
	}

	sort.Slice(allWords, func(i, j int) bool {
		return allWords[i].Start < allWords[j].Start
	})

	// Safety net: remove exact duplicates (same word, same timestamp within 10ms)
	// that slip through the ownership window filter from overlapping chunks.
	if len(allWords) > 1 {
		deduped := allWords[:1]
		for i := 1; i < len(allWords); i++ {
			prev := deduped[len(deduped)-1]
			curr := allWords[i]
			if math.Abs(curr.Start-prev.Start) < 0.01 && curr.Word == prev.Word {
				continue
			}
			deduped = append(deduped, curr)
		}
		allWords = deduped
	}

	// Split into segments at natural sentence boundaries (gaps > 2s between words).
	const sentenceGap = 2.0
	var segments []asr.Segment
	if len(allWords) > 0 {
		segStart := 0
		for i := 1; i <= len(allWords); i++ {
			if i == len(allWords) || allWords[i].Start-allWords[i-1].End > sentenceGap {
				segWords := allWords[segStart:i]
				segments = append(segments, asr.Segment{
					ID:    len(segments),
					Text:  cleanTokenText(segWords),
					Start: segWords[0].Start,
					End:   segWords[len(segWords)-1].End,
					Words: segWords,
				})
				segStart = i
			}
		}
	}

	return &asr.Result{
		Language: language,
		Segments: segments,
	}
}

// cleanTokenText joins word tokens into readable text, handling SentencePiece
// \u2581 space markers, and mutates each word's token to the cleaned form.
func cleanTokenText(words []asr.Word) string {
	var builder strings.Builder
	for i := range words {
		hasSpace := strings.HasPrefix(words[i].Word, "\u2581")
		cleanToken := strings.TrimPrefix(words[i].Word, "\u2581")

		if hasSpace && i > 0 {
			builder.WriteString(" ")
		}
		builder.WriteString(cleanToken)

		words[i].Word = cleanToken
	}
	return strings.TrimSpace(builder.String())
}
