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
// merging overlapping regions using a token-level LCS algorithm.
func StitchResults(chunks []ChunkResult) *speech.Result {
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

	// 2. Merge chunks sequentially
	allWords := make([]speech.Word, 0)
	var prevOrigEnd float64
	for i, cr := range chunks {
		if cr.Result == nil || len(cr.Result.Segments) == 0 {
			continue
		}

		chunkWords := make([]speech.Word, 0)
		for _, seg := range cr.Result.Segments {
			chunkWords = append(chunkWords, seg.Words...)
		}

		// The padding region is for context only; its transcription shouldn't leak.
		if cr.OrigEnd > cr.OrigStart {
			filtered := make([]speech.Word, 0, len(chunkWords))
			for _, w := range chunkWords {
				mid := w.Start + (w.End-w.Start)/2
				// Important: Use strictly less than (<) for OrigEnd to ensure a word
				// with its midpoint exactly on the boundary is not duplicated.
				if mid >= cr.OrigStart && mid < cr.OrigEnd {
					filtered = append(filtered, w)
				}
			}
			chunkWords = filtered
		}

		if len(chunkWords) == 0 {
			continue
		}

		if i == 0 {
			allWords = append(allWords, chunkWords...)
			prevOrigEnd = cr.OrigEnd
			continue
		}

		// Handle overlap with the words already in allWords
		if len(allWords) == 0 {
			allWords = append(allWords, chunkWords...)
			prevOrigEnd = cr.OrigEnd
			continue
		}

		// Find relevant words in the overlap period
		// We use a window of overlap to search for common tokens
		lastTokenGlobalEnd := allWords[len(allWords)-1].End
		overlapStart := chunkWords[0].Start

		if overlapStart >= lastTokenGlobalEnd || cr.OrigStart >= prevOrigEnd {
			allWords = append(allWords, chunkWords...)
			prevOrigEnd = cr.OrigEnd
			continue
		}

		// Find anchor in allWords and chunkWords using LCS
		// Look back/forward up to 2 seconds or 20 words
		const windowWords = 20
		aStart := len(allWords) - windowWords
		if aStart < 0 {
			aStart = 0
		}
		a := allWords[aStart:]

		bEnd := windowWords
		if bEnd > len(chunkWords) {
			bEnd = len(chunkWords)
		}
		b := chunkWords[:bEnd]

		// Perform LCS on word text
		aTexts := make([]string, len(a))
		for i, w := range a {
			aTexts[i] = normalizeForMatch(w.Word)
		}
		bTexts := make([]string, len(b))
		for i, w := range b {
			bTexts[i] = normalizeForMatch(w.Word)
		}

		lcsIndices := findLCS(aTexts, bTexts)
		if len(lcsIndices) == 0 {
			// Bug #3 fallback: No LCS found, use temporal overlap to find a cut point.
			// We split at the midpoint of the temporal overlap between existing results
			// and the new chunk.
			cutPoint := (overlapStart + lastTokenGlobalEnd) / 2

			// Trim allWords to cutPoint
			newAllWords := make([]speech.Word, 0, len(allWords))
			for _, w := range allWords {
				if (w.Start + (w.End-w.Start)/2) < cutPoint {
					newAllWords = append(newAllWords, w)
				}
			}
			allWords = newAllWords

			// Append chunkWords starting from cutPoint
			for _, w := range chunkWords {
				if (w.Start + (w.End-w.Start)/2) >= cutPoint {
					allWords = append(allWords, w)
				}
			}
			continue
		}

		// Pick the middle of the LCS as the split point
		midLCS := len(lcsIndices) / 2
		idxA := aStart + lcsIndices[midLCS].i
		idxB := lcsIndices[midLCS].j

		// Keep allWords up to idxA (exclusive) and chunkWords from idxB (exclusive)
		// Let's keep the LCS token from allWords.
		allWords = allWords[:idxA+1]
		if idxB+1 < len(chunkWords) {
			allWords = append(allWords, chunkWords[idxB+1:]...)
		}

		prevOrigEnd = cr.OrigEnd
	}

	// 3. Rebuild segments from merged words
	var b strings.Builder
	b.Grow(len(allWords) * 6)
	for _, w := range allWords {
		b.WriteString(w.Word)
	}
	resText := strings.TrimSpace(b.String())

	combined.Segments = []speech.Segment{
		{
			ID:    0,
			Start: 0,
			End:   0,
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

type lcsMatch struct {
	i, j int
}

func findLCS(a, b []string) []lcsMatch {
	n, m := len(a), len(b)
	if n == 0 || m == 0 {
		return nil
	}
	dp := make([][]int, n+1)
	for i := range dp {
		dp[i] = make([]int, m+1)
	}

	for i := 1; i <= n; i++ {
		for j := 1; j <= m; j++ {
			if a[i-1] == b[j-1] && a[i-1] != "" {
				dp[i][j] = dp[i-1][j-1] + 1
			} else {
				if dp[i-1][j] > dp[i][j-1] {
					dp[i][j] = dp[i-1][j]
				} else {
					dp[i][j] = dp[i][j-1]
				}
			}
		}
	}

	matches := make([]lcsMatch, 0)
	i, j := n, m
	for i > 0 && j > 0 {
		if a[i-1] == b[j-1] && a[i-1] != "" {
			matches = append([]lcsMatch{{i - 1, j - 1}}, matches...)
			i--
			j--
		} else if dp[i-1][j] > dp[i][j-1] {
			i--
		} else {
			j--
		}
	}
	return matches
}

func normalizeForMatch(s string) string {
	s = strings.ToLower(strings.TrimSpace(s))
	s = strings.Trim(s, ".,;!?:")
	return s
}
