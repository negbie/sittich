package asr

import (
	"context"
	"fmt"
	"math"
	"strings"
	"sync"
)

var punctuationReplacer = strings.NewReplacer(".", "", ",", "", "!", "", "?", "")

// Voter is an engine that wraps multiple engines and picks the best result
// for each chunk based on transcript length. This is effective for models
// that might miss speech but rarely hallucinate extra incorrect speech.
type Voter struct {
	engines []Engine
}

// NewVoter creates a new voter engine wrapper.
func NewVoter(engines ...Engine) *Voter {
	return &Voter{engines: engines}
}

func (v *Voter) Transcribe(ctx context.Context, audio []float32, sampleRate int, opts Options) (*Result, error) {
	results, err := v.TranscribeBatch(ctx, [][]float32{audio}, sampleRate, opts)
	if err != nil {
		return nil, err
	}
	if len(results) == 0 {
		return &Result{}, nil
	}
	return results[0], nil
}

func (v *Voter) TranscribeBatch(ctx context.Context, chunks [][]float32, sampleRate int, opts Options) ([]*Result, error) {
	if len(v.engines) == 0 {
		return nil, fmt.Errorf("voter: no engines configured")
	}
	if len(v.engines) == 1 {
		return v.engines[0].TranscribeBatch(ctx, chunks, sampleRate, opts)
	}

	type engineResult struct {
		index   int
		results []*Result
		err     error
	}

	resChan := make(chan engineResult, len(v.engines))
	for idx, e := range v.engines {
		go func(i int, engine Engine) {
			defer func() {
				if r := recover(); r != nil {
					resChan <- engineResult{index: i, err: fmt.Errorf("voter: engine %s panicked: %v", engine.ModelName(), r)}
				}
			}()
			res, err := engine.TranscribeBatch(ctx, chunks, sampleRate, opts)
			resChan <- engineResult{index: i, results: res, err: err}
		}(idx, e)
	}

	allResults := make([][]*Result, len(v.engines))
	var lastErr error
	receivedCount := 0
	for i := 0; i < len(v.engines); i++ {
		res := <-resChan
		if res.err != nil {
			lastErr = res.err
			continue
		}
		allResults[res.index] = res.results
		receivedCount++
	}

	if receivedCount == 0 {
		if lastErr != nil {
			return nil, fmt.Errorf("voter: all engines failed: %w", lastErr)
		}
		return nil, nil
	}

	// Pick the best result for each chunk index.
	// Strategy: always prefer the primary engine (index 0) for consistency across
	// chunks. Only fall back to a secondary engine if the primary produced empty
	// output or the secondary is substantially better (>20% score advantage).
	finalResults := make([]*Result, len(chunks))
	primaryIdx := 0

	for i := 0; i < len(chunks); i++ {
		var bestRes *Result
		var bestScore float64
		var bestIdx int
		var primaryRes *Result
		var primaryScore float64

		for idx, resSet := range allResults {
			if i >= len(resSet) || resSet[i] == nil {
				continue
			}

			score := scoreResult(resSet[i])
			if idx == primaryIdx {
				primaryRes = resSet[i]
				primaryScore = score
			}

			if opts.Debug {
				fmt.Printf("   [Voter] chunk=%d engine=%s confidence=%.4f score=%.2f text=%q\n",
					i, v.engines[idx].ModelName(), resSet[i].Confidence, score, resSet[i].FullText())
			}

			if score > bestScore {
				bestScore = score
				bestRes = resSet[i]
				bestIdx = idx
			}
		}

		// Prefer the primary engine unless it failed or the alternative is
		// substantially better. Cross-chunk engine consistency prevents tokenization
		// and capitalization mismatches in the stitcher.
		if primaryRes != nil && primaryRes.FullText() != "" {
			if bestIdx != primaryIdx && primaryScore < bestScore*0.8 {
				// Secondary is >20% better — use it.
				if opts.Debug {
					fmt.Printf("   [Voter] chunk=%d OVERRIDING primary (primary_score=%.2f best_score=%.2f by %.0f%%)\n",
						i, primaryScore, bestScore, (1-primaryScore/bestScore)*100)
				}
			} else {
				bestRes = primaryRes
				bestIdx = primaryIdx
			}
		}

		if opts.Debug && bestRes != nil {
			fmt.Printf("   [Voter] chunk=%d CHOSE %s\n", i, v.engines[bestIdx].ModelName())
		}
		finalResults[i] = bestRes
	}

	return finalResults, nil
}

func scoreResult(res *Result) float64 {
	text := res.FullText()
	if text == "" {
		return 0
	}

	// 1. Structure Score: Reward punctuation (periods and commas)
	// We use a smaller multiplier (2.0) to prefer structured text
	// without letting punctuation dominate the content score.
	periods := strings.Count(text, ".")
	commas := strings.Count(text, ",")
	score := float64(periods+commas) * 2.0

	// 2. Content Score: Reward information density
	// We strip punctuation and use the square root of word length
	// to reward long content words more than short function words.
	cleanText := punctuationReplacer.Replace(text)
	words := strings.Fields(cleanText)
	for _, w := range words {
		score += math.Sqrt(float64(len(w)))
	}

	if len(words) == 0 {
		return 0
	}

	// 3. Quality Signal: Type-Token Ratio (TTR)
	// Instead of a hard threshold, we use uniqueness as a continuous multiplier.
	uniqueWords := make(map[string]int)
	consecutiveRepeats := 0
	for i, w := range words {
		w = strings.ToLower(w)
		uniqueWords[w]++
		if i > 0 && w == strings.ToLower(words[i-1]) {
			consecutiveRepeats++
		}
	}

	ttr := float64(len(uniqueWords)) / float64(len(words))
	score *= ttr

	// Penalty for excessive consecutive repeats
	if consecutiveRepeats > 3 {
		score *= 0.5
	}

	// 4. Confidence Score: Scale by model certainty
	// res.Confidence is avg log-prob (<= 0). By multiplying confidence by 2.0
	// before exponentiating, we penalize low-confidence outputs (like
	// hallucinations) more aggressively.
	if res.Confidence != 0 {
		score *= math.Exp(res.Confidence * 2.0)
	}

	return score
}

func (v *Voter) SupportedLanguages() []string {
	if len(v.engines) == 0 {
		return nil
	}
	return v.engines[0].SupportedLanguages()
}

func (v *Voter) ModelName() string {
	names := make([]string, len(v.engines))
	for i, e := range v.engines {
		names[i] = e.ModelName()
	}
	return fmt.Sprintf("voter(%s)", strings.Join(names, "|"))
}

func (v *Voter) VADPath() string {
	if len(v.engines) == 0 {
		return ""
	}
	return v.engines[0].VADPath()
}

func (v *Voter) Close() error {
	var errs []string
	var wg sync.WaitGroup
	var mu sync.Mutex

	for _, e := range v.engines {
		wg.Add(1)
		go func(engine Engine) {
			defer wg.Done()
			if err := engine.Close(); err != nil {
				mu.Lock()
				errs = append(errs, err.Error())
				mu.Unlock()
			}
		}(e)
	}
	wg.Wait()

	if len(errs) > 0 {
		return fmt.Errorf("voter close errors: %s", strings.Join(errs, "; "))
	}
	return nil
}
