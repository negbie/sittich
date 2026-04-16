package pipeline

import (
	"context"
	"fmt"
	"math"
	"os"
	"sort"
	"strings"
	"time"

	"github.com/negbie/sittich/internal/audio"
	"github.com/negbie/sittich/internal/config"
	"github.com/negbie/sittich/internal/speech"
)

// Pipeline orchestrates audio processing and transcription.
type Pipeline struct {
	Engine speech.Engine
	Config config.Pipeline
}

// Process runs the transcription pipeline.
func (p *Pipeline) Process(ctx context.Context, path string, chunkDuration float64, soxFlags ...string) (*speech.Result, error) {
	const targetRate = 16000
	start := time.Now()
	
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("pipeline: open: %w", err)
	}
	defer f.Close()

	samples, err := audio.DecodeWAV(ctx, f, soxFlags...)
	if err != nil {
		return nil, fmt.Errorf("pipeline: decode: %w", err)
	}

	totalDur := float64(len(samples)) / float64(targetRate)
	if chunkDuration <= 0 {
		chunkDuration = p.Config.ChunkDuration
	}

	if p.Config.Debug {
		audio.DebugPlotWaveform(samples, "Input")
	}

	audio.ConditionAudioSignal(samples, targetRate)

	if p.Config.Debug {
		audio.DebugPlotWaveform(samples, "Processed")
		p.verifySignal(samples, targetRate)
	}

	overlap := p.Config.ChunkOverlapDuration
	padding := 0.6
	if overlap > 0 && overlap/2 > padding {
		padding = overlap / 2
	}

	chunks := ChunkAudioEnergyAware(samples, targetRate, chunkDuration, 5.0, overlap, padding)
	asrResults, err := p.transcribeChunks(ctx, samples, targetRate, chunks)
	if err != nil {
		return nil, err
	}

	if len(asrResults) == 0 {
		return &speech.Result{Duration: totalDur}, nil
	}

	sort.Slice(asrResults, func(i, j int) bool {
		return asrResults[i].Offset < asrResults[j].Offset
	})

	combined := StitchResults(asrResults, padding, p.Config.Debug)
	combined.Duration = totalDur

	for i := range combined.Segments {
		combined.Segments[i].Text = strings.TrimSpace(combined.Segments[i].Text)
	}

	if p.Config.Debug {
		fmt.Fprintf(os.Stderr, "   [Pipeline] Processed %d chunks in %s\n", 
			len(asrResults), time.Since(start).Round(time.Millisecond))
	}

	return combined, nil
}

func (p *Pipeline) transcribeChunks(ctx context.Context, samples []float32, targetRate int, chunks []Chunk) ([]ChunkResult, error) {
	batch := make([][]float32, 0, len(chunks))
	validIndices := make([]int, 0, len(chunks))

	for i, c := range chunks {
		start := int(c.Start * float64(targetRate))
		end := int(c.End * float64(targetRate))
		if start < 0 { start = 0 }
		if end > len(samples) { end = len(samples) }
		if start >= end { continue }

		chunk := make([]float32, end-start)
		copy(chunk, samples[start:end])
		batch = append(batch, chunk)
		validIndices = append(validIndices, i)
	}

	if len(batch) == 0 {
		return nil, nil
	}

	opts := speech.Options{
		Language:       p.Config.Language,
		WordTimestamps: p.Config.WordTimestamps,
		Debug:          p.Config.Debug,
	}

	results, err := p.Engine.TranscribeBatch(ctx, batch, targetRate, opts)
	if err != nil {
		return nil, err
	}

	chunkResults := make([]ChunkResult, len(results))
	for i, res := range results {
		ci := validIndices[i]
		chunkResults[i] = ChunkResult{
			Offset:    chunks[ci].Start,
			OrigStart: chunks[ci].OrigStart,
			OrigEnd:   chunks[ci].OrigEnd,
			Result:    res,
		}
	}
	return chunkResults, nil
}

func (p *Pipeline) verifySignal(samples []float32, targetRate int) {
	var peak float32
	for _, s := range samples {
		abs := float32(math.Abs(float64(s)))
		if abs > peak { peak = abs }
	}
	fmt.Fprintf(os.Stderr, "   [Pipeline] signal samples=%d duration=%.2fs peak=%.4f\n", 
		len(samples), float64(len(samples))/float64(targetRate), peak)
}
