package audio

import (
	"fmt"
	"os"
	"strings"
)

// DebugPlotWaveform prints a high-resolution waveform graph to the terminal.
func DebugPlotWaveform(samples []float32, title string) {
	if len(samples) == 0 {
		fmt.Fprintf(os.Stderr, "   [DSP] %s: No samples to plot\n", title)
		return
	}

	const termWidth = 100
	const termHeight = 11

	bucketSize := len(samples) / termWidth
	if bucketSize == 0 {
		bucketSize = 1
	}

	mins := make([]float32, termWidth)
	maxs := make([]float32, termWidth)

	var globalMax float32
	for i := 0; i < termWidth; i++ {
		start := i * bucketSize
		end := start + bucketSize
		if end > len(samples) {
			end = len(samples)
		}
		if start >= end {
			break
		}

		var bmin, bmax float32 = samples[start], samples[start]
		for j := start; j < end; j++ {
			if samples[j] < bmin {
				bmin = samples[j]
			}
			if samples[j] > bmax {
				bmax = samples[j]
			}

			absS := samples[j]
			if absS < 0 {
				absS = -absS
			}
			if absS > globalMax {
				globalMax = absS
			}
		}
		mins[i] = bmin
		maxs[i] = bmax
	}

	fmt.Fprintf(os.Stderr, "\n   [DSP] %s (length: %.2fs, peak: %.4f)\n", title, float64(len(samples))/16000.0, globalMax)

	// Map +1.0 to -1.0
	for row := 0; row < termHeight; row++ {
		rowMax := 1.0 - float32(row)*(2.0/float32(termHeight))
		rowMin := 1.0 - float32(row+1)*(2.0/float32(termHeight))

		var b strings.Builder
		// Y-axis labels mapping
		if row == 0 {
			b.WriteString(fmt.Sprintf("   %5.1f ▏", 1.0))
		} else if row == termHeight/2 {
			b.WriteString(fmt.Sprintf("   %5.1f ▏", 0.0))
		} else if row == termHeight-1 {
			b.WriteString(fmt.Sprintf("   %5.1f ▏", -1.0))
		} else {
			b.WriteString("         ▏")
		}

		for col := 0; col < termWidth; col++ {
			pMax := maxs[col]
			pMin := mins[col]

			// Fill if the envelope passes through this row
			if pMax >= rowMin && pMin <= rowMax {
				// Colored based on amplitude
				amp := pMax
				if -pMin > amp {
					amp = -pMin
				}

				if amp > 0.8 {
					b.WriteString("\033[31m█\033[0m") // Red
				} else if amp > 0.4 {
					b.WriteString("\033[33m█\033[0m") // Yellow
				} else if amp > 0.1 {
					b.WriteString("\033[32m█\033[0m") // Green
				} else {
					b.WriteString("\033[36m█\033[0m") // Cyan/Blueish for quiet
				}
			} else if rowMin < 0.0 && rowMax > 0.0 { // Center line
				b.WriteString("\033[90m-\033[0m")
			} else {
				b.WriteString(" ")
			}
		}
		fmt.Fprintln(os.Stderr, b.String())
	}
}
