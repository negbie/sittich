package audio

import (
	"fmt"
	"math"
	"os"
	"strings"
)

// DebugPlotWaveform prints a high-resolution Braille waveform graph to the terminal.
func DebugPlotWaveform(samples []float32, title string) {
	if len(samples) == 0 {
		fmt.Fprintf(os.Stderr, "   [DSP] %s: No samples to plot\n", title)
		return
	}

	const charWidth = 100
	const charHeight = 3 // Total 12 dots vertical
	const dotsPerCharW = 2
	const dotsPerCharH = 4

	totalDotsW := charWidth * dotsPerCharW
	totalDotsH := charHeight * dotsPerCharH

	bucketSize := len(samples) / totalDotsW
	if bucketSize == 0 {
		bucketSize = 1
	}

	// Track peaks and RMS for both visualization and stats
	peaks := make([]float32, totalDotsW)
	rmss := make([]float32, totalDotsW)
	var globalPeak float32
	var globalRMS float64
	var clippedSamples int
	var saturatedSamples int // Near 1.0 or limiter ceiling

	for i := 0; i < totalDotsW; i++ {
		start := i * bucketSize
		end := start + bucketSize
		if end > len(samples) {
			end = len(samples)
		}
		if start >= end {
			break
		}

		var bMax float32
		var bSumSq float64
		for j := start; j < end; j++ {
			v := samples[j]
			if v < 0 {
				v = -v
			}
			if v > bMax {
				bMax = v
			}
			bSumSq += float64(v * v)
		}
		peaks[i] = bMax
		rmss[i] = float32(math.Sqrt(bSumSq / float64(end-start)))

		if bMax > globalPeak {
			globalPeak = bMax
		}
	}

	for _, s := range samples {
		absS := float32(math.Abs(float64(s)))
		globalRMS += float64(s * s)
		if absS >= 0.999 {
			clippedSamples++
		}
		if absS >= 0.8 {
			saturatedSamples++
		}
	}
	globalRMS = math.Sqrt(globalRMS / float64(len(samples)))
	globalRMSdB := 20 * math.Log10(globalRMS+1e-9)
	globalPeakdB := 20 * math.Log10(float64(globalPeak)+1e-9)
	crestFactor := globalPeakdB - globalRMSdB

	fmt.Fprintf(os.Stderr, "\n   [DSP] %s\n", title)
	fmt.Fprintf(os.Stderr, "   %-18s %-18s %-18s %-18s\n",
		fmt.Sprintf("Length: %.2fs", float64(len(samples))/16000.0),
		fmt.Sprintf("Peak: %.2f dB", globalPeakdB),
		fmt.Sprintf("RMS: %.2f dB", globalRMSdB),
		fmt.Sprintf("Crest: %.2f dB", crestFactor))

	// Warning indicators
	if clippedSamples > 0 || saturatedSamples > 0 {
		warn := "   ⚠️  "
		if clippedSamples > 0 {
			warn += fmt.Sprintf("\033[31mCLIPPING DETECTED: %d samples hit 0dBFS!\033[0m ", clippedSamples)
		}
		if saturatedSamples > 100 {
			pct := float64(saturatedSamples) / float64(len(samples)) * 100
			warn += fmt.Sprintf("\033[33mSATURATION: %.1f%% of audio is in the limiter zone.\033[0m", pct)
		}
		fmt.Fprintln(os.Stderr, warn)
	}
	if crestFactor < 9.0 {
		fmt.Fprintf(os.Stderr, "   💡 \033[35mADVICE: Crest Factor is low (%.2f dB). Audio is very 'squashed/screamy'.\033[0m\n", crestFactor)
	}

	// Drawing buffer: grid of bits for each dot
	grid := make([][]bool, totalDotsH)
	for i := range grid {
		grid[i] = make([]bool, totalDotsW)
	}

	halfH := float32(totalDotsH) / 2.0
	for x := 0; x < totalDotsW; x++ {
		p := peaks[x]
		r := rmss[x]

		// Draw peak spikes (symmetrical about center)
		pHeight := int(p * halfH)
		for y := 0; y < pHeight; y++ {
			yTop := int(halfH) - 1 - y
			yBot := int(halfH) + y
			if yTop >= 0 && yTop < totalDotsH {
				grid[yTop][x] = true
			}
			if yBot >= 0 && yBot < totalDotsH {
				grid[yBot][x] = true
			}
		}

		// Draw RMS "body" (solid near center)
		rHeight := int(r * halfH)
		for y := 0; y < rHeight; y++ {
			yTop := int(halfH) - 1 - y
			yBot := int(halfH) + y
			if yTop >= 0 && yTop < totalDotsH {
				grid[yTop][x] = true
			}
			if yBot >= 0 && yBot < totalDotsH {
				grid[yBot][x] = true
			}
		}
	}

	// Find the last column that actually contains signal/dots
	lastActiveX := -1
	for x := totalDotsW - 1; x >= 0; x-- {
		active := false
		for y := 0; y < totalDotsH; y++ {
			if grid[y][x] {
				active = true
				break
			}
		}
		if active {
			lastActiveX = x
			break
		}
	}
	lastActiveChar := lastActiveX / dotsPerCharW

	// Render the grid using Braille characters
	for row := 0; row < charHeight; row++ {
		var b strings.Builder
		// Y-axis legend
		if row == 0 {
			b.WriteString("    1.0 ┤")
		} else if row == charHeight/2 {
			b.WriteString("    0.0 ┼")
		} else if row == charHeight-1 {
			b.WriteString("   -1.0 ┤")
		} else {
			b.WriteString("        │")
		}

		for col := 0; col < charWidth; col++ {
			// Braille dots numbering:
			// 1 4
			// 2 5
			// 3 6
			// 7 8
			char := rune(0x2800)
			dots := []struct {
				dx, dy int
				bit    rune
			}{
				{0, 0, 0x01}, {0, 1, 0x02}, {0, 2, 0x04}, {1, 0, 0x08},
				{1, 1, 0x10}, {1, 2, 0x20}, {0, 3, 0x40}, {1, 3, 0x80},
			}

			var maxAmp float32
			for _, d := range dots {
				gx := col*dotsPerCharW + d.dx
				gy := row*dotsPerCharH + d.dy
				if gx < totalDotsW && gy < totalDotsH && grid[gy][gx] {
					char |= d.bit
					// Track max amp in this character for coloring
					if peaks[gx] > maxAmp {
						maxAmp = peaks[gx]
					}
				}
			}

			if char == 0x2800 {
				if row == charHeight/2 && col <= lastActiveChar {
					b.WriteString("\033[90m─\033[0m") // Solid slim center line
				} else {
					b.WriteString(" ")
				}
			} else {
				// Apply coloring based on amplitude
				if maxAmp > 0.8 {
					b.WriteString(fmt.Sprintf("\033[31m%c\033[0m", char)) // Red
				} else if maxAmp > 0.4 {
					b.WriteString(fmt.Sprintf("\033[33m%c\033[0m", char)) // Yellow
				} else if maxAmp > 0.1 {
					b.WriteString(fmt.Sprintf("\033[32m%c\033[0m", char)) // Green
				} else {
					b.WriteString(fmt.Sprintf("\033[36m%c\033[0m", char)) // Cyan
				}
			}
		}
		fmt.Fprintln(os.Stderr, b.String())
	}
	fmt.Fprintln(os.Stderr)
}
