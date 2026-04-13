package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
)

var httpClient = &http.Client{
	Timeout: 60 * time.Second,
	Transport: &http.Transport{
		MaxIdleConns:        16,
		MaxIdleConnsPerHost: 16,
		IdleConnTimeout:     90 * time.Second,
	},
}

type BenchmarkConfig struct {
	AudioDir    string `json:"audio_dir"`
	GroundTruth string `json:"ground_truth"` // Path to JSON file mapping filename to text
}

type TranscribeResponse struct {
	Success bool   `json:"success"`
	Text    string `json:"text"`
	Error   string `json:"error"`
}

type Result struct {
	Flags    []string
	WER      float64
	Latency  time.Duration
	Filename string
}

type Summary struct {
	Flags      []string
	AverageWER float64
	MaxWER     float64
	AvgLatency time.Duration
}

func main() {
	serverURL := flag.String("url", "http://localhost:5092/transcribe", "Sittich server URL")
	audioDir := flag.String("audio", "./audio", "Directory containing WAV files")
	truthFile := flag.String("truth", "./truth.json", "JSON file mapping filename to ground truth text")
	parallel := flag.Int("parallel", 4, "Number of parallel benchmark workers")
	flag.Parse()

	// 1. Load Ground Truth
	truthData, err := os.ReadFile(*truthFile)
	if err != nil {
		fmt.Printf("Error reading truth file: %v\n", err)
		os.Exit(1)
	}
	var truth map[string]string
	if err := json.Unmarshal(truthData, &truth); err != nil {
		fmt.Printf("Error parsing truth JSON: %v\n", err)
		os.Exit(1)
	}

	// 2. Define Search Space
	searchSpace := generateSearchSpace()
	fmt.Printf("Searching through %d Sox flag combinations...\n", len(searchSpace))

	// 3. Prepare Jobs
	type Job struct {
		Filename string
		Audio    []byte
		Truth    string
		Flags    []string
	}

	resultsChan := make(chan Result, len(searchSpace)*len(truth))
	jobsChan := make(chan Job, 100)

	var wg sync.WaitGroup
	for i := 0; i < *parallel; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for job := range jobsChan {
				resText, latency, err := callServer(*serverURL, job.Filename, job.Audio, job.Flags)
				if err != nil {
					fmt.Printf("Error calling server for %s with %v: %v\n", job.Filename, job.Flags, err)
					continue
				}
				wer := calculateWER(job.Truth, resText)
				resultsChan <- Result{
					Flags:    job.Flags,
					WER:      wer,
					Latency:  latency,
					Filename: job.Filename,
				}
			}
		}()
	}

	// 4. Dispatch Jobs
	go func() {
		for filename, expected := range truth {
			audioPath := filepath.Join(*audioDir, filename)
			audioData, err := os.ReadFile(audioPath)
			if err != nil {
				fmt.Printf("Error reading audio %s: %v\n", audioPath, err)
				continue
			}

			for _, flags := range searchSpace {
				jobsChan <- Job{
					Filename: filename,
					Audio:    audioData,
					Truth:    expected,
					Flags:    flags,
				}
			}
		}
		close(jobsChan)
	}()

	wg.Wait()
	close(resultsChan)

	// 5. Aggregate Results
	summaries, bestPerFile := aggregateResults(resultsChan)
	sort.Slice(summaries, func(i, j int) bool {
		if summaries[i].AverageWER != summaries[j].AverageWER {
			return summaries[i].AverageWER < summaries[j].AverageWER
		}
		return summaries[i].AvgLatency < summaries[j].AvgLatency
	})

	// 6. Report Best Per File
	fmt.Printf("\n%s%-40s %-10s %-50s%s\n", "\033[1m", "Filename", "WER", "Best Sox Flags", "\033[0m")
	fmt.Println(strings.Repeat("-", 100))
	filenames := make([]string, 0, len(bestPerFile))
	for f := range bestPerFile {
		filenames = append(filenames, f)
	}
	sort.Strings(filenames)
	for _, f := range filenames {
		res := bestPerFile[f]
		flagStr := strings.Join(res.Flags, " ")
		if flagStr == "" {
			flagStr = "(default)"
		}
		fmt.Printf("%-40s %-10.2f %-50s\n", f, res.WER*100, flagStr)
	}

	// 7. Report Overall Best
	fmt.Printf("\n%s%-60s %-10s %-10s %-10s%s\n", "\033[1m", "Overall Best Sox Flags (Avg)", "Avg WER", "Max WER", "Avg Lat", "\033[0m")
	fmt.Println(strings.Repeat("-", 95))
	for i, s := range summaries {
		if i >= 3 { // Show top 3
			break
		}
		flagStr := strings.Join(s.Flags, " ")
		if flagStr == "" {
			flagStr = "(default)"
		}
		color := ""
		if i == 0 {
			color = "\033[32m" // Green for best
		}
		fmt.Printf("%s%-60s %-10.2f %-10.2f %-10v\033[0m\n", color, flagStr, s.AverageWER*100, s.MaxWER*100, s.AvgLatency.Round(time.Millisecond))
	}
}

func generateSearchSpace() [][]string {
	fadeOptions := [][]string{
		{"t", "0.1"},
		{"t", "0.25"},
	}
	vadOptions := [][]string{
		{}, // None
		{"-t", "5", "-p", "0.2", "-s", "0.2"},
		{"-t", "8", "-p", "0.5", "-s", "0.5"},
	}
	hpFreqs := []string{"", "20", "40", "80"}
	gainOptions := [][]string{{}, {"-n"}, {"-n", "-0.1"}, {"-n", "-1.5"}}

	var space [][]string

	for _, fOpt := range fadeOptions {
		for _, vOpt := range vadOptions {
			for _, hpf := range hpFreqs {
				for _, gOpt := range gainOptions {
					var effects [][]string
					effects = append(effects, append([]string{"fade"}, fOpt...))

					if len(vOpt) > 0 {
						effects = append(effects, append([]string{"vad"}, vOpt...))
					}
					if hpf != "" {
						effects = append(effects, []string{"highpass", hpf})
					}
					if len(gOpt) > 0 {
						effects = append(effects, append([]string{"gain"}, gOpt...))
					}

					// Generate all permutations of the current effects set
					space = append(space, permutations(effects)...)
				}
			}
		}
	}

	return space
}

func permutations(effects [][]string) [][]string {
	var res [][]string
	var p func([][]string, int)
	p = func(arr [][]string, n int) {
		if n == 1 {
			// Enforce boundary constraints:
			// 1. VAD must be AT THE FRONT (index 0)
			// 2. 'fade' and 'gain' must be at start or end.
			for i, e := range arr {
				if len(e) == 0 {
					continue
				}
				if e[0] == "vad" && i != 0 {
					return
				}
				if e[0] == "fade" || e[0] == "gain" {
					if i > 0 && i < len(arr)-1 {
						return // Skip if middle position
					}
				}
			}

			tmp := make([]string, 0)
			for _, e := range arr {
				tmp = append(tmp, e...)
			}
			res = append(res, tmp)
			return
		}
		for i := 0; i < n; i++ {
			p(arr, n-1)
			if n%2 == 1 {
				arr[0], arr[n-1] = arr[n-1], arr[0]
			} else {
				arr[i], arr[n-1] = arr[n-1], arr[i]
			}
		}
	}
	// Copy to avoid modifying original
	cpy := make([][]string, len(effects))
	copy(cpy, effects)
	p(cpy, len(cpy))
	return res
}

func callServer(url, filename string, audio []byte, flags []string) (string, time.Duration, error) {
	body := &bytes.Buffer{}
	writer := multipart.NewWriter(body)
	part, err := writer.CreateFormFile("file", filename)
	if err != nil {
		return "", 0, err
	}
	part.Write(audio)

	writer.WriteField("format", "json")
	for _, f := range flags {
		writer.WriteField("sox_flags", f)
	}
	writer.Close()

	start := time.Now()
	req, _ := http.NewRequest("POST", url, body)
	req.Header.Set("Content-Type", writer.FormDataContentType())

	client := httpClient
	resp, err := client.Do(req)
	if err != nil {
		return "", 0, err
	}
	defer resp.Body.Close()

	latency := time.Since(start)
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		return "", 0, fmt.Errorf("status %s: %s", resp.Status, string(b))
	}

	var tr TranscribeResponse
	json.NewDecoder(resp.Body).Decode(&tr)
	return tr.Text, latency, nil
}

func calculateWER(truth, hypothesis string) float64 {
	tWords := strings.Fields(strings.ToLower(truth))
	hWords := strings.Fields(strings.ToLower(hypothesis))

	if len(tWords) == 0 {
		if len(hWords) == 0 {
			return 0
		}
		return 1.0
	}

	// Levenshtein distance
	d := make([][]int, len(tWords)+1)
	for i := range d {
		d[i] = make([]int, len(hWords)+1)
		d[i][0] = i
	}
	for j := range d[0] {
		d[0][j] = j
	}

	for i := 1; i <= len(tWords); i++ {
		for j := 1; j <= len(hWords); j++ {
			cost := 1
			if tWords[i-1] == hWords[j-1] {
				cost = 0
			}
			d[i][j] = min(d[i-1][j]+1, d[i][j-1]+1, d[i-1][j-1]+cost)
		}
	}

	return float64(d[len(tWords)][len(hWords)]) / float64(len(tWords))
}

func min(a ...int) int {
	m := a[0]
	for _, x := range a[1:] {
		if x < m {
			m = x
		}
	}
	return m
}

func aggregateResults(results chan Result) ([]Summary, map[string]Result) {
	stats := make(map[string][]Result)
	bestPerFile := make(map[string]Result)

	for r := range results {
		fKey := strings.Join(r.Flags, "|")
		stats[fKey] = append(stats[fKey], r)

		// Track best per file
		currentBest, exists := bestPerFile[r.Filename]
		if !exists || r.WER < currentBest.WER {
			bestPerFile[r.Filename] = r
		} else if r.WER == currentBest.WER && r.Latency < currentBest.Latency {
			bestPerFile[r.Filename] = r
		}
	}

	var summaries []Summary
	for fKey, resList := range stats {
		var totalWER float64
		var maxWER float64
		var totalLat time.Duration
		for _, r := range resList {
			totalWER += r.WER
			if r.WER > maxWER {
				maxWER = r.WER
			}
			totalLat += r.Latency
		}
		summaries = append(summaries, Summary{
			Flags:      strings.Split(fKey, "|"),
			AverageWER: totalWER / float64(len(resList)),
			MaxWER:     maxWER,
			AvgLatency: totalLat / time.Duration(len(resList)),
		})
	}
	return summaries, bestPerFile
}
