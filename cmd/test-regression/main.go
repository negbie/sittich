package main

import (
	"bytes"
	"crypto/tls"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
)

type TranscribeResponse struct {
	Success bool   `json:"success"`
	Text    string `json:"text"`
	Error   string `json:"error"`
}

type TestResult struct {
	WER      float64
	Latency  time.Duration
}

func main() {
	serverURL := flag.String("url", "https://localhost:5092/transcribe", "Sittich server URL")
	audioDir := flag.String("audio", "./bin/audio", "Directory containing WAV files")
	truthFile := flag.String("truth", "./bin/truth.json", "JSON file mapping filename to ground truth text")
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

	fmt.Printf("\n\033[1mRunning Regression Test Suite\033[0m\n")
	fmt.Printf("Server: %s\n", *serverURL)
	fmt.Printf("Audio:  %s\n", *audioDir)
	fmt.Println(strings.Repeat("-", 100))

	fmt.Printf("%-20s %-10s %-10s %-40s\n", "Filename", "WER %", "Time ms", "Extracted Text (Truncated)")
	fmt.Println(strings.Repeat("-", 100))

	for filename, expected := range truth {
		audioPath := filepath.Join(*audioDir, filename)
		audioData, err := os.ReadFile(audioPath)
		if err != nil {
			fmt.Printf("%-20s Error reading audio: %v\n", filename, err)
			continue
		}

		// Benchmark transcription
		res, lat, err := callServer(*serverURL, filename, audioData)
		if err != nil {
			fmt.Printf("%-20s Error: %v\n", filename, err)
		} else {
			wer := calculateWER(expected, res)
			printResult(filename, wer, lat, res)
		}
		fmt.Println(strings.Repeat("-", 100))
	}
}

func printResult(filename string, wer float64, latency time.Duration, text string) {

	color := ""
	if wer == 0 {
		color = "\033[32m" // Green
	} else if wer > 0.1 {
		color = "\033[31m" // Red
	}

	truncatedText := text
	if len(truncatedText) > 80 {
		truncatedText = truncatedText[:77] + "..."
	}
	truncatedText = strings.ReplaceAll(truncatedText, "\n", " ")

	fmt.Printf("%-20s %s%-10.2f\033[0m %-10d %-40s\n",
		filename, color, wer*100, latency.Milliseconds(), truncatedText)
}

func callServer(url, filename string, audio []byte) (string, time.Duration, error) {
	body := &bytes.Buffer{}
	writer := multipart.NewWriter(body)
	part, err := writer.CreateFormFile("file", filename)
	if err != nil {
		return "", 0, err
	}
	part.Write(audio)

	writer.WriteField("format", "json")
	writer.Close()

	start := time.Now()
	req, _ := http.NewRequest("POST", url, body)
	req.Header.Set("Content-Type", writer.FormDataContentType())

	client := &http.Client{
		Timeout: 60 * time.Second,
		Transport: &http.Transport{
			TLSClientConfig: &tls.Config{InsecureSkipVerify: true},
		},
	}
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
	if err := json.NewDecoder(resp.Body).Decode(&tr); err != nil {
		return "", 0, err
	}
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
