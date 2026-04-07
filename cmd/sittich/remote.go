package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"

	"github.com/negbie/sittich/internal/types"
)

type remoteJSONResponse struct {
	Success  bool            `json:"success"`
	Error    string          `json:"error,omitempty"`
	Duration float64         `json:"duration_seconds"`
	Text     string          `json:"text"`
	Segments []types.Segment `json:"segments,omitempty"`
}

func resolveRemoteURL() (string, error) {
	raw := strings.TrimSpace(os.Getenv("sittich_URL"))
	if raw == "" {
		return "", fmt.Errorf("--remote requires sittich_URL (e.g. sittich_URL=http://localhost:8080)")
	}

	u, err := url.Parse(raw)
	if err != nil {
		return "", fmt.Errorf("invalid sittich_URL: %w", err)
	}
	if u.Scheme != "http" && u.Scheme != "https" {
		return "", fmt.Errorf("invalid sittich_URL %q: must start with http:// or https://", raw)
	}
	if u.Host == "" {
		return "", fmt.Errorf("invalid sittich_URL %q: missing host", raw)
	}

	return strings.TrimRight(raw, "/"), nil
}

func transcribeRemote(serverURL, audioFile string, chunkSize int) ([]types.Segment, float64, error) {
	body := &bytes.Buffer{}
	writer := multipart.NewWriter(body)

	fileWriter, err := writer.CreateFormFile("file", filepath.Base(audioFile))
	if err != nil {
		return nil, 0, fmt.Errorf("failed to create multipart file field: %w", err)
	}

	file, err := os.Open(audioFile)
	if err != nil {
		return nil, 0, fmt.Errorf("failed to open audio file: %w", err)
	}
	defer file.Close()

	if _, err := io.Copy(fileWriter, file); err != nil {
		return nil, 0, fmt.Errorf("failed to stream audio file: %w", err)
	}

	if err := writer.WriteField("format", "json"); err != nil {
		return nil, 0, fmt.Errorf("failed to set format: %w", err)
	}
	if err := writer.WriteField("chunk_size", fmt.Sprintf("%d", chunkSize)); err != nil {
		return nil, 0, fmt.Errorf("failed to set chunk_size: %w", err)
	}
	if err := writer.Close(); err != nil {
		return nil, 0, fmt.Errorf("failed to finalize multipart body: %w", err)
	}

	endpoint := serverURL + "/transcribe"
	req, err := http.NewRequest(http.MethodPost, endpoint, body)
	if err != nil {
		return nil, 0, fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", writer.FormDataContentType())

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, 0, fmt.Errorf("remote request failed to %s: %w", endpoint, err)
	}
	defer resp.Body.Close()

	var parsed remoteJSONResponse
	if err := json.NewDecoder(resp.Body).Decode(&parsed); err != nil {
		return nil, 0, fmt.Errorf("failed to decode remote JSON response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		if parsed.Error != "" {
			return nil, 0, fmt.Errorf("remote server returned %s: %s", resp.Status, parsed.Error)
		}
		return nil, 0, fmt.Errorf("remote server returned %s", resp.Status)
	}

	if !parsed.Success {
		if parsed.Error != "" {
			return nil, 0, fmt.Errorf("remote transcription failed: %s", parsed.Error)
		}
		return nil, 0, fmt.Errorf("remote transcription failed")
	}

	if len(parsed.Segments) > 0 {
		return parsed.Segments, parsed.Duration, nil
	}

	if strings.TrimSpace(parsed.Text) == "" {
		return []types.Segment{}, parsed.Duration, nil
	}

	return []types.Segment{{Text: parsed.Text}}, parsed.Duration, nil
}
