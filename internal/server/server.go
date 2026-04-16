package server

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/negbie/sittich/internal/asr"
	"github.com/negbie/sittich/internal/config"
	"github.com/negbie/sittich/internal/models"
	"github.com/negbie/sittich/internal/output"
	"github.com/negbie/sittich/internal/pipeline"
	"github.com/negbie/sittich/internal/speech"
)

// Server is the HTTP server
type Server struct {
	httpServer *http.Server
	engine     speech.Engine
	options    *config.Server
	pipeline   config.Pipeline
	sem        chan struct{}
	version    string
	startTime  time.Time
}

// NewServer creates a new HTTP server
func NewServer(options *config.Server, pipeline config.Pipeline, engine speech.Engine, version string) *Server {
	s := &Server{
		engine:    engine,
		options:   options,
		pipeline:  pipeline,
		sem:       make(chan struct{}, options.Workers),
		version:   version,
		startTime: time.Now(),
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/transcribe", s.handleTranscribe)
	mux.HandleFunc("/health", s.handleHealth)

	s.httpServer = &http.Server{
		Addr:    options.ListenAddr,
		Handler: s.withMiddleware(mux),
	}

	return s
}

// SetDefaults configures the default response format and chunk size used when
// a request does not explicitly provide them.
func (s *Server) SetDefaults(format string, chunkSize int) {
	if format != "" {
		s.options.DefaultFormat = strings.ToLower(format)
	}
	if chunkSize > 0 {
		s.options.DefaultChunkSize = chunkSize
	}
}

// Start starts the HTTP server
func (s *Server) Start() error {
	return s.httpServer.ListenAndServe()
}

// Shutdown gracefully shuts down the server
func (s *Server) Shutdown(ctx context.Context) error {
	return s.httpServer.Shutdown(ctx)
}

func (s *Server) withMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type")

		if r.Method == "OPTIONS" {
			w.WriteHeader(http.StatusOK)
			return
		}

		start := time.Now()
		next.ServeHTTP(w, r)
		fmt.Fprintf(os.Stderr, "%s %s %s\n", r.Method, r.URL.Path, time.Since(start))
	})
}

func (s *Server) handleTranscribe(w http.ResponseWriter, r *http.Request) {
	// If the loop-prevention header is present, we ALWAYS process locally.
	if r.Header.Get("X-Sittich-Proxy-Loop") == "true" {
		s.processLocalTranscribe(w, r)
		return
	}

	// If a proxy is configured, forward the request.
	if s.options != nil && s.options.Proxy != "" {
		s.proxyRequest(w, r, s.options.Proxy)
		return
	}

	// Default fallback to local processing.
	s.processLocalTranscribe(w, r)
}

func (s *Server) processLocalTranscribe(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		s.sendError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}

	requestStart := time.Now()

	filePath, format, chunkSize, soxFlags, cleanup, err := s.parseRequest(r)
	if err != nil {
		s.sendError(w, http.StatusBadRequest, err.Error())
		return
	}
	defer cleanup()

	ctx, cancel := context.WithTimeout(r.Context(), 10*time.Minute)
	defer cancel()

	if s.options != nil && s.options.Debug {
		fmt.Fprintf(os.Stderr, "[HTTP] parse_request=%s format=%s chunk_size=%d\n", time.Since(requestStart).Round(time.Millisecond), format, chunkSize)
	}

	// Acquire a worker slot. This provides backpressure and limits concurrent DSP/ASR.
	select {
	case s.sem <- struct{}{}:
		defer func() { <-s.sem }()
	case <-ctx.Done():
		s.sendError(w, http.StatusServiceUnavailable, "server too busy")
		return
	}

	jobID := strconv.FormatInt(time.Now().UnixNano(), 10)
	if s.options != nil && s.options.Debug {
		fmt.Fprintf(os.Stderr, "[HTTP] job_start=%s job_id=%s\n", time.Since(requestStart).Round(time.Millisecond), jobID)
	}

	// Initialise a temporary Pipeline for this request.
	pipe := &pipeline.Pipeline{
		Engine: s.engine,
		Config: s.pipeline,
	}

	startTime := time.Now()
	result, err := pipe.Process(ctx, filePath, float64(chunkSize), soxFlags...)
	if err != nil {
		if s.options != nil && s.options.Debug {
			fmt.Fprintf(os.Stderr, "[HTTP] job_failed=%s job_id=%s err=%v\n", time.Since(requestStart).Round(time.Millisecond), jobID, err)
		}
		s.sendError(w, http.StatusInternalServerError, err.Error())
		return
	}

	processingTime := time.Since(startTime).Seconds()
	rtFactor := result.Duration / processingTime
	if rtFactor < 0 {
		rtFactor = 0
	}

	if s.options != nil && s.options.Debug {
		processingElapsed := time.Since(startTime).Round(time.Millisecond)
		totalElapsed := time.Since(requestStart).Round(time.Millisecond)
		fmt.Fprintf(os.Stderr, "[HTTP] job_done=%s job_id=%s processing_time=%s total_time=%s rtf=%.2f\n", jobID, time.Since(startTime).Round(time.Millisecond), processingElapsed, totalElapsed, rtFactor)
	}

	s.sendFormattedResponse(w, format, result, processingTime, rtFactor)
}

func (s *Server) proxyRequest(w http.ResponseWriter, r *http.Request, targetURL string) {
	target, err := url.Parse(targetURL)
	if err != nil {
		s.sendError(w, http.StatusInternalServerError, fmt.Sprintf("invalid proxy target: %v", err))
		return
	}

	if s.options != nil && s.options.Debug {
		fmt.Fprintf(os.Stderr, "[HTTP] proxying request to %s\n", targetURL)
	}

	proxy := &httputil.ReverseProxy{
		Director: func(req *http.Request) {
			req.URL.Scheme = target.Scheme
			req.URL.Host = target.Host
			req.URL.Path = target.Path
			req.URL.RawQuery = target.RawQuery
			req.Host = target.Host

			// Add loop prevention header
			req.Header.Set("X-Sittich-Proxy-Loop", "true")
		},
	}

	proxy.ServeHTTP(w, r)
}

func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		s.sendError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}

	uptime := time.Since(s.startTime)
	s.sendJSON(w, http.StatusOK, HealthResponse{
		Status:      "healthy",
		ModelLoaded: true,
		Version:     s.version,
		Uptime:      uptime.Round(time.Second).String(),
		Workers:     cap(s.sem),
		BusyWorkers: len(s.sem),
		Proxy:       s.options.Proxy,
	})
}

func (s *Server) parseRequest(r *http.Request) (filePath, format string, chunkSize int, soxFlags []string, cleanup func(), err error) {
	format = s.options.DefaultFormat
	if format == "" {
		format = "text"
	}
	chunkSize = s.options.DefaultChunkSize
	if chunkSize <= 0 {
		chunkSize = 40
	}

	contentType := r.Header.Get("Content-Type")

	if strings.HasPrefix(contentType, "multipart/form-data") {
		if err := r.ParseMultipartForm(s.options.MaxUploadMB * 1024 * 1024); err != nil {
			return "", "", 0, nil, nil, fmt.Errorf("failed to parse multipart form: %w", err)
		}

		file, header, err := r.FormFile("file")
		if err != nil {
			return "", "", 0, nil, nil, fmt.Errorf("missing file field: %w", err)
		}
		defer file.Close()

		tmpFile, err := os.CreateTemp("", "sittich-upload-*-"+filepath.Base(header.Filename))
		if err != nil {
			return "", "", 0, nil, nil, fmt.Errorf("failed to create temp file: %w", err)
		}

		if _, err := io.Copy(tmpFile, file); err != nil {
			tmpFile.Close()
			os.Remove(tmpFile.Name())
			return "", "", 0, nil, nil, fmt.Errorf("failed to save file: %w", err)
		}
		if err := tmpFile.Close(); err != nil {
			os.Remove(tmpFile.Name())
			return "", "", 0, nil, nil, fmt.Errorf("failed to close temp file: %w", err)
		}

		filePath = tmpFile.Name()

		if f := r.FormValue("format"); f != "" {
			format = strings.ToLower(f)
		}
		if c := r.FormValue("chunk_size"); c != "" {
			if n, err := strconv.Atoi(c); err == nil && n > 0 {
				chunkSize = n
			}
		}
		if sf := r.MultipartForm.Value["sox_flags"]; len(sf) > 0 {
			for _, f := range sf {
				soxFlags = append(soxFlags, strings.Fields(f)...)
			}
		}

		cleanup = func() {
			os.Remove(filePath)
		}

	} else if strings.HasPrefix(contentType, "application/json") {
		var req TranscribeRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			return "", "", 0, nil, nil, fmt.Errorf("invalid JSON: %w", err)
		}

		if req.URL != "" {
			filePath, err = s.downloadFromURL(req.URL)
			if err != nil {
				return "", "", 0, nil, nil, err
			}
		} else if req.Base64 != "" {
			data, err := base64.StdEncoding.DecodeString(req.Base64)
			if err != nil {
				return "", "", 0, nil, nil, fmt.Errorf("invalid base64: %w", err)
			}

			tmpFile, err := os.CreateTemp("", "sittich-b64-*")
			if err != nil {
				return "", "", 0, nil, nil, fmt.Errorf("failed to create temp file: %w", err)
			}

			if _, err := tmpFile.Write(data); err != nil {
				tmpFile.Close()
				os.Remove(tmpFile.Name())
				return "", "", 0, nil, nil, fmt.Errorf("failed to write file: %w", err)
			}
			if err := tmpFile.Close(); err != nil {
				os.Remove(tmpFile.Name())
				return "", "", 0, nil, nil, fmt.Errorf("failed to close temp file: %w", err)
			}

			filePath = tmpFile.Name()
		} else {
			return "", "", 0, nil, nil, fmt.Errorf("missing url or base64 in request")
		}

		if req.Format != "" {
			format = strings.ToLower(req.Format)
		}
		if req.ChunkSize > 0 {
			chunkSize = req.ChunkSize
		}
		if len(req.SoxFlags) > 0 {
			for _, f := range req.SoxFlags {
				soxFlags = append(soxFlags, strings.Fields(f)...)
			}
		}

		cleanup = func() {
			os.Remove(filePath)
		}

	} else {
		return "", "", 0, nil, nil, fmt.Errorf("unsupported content type: %s", contentType)
	}

	if format != "text" && format != "json" && format != "vtt" {
		if filePath != "" {
			os.Remove(filePath)
		}
		return "", "", 0, nil, nil, fmt.Errorf("invalid format: %s (must be text, json, or vtt)", format)
	}

	return filePath, format, chunkSize, soxFlags, cleanup, nil
}

func (s *Server) downloadFromURL(url string) (string, error) {
	client := &http.Client{
		Timeout: 5 * time.Minute,
	}

	resp, err := client.Get(url)
	if err != nil {
		return "", fmt.Errorf("failed to download: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("download failed: %s", resp.Status)
	}

	maxBytes := s.options.MaxUploadMB * 1024 * 1024
	if resp.ContentLength > maxBytes {
		return "", fmt.Errorf("file too large (max %d MB)", s.options.MaxUploadMB)
	}

	tmpFile, err := os.CreateTemp("", "sittich-url-*")
	if err != nil {
		return "", fmt.Errorf("failed to create temp file: %w", err)
	}

	written, err := io.Copy(tmpFile, io.LimitReader(resp.Body, maxBytes+1))
	if err != nil {
		tmpFile.Close()
		os.Remove(tmpFile.Name())
		return "", fmt.Errorf("failed to download: %w", err)
	}

	if err := tmpFile.Close(); err != nil {
		os.Remove(tmpFile.Name())
		return "", fmt.Errorf("failed to close file: %w", err)
	}

	if written > maxBytes {
		os.Remove(tmpFile.Name())
		return "", fmt.Errorf("file too large (max %d MB)", s.options.MaxUploadMB)
	}

	if written < 44 {
		os.Remove(tmpFile.Name())
		return "", fmt.Errorf("downloaded file too small")
	}

	return tmpFile.Name(), nil
}

func (s *Server) sendJSON(w http.ResponseWriter, status int, data interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(data)
}

func (s *Server) sendError(w http.ResponseWriter, status int, message string) {
	s.sendJSON(w, status, TranscribeResponse{
		Success: false,
		Error:   message,
	})
}

func (s *Server) sendFormattedResponse(w http.ResponseWriter, format string, result *speech.Result, processingTime, rtFactor float64) {
	switch format {
	case "text":
		w.Header().Set("Content-Type", "text/plain; charset=utf-8")
		w.WriteHeader(http.StatusOK)
		output.WriteText(w, result)
	case "vtt":
		w.Header().Set("Content-Type", "text/vtt; charset=utf-8")
		w.WriteHeader(http.StatusOK)
		output.WriteVTT(w, result)
	default: // json
		s.sendJSON(w, http.StatusOK, TranscribeResponse{
			Success:        true,
			Duration:       result.Duration,
			ProcessingTime: processingTime,
			RealtimeFactor: rtFactor,
			Text:           result.FullText(),
			Segments:       result.Segments,
		})
	}
}

func LoadRecognizer(cfg *config.ASR) (*asr.Recognizer, error) {
	modelPath, err := models.GetModelPath(cfg.ModelPath)
	if err != nil {
		return nil, fmt.Errorf("failed to get model: %w", err)
	}

	if cfg == nil {
		cfg = &config.ASR{
			ModelPath: modelPath,
		}
	} else {
		cfg.ModelPath = modelPath
	}

	recognizer, err := asr.NewRecognizer(cfg)
	if err != nil {
		return nil, fmt.Errorf("failed to load model: %w", err)
	}

	return recognizer, nil
}
