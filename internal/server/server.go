package server

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"crypto/tls"
	"mime/multipart"

	"github.com/negbie/sittich/internal/asr"
	"github.com/negbie/sittich/internal/config"
	"github.com/negbie/sittich/internal/models"
	"github.com/negbie/sittich/internal/output"
	"github.com/negbie/sittich/internal/pipeline"
)

// Server provides the HTTP interface for transcription.
type Server struct {
	httpServer *http.Server
	mux        *http.ServeMux
	ctx        context.Context
	engine     asr.Engine
	options    *config.Server
	pipeline   config.Pipeline
	sem        chan struct{}
	version    string
	startTime  time.Time
}

func NewServer(ctx context.Context, options *config.Server, pipeline config.Pipeline, engine asr.Engine, version string) *Server {
	s := &Server{
		ctx:       ctx,
		engine:    engine,
		options:   options,
		pipeline:  pipeline,
		sem:       make(chan struct{}, options.Workers),
		version:   version,
		startTime: time.Now(),
		mux:       http.NewServeMux(),
	}

	s.mux.HandleFunc("/transcribe", s.handleTranscribe)
	s.mux.HandleFunc("/health", s.handleHealth)

	s.httpServer = &http.Server{
		Addr:    options.ListenAddr,
		Handler: s.withMiddleware(s.mux),
		BaseContext: func(_ net.Listener) context.Context {
			return s.ctx
		},
	}

	return s
}

func (s *Server) SetS3Handler(h http.Handler) {
	if h != nil {
		s.mux.Handle("/", h)
	}
}

func (s *Server) SetDefaults(format string, chunkSize int) {
	if format != "" {
		s.options.DefaultFormat = strings.ToLower(format)
	}
	if chunkSize > 0 {
		s.options.DefaultChunkSize = chunkSize
	}
}

func (s *Server) Start() error {
	if s.options.DisableHTTPS {
		return s.httpServer.ListenAndServe()
	}

	// Ensure we have certificate paths
	if s.options.CertFile == "" || s.options.KeyFile == "" {
		return fmt.Errorf("certificate and key paths are required for HTTPS")
	}

	// Check if certs exist, generate if not
	if _, err := os.Stat(s.options.CertFile); os.IsNotExist(err) {
		fmt.Fprintf(os.Stderr, "Generating self-signed certificate: %s\n", s.options.CertFile)
		if err := GenerateSelfSignedCert(s.options.CertFile, s.options.KeyFile); err != nil {
			return fmt.Errorf("failed to generate certificates: %w", err)
		}
	} else if _, err := os.Stat(s.options.KeyFile); os.IsNotExist(err) {
		fmt.Fprintf(os.Stderr, "Generating missing key for certificate: %s\n", s.options.KeyFile)
		if err := GenerateSelfSignedCert(s.options.CertFile, s.options.KeyFile); err != nil {
			return fmt.Errorf("failed to generate certificates: %w", err)
		}
	}

	return s.httpServer.ListenAndServeTLS(s.options.CertFile, s.options.KeyFile)
}

func (s *Server) Shutdown(ctx context.Context) error { return s.httpServer.Shutdown(ctx) }

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
	if r.Header.Get("X-Sittich-Proxy-Loop") == "true" {
		s.processLocalTranscribe(w, r)
		return
	}

	if s.options != nil && s.options.Proxy != "" {
		s.proxyRequest(w, r, s.options.Proxy)
		return
	}

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

	result, err := s.transcribeLocal(ctx, filePath, chunkSize, soxFlags)
	if err != nil {
		s.sendError(w, http.StatusInternalServerError, err.Error())
		return
	}

	processingTime := time.Since(requestStart).Seconds()
	rtFactor := result.Duration / processingTime
	if rtFactor < 0 {
		rtFactor = 0
	}

	if s.options != nil && s.options.Debug {
		fmt.Fprintf(os.Stderr, "[HTTP] job_done=%s processing_time=%.2fs rtf=%.2f\n", r.URL.Path, processingTime, rtFactor)
	}

	s.sendFormattedResponse(w, format, result, processingTime, rtFactor)
}

// ProcessTranscribe handles the transcription of a file, respecting proxy settings if configured.
func (s *Server) ProcessTranscribe(ctx context.Context, filePath string, format string, chunkSize int, soxFlags []string) (*asr.Result, error) {
	if s.options != nil && s.options.Proxy != "" {
		return s.transcribeWithProxy(ctx, filePath, format, chunkSize, soxFlags)
	}

	return s.transcribeLocal(ctx, filePath, chunkSize, soxFlags)
}

func (s *Server) transcribeLocal(ctx context.Context, filePath string, chunkSize int, soxFlags []string) (*asr.Result, error) {
	// Acquire a worker slot.
	select {
	case s.sem <- struct{}{}:
		defer func() { <-s.sem }()
	case <-ctx.Done():
		return nil, fmt.Errorf("server too busy")
	}

	pipe := &pipeline.Pipeline{
		Engine: s.engine,
		Config: s.pipeline,
	}

	return pipe.Process(ctx, filePath, float64(chunkSize), soxFlags...)
}

func (s *Server) transcribeWithProxy(ctx context.Context, filePath string, format string, chunkSize int, soxFlags []string) (*asr.Result, error) {
	proxyURL := s.options.Proxy
	if !strings.HasPrefix(proxyURL, "http") {
		proxyURL = "http://" + proxyURL
	}
	u, err := url.Parse(proxyURL)
	if err != nil {
		return nil, fmt.Errorf("invalid proxy URL: %w", err)
	}
	u.Path = "/transcribe"

	return s.doProxyPOST(ctx, u.String(), filePath, format, chunkSize, soxFlags)
}

func (s *Server) proxyRequest(w http.ResponseWriter, r *http.Request, targetURL string) {
	if !strings.HasPrefix(targetURL, "http") {
		targetURL = "http://" + targetURL
	}
	target, err := url.Parse(targetURL)
	if err != nil {
		s.sendError(w, http.StatusInternalServerError, fmt.Sprintf("invalid proxy target: %v", err))
		return
	}

	proxy := httputil.NewSingleHostReverseProxy(target)
	proxy.Transport = &http.Transport{
		TLSClientConfig: &tls.Config{InsecureSkipVerify: true},
	}

	originalDirector := proxy.Director
	proxy.Director = func(req *http.Request) {
		originalDirector(req)
		req.Header.Set("X-Sittich-Proxy-Loop", "true")
		req.Host = target.Host
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
		tmpFile.Close()
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

		cleanup = func() { os.Remove(filePath) }

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

			tmpFile.Write(data)
			tmpFile.Close()
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

		cleanup = func() { os.Remove(filePath) }

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
	resp, err := http.Get(url)
	if err != nil {
		return "", fmt.Errorf("failed to download: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("download failed: %s", resp.Status)
	}

	maxBytes := s.options.MaxUploadMB * 1024 * 1024
	tmpFile, err := os.CreateTemp("", "sittich-url-*")
	if err != nil {
		return "", fmt.Errorf("failed to create temp file: %w", err)
	}

	written, err := io.Copy(tmpFile, io.LimitReader(resp.Body, maxBytes+1))
	tmpFile.Close()

	if err != nil || written > maxBytes {
		os.Remove(tmpFile.Name())
		return "", fmt.Errorf("download failed or file too large")
	}

	return tmpFile.Name(), nil
}

func (s *Server) sendJSON(w http.ResponseWriter, status int, data interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(data)
}

func (s *Server) sendError(w http.ResponseWriter, status int, message string) {
	s.sendJSON(w, status, TranscribeResponse{Success: false, Error: message})
}

func (s *Server) sendFormattedResponse(w http.ResponseWriter, format string, result *asr.Result, processingTime, rtFactor float64) {
	switch format {
	case "text":
		w.Header().Set("Content-Type", "text/plain; charset=utf-8")
		w.WriteHeader(http.StatusOK)
		output.WriteText(w, result)
	case "vtt":
		w.Header().Set("Content-Type", "text/vtt; charset=utf-8")
		w.WriteHeader(http.StatusOK)
		output.WriteVTT(w, result)
	default:
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

func (s *Server) doProxyPOST(ctx context.Context, targetURL, filePath, format string, chunkSize int, soxFlags []string) (*asr.Result, error) {
	f, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	body, pw := io.Pipe()

	writer := multipart.NewWriter(pw)

	errChan := make(chan error, 1)
	go func() {
		defer pw.Close()
		defer writer.Close()

		part, err := writer.CreateFormFile("file", filepath.Base(filePath))
		if err != nil {
			errChan <- err
			return
		}
		if _, err := io.Copy(part, f); err != nil {
			errChan <- err
			return
		}

		if format != "" {
			writer.WriteField("format", format)
		}
		if chunkSize > 0 {
			writer.WriteField("chunk_size", strconv.Itoa(chunkSize))
		}
		for _, sf := range soxFlags {
			writer.WriteField("sox_flags", sf)
		}
	}()

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, targetURL, body)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", writer.FormDataContentType())
	req.Header.Set("X-Sittich-Proxy-Loop", "true")

	client := &http.Client{
		Transport: &http.Transport{
			TLSClientConfig: &tls.Config{InsecureSkipVerify: true},
		},
	}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("proxy failed with status %s: %s", resp.Status, string(b))
	}

	var res TranscribeResponse
	if err := json.NewDecoder(resp.Body).Decode(&res); err != nil {
		return nil, fmt.Errorf("failed to decode proxy response: %w", err)
	}

	if !res.Success {
		return nil, fmt.Errorf("proxy error: %s", res.Error)
	}

	return &asr.Result{
		Duration: res.Duration,
		Segments: res.Segments,
	}, nil
}

func LoadRecognizer(cfg *config.ASR) (asr.Engine, error) {
	baseDir := cfg.ModelPath
	if baseDir == "" {
		baseDir = "./data"
	}
 
	parakeetFolder := filepath.Join(baseDir, "parakeet")
	parakeetPath, err := models.GetModelPath(parakeetFolder, models.ModelURL)
	if err != nil {
		return nil, fmt.Errorf("failed to resolve parakeet model: %w", err)
	}
 
	if cfg.VADEnabled {
		vadFolder := filepath.Join(baseDir, "vad")
		vadPath, err := models.GetVADPath(vadFolder)
		if err != nil {
			return nil, fmt.Errorf("failed to resolve VAD model: %w", err)
		}
		cfg.VADPath = vadPath
	}
 
	pCfg := *cfg
	pCfg.ModelPath = parakeetPath
	engParakeet, err := asr.NewRecognizer(&pCfg)
	if err != nil {
		return nil, fmt.Errorf("failed to load parakeet: %w", err)
	}
 
	if !cfg.DualModel {
		return engParakeet, nil
	}
 
	nemoFolder := filepath.Join(baseDir, "nemo")
	nemoPath, err := models.GetModelPath(nemoFolder, models.NemoModelURL)
	if err != nil {
		engParakeet.Close()
		return nil, fmt.Errorf("failed to resolve nemo model: %w", err)
	}
 
	nCfg := *cfg
	nCfg.ModelPath = nemoPath
	engNemo, err := asr.NewRecognizer(&nCfg)
	if err != nil {
		engParakeet.Close()
		return nil, fmt.Errorf("failed to load nemo: %w", err)
	}
 
	return asr.NewVoter(engParakeet, engNemo), nil
}
