package main

import (
	"context"
	"fmt"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"
	"time"

	"github.com/negbie/sittich/internal/asr"
	"github.com/negbie/sittich/internal/config"
	"github.com/negbie/sittich/internal/models"
	"github.com/negbie/sittich/internal/pipeline"
	"github.com/negbie/sittich/internal/server"
	"github.com/negbie/sittich/internal/worker"
)

func runServer(opts *cliOptions) error {
	// Load model
	hideCursor()
	defer showCursor()

	fmt.Fprint(os.Stderr, "Loading model...\r")
	actualDataFolder, err := models.GetModelPath(opts.DataFolder)
	if err != nil {
		fmt.Fprintln(os.Stderr)
		return err
	}

	cfg := recognizerConfigFromCLI(*opts, actualDataFolder)

	recognizer, err := server.LoadRecognizer(cfg)
	if err != nil {
		fmt.Fprintln(os.Stderr)
		return err
	}
	defer recognizer.Close()
	fmt.Fprintln(os.Stderr, "Model loaded!   ")

	// Global ASR Dispatcher with 4 workers and 20ms batch window (Parallel Batching mode)
	dispatcher := asr.NewDispatcher(recognizer, 4, 16, 5*time.Millisecond, opts.Debug)
	defer dispatcher.Close()

	// Initialize VAD once (shared)
	var sharedVAD *pipeline.VAD
	if opts.UseVAD {
		if err := models.EnsureVAD(actualDataFolder); err != nil {
			fmt.Fprintf(os.Stderr, "   VAD download error: %v, falling back to blind chunking\n", err)
		} else {
			vadModelPath := filepath.Join(actualDataFolder, models.VADModelFile)
			sharedVAD, err = pipeline.NewVAD(
				vadModelPath,
				float32(opts.VADThreshold),
				float32(opts.VADMinSilence),
				float32(opts.VADMinSpeech),
				1,
			)
			if err != nil {
				fmt.Fprintf(os.Stderr, "   VAD error: %v, falling back to blind chunking\n", err)
			} else {
				defer sharedVAD.Close()
			}
		}
	}

	// Create worker pool
	serverCfg := &config.Server{
		ListenAddr:   opts.ListenAddr,
		MaxUploadMB:  int64(opts.MaxUploadMB),
		Workers:      opts.Workers,
		MaxQueueSize: 10,
		Debug:        opts.Debug,
	}
	pool := worker.NewPool(
		opts.Workers,
		10,
		dispatcher,
		config.Pipeline{
			ChunkDuration:         float64(opts.ChunkSize),
			ChunkOverlapDuration:  opts.ChunkOverlapDuration,
			WordTimestamps:        true,
			Debug:                 opts.Debug,
			UseVAD:                opts.UseVAD,
			VADModelPath:          filepath.Join(actualDataFolder, models.VADModelFile),
			VADThreshold:          float32(opts.VADThreshold),
			VADMinSilenceDuration: float32(opts.VADMinSilence),
			VADMinSpeechDuration:  float32(opts.VADMinSpeech),
		},
		opts.Debug,
		actualDataFolder,
		sharedVAD,
	)
	defer pool.Shutdown()

	// Create HTTP server
	srv := server.NewServer(serverCfg, pool, version)
	srv.SetDefaults(opts.Format, opts.ChunkSize)

	// Start server
	fmt.Fprintf(os.Stderr, "   Server running on http://%s\n", opts.ListenAddr)
	fmt.Fprintf(os.Stderr, "   POST /transcribe - Transcribe audio\n")
	fmt.Fprintf(os.Stderr, "   GET  /health     - Health check\n")
	fmt.Fprintf(os.Stderr, "\nPress Ctrl+C to stop\n")

	// Handle shutdown
	errChan := make(chan error, 1)
	go func() {
		errChan <- srv.Start()
	}()

	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	select {
	case err := <-errChan:
		return err
	case <-sigChan:
		fmt.Fprintln(os.Stderr, "\n Shutting down...")
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		return srv.Shutdown(ctx)
	}
}
