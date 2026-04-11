package main

import (
	"context"
	"fmt"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/negbie/sittich/internal/asr"
	"github.com/negbie/sittich/internal/config"
	"github.com/negbie/sittich/internal/server"
	"github.com/negbie/sittich/internal/worker"
)

func runServer(opts *cliOptions) error {
	// Load model
	hideCursor()
	defer showCursor()

	fmt.Fprint(os.Stderr, "Loading model...\r")
	cfg := recognizerConfigFromCLI(*opts, opts.DataFolder)

	recognizer, err := server.LoadRecognizer(cfg)
	if err != nil {
		fmt.Fprintln(os.Stderr)
		return err
	}
	defer recognizer.Close()
	fmt.Fprintln(os.Stderr, "Model loaded!   ")

	// Global ASR Dispatcher with 4 workers and 20ms batch window (Parallel Batching mode)
	dispatcher := asr.NewDispatcher(recognizer, 4, 16, 20*time.Millisecond, opts.Debug)
	defer dispatcher.Close()

	// Create worker pool
	serverCfg := &config.Server{
		ListenAddr:   opts.ListenAddr,
		MaxUploadMB:  int64(opts.MaxUploadMB),
		Workers:      opts.Workers,
		MaxQueueSize: config.DefaultMaxQueueSize,
		Debug:        opts.Debug,
	}
	pool := worker.NewPool(
		opts.Workers,
		config.DefaultMaxQueueSize,
		dispatcher,
		config.Pipeline{
			ChunkDuration:        float64(opts.ChunkSize),
			ChunkOverlapDuration: opts.ChunkOverlapDuration,
			ChunkMinTailDuration: opts.ChunkMinTailDuration,
			WordTimestamps:       true,
			Debug:                opts.Debug,
		},
		opts.Debug,
		opts.DataFolder,
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
