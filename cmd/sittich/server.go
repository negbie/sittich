package main

import (
	"context"
	"fmt"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/negbie/sittich/internal/asr"
	"github.com/negbie/sittich/internal/server"
	"github.com/negbie/sittich/internal/worker"
)

func runServer(opts *cliOptions) error {
	// Load model
	hideCursor()
	defer showCursor()

	fmt.Fprint(os.Stderr, "Loading model...\r")
	cfg := &asr.Config{
		ModelPath:      opts.DataFolder,
		DecodingMethod: opts.DecodingMethod,
		MaxActivePaths: opts.MaxActivePaths,
	}

	recognizer, err := server.LoadRecognizer(cfg)
	if err != nil {
		fmt.Fprintln(os.Stderr)
		return err
	}
	defer recognizer.Close()
	fmt.Fprintln(os.Stderr, "Model loaded!   ")

	// Create worker pool
	serverOpts := &server.ServerOptions{
		ListenAddr:   opts.ListenAddr,
		MaxUploadMB:  int64(opts.MaxUploadMB),
		Workers:      opts.Workers,
		MaxQueueSize: 10,
		Debug:        opts.Debug,
	}
	pool := worker.NewPool(opts.Workers, 10, recognizer, opts.Debug, opts.DataFolder)
	defer pool.Shutdown()

	// Create HTTP server
	srv := server.NewServer(serverOpts, pool, version)
	srv.SetDefaults(opts.Format, opts.ChunkSize)

	// Start server
	fmt.Fprintf(os.Stderr, "   Server running on http://%s\n", opts.ListenAddr)
	fmt.Fprintf(os.Stderr, "   POST /transcribe - Transcribe audio\n")
	fmt.Fprintf(os.Stderr, "   GET  /health     - Health check\n")
	fmt.Fprintf(os.Stderr, "   defaults: format=%s chunk-size=%ds decoding=%s max-active-paths=%d\n",
		opts.Format, opts.ChunkSize, opts.DecodingMethod, opts.MaxActivePaths)
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
