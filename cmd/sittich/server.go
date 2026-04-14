package main

import (
	"context"
	"fmt"
	"os"
	"os/signal"
	"runtime"
	"syscall"
	"time"

	"github.com/negbie/sittich/internal/asr"
	"github.com/negbie/sittich/internal/config"
	"github.com/negbie/sittich/internal/models"
	"github.com/negbie/sittich/internal/server"
	"github.com/negbie/sittich/internal/worker"
)

func runServer(opts *cliOptions) error {
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

	dispatcher := asr.NewDispatcher(recognizer, opts.DispatcherWorkers, opts.MaxBatchSize, 5*time.Millisecond, opts.Debug)
	defer dispatcher.Close()

	maxQueue := opts.MaxQueueSize
	if maxQueue <= 0 {
		maxQueue = opts.Workers * 2
	}

	pool := setupPool(opts, dispatcher, actualDataFolder, maxQueue)
	defer pool.Shutdown()

	serverCfg := &config.Server{
		ListenAddr:   opts.ListenAddr,
		MaxUploadMB:  int64(opts.MaxUploadMB),
		Workers:      opts.Workers,
		MaxQueueSize: maxQueue,
		Debug:        opts.Debug,
		Proxy:        opts.Proxy,
	}

	srv := server.NewServer(serverCfg, pool, version)
	srv.SetDefaults(opts.Format, opts.ChunkSize)

	fmt.Fprintf(os.Stderr, "   Concurrency: workers=%d dispatcher=%d max_active=%d num_threads=%d queue=%d (NumCPU=%d)\n",
		opts.Workers, opts.DispatcherWorkers, cfg.MaxActive, opts.NumThreads, maxQueue, runtime.NumCPU())
	fmt.Fprintf(os.Stderr, "   Server running on http://%s\n", opts.ListenAddr)
	if opts.Proxy != "" {
		fmt.Fprintf(os.Stderr, "   Proxy mode: ON (target: %s)\n", opts.Proxy)
	}
	fmt.Fprintf(os.Stderr, "   POST /transcribe - Transcription endpoint (Proxy-aware)\n")
	fmt.Fprintf(os.Stderr, "   GET  /health     - Health check\n\nPress Ctrl+C to stop\n")

	errChan := make(chan error, 1)
	go func() { errChan <- srv.Start() }()

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

func setupPool(opts *cliOptions, dispatcher *asr.Dispatcher, dataFolder string, queueSize int) *worker.Pool {
	return worker.NewPool(
		opts.Workers,
		queueSize,
		dispatcher,
		config.Pipeline{
			ChunkDuration:        float64(opts.ChunkSize),
			ChunkOverlapDuration: opts.ChunkOverlapDuration,
			WordTimestamps:       true,
			Debug:                opts.Debug,
			DSPMode:              opts.DSPMode,
		},
		opts.Debug,
		dataFolder,
	)
}
