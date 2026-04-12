package main

import (
	"context"
	"fmt"
	"os"
	"os/signal"
	"path/filepath"
	"runtime"
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

	// Global ASR Dispatcher with focused parallel batching
	dispatcher := asr.NewDispatcher(recognizer, opts.Workers, 16, 5*time.Millisecond, opts.Debug)
	defer dispatcher.Close()

	sharedVAD := setupVAD(opts, actualDataFolder)
	if sharedVAD != nil {
		defer sharedVAD.Close()
	}

	maxQueue := opts.MaxQueueSize
	if maxQueue <= 0 {
		maxQueue = opts.Workers * 2
	}

	pool := setupPool(opts, dispatcher, actualDataFolder, sharedVAD, maxQueue)
	defer pool.Shutdown()

	serverCfg := &config.Server{
		ListenAddr:   opts.ListenAddr,
		MaxUploadMB:  int64(opts.MaxUploadMB),
		Workers:      opts.Workers,
		MaxQueueSize: maxQueue,
		Debug:        opts.Debug,
	}

	srv := server.NewServer(serverCfg, pool, version)
	srv.SetDefaults(opts.Format, opts.ChunkSize)

	fmt.Fprintf(os.Stderr, "   Concurrency: workers=%d dispatcher=%d max_active=%d num_threads=%d queue=%d (NumCPU=%d)\n",
		opts.Workers, opts.Workers, cfg.MaxActive, opts.NumThreads, maxQueue, runtime.NumCPU())

	fmt.Fprintf(os.Stderr, "   Server running on http://%s\n", opts.ListenAddr)
	fmt.Fprintf(os.Stderr, "   POST /transcribe - Transcribe audio\n")
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

func setupVAD(opts *cliOptions, dataFolder string) *pipeline.VAD {
	if !opts.UseVAD {
		return nil
	}
	if err := models.EnsureVAD(dataFolder); err != nil {
		fmt.Fprintf(os.Stderr, "   VAD download error: %v, falling back to blind chunking\n", err)
		return nil
	}
	vadModelPath := filepath.Join(dataFolder, models.VADModelFile)
	vad, err := pipeline.NewVAD(vadModelPath, float32(opts.VADThreshold), float32(opts.VADMinSilence), float32(opts.VADMinSpeech), 1)
	if err != nil {
		fmt.Fprintf(os.Stderr, "   VAD error: %v, falling back to blind chunking\n", err)
		return nil
	}
	return vad
}

func setupPool(opts *cliOptions, dispatcher *asr.Dispatcher, dataFolder string, vad *pipeline.VAD, queueSize int) *worker.Pool {
	return worker.NewPool(
		opts.Workers,
		queueSize,
		dispatcher,
		config.Pipeline{
			ChunkDuration:         float64(opts.ChunkSize),
			ChunkOverlapDuration:  opts.ChunkOverlapDuration,
			WordTimestamps:        true,
			Debug:                 opts.Debug,
			UseVAD:                opts.UseVAD,
			VADModelPath:          filepath.Join(dataFolder, models.VADModelFile),
			VADThreshold:          float32(opts.VADThreshold),
			VADMinSilenceDuration: float32(opts.VADMinSilence),
			VADMinSpeechDuration:  float32(opts.VADMinSpeech),
		},
		opts.Debug,
		dataFolder,
		vad,
	)
}
