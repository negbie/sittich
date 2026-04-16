package main

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/negbie/sittich/internal/asr"
	"github.com/negbie/sittich/internal/config"
	"github.com/negbie/sittich/internal/server"
)

func runServer(opts *cliOptions) error {
	asrCfg := &config.ASR{
		NumThreads:     opts.NumThreads,
		DecodingMethod: opts.DecodingMethod,
		MaxActivePaths: opts.MaxActivePaths,
		MaxActive:      opts.MaxActiveStreams,
		ModelPath:      opts.DataFolder,
	}

	recognizer, err := server.LoadRecognizer(asrCfg)
	if err != nil {
		return err
	}
	defer recognizer.Close()

	dispatcher := asr.NewDispatcher(recognizer, opts.MaxActiveStreams, opts.Debug)
	defer dispatcher.Close()

	pipeCfg := config.Pipeline{
		ChunkDuration:        float64(opts.ChunkSize),
		ChunkOverlapDuration: opts.ChunkOverlapDuration,
		WordTimestamps:       true,
		Debug:                opts.Debug,
	}

	serverCfg := &config.Server{
		ListenAddr:         opts.ListenAddr,
		Workers:            opts.MaxActiveStreams,
		MaxUploadMB:        int64(opts.MaxUploadMB),
		DefaultFormat:      opts.Format,
		DefaultChunkSize:   opts.ChunkSize,
		Proxy:              opts.Proxy,
		Debug:              opts.Debug,
	}

	srv := server.NewServer(serverCfg, pipeCfg, dispatcher, version)

	stop := make(chan os.Signal, 1)
	signal.Notify(stop, os.Interrupt, syscall.SIGTERM)

	go func() {
		fmt.Fprintf(os.Stderr, "Model loaded!   \n")
		fmt.Fprintf(os.Stderr, "   Concurrency: workers=%d dispatcher=%d max_active=%d num_threads=%d (NumCPU=%s)\n", 
			opts.MaxActiveStreams, opts.MaxActiveStreams, opts.MaxActiveStreams, opts.NumThreads, os.Getenv("NUM_CPU"))
		fmt.Fprintf(os.Stderr, "   Server running on http://%s\n", opts.ListenAddr)
		fmt.Fprintf(os.Stderr, "   POST /transcribe - Transcription endpoint (Proxy-aware)\n")
		fmt.Fprintf(os.Stderr, "   GET  /health     - Health check\n\n")
		fmt.Fprintf(os.Stderr, "Press Ctrl+C to stop\n")

		if err := srv.Start(); err != nil && err != http.ErrServerClosed {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}
	}()

	<-stop
	fmt.Fprintf(os.Stderr, "\n Shutting down...\n")

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	return srv.Shutdown(ctx)
}
