package main

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"path/filepath"
	"strings"

	"github.com/negbie/sittich/internal/asr"
	"github.com/negbie/sittich/internal/config"
	"github.com/negbie/sittich/internal/s3"
	"github.com/negbie/sittich/internal/server"
)

func runServer(opts *cliOptions) error {
	asrCfg := recognizerConfigFromCLI(*opts, opts.DataFolder)

	recognizer, err := server.LoadRecognizer(asrCfg)
	if err != nil {
		return err
	}
	defer recognizer.Close()

	denoiser, err := server.LoadDenoiser(asrCfg)
	if err != nil {
		return err
	}
	if denoiser != nil {
		defer denoiser.Close()
	}

	dispatcher := asr.NewDispatcher(recognizer, opts.MaxActiveStreams, opts.Debug)
	defer dispatcher.Close()

	pipeCfg := config.Pipeline{
		ChunkDuration:        float64(opts.ChunkSize),
		ChunkOverlapDuration: opts.ChunkOverlapDuration,
		WordTimestamps:       true,
		Denoise:              opts.Denoise,
		Debug:                opts.Debug,
	}

	dataDir := opts.DataFolder
	if dataDir == "" {
		dataDir = "./data"
	}

	certPath := opts.CertPath
	if certPath == "" {
		certPath = filepath.Join(dataDir, "cert.pem")
	}
	keyPath := opts.KeyPath
	if keyPath == "" {
		keyPath = filepath.Join(dataDir, "key.pem")
	}

	serverCfg := &config.Server{
		ListenAddr:       opts.ListenAddr,
		Workers:          opts.MaxActiveStreams,
		MaxUploadMB:      int64(opts.MaxUploadMB),
		DefaultFormat:    opts.Format,
		DefaultChunkSize: opts.ChunkSize,
		Proxy:            opts.Proxy,
		Debug:            opts.Debug,
		CertFile:         certPath,
		KeyFile:          keyPath,
		DisableHTTPS:     opts.DisableHTTPS,
	}

	srv := server.NewServer(serverCfg, pipeCfg, dispatcher, denoiser, version)

	stop := make(chan os.Signal, 1)
	signal.Notify(stop, os.Interrupt, syscall.SIGTERM)

	go func() {
		if opts.Lazy {
			fmt.Fprintf(os.Stderr, "Lazy Mode active (Models will load on demand)   \n")
		} else {
			fmt.Fprintf(os.Stderr, "Models loaded!   \n")
		}

		protocol := "https"
		if opts.DisableHTTPS {
			protocol = "http"
		}

		fmt.Fprintf(os.Stderr, "   Concurrency: workers=%d dispatcher=%d max_active=%d threads=%d\n",
			opts.MaxActiveStreams, opts.MaxActiveStreams, opts.MaxActiveStreams, opts.NumThreads)
		fmt.Fprintf(os.Stderr, "   Server running on %s://%s\n", protocol, opts.ListenAddr)
		fmt.Fprintf(os.Stderr, "   POST /transcribe - Transcription endpoint (Proxy-aware)\n")
		fmt.Fprintf(os.Stderr, "   GET  /health     - Health check\n")

		if opts.S3Enabled {
			s3cfg := s3.ServerConfig{
				DataDir: opts.S3DataDir,
				Debug:   opts.Debug,
				OnUpload: func(bucket, key, localPath string) {
					if !strings.HasSuffix(strings.ToLower(key), ".wav") {
						return
					}

					fmt.Fprintf(os.Stderr, "[S3] Job start: bucket=%s key=%s\n", bucket, key)
					ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
					defer cancel()

					result, err := srv.ProcessTranscribe(ctx, localPath, opts.Format, opts.ChunkSize, nil)
					if err != nil {
						fmt.Fprintf(os.Stderr, "[S3] Job failed: bucket=%s key=%s err=%v\n", bucket, key, err)
						return
					}
					fmt.Fprintf(os.Stderr, "[S3] Job done: bucket=%s key=%s text=%q\n", bucket, key, result.FullText())
				},
			}

			s3Srv, err := s3.NewServer(s3cfg)
			if err != nil {
				fmt.Fprintf(os.Stderr, "Error: failed to create S3 server: %v\n", err)
				os.Exit(1)
			}

			srv.SetS3Handler(s3Srv.Handler())
			fmt.Fprintf(os.Stderr, "   S3 API enabled (Data: %s)\n", opts.S3DataDir)
		}

		fmt.Fprintf(os.Stderr, "\nPress Ctrl+C to stop\n")

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
