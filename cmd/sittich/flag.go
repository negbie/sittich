package main

import (
	"errors"
	"flag"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/negbie/sittich/internal/config"
)

var (
	errShowHelp    = flag.ErrHelp
	errInvalidArgs = errors.New("invalid arguments")
)

type cliOptions struct {
	ListenAddr           string
	ChunkSize            int
	ChunkOverlapDuration float64
	Format               string
	MaxActivePaths       int
	MaxConcurrency       int
	DecodingMethod       string
	NumThreads           int
	MaxUploadMB          int
	Debug                bool
	Lazy                 bool
	ShowVersion          bool
	DataFolder           string
	Proxy                string
	S3DataDir            string
	S3Enabled            bool
	CertPath             string
	KeyPath              string
	DisableHTTPS         bool
	IdleTimeout          time.Duration
	VAD                  bool
	DualModel            bool
}

func defineFlags(fs *flag.FlagSet, opts *cliOptions) {
	fs.StringVar(&opts.ListenAddr, "listen", ":5092", "listen address")
	fs.IntVar(&opts.NumThreads, "num-threads", 4, "ONNX thread pool size per stream")
	fs.IntVar(&opts.MaxUploadMB, "max-upload", 16, "max upload size in MB")
	fs.StringVar(&opts.Format, "format", "text", "output format: text, json, vtt")
	fs.IntVar(&opts.ChunkSize, "chunk-size", 40, "chunk size in seconds")
	fs.Float64Var(&opts.ChunkOverlapDuration, "chunk-overlap", 0.4, "overlap in seconds")
	fs.IntVar(&opts.MaxActivePaths, "max-active-paths", 4, "active paths for beam search")
	fs.IntVar(&opts.MaxConcurrency, "concurrency", 1, "max concurrent ONNX streams per engine")
	fs.StringVar(&opts.DecodingMethod, "decoding-method", "greedy_search", "greedy_search or modified_beam_search")
	fs.StringVar(&opts.DataFolder, "data-folder", "", "path to model directory")
	fs.StringVar(&opts.Proxy, "proxy", "", "proxy URL for remote transcription")
	fs.StringVar(&opts.S3DataDir, "s3-data", "./data/s3", "directory for S3 storage")
	fs.BoolVar(&opts.S3Enabled, "s3-enabled", false, "enable S3 server")
	fs.BoolVar(&opts.Debug, "debug", false, "detailed debug logs")
	fs.BoolVar(&opts.Lazy, "lazy", false, "lazy mode (load model on demand and unload immediately)")
	fs.BoolVar(&opts.ShowVersion, "version", false, "show version")
	fs.StringVar(&opts.CertPath, "cert", "", "path to HTTPS certificate")
	fs.StringVar(&opts.KeyPath, "key", "", "path to HTTPS private key")
	fs.BoolVar(&opts.DisableHTTPS, "disable-https", false, "disable HTTPS and use plain HTTP")
	fs.DurationVar(&opts.IdleTimeout, "idle-timeout", 6*time.Hour, "idle time before unloading models (e.g. 5m, 0 to disable)")
	fs.BoolVar(&opts.VAD, "vad", false, "enable Voice Activity Detection")
	fs.BoolVar(&opts.DualModel, "dual-model", false, "use a second model engine for voting (more accurate but slower/more RAM)")
}

func parseCLI(args []string) (cliOptions, error) {
	fs := flag.NewFlagSet("sittich", flag.ContinueOnError)
	opts := cliOptions{}
	defineFlags(fs, &opts)

	if err := fs.Parse(args); err != nil {
		return opts, err
	}

	opts.Format = strings.ToLower(opts.Format)
	opts.DecodingMethod = strings.ToLower(opts.DecodingMethod)

	return opts, nil
}

func printUsage() {
	fmt.Fprintf(os.Stderr, "Usage: sittich [flags]\n\nFlags:\n")
	fs := flag.NewFlagSet("sittich", flag.ContinueOnError)
	defineFlags(fs, &cliOptions{})
	fs.PrintDefaults()
}

func recognizerConfigFromCLI(opts cliOptions, modelPath string) *config.ASR {
	return &config.ASR{
		ModelPath:      modelPath,
		NumThreads:     opts.NumThreads,
		DecodingMethod: opts.DecodingMethod,
		MaxActivePaths: opts.MaxActivePaths,
		MaxConcurrency: opts.MaxConcurrency,
		Lazy:           opts.Lazy,
		IdleTimeout:    opts.IdleTimeout,
		VADEnabled:     opts.VAD,
		DualModel:      opts.DualModel,
	}
}
