package main

import (
	"errors"
	"flag"
	"fmt"
	"os"
	"runtime"
	"strings"

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
	MaxActiveStreams     int
	MaxQueueSize         int
	DecodingMethod       string
	Workers              int
	DispatcherWorkers    int
	NumThreads           int
	MaxBatchSize         int
	MaxUploadMB          int
	Debug                bool
	ShowVersion          bool
	DataFolder           string
	DSPMode              string
	Proxy                string
}

func defineFlags(fs *flag.FlagSet, opts *cliOptions) {
	fs.StringVar(&opts.ListenAddr, "listen", ":5092", "listen address")
	fs.IntVar(&opts.Workers, "workers", 4, "concurrent workers for audio processing")
	fs.IntVar(&opts.DispatcherWorkers, "dispatcher-workers", 4, "concurrent workers for dispatcher")
	fs.IntVar(&opts.NumThreads, "num-threads", 2, "ONNX thread pool size per stream")
	fs.IntVar(&opts.MaxActiveStreams, "max-active-streams", 4, "max concurrent streams (0 = auto)")
	fs.IntVar(&opts.MaxBatchSize, "max-batch-size", 1, "max chunks to batch")
	fs.IntVar(&opts.MaxQueueSize, "max-queue-size", 12, "max queued jobs")
	fs.IntVar(&opts.MaxUploadMB, "max-upload", 16, "max upload size in MB")
	fs.StringVar(&opts.Format, "format", "text", "output format: text, json, vtt")
	fs.IntVar(&opts.ChunkSize, "chunk-size", 40, "chunk size in seconds")
	fs.Float64Var(&opts.ChunkOverlapDuration, "chunk-overlap", 0.4, "overlap in seconds")
	fs.IntVar(&opts.MaxActivePaths, "max-active-paths", 5, "active paths for beam search")
	fs.StringVar(&opts.DecodingMethod, "decoding-method", "greedy_search", "greedy_search or modified_beam_search")
	fs.StringVar(&opts.DataFolder, "data-folder", "", "path to model directory")
	fs.StringVar(&opts.DSPMode, "dsp-mode", "aggressive", "DSP chain: minimal, gentle, aggressive")
	fs.StringVar(&opts.Proxy, "proxy", "", "proxy URL for remote transcription")
	fs.BoolVar(&opts.Debug, "debug", false, "detailed debug logs")
	fs.BoolVar(&opts.ShowVersion, "version", false, "show version")
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
	opts.DSPMode = strings.ToLower(opts.DSPMode)

	if opts.MaxActiveStreams == 0 {
		opts.MaxActiveStreams = runtime.NumCPU() / opts.NumThreads
		if opts.MaxActiveStreams < 1 {
			opts.MaxActiveStreams = 1
		}
	}
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
		MaxActive:      opts.MaxActiveStreams,
		DecodingMethod: opts.DecodingMethod,
		MaxActivePaths: opts.MaxActivePaths,
	}
}
