package main

import (
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"
	"unicode/utf8"

	"github.com/negbie/sittich/internal/config"
)

var (
	reset   = "\033[0m"
	bold    = "\033[1m"
	dim     = "\033[2m"
	gray    = "\033[38;5;250m"
	green   = "\033[32m"
	yellow  = "\033[33m"
	magenta = "\033[35m"
	cyan    = "\033[36m"
)

var (
	errShowHelp    = errors.New("show help")
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
	UseVAD               bool
	VADThreshold         float64
	VADMinSilence        float64
	VADMinSpeech         float64
	DataFolder           string
	Workers              int
	NumThreads           int
	MaxUploadMB          int
	Debug                bool
	ShowVersion          bool
}

type cliFlag struct {
	short       string
	long        string
	arg         string
	description string
	defaultVal  string
}

type usageRow struct {
	label      string
	plainLabel string
	desc       string
}

var allFlags = []cliFlag{
	{long: "listen", arg: "address", description: "listen address", defaultVal: ":5092"},
	{long: "workers", arg: "int", description: "concurrent workers", defaultVal: "4"},
	{long: "num-threads", arg: "int", description: "ONNX thread pool size per decode stream", defaultVal: "2"},
	{long: "max-active-streams", arg: "int", description: "max concurrent decode streams", defaultVal: "4"},
	{long: "max-queue-size", arg: "int", description: "max queued jobs", defaultVal: "12"},
	{long: "max-upload", arg: "int", description: "max upload size in MB", defaultVal: "16"},
	{long: "format", arg: "string", description: "default output format: text, json, vtt", defaultVal: "text"},
	{long: "chunk-size", arg: "int", description: "default chunk size in seconds", defaultVal: "40"},
	{long: "chunk-overlap", arg: "float", description: "overlap duration between chunks in seconds", defaultVal: "0.4"},
	{long: "use-vad", description: "use voice activity detection for chunking"},
	{long: "vad-threshold", arg: "float", description: "VAD probability threshold (0.0 to 1.0)", defaultVal: "0.5"},
	{long: "vad-min-silence", arg: "float", description: "VAD minimum silence duration in seconds", defaultVal: "0.2"},
	{long: "vad-min-speech", arg: "float", description: "VAD minimum speech duration in seconds", defaultVal: "0.2"},
	{long: "debug", description: "show detailed debug logs"},
	{long: "data-folder", arg: "path", description: "path to model directory"},
	{long: "version", description: "show version"},
}

func init() {
	if !isStderrTTY() {
		disableColors()
	}
}

func isStderrTTY() bool {
	info, err := os.Stderr.Stat()
	if err != nil {
		return false
	}
	return info.Mode()&os.ModeCharDevice != 0
}

func disableColors() {
	reset = ""
	bold = ""
	dim = ""
	gray = ""
	green = ""
	yellow = ""
	magenta = ""
	cyan = ""
}

func hideCursor() {
	if isStderrTTY() {
		fmt.Fprint(os.Stderr, "\033[?25l")
	}
}

func showCursor() {
	if isStderrTTY() {
		fmt.Fprint(os.Stderr, "\033[?25h")
	}
}

func parseCLI(args []string) (cliOptions, error) {
	fs := flag.NewFlagSet("sittich", flag.ContinueOnError)
	fs.SetOutput(io.Discard)

	listenAddr := fs.String("listen", ":5092", "listen address")
	workers := fs.Int("workers", 4, "concurrent workers")
	numThreads := fs.Int("num-threads", 2, "ONNX thread pool size per decode stream")
	maxActiveStreams := fs.Int("max-active-streams", 4, "max concurrent decode streams")
	maxQueueSize := fs.Int("max-queue-size", 12, "max queued jobs")
	maxUploadMB := fs.Int("max-upload", 16, "max upload size in MB")
	format := fs.String("format", "text", "default output format (text, json, vtt)")
	chunkSize := fs.Int("chunk-size", 40, "default chunk size in seconds")
	chunkOverlap := fs.Float64("chunk-overlap", 0.4, "overlap duration between adjacent chunks in seconds")
	maxActivePaths := fs.Int("max-active-paths", 4, "number of active paths for modified beam search")
	decodingMethod := fs.String("decoding-method", "modified_beam_search", "decoding method: greedy_search or modified_beam_search")
	useVAD := fs.Bool("use-vad", false, "use voice activity detection for chunking")
	vadThreshold := fs.Float64("vad-threshold", 0.5, "VAD probability threshold (0.0 to 1.0)")
	vadMinSilence := fs.Float64("vad-min-silence", 0.2, "VAD minimum silence duration in seconds")
	vadMinSpeech := fs.Float64("vad-min-speech", 0.2, "VAD minimum speech duration in seconds")
	dataFolder := fs.String("data-folder", "", "path to model directory")
	debug := fs.Bool("debug", false, "show detailed debug logs")
	showVersion := fs.Bool("version", false, "show version")

	fs.Usage = func() {
		printUsage()
	}

	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return cliOptions{}, errShowHelp
		}
		return cliOptions{}, fmt.Errorf("%w: %v", errInvalidArgs, err)
	}

	opts := cliOptions{
		ListenAddr:           *listenAddr,
		Workers:              *workers,
		NumThreads:           *numThreads,
		MaxActiveStreams:     *maxActiveStreams,
		MaxQueueSize:         *maxQueueSize,
		MaxUploadMB:          *maxUploadMB,
		Format:               strings.ToLower(*format),
		ChunkSize:            *chunkSize,
		ChunkOverlapDuration: *chunkOverlap,
		MaxActivePaths:       *maxActivePaths,
		DecodingMethod:       strings.ToLower(*decodingMethod),
		UseVAD:               *useVAD,
		VADThreshold:         *vadThreshold,
		VADMinSilence:        *vadMinSilence,
		VADMinSpeech:         *vadMinSpeech,
		DataFolder:           *dataFolder,
		Debug:                *debug,
		ShowVersion:          *showVersion,
	}

	if opts.ShowVersion {
		return opts, nil
	}

	switch opts.Format {
	case "text", "json", "vtt":
	default:
		return cliOptions{}, fmt.Errorf("%w: unknown format %q (valid: text, json, vtt)", errInvalidArgs, opts.Format)
	}

	if opts.MaxActivePaths < 1 {
		return cliOptions{}, fmt.Errorf("%w: max-active-paths must be >= 1", errInvalidArgs)
	}
	if opts.ChunkOverlapDuration < 0 {
		return cliOptions{}, fmt.Errorf("%w: chunk-overlap must be >= 0", errInvalidArgs)
	}
	if opts.VADThreshold < 0 || opts.VADThreshold > 1 {
		return cliOptions{}, fmt.Errorf("%w: vad-threshold must be between 0 and 1", errInvalidArgs)
	}

	switch opts.DecodingMethod {
	case "greedy_search", "modified_beam_search":
	default:
		return cliOptions{}, fmt.Errorf("%w: unknown decoding method %q (valid: greedy_search, modified_beam_search)", errInvalidArgs, opts.DecodingMethod)
	}

	return opts, nil
}

func recognizerConfigFromCLI(opts cliOptions, modelPath string) *config.ASR {
	maxActive := opts.MaxActiveStreams
	if maxActive <= 0 {
		maxActive = 4
	}

	return &config.ASR{
		ModelPath:      modelPath,
		NumThreads:     opts.NumThreads,
		MaxActive:      maxActive,
		DecodingMethod: opts.DecodingMethod,
		MaxActivePaths: opts.MaxActivePaths,
	}
}

func formatFlagLabel(f cliFlag) string {
	parts := make([]string, 0, 2)
	if f.short != "" {
		parts = append(parts, "-"+f.short)
	}
	if f.long != "" {
		parts = append(parts, "--"+f.long)
	}
	return strings.Join(parts, ", ")
}

func plainFlagLabel(f cliFlag) string {
	label := formatFlagLabel(f)
	if f.arg != "" {
		label += " " + f.arg
	}
	return label
}

func coloredFlagLabel(f cliFlag) string {
	base := formatFlagLabel(f)
	if f.arg == "" {
		return cyan + base + reset
	}
	return cyan + base + " " + yellow + f.arg + reset
}

func printAlignedRows(rows []usageRow) {
	maxLabelWidth := 0
	for _, row := range rows {
		if w := utf8.RuneCountInString(row.plainLabel); w > maxLabelWidth {
			maxLabelWidth = w
		}
	}

	for _, row := range rows {
		pad := strings.Repeat(" ", maxLabelWidth-utf8.RuneCountInString(row.plainLabel))
		fmt.Fprintf(os.Stderr, "  %s%s  %s\n", row.label, pad, row.desc)
	}
}

func printUsage() {
	fmt.Fprintf(os.Stderr, "%s🐦‍⬛ %ssittich%s\n\n", bold, magenta, reset)

	fmt.Fprintf(os.Stderr, "%sUsage:%s\n", bold, reset)
	fmt.Fprintln(os.Stderr, "  sittich [flags]")
	fmt.Fprintln(os.Stderr)

	flagRows := make([]usageRow, 0, len(allFlags))
	for _, f := range allFlags {
		desc := f.description
		if f.defaultVal != "" {
			desc += fmt.Sprintf(" %s(default: %s)%s", dim, f.defaultVal, reset)
		}
		flagRows = append(flagRows, usageRow{
			label:      coloredFlagLabel(f),
			plainLabel: plainFlagLabel(f),
			desc:       desc,
		})
	}

	fmt.Fprintf(os.Stderr, "%sFlags:%s\n", bold, reset)
	printAlignedRows(flagRows)
	fmt.Fprintln(os.Stderr)

	fmt.Fprintf(os.Stderr, "%sExamples:%s\n", bold, reset)
	exampleRows := []usageRow{
		{label: fmt.Sprintf("%s$%s sittich", green, reset), plainLabel: "$ sittich", desc: fmt.Sprintf("%s# Run server on port 8080%s", dim, reset)},
		{label: fmt.Sprintf("%s$%s sittich --listen :5000", green, reset), plainLabel: "$ sittich --listen :5000", desc: fmt.Sprintf("%s# Run server on port 5000%s", dim, reset)},
		{label: fmt.Sprintf("%s$%s sittich --debug --workers 8", green, reset), plainLabel: "$ sittich --debug --workers 8", desc: fmt.Sprintf("%s# Run with 8 workers and debug logs%s", dim, reset)},
	}
	printAlignedRows(exampleRows)
	fmt.Fprintln(os.Stderr)
}
