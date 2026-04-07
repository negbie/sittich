package main

import (
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"
	"unicode/utf8"
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

const (
	defaultChunkSize      = 60
	defaultFormat         = "text"
	defaultMaxActivePaths = 2
	defaultDecodingMethod = "greedy_search"
	defaultWorkers        = 4
	defaultMaxUploadMB    = 32
)

type cliOptions struct {
	// CLI mode
	AudioFile      string
	ChunkSize      int
	Format         string
	OutputFile     string
	ShowVersion    bool
	MaxActivePaths int
	DecodingMethod string

	DataFolder  string
	RemoteURL   string
	ListenAddr  string
	Workers     int
	MaxUploadMB int
	Debug       bool
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
	{long: "chunk-size", arg: "int", description: "chunk size in seconds", defaultVal: "60"},
	{long: "format", arg: "string", description: "output format: text, json, vtt", defaultVal: defaultFormat},
	{long: "output", arg: "file", description: "output file", defaultVal: "stdout"},
	{long: "remote-url", arg: "string", description: "transcribe via remote server"},
	{long: "listen", arg: "address", description: "listen address"},
	{long: "workers", arg: "int", description: "concurrent workers", defaultVal: "2"},
	{long: "max-upload", arg: "int", description: "max upload size in MB", defaultVal: "1024"},
	{long: "debug", description: "show detailed debug logs"},
	{long: "max-active-paths", arg: "int", description: "number of active paths for modified beam search", defaultVal: "4"},
	{long: "decoding-method", arg: "string", description: "decoding method: greedy_search or modified_beam_search", defaultVal: defaultDecodingMethod},
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

func parseCLI(args []string) (cliOptions, error) {
	fs := flag.NewFlagSet("sittich", flag.ContinueOnError)
	fs.SetOutput(io.Discard)

	// CLI flags
	chunkSize := fs.Int("chunk-size", defaultChunkSize, "chunk size in seconds")
	format := fs.String("format", defaultFormat, "output format (text, json, vtt)")
	outputFile := fs.String("output", "", "output file")
	showVersion := fs.Bool("version", false, "show version")
	remoteURL := fs.String("remote-url", "", "transcribe via remote server")
	maxActivePaths := fs.Int("max-active-paths", defaultMaxActivePaths, "number of active paths for modified beam search")
	decodingMethod := fs.String("decoding-method", defaultDecodingMethod, "decoding method: greedy_search or modified_beam_search")

	// Server flags
	listenAddr := fs.String("listen", "", "listen address")
	workers := fs.Int("workers", defaultWorkers, "concurrent workers")
	maxUploadMB := fs.Int("max-upload", defaultMaxUploadMB, "max upload size in MB")
	debug := fs.Bool("debug", false, "show detailed debug logs")
	dataFolder := fs.String("data-folder", "", "path to model directory")

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
		ChunkSize:      *chunkSize,
		Format:         strings.ToLower(*format),
		OutputFile:     *outputFile,
		ShowVersion:    *showVersion,
		MaxActivePaths: *maxActivePaths,
		DecodingMethod: strings.ToLower(*decodingMethod),
		Workers:        *workers,
		MaxUploadMB:    *maxUploadMB,
		Debug:          *debug,
		DataFolder:     *dataFolder,
		RemoteURL:      *remoteURL,
		ListenAddr:     *listenAddr,
	}

	if opts.ShowVersion || opts.ListenAddr != "" {
		return opts, nil
	}

	if fs.NArg() < 1 {
		// Check if stdin has data for pipe mode
		stat, err := os.Stdin.Stat()
		if err != nil || (stat.Mode()&os.ModeCharDevice) != 0 {
			return cliOptions{}, fmt.Errorf("%w: audio file is required (or pipe audio to stdin)", errInvalidArgs)
		}
		opts.AudioFile = "-"
	} else {
		opts.AudioFile = fs.Arg(0)
	}

	switch opts.Format {
	case "text", "json", "vtt":
	default:
		return cliOptions{}, fmt.Errorf("%w: unknown format %q (valid: text, json, vtt)", errInvalidArgs, opts.Format)
	}

	if opts.MaxActivePaths < 1 {
		return cliOptions{}, fmt.Errorf("%w: max-active-paths must be >= 1", errInvalidArgs)
	}

	switch opts.DecodingMethod {
	case "greedy_search", "modified_beam_search":
	default:
		return cliOptions{}, fmt.Errorf("%w: unknown decoding method %q (valid: greedy_search, modified_beam_search)", errInvalidArgs, opts.DecodingMethod)
	}

	return opts, nil
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
	fmt.Fprintln(os.Stderr, "  sittich [flags] [audio-file]")
	fmt.Fprintln(os.Stderr, "  cat audio | sittich [flags]")
	fmt.Fprintln(os.Stderr, "  sittich --server [flags]")
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
		{label: fmt.Sprintf("%s$%s sittich audio.wav", green, reset), plainLabel: "$ sittich audio.wav", desc: fmt.Sprintf("%s# 60s chunks, text output%s", dim, reset)},
		{label: fmt.Sprintf("%s$%s cat audio.wav | sittich", green, reset), plainLabel: "$ cat audio.wav | sittich", desc: fmt.Sprintf("%s# transcribe from pipe%s", dim, reset)},
		{label: fmt.Sprintf("%s$%s sittich --chunk-size 30 talk.wav", green, reset), plainLabel: "$ sittich --chunk-size 30 talk.wav", desc: fmt.Sprintf("%s# 30s chunks%s", dim, reset)},
		{label: fmt.Sprintf("%s$%s sittich --remote-url http://localhost:8080 audio.wav", green, reset), plainLabel: "$ sittich --remote-url http://localhost:8080 audio.wav", desc: fmt.Sprintf("%s# transcribe via remote server%s", dim, reset)},
		{label: fmt.Sprintf("%s$%s sittich --format vtt --output subs.vtt audio.wav", green, reset), plainLabel: "$ sittich --format vtt --output subs.vtt audio.wav", desc: fmt.Sprintf("%s# WebVTT to file%s", dim, reset)},
		{label: fmt.Sprintf("%s$%s sittich --listen :8080", green, reset), plainLabel: "$ sittich --listen :8080", desc: fmt.Sprintf("%s# Run server on port 8080%s", dim, reset)},
	}
	printAlignedRows(exampleRows)
	fmt.Fprintln(os.Stderr)
}
