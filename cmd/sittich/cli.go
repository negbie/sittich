package main

import (
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
	"unicode/utf8"

	"github.com/negbie/sittich/internal/asr"
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
	defaultChunkSize              = 40
	defaultChunkMinTail           = 1.5
	defaultFormat                 = "text"
	defaultMaxActivePaths         = 1
	defaultDecodingMethod         = "modified_beam_search"
	defaultWorkers                = 4
	defaultMaxUploadMB            = 32
	defaultVADThreshold           = 0.5
	defaultVADMinSilence          = 0.7
	defaultVADMinSpeech           = 0.25
	defaultVADSegmentPadding      = 0.0
	defaultCalibrationTargetPeak  = 120.0
	defaultCalibrationMaxGain     = 180.0
	defaultCalibrationMaxMSamples = 8192
)

type cliOptions struct {
	// CLI mode
	AudioFile              string
	ChunkSize              int
	ChunkMinTailDuration   float64
	Format                 string
	OutputFile             string
	ShowVersion            bool
	MaxActivePaths         int
	DecodingMethod         string
	NoVAD                  bool
	VADThreshold           float64
	VADMinSilenceDuration  float64
	VADMinSpeechDuration   float64
	VADSegmentPadding      float64
	CalibrationTargetPeak  float64
	CalibrationMaxGain     float64
	CalibrationMaxMSamples int

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
	{long: "chunk-size", arg: "int", description: "chunk size in seconds", defaultVal: strconv.Itoa(defaultChunkSize)},
	{long: "chunk-min-tail", arg: "float", description: "minimum tail duration when balancing oversized chunks in seconds", defaultVal: strconv.FormatFloat(defaultChunkMinTail, 'f', 1, 64)},
	{long: "format", arg: "string", description: "output format: text, json, vtt", defaultVal: defaultFormat},
	{long: "output", arg: "file", description: "output file", defaultVal: "stdout"},
	{long: "remote-url", arg: "string", description: "transcribe via remote server"},
	{long: "listen", arg: "address", description: "listen address"},
	{long: "workers", arg: "int", description: "concurrent workers", defaultVal: strconv.Itoa(defaultWorkers)},
	{long: "max-upload", arg: "int", description: "max upload size in MB", defaultVal: strconv.Itoa(defaultMaxUploadMB)},
	{long: "debug", description: "show detailed debug logs"},
	{long: "max-active-paths", arg: "int", description: "number of active paths for modified beam search", defaultVal: strconv.Itoa(defaultMaxActivePaths)},
	{long: "decoding-method", arg: "string", description: "decoding method: greedy_search or modified_beam_search", defaultVal: defaultDecodingMethod},
	{long: "no-vad", description: "disable VAD for local transcription"},
	{long: "vad-threshold", arg: "float", description: "VAD speech threshold", defaultVal: strconv.FormatFloat(defaultVADThreshold, 'f', 1, 64)},
	{long: "vad-min-silence", arg: "float", description: "minimum silence duration for VAD splits in seconds", defaultVal: strconv.FormatFloat(defaultVADMinSilence, 'f', 1, 64)},
	{long: "vad-min-speech", arg: "float", description: "minimum speech duration for VAD segments in seconds", defaultVal: strconv.FormatFloat(defaultVADMinSpeech, 'f', 2, 64)},
	{long: "vad-segment-padding", arg: "float", description: "padding added to both sides of each VAD segment in seconds", defaultVal: strconv.FormatFloat(defaultVADSegmentPadding, 'f', 1, 64)},
	{long: "calibration-target-peak", arg: "float", description: "target robust peak used for audio calibration", defaultVal: strconv.FormatFloat(defaultCalibrationTargetPeak, 'f', 1, 64)},
	{long: "calibration-max-gain", arg: "float", description: "maximum gain applied during audio calibration", defaultVal: strconv.FormatFloat(defaultCalibrationMaxGain, 'f', 1, 64)},
	{long: "calibration-max-samples", arg: "int", description: "maximum sampled points used for calibration peak estimation", defaultVal: strconv.Itoa(defaultCalibrationMaxMSamples)},
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
	chunkMinTail := fs.Float64("chunk-min-tail", defaultChunkMinTail, "minimum tail duration when balancing oversized chunks in seconds")
	format := fs.String("format", defaultFormat, "output format (text, json, vtt)")
	outputFile := fs.String("output", "", "output file")
	showVersion := fs.Bool("version", false, "show version")
	remoteURL := fs.String("remote-url", "", "transcribe via remote server")
	maxActivePaths := fs.Int("max-active-paths", defaultMaxActivePaths, "number of active paths for modified beam search")
	decodingMethod := fs.String("decoding-method", defaultDecodingMethod, "decoding method: greedy_search or modified_beam_search")
	noVAD := fs.Bool("no-vad", false, "disable VAD for local transcription")
	vadThreshold := fs.Float64("vad-threshold", defaultVADThreshold, "VAD speech threshold")
	vadMinSilence := fs.Float64("vad-min-silence", defaultVADMinSilence, "minimum silence duration for VAD splits in seconds")
	vadMinSpeech := fs.Float64("vad-min-speech", defaultVADMinSpeech, "minimum speech duration for VAD segments in seconds")
	vadSegmentPadding := fs.Float64("vad-segment-padding", defaultVADSegmentPadding, "padding added to both sides of each VAD segment in seconds")
	calibrationTargetPeak := fs.Float64("calibration-target-peak", defaultCalibrationTargetPeak, "target robust peak used for audio calibration")
	calibrationMaxGain := fs.Float64("calibration-max-gain", defaultCalibrationMaxGain, "maximum gain applied during audio calibration")
	calibrationMaxSamples := fs.Int("calibration-max-samples", defaultCalibrationMaxMSamples, "maximum sampled points used for calibration peak estimation")

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
		ChunkSize:              *chunkSize,
		ChunkMinTailDuration:   *chunkMinTail,
		Format:                 strings.ToLower(*format),
		OutputFile:             *outputFile,
		ShowVersion:            *showVersion,
		MaxActivePaths:         *maxActivePaths,
		DecodingMethod:         strings.ToLower(*decodingMethod),
		NoVAD:                  *noVAD,
		VADThreshold:           *vadThreshold,
		VADMinSilenceDuration:  *vadMinSilence,
		VADMinSpeechDuration:   *vadMinSpeech,
		VADSegmentPadding:      *vadSegmentPadding,
		CalibrationTargetPeak:  *calibrationTargetPeak,
		CalibrationMaxGain:     *calibrationMaxGain,
		CalibrationMaxMSamples: *calibrationMaxSamples,
		Workers:                *workers,
		MaxUploadMB:            *maxUploadMB,
		Debug:                  *debug,
		DataFolder:             *dataFolder,
		RemoteURL:              *remoteURL,
		ListenAddr:             *listenAddr,
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
	if opts.ChunkMinTailDuration < 0 {
		return cliOptions{}, fmt.Errorf("%w: chunk-min-tail must be >= 0", errInvalidArgs)
	}
	if opts.VADThreshold <= 0 {
		return cliOptions{}, fmt.Errorf("%w: vad-threshold must be > 0", errInvalidArgs)
	}
	if opts.VADMinSilenceDuration <= 0 {
		return cliOptions{}, fmt.Errorf("%w: vad-min-silence must be > 0", errInvalidArgs)
	}
	if opts.VADMinSpeechDuration <= 0 {
		return cliOptions{}, fmt.Errorf("%w: vad-min-speech must be > 0", errInvalidArgs)
	}
	if opts.VADSegmentPadding < 0 {
		return cliOptions{}, fmt.Errorf("%w: vad-segment-padding must be >= 0", errInvalidArgs)
	}
	if opts.CalibrationTargetPeak <= 0 {
		return cliOptions{}, fmt.Errorf("%w: calibration-target-peak must be > 0", errInvalidArgs)
	}
	if opts.CalibrationMaxGain <= 0 {
		return cliOptions{}, fmt.Errorf("%w: calibration-max-gain must be > 0", errInvalidArgs)
	}
	if opts.CalibrationMaxMSamples <= 0 {
		return cliOptions{}, fmt.Errorf("%w: calibration-max-samples must be > 0", errInvalidArgs)
	}

	switch opts.DecodingMethod {
	case "greedy_search", "modified_beam_search":
	default:
		return cliOptions{}, fmt.Errorf("%w: unknown decoding method %q (valid: greedy_search, modified_beam_search)", errInvalidArgs, opts.DecodingMethod)
	}

	return opts, nil
}

func recognizerConfigFromCLI(opts cliOptions, modelPath string) *asr.Config {
	return &asr.Config{
		ModelPath:              modelPath,
		DecodingMethod:         opts.DecodingMethod,
		MaxActivePaths:         opts.MaxActivePaths,
		CalibrationTargetPeak:  float32(opts.CalibrationTargetPeak),
		CalibrationMaxGain:     float32(opts.CalibrationMaxGain),
		CalibrationMaxMSamples: opts.CalibrationMaxMSamples,
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
		{label: fmt.Sprintf("%s$%s sittich audio.wav", green, reset), plainLabel: "$ sittich audio.wav", desc: fmt.Sprintf("%s# %ds chunks, text output%s", dim, defaultChunkSize, reset)},
		{label: fmt.Sprintf("%s$%s cat audio.wav | sittich", green, reset), plainLabel: "$ cat audio.wav | sittich", desc: fmt.Sprintf("%s# transcribe from pipe%s", dim, reset)},
		{label: fmt.Sprintf("%s$%s sittich --chunk-size 30 talk.wav", green, reset), plainLabel: "$ sittich --chunk-size 30 talk.wav", desc: fmt.Sprintf("%s# 30s chunks%s", dim, reset)},
		{label: fmt.Sprintf("%s$%s sittich --remote-url http://localhost:8080 audio.wav", green, reset), plainLabel: "$ sittich --remote-url http://localhost:8080 audio.wav", desc: fmt.Sprintf("%s# transcribe via remote server%s", dim, reset)},
		{label: fmt.Sprintf("%s$%s sittich --format vtt --output subs.vtt audio.wav", green, reset), plainLabel: "$ sittich --format vtt --output subs.vtt audio.wav", desc: fmt.Sprintf("%s# WebVTT to file%s", dim, reset)},
		{label: fmt.Sprintf("%s$%s sittich --listen :8080", green, reset), plainLabel: "$ sittich --listen :8080", desc: fmt.Sprintf("%s# Run server on port 8080%s", dim, reset)},
	}
	printAlignedRows(exampleRows)
	fmt.Fprintln(os.Stderr)
}
