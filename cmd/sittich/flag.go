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
	ListenAddr             string
	ChunkSize              int
	ChunkMinTailDuration   float64
	Format                 string
	MaxActivePaths         int
	DecodingMethod         string
	NoVAD                  bool
	VADThreshold           float64
	VADMinSilenceDuration  float64
	VADMinSpeechDuration   float64
	CalibrationTargetPeak  float64
	CalibrationMaxGain     float64
	CalibrationMaxMSamples int
	DataFolder             string
	Workers                int
	NumThreads             int
	MaxUploadMB            int
	Debug                  bool
	ShowVersion            bool
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
	{long: "listen", arg: "address", description: "listen address", defaultVal: config.DefaultListenAddr},
	{long: "workers", arg: "int", description: "concurrent workers", defaultVal: strconv.Itoa(config.DefaultWorkers)},
	{long: "num-threads", arg: "int", description: "number of threads for recognizer and VAD", defaultVal: strconv.Itoa(config.DefaultNumThreads)},
	{long: "max-upload", arg: "int", description: "max upload size in MB", defaultVal: strconv.Itoa(config.DefaultMaxUploadMB)},
	{long: "format", arg: "string", description: "default output format: text, json, vtt", defaultVal: config.DefaultFormat},
	{long: "chunk-size", arg: "int", description: "default chunk size in seconds", defaultVal: strconv.Itoa(config.DefaultChunkSize)},
	{long: "chunk-min-tail", arg: "float", description: "minimum tail duration when balancing oversized chunks in seconds", defaultVal: strconv.FormatFloat(config.DefaultChunkMinTail, 'f', 1, 64)},
	{long: "debug", description: "show detailed debug logs"},
	{long: "max-active-paths", arg: "int", description: "number of active paths for modified beam search", defaultVal: strconv.Itoa(config.DefaultMaxActivePaths)},
	{long: "decoding-method", arg: "string", description: "decoding method: greedy_search or modified_beam_search", defaultVal: config.DefaultDecodingMethod},
	{long: "no-vad", description: "disable VAD"},
	{long: "vad-threshold", arg: "float", description: "VAD speech threshold", defaultVal: strconv.FormatFloat(config.DefaultVADThreshold, 'f', 1, 64)},
	{long: "vad-min-silence", arg: "float", description: "minimum silence duration for VAD splits in seconds", defaultVal: strconv.FormatFloat(config.DefaultVADMinSilence, 'f', 1, 64)},
	{long: "vad-min-speech", arg: "float", description: "minimum speech duration for VAD segments in seconds", defaultVal: strconv.FormatFloat(config.DefaultVADMinSpeech, 'f', 2, 64)},
	{long: "calibration-target-peak", arg: "float", description: "target robust peak used for audio calibration", defaultVal: strconv.FormatFloat(config.DefaultCalibrationTargetPeak, 'f', 1, 64)},
	{long: "calibration-max-gain", arg: "float", description: "maximum gain applied during audio calibration", defaultVal: strconv.FormatFloat(config.DefaultCalibrationMaxGain, 'f', 1, 64)},
	{long: "calibration-max-samples", arg: "int", description: "maximum sampled points used for calibration peak estimation", defaultVal: strconv.Itoa(config.DefaultCalibrationMaxMSamples)},
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

	listenAddr := fs.String("listen", config.DefaultListenAddr, "listen address")
	workers := fs.Int("workers", config.DefaultWorkers, "concurrent workers")
	numThreads := fs.Int("num-threads", config.DefaultNumThreads, "number of threads for recognizer and VAD")
	maxUploadMB := fs.Int("max-upload", config.DefaultMaxUploadMB, "max upload size in MB")
	format := fs.String("format", config.DefaultFormat, "default output format (text, json, vtt)")
	chunkSize := fs.Int("chunk-size", config.DefaultChunkSize, "default chunk size in seconds")
	chunkMinTail := fs.Float64("chunk-min-tail", config.DefaultChunkMinTail, "minimum tail duration when balancing oversized chunks in seconds")
	maxActivePaths := fs.Int("max-active-paths", config.DefaultMaxActivePaths, "number of active paths for modified beam search")
	decodingMethod := fs.String("decoding-method", config.DefaultDecodingMethod, "decoding method: greedy_search or modified_beam_search")
	noVAD := fs.Bool("no-vad", false, "disable VAD")
	vadThreshold := fs.Float64("vad-threshold", config.DefaultVADThreshold, "VAD speech threshold")
	vadMinSilence := fs.Float64("vad-min-silence", config.DefaultVADMinSilence, "minimum silence duration for VAD splits in seconds")
	vadMinSpeech := fs.Float64("vad-min-speech", config.DefaultVADMinSpeech, "minimum speech duration for VAD segments in seconds")
	calibrationTargetPeak := fs.Float64("calibration-target-peak", config.DefaultCalibrationTargetPeak, "target robust peak used for audio calibration")
	calibrationMaxGain := fs.Float64("calibration-max-gain", config.DefaultCalibrationMaxGain, "maximum gain applied during audio calibration")
	calibrationMaxSamples := fs.Int("calibration-max-samples", config.DefaultCalibrationMaxMSamples, "maximum sampled points used for calibration peak estimation")
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
		ListenAddr:             *listenAddr,
		Workers:                *workers,
		NumThreads:             *numThreads,
		MaxUploadMB:            *maxUploadMB,
		Format:                 strings.ToLower(*format),
		ChunkSize:              *chunkSize,
		ChunkMinTailDuration:   *chunkMinTail,
		MaxActivePaths:         *maxActivePaths,
		DecodingMethod:         strings.ToLower(*decodingMethod),
		NoVAD:                  *noVAD,
		VADThreshold:           *vadThreshold,
		VADMinSilenceDuration:  *vadMinSilence,
		VADMinSpeechDuration:   *vadMinSpeech,
		CalibrationTargetPeak:  *calibrationTargetPeak,
		CalibrationMaxGain:     *calibrationMaxGain,
		CalibrationMaxMSamples: *calibrationMaxSamples,
		DataFolder:             *dataFolder,
		Debug:                  *debug,
		ShowVersion:            *showVersion,
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

func recognizerConfigFromCLI(opts cliOptions, modelPath string) *config.ASR {
	return &config.ASR{
		ModelPath:              modelPath,
		NumThreads:             opts.NumThreads,
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
