package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"time"

	"github.com/negbie/sittich/internal/asr"
	"github.com/negbie/sittich/internal/models"
	"github.com/negbie/sittich/internal/output"
	"github.com/negbie/sittich/internal/pipeline"
	"github.com/negbie/sittich/internal/types"
)

func run(args []string) error {
	opts, err := parseCLI(args)
	if err != nil {
		switch {
		case errors.Is(err, errShowHelp):
			return nil
		case errors.Is(err, errInvalidArgs):
			printUsage()
			return err
		default:
			return err
		}
	}

	if opts.ShowVersion {
		fmt.Println(version)
		return nil
	}

	// Server mode
	if opts.ListenAddr != "" {
		return runServer(&opts)
	}

	// Handle stdin input by copying to temp file
	audioFile := opts.AudioFile
	var tempFile string
	if opts.AudioFile == "-" {
		tempFile, err = copyStdinToTemp()
		if err != nil {
			return fmt.Errorf("failed to read stdin: %w", err)
		}
		audioFile = tempFile
		defer os.Remove(tempFile)
	}

	var results []types.Segment
	var duration float64

	if opts.RemoteURL != "" {
		serverURL, err := validateRemoteURL(opts.RemoteURL)
		if err != nil {
			return err
		}
		fmt.Fprintf(os.Stderr, "mode: %sremote%s %s•%s url: %s\n", cyan, reset, dim, reset, serverURL)
		srcInfo := opts.AudioFile
		if srcInfo == "-" {
			srcInfo = "stdin"
		}
		fmt.Fprintf(os.Stderr, "audio: %s %s•%s chunks: %ds %s•%s format: %s\n", srcInfo, dim, reset, opts.ChunkSize, dim, reset, opts.Format)

		results, duration, err = transcribeRemote(serverURL, audioFile, opts.ChunkSize)
		if err != nil {
			return err
		}
	} else {
		fmt.Fprintf(os.Stderr, "mode: %slocal%s\n", cyan, reset)

		// 1. Load Model & VAD
		recognizer, err := loadRecognizer(opts)
		if err != nil {
			return err
		}
		defer recognizer.Close()

		vadPath, err := models.GetVADPath(opts.DataFolder)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Warning: failed to get VAD model: %v. Proceeding without VAD.\n", err)
		}

		// 2. Setup Pipeline
		pipe, err := pipeline.NewPipeline(recognizer, pipeline.PipelineConfig{
			VADEnabled:    vadPath != "",
			VADModelPath:  vadPath,
			ChunkDuration: float64(opts.ChunkSize),
			Debug:         opts.Debug,
		})
		if err != nil {
			return fmt.Errorf("failed to initialize pipeline: %w", err)
		}
		defer pipe.Close()

		fmt.Fprintf(os.Stderr, "audio: %s %s•%s chunks: %ds %s•%s format: %s %s•%s VAD: %v\n",
			opts.AudioFile, dim, reset, opts.ChunkSize, dim, reset, opts.Format, dim, reset, vadPath != "")

		// 3. Transcribe
		startTime := time.Now()
		fmt.Fprintf(os.Stderr, "Transcribing...\r")

		result, err := pipe.Process(context.Background(), audioFile)
		if err != nil {
			return fmt.Errorf("transcription failed: %w", err)
		}
		elapsed := time.Since(startTime)

		results = result.Segments
		duration = result.Duration

		// 4. Report stats
		rtFactor := duration / elapsed.Seconds()
		rtColor := green
		if rtFactor < 5 {
			rtColor = yellow
		}
		fmt.Fprintf(os.Stderr, "Done!        \n")
		fmt.Fprintf(os.Stderr, "%s⚡%s Processed %.1fs in %s%.1fs%s %s(%s%.1fx%s realtime)%s\n\n",
			yellow, reset, duration, bold, elapsed.Seconds(), reset, dim, rtColor, rtFactor, reset, reset)
	}

	// 5. Output
	out, closeFn, err := openOutput(opts.OutputFile)
	if err != nil {
		return err
	}
	defer closeFn()

	finalResult := &types.Result{
		Duration: duration,
		Segments: results,
	}

	if err := output.Write(out, opts.Format, finalResult); err != nil {
		return fmt.Errorf("error writing output: %w", err)
	}

	return nil
}

func loadRecognizer(opts cliOptions) (*asr.Recognizer, error) {
	hideCursor()
	defer showCursor()

	fmt.Fprint(os.Stderr, "Loading ASR model...\r")
	modelPath, err := models.GetModelPath(opts.DataFolder)
	if err != nil {
		fmt.Fprintln(os.Stderr)
		return nil, fmt.Errorf("failed to get model: %w", err)
	}

	cfg := &asr.Config{
		ModelPath:      modelPath,
		DecodingMethod: opts.DecodingMethod,
		MaxActivePaths: opts.MaxActivePaths,
	}

	recognizer, err := asr.NewRecognizer(cfg)
	if err != nil {
		fmt.Fprintln(os.Stderr)
		return nil, fmt.Errorf("failed to load model: %w", err)
	}

	fmt.Fprintln(os.Stderr, "ASR Model loaded!   ")
	return recognizer, nil
}

func openOutput(path string) (io.Writer, func(), error) {
	if path == "" {
		return os.Stdout, func() {}, nil
	}

	file, err := os.Create(path)
	if err != nil {
		return nil, nil, fmt.Errorf("error creating output file: %w", err)
	}
	fmt.Fprintf(os.Stderr, "Output: %s\n", path)
	return file, func() { file.Close() }, nil
}

func copyStdinToTemp() (string, error) {
	tmpFile, err := os.CreateTemp("", "sittich-stdin-*")
	if err != nil {
		return "", fmt.Errorf("failed to create temp file: %w", err)
	}
	defer tmpFile.Close()

	if _, err := io.Copy(tmpFile, os.Stdin); err != nil {
		return "", fmt.Errorf("failed to copy stdin: %w", err)
	}

	return tmpFile.Name(), nil
}
