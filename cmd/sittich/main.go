package main

import (
	"errors"
	"fmt"
	"os"
	"runtime/debug"
)

var version = "dev"

func main() {
	// Tune GC for this memory-intensive binary:
	// - Lower GC threshold to collect more frequently
	// - Set a soft memory limit so the GC aggressively returns memory to the OS.
	//   The model (~700 MB) is allocated via CGO (outside Go heap), so 1 GiB is
	//   a reasonable ceiling for the Go-side heap (audio buffers, results, etc.).
	debug.SetGCPercent(50)
	debug.SetMemoryLimit(1 << 30)

	if err := run(os.Args[1:]); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}

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

	return runServer(&opts)
}
