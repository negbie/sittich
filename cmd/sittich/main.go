package main

import (
	"errors"
	"fmt"
	"os"
)

var version = "dev"

func main() {
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
