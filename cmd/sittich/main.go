package main

import (
	"fmt"
	"os"

	"github.com/negbie/sittich/internal/libbundle"
)

var version = "dev"

func main() {
	// Bootstrap ensures shared libraries are available before continuing.
	libbundle.Bootstrap()

	if err := run(os.Args[1:]); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}
