package main

import (
	"fmt"
	"os"
)

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
