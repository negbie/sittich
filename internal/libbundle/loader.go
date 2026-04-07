package libbundle

import (
	"embed"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
)

//go:embed libs/*
var libFiles embed.FS

const (
	envSignal = "SITTICH_EMBEDDED"
	libSubdir = ".libs"
)

// Bootstrap ensures that mandated shared libraries are extracted and accessible
// by the dynamic linker before the Go runtime attempts to load cgo-linked objects.
// If the environment is not set up, it extracts the libs and re-executes the process.
func Bootstrap() {
	// If we've already re-executed, we're done.
	if os.Getenv(envSignal) == "1" {
		return
	}

	// 1. Identify where we should extract the libs.
	// We peek at the --data-folder flag to keep libs in sync with models.
	dataDir := peekDataFolder()
	if dataDir == "" {
		dataDir = "data"
	}
	absData, _ := filepath.Abs(dataDir)
	libDir := filepath.Join(absData, libSubdir, fmt.Sprintf("%s_%s", runtime.GOOS, runtime.GOARCH))

	// 2. Extract libraries if they are missing or outdated.
	// For simplicity, we currently extract if the directory doesn't exist.
	if err := extractLibs(libDir); err != nil {
		fmt.Fprintf(os.Stderr, "Error bootstrapping libraries: %v\n", err)
		return
	}

	// 3. Prepare environment for re-execution.
	newPath := libDir
	if current := os.Getenv(envPathKey()); current != "" {
		newPath = libDir + string(os.PathListSeparator) + current
	}

	// 4. Re-execute the binary.
	cmd := exec.Command(os.Args[0], os.Args[1:]...)
	cmd.Env = append(os.Environ(), 
		fmt.Sprintf("%s=1", envSignal),
		fmt.Sprintf("%s=%s", envPathKey(), newPath),
	)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.Stdin = os.Stdin

	if err := cmd.Run(); err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok {
			os.Exit(exitErr.ExitCode())
		}
		fmt.Fprintf(os.Stderr, "Error re-executing for embedded libs: %v\n", err)
		os.Exit(1)
	}
	os.Exit(0)
}

func envPathKey() string {
	switch runtime.GOOS {
	case "darwin":
		return "DYLD_LIBRARY_PATH"
	case "windows":
		return "PATH"
	default:
		return "LD_LIBRARY_PATH"
	}
}


func peekDataFolder() string {
	for i := 1; i < len(os.Args); i++ {
		arg := os.Args[i]
		if arg == "-data-folder" || arg == "--data-folder" {
			if i+1 < len(os.Args) {
				return os.Args[i+1]
			}
		}
		if strings.HasPrefix(arg, "-data-folder=") {
			return strings.TrimPrefix(arg, "-data-folder=")
		}
		if strings.HasPrefix(arg, "--data-folder=") {
			return strings.TrimPrefix(arg, "--data-folder=")
		}
	}
	return ""
}

func extractLibs(targetDir string) error {
	platformDir := fmt.Sprintf("libs/%s_%s", runtime.GOOS, runtime.GOARCH)
	entries, err := libFiles.ReadDir(platformDir)
	if err != nil {
		// If no libs are found for this platform, assume they are already on the system
		// or not required for this build.
		return nil
	}

	if _, err := os.Stat(targetDir); err == nil {
		// Already exists. In production we might want to check versions/checksums.
		return nil
	}

	if err := os.MkdirAll(targetDir, 0755); err != nil {
		return fmt.Errorf("create lib dir: %w", err)
	}

	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		
		srcPath := filepath.Join(platformDir, entry.Name())
		dstPath := filepath.Join(targetDir, entry.Name())
		
		if err := copyFile(srcPath, dstPath); err != nil {
			return fmt.Errorf("extract %s: %w", entry.Name(), err)
		}
	}
	return nil
}

func copyFile(src string, dst string) error {
	srcFile, err := libFiles.Open(src)
	if err != nil {
		return err
	}
	defer srcFile.Close()

	dstFile, err := os.OpenFile(dst, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0755)
	if err != nil {
		return err
	}
	defer dstFile.Close()

	_, err = io.Copy(dstFile, srcFile)
	return err
}
