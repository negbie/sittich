package models

import (
	"archive/tar"
	"compress/bzip2"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
)

const (
	DefaultModelName = "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"
	ModelURL         = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2"

	EncoderFile = "encoder.int8.onnx"
	DecoderFile = "decoder.int8.onnx"
	JoinerFile  = "joiner.int8.onnx"
	TokensFile  = "tokens.txt"

	VADModelName = "silero-vad"
	VADModelURL  = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx"
	VADFile      = "silero_vad.onnx"
)

var requiredModelFiles = []string{EncoderFile, DecoderFile, JoinerFile, TokensFile}

// GetVADPath returns the path to the VAD model file, downloading if necessary
func GetVADPath() (string, error) {
	cacheDir := getCacheDir()
	vadDir := filepath.Join(cacheDir, "sittich", "models", VADModelName)
	vadPath := filepath.Join(vadDir, VADFile)

	if _, err := os.Stat(vadPath); err == nil {
		return vadPath, nil
	}

	fmt.Fprintf(os.Stderr, "Downloading VAD model to %s...\n", vadPath)
	if err := os.MkdirAll(vadDir, 0755); err != nil {
		return "", err
	}

	if err := downloadFile(VADModelURL, vadPath); err != nil {
		return "", fmt.Errorf("failed to download VAD model: %w", err)
	}

	return vadPath, nil
}

// downloadFile downloads a single file from url to path
func downloadFile(url, path string) error {
	resp, err := http.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("download failed: %s", resp.Status)
	}

	tmpPath := path + ".part"
	out, err := os.Create(tmpPath)
	if err != nil {
		return err
	}

	if _, err := io.Copy(out, resp.Body); err != nil {
		out.Close()
		os.Remove(tmpPath)
		return err
	}

	if err := out.Close(); err != nil {
		os.Remove(tmpPath)
		return err
	}

	if err := os.Rename(tmpPath, path); err != nil {
		os.Remove(tmpPath)
		return err
	}

	return nil
}

// GetModelPath returns the path to the model directory, downloading if necessary
func GetModelPath() (string, error) {
	// 1. Prefer local ./data directory
	localModelDir := "./data"
	if isValidModel(localModelDir) {
		return localModelDir, nil
	}

	// 2. Check sittich_MODEL env var
	if envPath := os.Getenv("sittich_MODEL"); envPath != "" {
		if isValidModel(envPath) {
			return envPath, nil
		}
		fmt.Fprintf(os.Stderr, "Warning: sittich_MODEL=%s not found or invalid\n", envPath)
	}

	// 3. Check cache directory
	cacheDir := getCacheDir()
	modelDir := filepath.Join(cacheDir, "sittich", "models", DefaultModelName)

	if isValidModel(modelDir) {
		return modelDir, nil
	}

	// 4. Download model
	fmt.Fprintf(os.Stderr, "Downloading model to %s...\n", modelDir)
	if err := downloadAndExtract(modelDir); err != nil {
		return "", fmt.Errorf("failed to download model: %w", err)
	}

	return modelDir, nil
}

func getCacheDir() string {
	if xdg := os.Getenv("XDG_CACHE_HOME"); xdg != "" {
		return xdg
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return ".cache"
	}
	return filepath.Join(home, ".cache")
}

func isValidModel(path string) bool {
	for _, file := range requiredModelFiles {
		if _, err := os.Stat(filepath.Join(path, file)); err != nil {
			return false
		}
	}
	return true
}

func downloadAndExtract(targetDir string) error {
	if err := os.MkdirAll(filepath.Dir(targetDir), 0755); err != nil {
		return err
	}

	tmpFile, err := os.CreateTemp(filepath.Dir(targetDir), "sittich-model-*.tar.bz2")
	if err != nil {
		return err
	}
	tmpPath := tmpFile.Name()
	defer os.Remove(tmpPath)

	resp, err := http.Get(ModelURL)
	if err != nil {
		tmpFile.Close()
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		tmpFile.Close()
		return fmt.Errorf("download failed: %s", resp.Status)
	}

	// Download with single-line progress bar
	size := resp.ContentLength
	written := int64(0)
	buf := make([]byte, 64*1024)
	lastPercent := -1

	for {
		nr, rerr := resp.Body.Read(buf)
		if nr > 0 {
			nw, werr := tmpFile.Write(buf[:nr])
			if nw > 0 {
				written += int64(nw)
			}
			if werr != nil {
				tmpFile.Close()
				return werr
			}
			// Update progress every 5%
			if size > 0 {
				percent := int(float64(written) * 100 / float64(size))
				if percent != lastPercent && percent%5 == 0 {
					mb := float64(written) / (1024 * 1024)
					totalMb := float64(size) / (1024 * 1024)
					fmt.Fprintf(os.Stderr, "\r  Downloading: %.1f / %.1f MB (%d%%)", mb, totalMb, percent)
					lastPercent = percent
				}
			}
		}
		if rerr == io.EOF {
			break
		}
		if rerr != nil {
			tmpFile.Close()
			return rerr
		}
	}
	if err := tmpFile.Close(); err != nil {
		return err
	}
	fmt.Fprintln(os.Stderr) // New line after progress

	stagingDir := targetDir + ".tmp"
	_ = os.RemoveAll(stagingDir)
	if err := os.MkdirAll(stagingDir, 0755); err != nil {
		return err
	}

	fmt.Fprintf(os.Stderr, "Extracting...\n")

	if err := extractTarBz2(tmpPath, stagingDir); err != nil {
		os.RemoveAll(stagingDir)
		return fmt.Errorf("extraction failed: %w", err)
	}

	if !isValidModel(stagingDir) {
		os.RemoveAll(stagingDir)
		return fmt.Errorf("extraction failed: required model files missing")
	}

	_ = os.RemoveAll(targetDir)
	if err := os.Rename(stagingDir, targetDir); err != nil {
		os.RemoveAll(stagingDir)
		return err
	}

	fmt.Fprintf(os.Stderr, "Model ready\n")
	return nil
}

func extractTarBz2(archivePath, targetDir string) error {
	if err := os.MkdirAll(targetDir, 0755); err != nil {
		return err
	}

	baseDir, err := filepath.Abs(targetDir)
	if err != nil {
		return err
	}

	file, err := os.Open(archivePath)
	if err != nil {
		return err
	}
	defer file.Close()

	bz2Reader := bzip2.NewReader(file)
	tarReader := tar.NewReader(bz2Reader)

	for {
		header, err := tarReader.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}

		// Strip root directory from path
		cleanName := filepath.Clean(header.Name)
		parts := strings.SplitN(cleanName, string(filepath.Separator), 2)
		if len(parts) == 1 {
			// Root directory entry - skip
			continue
		}
		cleanName = parts[1]
		if cleanName == "" || cleanName == "." {
			continue
		}

		target := filepath.Join(baseDir, cleanName)
		absTarget, err := filepath.Abs(target)
		if err != nil {
			return err
		}
		rel, err := filepath.Rel(baseDir, absTarget)
		if err != nil {
			return err
		}
		if strings.HasPrefix(rel, "..") || filepath.IsAbs(rel) {
			return fmt.Errorf("archive entry escapes target directory: %s", header.Name)
		}

		switch header.Typeflag {
		case tar.TypeDir:
			if err := os.MkdirAll(absTarget, 0755); err != nil {
				return err
			}
		case tar.TypeReg:
			if err := os.MkdirAll(filepath.Dir(absTarget), 0755); err != nil {
				return err
			}
			outFile, err := os.Create(absTarget)
			if err != nil {
				return err
			}
			if _, err := io.Copy(outFile, tarReader); err != nil {
				outFile.Close()
				return err
			}
			if err := outFile.Close(); err != nil {
				return err
			}
		}
	}

	return nil
}
