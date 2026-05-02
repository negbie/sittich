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
	ModelURL     = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2"
	NemoModelURL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-transducer-stt_de_fastconformer_hybrid_large_pc-int8.tar.bz2"
	EncoderFile  = "encoder.int8.onnx"
	DecoderFile  = "decoder.int8.onnx"
	JoinerFile   = "joiner.int8.onnx"
	TokensFile   = "tokens.txt"
	VADFile      = "silero_vad.onnx"
	SileroVADURL = "https://huggingface.co/istupakov/silero-vad-onnx/resolve/main/silero_vad_16k_op15.onnx?download=true"
)

var requiredModelFiles = []string{EncoderFile, DecoderFile, JoinerFile, TokensFile}

func GetModelPath(dataDir, url string) (string, error) {
	if dataDir == "" {
		dataDir = "./data"
	}

	if !isValidModel(dataDir) {
		fmt.Fprintf(os.Stderr, "Downloading model to %s...\n", dataDir)
		if err := downloadAndExtract(dataDir, url); err != nil {
			return "", fmt.Errorf("failed to download model: %w", err)
		}
	}

	return dataDir, nil
}

func GetVADPath(dataDir string) (string, error) {
	if dataDir == "" {
		dataDir = "./data/vad"
	}
	targetPath := filepath.Join(dataDir, VADFile)

	if _, err := os.Stat(targetPath); err != nil {
		fmt.Fprintf(os.Stderr, "Downloading VAD model to %s...\n", targetPath)
		if err := downloadToFile(targetPath, SileroVADURL); err != nil {
			return "", fmt.Errorf("failed to download VAD model: %w", err)
		}
	}

	return targetPath, nil
}

func isValidModel(path string) bool {
	for _, file := range requiredModelFiles {
		if _, err := os.Stat(filepath.Join(path, file)); err != nil {
			return false
		}
	}
	return true
}

func downloadAndExtract(targetDir, url string) error {
	if err := os.MkdirAll(filepath.Dir(targetDir), 0755); err != nil {
		return err
	}

	tmpFile, err := os.CreateTemp(filepath.Dir(targetDir), "sittich-model-*.tar.bz2")
	if err != nil {
		return err
	}
	tmpPath := tmpFile.Name()
	defer os.Remove(tmpPath)

	if err := download(tmpFile, url); err != nil {
		return err
	}

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

func downloadToFile(targetPath, url string) error {
	if err := os.MkdirAll(filepath.Dir(targetPath), 0755); err != nil {
		return err
	}

	f, err := os.Create(targetPath + ".tmp")
	if err != nil {
		return err
	}
	defer os.Remove(f.Name())

	if err := download(f, url); err != nil {
		f.Close()
		return err
	}
	f.Close()

	return os.Rename(f.Name(), targetPath)
}

func download(w io.WriteSeeker, url string) error {
	resp, err := http.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
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
			nw, werr := w.Write(buf[:nr])
			if nw > 0 {
				written += int64(nw)
			}
			if werr != nil {
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
			return rerr
		}
	}
	fmt.Fprintln(os.Stderr) // New line after progress
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
