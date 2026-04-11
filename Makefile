# sittich - Fast chunked ASR CLI
# Static Build Makefile (Linux Only)

BINARY_NAME := sittich
BIN_DIR := bin
VERSION := $(shell git describe --tags --always --dirty 2>/dev/null || echo "dev")

# Go build flags for size optimization (strip symbols and debug info)
# We also link statically using CGO_LDFLAGS provided in internal/asr/cgo.go
GO_LDFLAGS := -s -w -X main.version=$(VERSION)

.PHONY: all build release clean help

all: build

help:
	@echo "sittich build system (Linux Static)"
	@echo ""
	@echo "Targets:"
	@echo "  build          Build the static binary to $(BIN_DIR)/"
	@echo "  release        Build and compress with UPX"
	@echo "  clean          Remove build artifacts"
	@echo ""

# Build the Go binary
build:
	@echo "Building static $(BINARY_NAME) $(VERSION)..."
	@mkdir -p $(BIN_DIR)
	go build -ldflags="$(GO_LDFLAGS)" -o $(BIN_DIR)/$(BINARY_NAME) ./cmd/sittich
	@echo "Binary ready in $(BIN_DIR)/"
	@ls -sh $(BIN_DIR)/

# Final release optimization (Strip + UPX)
release: build
	@if command -v upx > /dev/null; then \
		echo "Compressing with UPX..."; \
		upx $(BIN_DIR)/$(BINARY_NAME); \
	else \
		echo "UPX not found, skipping compression."; \
	fi
	@echo "Release binary ready in $(BIN_DIR)/"
	@ls -sh $(BIN_DIR)/

# Clean build artifacts
clean:
	@echo "Cleaning..."
	rm -rf $(BIN_DIR)
	go clean -cache
