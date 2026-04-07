# sittich - Fast chunked ASR CLI
# Build & Bundle Makefile

BINARY_NAME := sittich
BIN_DIR := bin
GO_OS := $(shell go env GOOS)
GO_ARCH := $(shell go env GOARCH)
VERSION := $(shell git describe --tags --always --dirty 2>/dev/null || echo "dev")

# Determine shared library extension based on OS
ifeq ($(GO_OS),darwin)
	LIB_EXT := dylib
	LIB_PLATFORM := darwin_$(GO_ARCH)
else ifeq ($(GO_OS),windows)
	LIB_EXT := dll
	LIB_PLATFORM := windows_$(GO_ARCH)
else
	LIB_EXT := so
	LIB_PLATFORM := linux_$(GO_ARCH)
endif

# Go build flags for size optimization (strip symbols and debug info)
GO_LDFLAGS := -s -w -X main.version=$(VERSION)

.PHONY: all build build-local stage-libs stage-libs-embedded bundle build-embedded release run clean help

all: build-embedded

help:
	@echo "sittich build system"
	@echo ""
	@echo "Targets:"
	@echo "  build          Build the binary to $(BIN_DIR)/"
	@echo "  stage-libs     Extract shared libraries from Go mod cache"
	@echo "  bundle         Build binary, copy libs, and patch runtime paths (Linux/macOS)"
	@echo "  build-embedded Build portable binary with embedded host-only libs"
	@echo "  release        Build, strip, and compress with UPX"
	@echo "  run            Build and run locally with LD_LIBRARY_PATH"
	@echo "  clean          Remove build artifacts"
	@echo ""

# Build the Go binary
build:
	@echo "Building $(BINARY_NAME) $(VERSION)..."
	@mkdir -p $(BIN_DIR)
	go build -ldflags="$(GO_LDFLAGS)" -o $(BIN_DIR)/$(BINARY_NAME) ./cmd/sittich

# Step 1: Stage libraries from Go mod cache to /tmp/sittich-libbundle
stage-libs:
	@echo "Staging sherpa-onnx shared libraries to /tmp..."
	@bash ./scripts/stage-sherpa-libs.sh /tmp/sittich-libbundle

# Stage libraries for embedding into the binary (Host platform only)
stage-libs-embedded:
	@SITTICH_HOST_ONLY=1 bash ./scripts/stage-sherpa-libs.sh internal/libbundle/libs

# Step 2: Full bundle (build + libs + rpath fix)
bundle: build stage-libs
	@echo "Bundling libraries for $(LIB_PLATFORM)..."
	@cp -f /tmp/sittich-libbundle/$(LIB_PLATFORM)/*.$(LIB_EXT) $(BIN_DIR)/
	@echo "Patching runtime paths..."
	@bash ./scripts/fix-runtime-paths.sh $(BIN_DIR)/$(BINARY_NAME) $(GO_OS) $(GO_ARCH)
	@echo "Bundle ready in $(BIN_DIR)/"
	@ls -sh $(BIN_DIR)/

# Single portable binary with embedded libs (Host platform only)
build-embedded: stage-libs-embedded build
	@echo "Portable binary ready in $(BIN_DIR)/"
	@ls -sh $(BIN_DIR)/

# Final release optimization (Strip + UPX)
release: build-embedded
	@if command -v upx > /dev/null; then \
		echo "Compressing with UPX..."; \
		upx $(BIN_DIR)/$(BINARY_NAME); \
	else \
		echo "UPX not found, skipping compression."; \
	fi
	@echo "Release binary ready in $(BIN_DIR)/"
	@ls -sh $(BIN_DIR)/

# Local build Alias
build-local: build

# Run locally without patching (using LD_LIBRARY_PATH)
run: build stage-libs
	@echo "Running $(BINARY_NAME)..."
	@LD_LIBRARY_PATH=$(LD_LIBRARY_PATH):/tmp/sittich-libbundle/$(LIB_PLATFORM) ./$(BIN_DIR)/$(BINARY_NAME) $(ARGS)

# Clean build artifacts
clean:
	@echo "Cleaning..."
	rm -rf $(BIN_DIR)
	go clean -cache

