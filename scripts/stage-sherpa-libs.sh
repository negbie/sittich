#!/usr/bin/env bash
set -euo pipefail

mod_cache="$(go env GOMODCACHE)"
if command -v cygpath >/dev/null 2>&1; then
	mod_cache="$(cygpath -u "$mod_cache")"
fi

linux_version="$(go list -m -f '{{.Version}}' github.com/k2-fsa/sherpa-onnx-go-linux)"
macos_version="$(go list -m -f '{{.Version}}' github.com/k2-fsa/sherpa-onnx-go-macos)"
windows_version="$(go list -m -f '{{.Version}}' github.com/k2-fsa/sherpa-onnx-go-windows)"

bundle_root="${1:-/tmp/sittich-libbundle}"
host_only="${SITTICH_HOST_ONLY:-0}"
go_os="$(go env GOOS)"
go_arch="$(go env GOARCH)"

copy_libs() {
	local platform="$1"
	local arch="$2"
	local pkg="$3"
	local lib_src="$4"

	if [[ "$host_only" == "1" ]]; then
		if [[ "$platform" != "$go_os" ]] || [[ "$arch" != "$go_arch" ]]; then
			return
		fi
	fi

	local target_dir="$bundle_root/${platform}_${arch}"
	mkdir -p "$target_dir"
	
	local version
	version="$(go list -m -f '{{.Version}}' "github.com/k2-fsa/$pkg")"
	local src_dir="${mod_cache}/github.com/k2-fsa/${pkg}@${version}/lib/${lib_src}"
	
	if [[ "$platform" == "windows" ]]; then
		cp -f "$src_dir"/*.dll "$target_dir/"
	elif [[ "$platform" == "darwin" ]]; then
		cp -f "$src_dir"/*.dylib "$target_dir/"
	else
		cp -f "$src_dir"/*.so "$target_dir/"
	fi
}

copy_libs "linux" "amd64" "sherpa-onnx-go-linux" "x86_64-unknown-linux-gnu"
copy_libs "darwin" "amd64" "sherpa-onnx-go-macos" "x86_64-apple-darwin"
copy_libs "darwin" "arm64" "sherpa-onnx-go-macos" "aarch64-apple-darwin"
copy_libs "windows" "amd64" "sherpa-onnx-go-windows" "x86_64-pc-windows-gnu"

