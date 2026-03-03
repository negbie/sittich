# Build stage
FROM golang:latest AS builder
WORKDIR /build
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=1 go build -ldflags="-w -s" -o chough ./cmd/chough

# Runtime stage
FROM debian:stable-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg ca-certificates && rm -rf /var/lib/apt/lists/*

# Copy binary and libraries to /opt/chough/
# Binary has rpath=$ORIGIN, so it looks for libs in same directory
COPY --from=builder /build/chough /opt/chough/
COPY --from=builder /go/pkg/mod/github.com/k2-fsa/sherpa-onnx-go-linux@v1.12.26/lib/x86_64-unknown-linux-gnu/*.so /opt/chough/

EXPOSE 8080
ENTRYPOINT ["/opt/chough/chough"]
CMD ["--server", "--host", "0.0.0.0", "--port", "8080"]
