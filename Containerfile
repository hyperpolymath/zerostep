# SPDX-FileCopyrightText: 2024 Joshua Jewell
# SPDX-License-Identifier: MIT
#
# VAE Dataset Normalizer Container
# RSR Compliant: Chainguard Wolfi base image
#
# Build:
#   podman build -t vae-normalizer -f Containerfile .
#
# Run:
#   podman run --rm -v ./data:/data:Z vae-normalizer normalize -d /data/input -o /data/output

# Stage 1: Build
FROM cgr.dev/chainguard/wolfi-base:latest AS builder

# Install Rust toolchain
RUN apk add --no-cache \
    rust \
    cargo \
    build-base \
    openssl-dev \
    pkgconf

WORKDIR /build
COPY Cargo.toml Cargo.lock* ./
COPY src/ ./src/

# Build release binary
RUN cargo build --release

# Stage 2: Runtime
FROM cgr.dev/chainguard/wolfi-base:latest

# Security: Run as non-root user
RUN adduser -D -u 1000 vae
USER vae

WORKDIR /app

# Copy binary from builder
COPY --from=builder /build/target/release/vae-normalizer /app/vae-normalizer

# Data volume
VOLUME ["/data"]
WORKDIR /data

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD ["/app/vae-normalizer", "--version"]

ENTRYPOINT ["/app/vae-normalizer"]
CMD ["--help"]

# Labels (OCI Image Spec)
LABEL org.opencontainers.image.title="VAE Dataset Normalizer"
LABEL org.opencontainers.image.description="Normalize VAE-decoded image datasets with formal verification"
LABEL org.opencontainers.image.version="1.0.0"
LABEL org.opencontainers.image.authors="Joshua Jewell"
LABEL org.opencontainers.image.licenses="MIT"
LABEL org.opencontainers.image.source="https://huggingface.co/datasets/joshuajewell/VAEDecodedImages-SDXL"
LABEL org.opencontainers.image.base.name="cgr.dev/chainguard/wolfi-base:latest"
