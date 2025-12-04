<!-- SPDX-FileCopyrightText: 2024 Joshua Jewell -->
<!-- SPDX-License-Identifier: MIT -->

# Roadmap

This document outlines the planned development direction for VAE Dataset Normalizer.

## Vision

Enable robust, verifiable training of AI image detection models by providing
high-quality paired datasets with formal guarantees.

## Current Status: v1.0.0

✅ **Complete**:
- Core normalization functionality
- SHAKE256 checksums
- Train/test/val/cal splits (random + stratified)
- CUE metadata with Dublin Core
- Isabelle/HOL formal proofs
- Julia/Flux training utilities
- RSR compliance

## Short-Term (v1.1.0)

### Planned Features

- [ ] **Multi-VAE support**: Process datasets through different VAEs
  - SD 1.5 VAE
  - SDXL VAE
  - Flux VAE
  - Custom VAE paths

- [ ] **Parallel processing**: Configurable worker threads
  - `--jobs N` flag implementation
  - Rayon thread pool optimization

- [ ] **Export formats**: Additional output formats
  - Parquet export
  - HuggingFace datasets format
  - TFRecord format

### Improvements

- [ ] Better progress reporting
- [ ] Memory-mapped file I/O for large datasets
- [ ] Incremental processing (resume interrupted jobs)

## Medium-Term (v1.2.0)

### Planned Features

- [ ] **Image preprocessing**:
  - Automatic resizing
  - Format conversion
  - Quality filtering

- [ ] **Augmentation integration**:
  - Document augmentation effects on VAE artifacts
  - Augmentation-aware split generation

- [ ] **Metrics computation**:
  - PSNR/SSIM between original and VAE
  - Artifact intensity scoring
  - Statistical summaries

### Infrastructure

- [ ] GitHub Actions / GitLab CI templates
- [ ] Pre-built binaries for major platforms
- [ ] Homebrew formula

## Long-Term (v2.0.0)

### Vision Features

- [ ] **Multi-model artifact detection**:
  - Support for non-VAE generative models
  - GAN artifact datasets
  - Autoregressive model artifacts

- [ ] **Federated dataset management**:
  - Distributed split generation
  - Cross-institution dataset pooling
  - Privacy-preserving checksums

- [ ] **Active learning integration**:
  - Uncertainty-based sample selection
  - Human-in-the-loop verification
  - Continuous model improvement

### Research Directions

- [ ] VAE artifact taxonomy
- [ ] Detection model benchmarks
- [ ] Adversarial robustness testing

## End-of-Life Planning

### Maintenance Commitment

- **Active development**: Until stated otherwise
- **Security fixes**: Minimum 2 years from v1.0.0
- **Critical bugs**: Minimum 3 years from v1.0.0

### Succession

If primary maintainer becomes unavailable:

1. Repository remains MIT licensed (forkable)
2. Archive on Software Heritage
3. Transfer to community organization if interest exists
4. Data export always available

### Archive Strategy

- Full source history preserved
- Binary releases archived
- Documentation snapshots
- Dataset compatibility notes

## Contributing to Roadmap

### Suggesting Features

1. Open issue with `roadmap` label
2. Describe use case
3. Propose implementation approach (optional)
4. Community discussion period

### Prioritization

Features prioritized by:

1. User demand (issue reactions)
2. Alignment with vision
3. Implementation complexity
4. Maintainer capacity

## Version Policy

- **Major** (x.0.0): Breaking changes, major features
- **Minor** (1.x.0): New features, backward compatible
- **Patch** (1.0.x): Bug fixes, security patches

## Contact

Roadmap questions? Open an issue or discussion.
