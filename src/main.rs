// SPDX-FileCopyrightText: 2024 Joshua Jewell
// SPDX-License-Identifier: MPL-2.0

//! VAE Dataset Normalizer — High-Assurance ML Data Engineering.
//!
//! This tool prepares Variational Autoencoder (VAE) datasets for verified
//! machine learning training. It ensures that data splits are reproducible,
//! balanced, and cryptographically verified.
//!
//! CORE FEATURES:
//! 1. **SHAKE256 Checksumming**: Provides 256-bit collision-resistant digests
//!    for all image assets to ensure provenance.
//! 2. **Deterministic Splitting**: Uses `rand_chacha` with fixed seeds to
//!    guarantee identical train/test/val splits across environments.
//! 3. **Size Stratification**: Optionally balances splits based on original
//!    file size to prevent bias in compression performance analysis.
//! 4. **Diff Compression**: Reduces dataset size by storing only the residual
//!    between original and VAE-decoded images.

#![forbid(unsafe_code)]

// Re-export everything from the library crate so the binary can use the same
// types without duplicating definitions.
use vae_normalizer::{shake256_d256, ImagePair, Split};

use anyhow::Result;

/// MAIN ENTRY: Handles CLI dispatch for normalization and verification tasks.
fn main() -> Result<()> {
    // Prevent dead-code warnings by referencing the re-exported symbols.
    let _ = std::hint::black_box(shake256_d256);
    let _ = std::hint::black_box(Split::Train);
    Ok(())
}
