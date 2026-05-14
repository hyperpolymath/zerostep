// SPDX-License-Identifier: PMPL-1.0-or-later

//! VAE Dataset Normalizer library — public types and pure functions.
//!
//! This module exposes the core domain types and deterministic algorithms used
//! by the normalizer binary, making them available for unit and integration
//! testing without depending on file-system I/O or a running CLI.

pub mod metadata;

use tiny_keccak::{Hasher, Shake};

// ---------------------------------------------------------------------------
// Domain types
// ---------------------------------------------------------------------------

/// ASSIGNMENT LOGIC: Partition the dataset into four distinct subsets.
///
/// Proportions follow the standard ML convention used by this tool:
/// - `Train`:       70% — model optimisation
/// - `Test`:        15% — unseen performance evaluation
/// - `Val`:         10% — hyperparameter tuning
/// - `Calibration`:  5% — uncertainty / quantisation calibration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Split {
    /// 70% split — primary training data.
    Train,
    /// 15% split — unseen performance evaluation.
    Test,
    /// 10% split — hyperparameter tuning.
    Val,
    /// 5% split — uncertainty / quantisation calibration.
    Calibration,
}

impl Split {
    /// Return the canonical lowercase string label for this split variant.
    pub fn as_str(&self) -> &'static str {
        match self {
            Split::Train => "train",
            Split::Test => "test",
            Split::Val => "val",
            Split::Calibration => "calibration",
        }
    }

    /// Return all split variants in a fixed, deterministic order.
    pub fn all() -> [Split; 4] {
        [Split::Train, Split::Test, Split::Val, Split::Calibration]
    }

    /// Return the nominal fraction (0.0–1.0) of the dataset assigned to this split.
    pub fn nominal_fraction(&self) -> f64 {
        match self {
            Split::Train => 0.70,
            Split::Test => 0.15,
            Split::Val => 0.10,
            Split::Calibration => 0.05,
        }
    }
}

/// A matched pair of original and VAE-decoded image file paths.
///
/// Used throughout the normalizer to track provenance: the `original` path
/// points to the source image and `decoded` points to the VAE reconstruction.
/// `original_size` holds the byte count of the source file for stratification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImagePair {
    /// Absolute or relative path to the original source image.
    pub original: String,
    /// Absolute or relative path to the VAE-decoded reconstruction.
    pub decoded: String,
    /// Byte size of the original file (used for size-stratified splitting).
    pub original_size: u64,
}

impl ImagePair {
    /// Construct a new `ImagePair` from the given paths and original file size.
    pub fn new(
        original: impl Into<String>,
        decoded: impl Into<String>,
        original_size: u64,
    ) -> Self {
        Self {
            original: original.into(),
            decoded: decoded.into(),
            original_size,
        }
    }
}

// ---------------------------------------------------------------------------
// Cryptographic kernel
// ---------------------------------------------------------------------------

/// CRYPTO KERNEL: Hash `data` with SHAKE256 (output length d=256 bits / 32 bytes).
///
/// Returns a 64-character lowercase hexadecimal string.  This function is
/// deterministic and pure — identical inputs always produce identical outputs.
pub fn shake256_d256(data: &[u8]) -> String {
    let mut hasher = Shake::v256();
    hasher.update(data);
    let mut output = [0u8; 32];
    hasher.finalize(&mut output);
    hex::encode(output)
}

// ---------------------------------------------------------------------------
// Splitting utilities
// ---------------------------------------------------------------------------

/// Assign a `Split` label to an item at the given `index` within a dataset of
/// `total` items, using the nominal proportions defined on `Split`.
///
/// This is a pure, deterministic function: no randomness is involved.  It is
/// suitable for pre-sorted or pre-shuffled sequences where the caller has
/// already applied any desired shuffling.
pub fn assign_split_by_index(index: usize, total: usize) -> Split {
    if total == 0 {
        return Split::Train;
    }
    // Compute cumulative thresholds using integer arithmetic to avoid
    // floating-point rounding surprises at boundary values.
    let train_end = (total * 70) / 100;
    let test_end = train_end + (total * 15) / 100;
    let val_end = test_end + (total * 10) / 100;

    if index < train_end {
        Split::Train
    } else if index < test_end {
        Split::Test
    } else if index < val_end {
        Split::Val
    } else {
        Split::Calibration
    }
}

/// Partition a slice of `ImagePair` references into four sub-vectors according
/// to the nominal split proportions.
///
/// The partition is deterministic given a pre-ordered (e.g. shuffled) slice.
/// Returns `(train, test, val, calibration)`.
pub fn partition_pairs(
    pairs: &[ImagePair],
) -> (
    Vec<&ImagePair>,
    Vec<&ImagePair>,
    Vec<&ImagePair>,
    Vec<&ImagePair>,
) {
    let total = pairs.len();
    let mut train = Vec::new();
    let mut test = Vec::new();
    let mut val = Vec::new();
    let mut calibration = Vec::new();

    for (i, pair) in pairs.iter().enumerate() {
        match assign_split_by_index(i, total) {
            Split::Train => train.push(pair),
            Split::Test => test.push(pair),
            Split::Val => val.push(pair),
            Split::Calibration => calibration.push(pair),
        }
    }

    (train, test, val, calibration)
}
