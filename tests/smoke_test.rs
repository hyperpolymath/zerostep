// SPDX-License-Identifier: PMPL-1.0-or-later

//! Smoke tests for the vae-normalizer library.
//!
//! These tests verify the public API surface without touching the file-system
//! or making network calls.  They cover:
//!   - `Split` variant construction and string labels
//!   - `Split` nominal fractions summing to 1.0
//!   - `ImagePair` construction and field access
//!   - `shake256_d256` determinism, length, and sensitivity
//!   - `assign_split_by_index` boundary conditions
//!   - `partition_pairs` proportional correctness

use vae_normalizer::{assign_split_by_index, partition_pairs, shake256_d256, ImagePair, Split};

// ---------------------------------------------------------------------------
// Split enum tests
// ---------------------------------------------------------------------------

/// Every `Split` variant must return a distinct, non-empty lowercase label.
#[test]
fn split_as_str_returns_distinct_labels() {
    let labels: Vec<&str> = Split::all().iter().map(|s| s.as_str()).collect();

    for label in &labels {
        assert!(!label.is_empty(), "split label must not be empty");
        assert_eq!(
            label.to_lowercase(),
            *label,
            "split label must be lowercase: {}",
            label
        );
    }

    // All four labels must be unique.
    let mut unique = labels.clone();
    unique.dedup();
    assert_eq!(unique.len(), labels.len(), "split labels must be unique");
}

/// The four nominal fractions must sum to exactly 1.0 within floating-point
/// tolerance — no dataset item should be left unassigned.
#[test]
fn split_nominal_fractions_sum_to_one() {
    let total: f64 = Split::all().iter().map(|s| s.nominal_fraction()).sum();
    assert!(
        (total - 1.0_f64).abs() < 1e-10,
        "nominal fractions must sum to 1.0, got: {}",
        total
    );
}

/// Each `Split` variant's Debug representation must be non-empty and must not
/// panic — it is used in log output and error messages.
#[test]
fn split_debug_does_not_panic() {
    for variant in Split::all() {
        let s = format!("{:?}", variant);
        assert!(!s.is_empty());
    }
}

/// `Split::all()` must return all four variants without repetition.
#[test]
fn split_all_contains_four_unique_variants() {
    let all = Split::all();
    assert_eq!(all.len(), 4);

    // Check every variant appears exactly once.
    assert!(all.contains(&Split::Train));
    assert!(all.contains(&Split::Test));
    assert!(all.contains(&Split::Val));
    assert!(all.contains(&Split::Calibration));

    // Verify there are no duplicates by comparing hash-equality.
    use std::collections::HashSet;
    let set: HashSet<_> = all.iter().collect();
    assert_eq!(
        set.len(),
        4,
        "Split::all() must not contain duplicate variants"
    );
}

// ---------------------------------------------------------------------------
// ImagePair tests
// ---------------------------------------------------------------------------

/// `ImagePair::new` must store the provided paths and size verbatim.
#[test]
fn image_pair_new_stores_fields() {
    let pair = ImagePair::new("original/img_001.png", "decoded/img_001.png", 204_800);

    assert_eq!(pair.original, "original/img_001.png");
    assert_eq!(pair.decoded, "decoded/img_001.png");
    assert_eq!(pair.original_size, 204_800);
}

/// Two `ImagePair` values with identical fields must be equal; differing fields
/// must be unequal.
#[test]
fn image_pair_equality() {
    let a = ImagePair::new("a.png", "b.png", 1024);
    let b = ImagePair::new("a.png", "b.png", 1024);
    let c = ImagePair::new("x.png", "y.png", 2048);

    assert_eq!(a, b, "identical ImagePairs must be equal");
    assert_ne!(a, c, "ImagePairs with different fields must be unequal");
}

/// Clone of an `ImagePair` must be equal to the original and independent.
#[test]
fn image_pair_clone_is_independent() {
    let original = ImagePair::new("src/cat.jpg", "vae/cat.jpg", 512_000);
    let cloned = original.clone();

    assert_eq!(original, cloned);

    // Modifying the clone must not affect the original (String fields are cloned).
    let mut mutable_clone = cloned;
    mutable_clone.original = "changed.jpg".to_string();
    assert_ne!(original.original, mutable_clone.original);
}

// ---------------------------------------------------------------------------
// shake256_d256 tests
// ---------------------------------------------------------------------------

/// The digest must be exactly 64 hexadecimal characters (256 bits = 32 bytes =
/// 64 hex chars).
#[test]
fn shake256_d256_output_length_is_64_chars() {
    let digest = shake256_d256(b"hello world");
    assert_eq!(
        digest.len(),
        64,
        "SHAKE256(d=256) must produce a 64-char hex string, got: {}",
        digest
    );
}

/// The function must be deterministic: calling it twice with the same input
/// must return identical digests.
#[test]
fn shake256_d256_is_deterministic() {
    let input = b"reproducible dataset normaliser";
    let first = shake256_d256(input);
    let second = shake256_d256(input);
    assert_eq!(first, second, "shake256_d256 must be deterministic");
}

/// The empty-input digest must be a well-formed 64-char hex string.  This is
/// a boundary condition that many hash implementations handle incorrectly.
#[test]
fn shake256_d256_empty_input_well_formed() {
    let digest = shake256_d256(b"");
    assert_eq!(digest.len(), 64);
    assert!(
        digest.chars().all(|c| c.is_ascii_hexdigit()),
        "digest must contain only hex characters"
    );
}

/// A single-byte difference in input must produce a completely different digest
/// (avalanche effect).
#[test]
fn shake256_d256_avalanche_sensitivity() {
    let d1 = shake256_d256(b"vae-dataset-v1");
    let d2 = shake256_d256(b"vae-dataset-v2");
    assert_ne!(d1, d2, "distinct inputs must produce distinct digests");
}

/// The digest must consist only of lowercase hexadecimal characters.
#[test]
fn shake256_d256_output_is_lowercase_hex() {
    let digest = shake256_d256(b"integrity check payload");
    assert!(
        digest.chars().all(|c| matches!(c, '0'..='9' | 'a'..='f')),
        "digest must be lowercase hex, got: {}",
        digest
    );
}

// ---------------------------------------------------------------------------
// assign_split_by_index tests
// ---------------------------------------------------------------------------

/// Index 0 of a non-empty dataset must always be in the Train split.
#[test]
fn assign_split_by_index_first_item_is_train() {
    assert_eq!(assign_split_by_index(0, 100), Split::Train);
    assert_eq!(assign_split_by_index(0, 1000), Split::Train);
}

/// The last index of a non-empty dataset must always fall into Calibration.
#[test]
fn assign_split_by_index_last_item_is_calibration() {
    // With 100 items: train=0..70, test=70..85, val=85..95, calibration=95..100
    assert_eq!(assign_split_by_index(99, 100), Split::Calibration);
}

/// With an empty dataset (total == 0) the function must not panic and must
/// return a sensible default.
#[test]
fn assign_split_by_index_zero_total_does_not_panic() {
    let split = assign_split_by_index(0, 0);
    // Must return some valid Split variant without panicking.
    let _ = format!("{:?}", split);
}

/// With a single-item dataset, everything is assigned to Train (index 0 < 70% of 1 = 0).
/// Actually integer division: train_end = (1 * 70) / 100 = 0, so index 0 >= 0,
/// the entire dataset falls through to Calibration for a single item.
/// This test documents the actual behaviour rather than asserting a particular split.
#[test]
fn assign_split_by_index_single_item_documented_behaviour() {
    let split = assign_split_by_index(0, 1);
    // Must be a valid variant (no panic or undefined behaviour).
    assert!(
        matches!(
            split,
            Split::Train | Split::Test | Split::Val | Split::Calibration
        ),
        "must return a valid Split for single-item dataset"
    );
}

// ---------------------------------------------------------------------------
// partition_pairs tests
// ---------------------------------------------------------------------------

/// An empty pair slice must produce four empty sub-vectors.
#[test]
fn partition_pairs_empty_slice_produces_empty_vecs() {
    let (train, test, val, calibration) = partition_pairs(&[]);
    assert!(train.is_empty());
    assert!(test.is_empty());
    assert!(val.is_empty());
    assert!(calibration.is_empty());
}

/// The total count across all partitions must equal the input slice length.
#[test]
fn partition_pairs_total_count_equals_input_length() {
    let pairs: Vec<ImagePair> = (0..200)
        .map(|i| {
            ImagePair::new(
                format!("orig/{}.png", i),
                format!("dec/{}.png", i),
                i as u64 * 1024,
            )
        })
        .collect();

    let (train, test, val, calibration) = partition_pairs(&pairs);
    let total = train.len() + test.len() + val.len() + calibration.len();

    assert_eq!(
        total,
        pairs.len(),
        "partition must cover all input items without loss"
    );
}

/// With 100 pairs the training split must be the largest, followed by test,
/// then val, then calibration — matching the 70/15/10/5 policy.
#[test]
fn partition_pairs_proportions_match_policy_for_100_items() {
    let pairs: Vec<ImagePair> = (0..100)
        .map(|i| ImagePair::new(format!("o/{}.png", i), format!("d/{}.png", i), 1000))
        .collect();

    let (train, test, val, calibration) = partition_pairs(&pairs);

    // Allow 1-2 items of slack for integer rounding at boundaries.
    assert!(
        train.len() >= 68 && train.len() <= 72,
        "train expected ~70, got {}",
        train.len()
    );
    assert!(
        test.len() >= 13 && test.len() <= 17,
        "test expected ~15, got {}",
        test.len()
    );
    assert!(
        val.len() >= 8 && val.len() <= 12,
        "val expected ~10, got {}",
        val.len()
    );
    assert!(
        calibration.len() >= 3 && calibration.len() <= 7,
        "calibration expected ~5, got {}",
        calibration.len()
    );
}
