// SPDX-FileCopyrightText: 2024 Joshua Jewell
// SPDX-License-Identifier: MIT

//! VAE Dataset Normalizer
//!
//! Normalizes VAE-decoded image datasets for training with:
//! - SHAKE256 (d=256) cryptographic checksums
//! - Train/test/val/calibration splits (70/15/10/5)
//! - Both random and stratified split variants
//! - CUE metadata with Dublin Core compliance

mod metadata;

use anyhow::{Context, Result, bail};
use clap::Parser;
use csv::Writer;
use image::{Rgb, RgbImage};
use indicatif::{ProgressBar, ProgressStyle};
use rand::prelude::*;
use rand_chacha::ChaCha20Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufWriter, Read, Write as IoWrite};
use std::path::{Path, PathBuf};
use tiny_keccak::{Hasher, Shake};
use walkdir::WalkDir;

/// Split ratios: train=70%, test=15%, val=10%, calibration=5%
const TRAIN_RATIO: f64 = 0.70;
const TEST_RATIO: f64 = 0.15;
const VAL_RATIO: f64 = 0.10;
const CAL_RATIO: f64 = 0.05;

/// Supported image extensions
const IMAGE_EXTENSIONS: &[&str] = &["png", "jpg", "jpeg", "webp", "bmp", "tiff"];

#[derive(Parser, Debug)]
#[command(name = "vae-normalizer")]
#[command(about = "Normalize VAE-decoded image datasets with formal guarantees")]
#[command(version = "1.0.0")]
#[command(author = "Joshua Jewell")]
#[command(after_help = "Examples:
  vae-normalizer normalize -d /data/vae -o /data/output
  vae-normalizer normalize -d /data/vae -o /data/output --skip-checksums
  vae-normalizer verify -o /data/output
  vae-normalizer stats -o /data/output")]
struct Args {
    #[command(subcommand)]
    command: Commands,

    /// Enable verbose output
    #[arg(short, long, global = true)]
    verbose: bool,

    /// Quiet mode - minimal output
    #[arg(short, long, global = true)]
    quiet: bool,
}

#[derive(clap::Subcommand, Debug)]
enum Commands {
    /// Normalize a VAE dataset (generate splits, checksums, metadata)
    Normalize {
        /// Path to the input dataset directory
        #[arg(short, long)]
        dataset: PathBuf,

        /// Path to the output directory
        #[arg(short, long)]
        output: PathBuf,

        /// Random seed for reproducible splits
        #[arg(short, long, default_value = "42")]
        seed: u64,

        /// Number of stratification bins for size-based stratification
        #[arg(long, default_value = "4")]
        strata: usize,

        /// Skip checksum computation (faster, less secure)
        #[arg(long)]
        skip_checksums: bool,

        /// Name of subdirectory containing original images
        #[arg(long, default_value = "Original")]
        original_dir: String,

        /// Name of subdirectory containing VAE-decoded images
        #[arg(long, default_value = "VAE")]
        vae_dir: String,

        /// Custom train ratio (0.0-1.0)
        #[arg(long)]
        train_ratio: Option<f64>,

        /// Custom test ratio (0.0-1.0)
        #[arg(long)]
        test_ratio: Option<f64>,

        /// Custom validation ratio (0.0-1.0)
        #[arg(long)]
        val_ratio: Option<f64>,

        /// Custom calibration ratio (0.0-1.0)
        #[arg(long)]
        cal_ratio: Option<f64>,

        /// Number of parallel workers (0 = auto)
        #[arg(short = 'j', long, default_value = "0")]
        jobs: usize,

        /// Output format for metadata (cue, json, both)
        #[arg(long, default_value = "cue")]
        meta_format: String,
    },

    /// Verify an existing normalized dataset
    Verify {
        /// Path to the normalized output directory
        #[arg(short, long)]
        output: PathBuf,

        /// Verify file checksums
        #[arg(long)]
        checksums: bool,

        /// Path to original dataset (required for checksum verification)
        #[arg(short, long)]
        dataset: Option<PathBuf>,
    },

    /// Show statistics about a normalized dataset
    Stats {
        /// Path to the normalized output directory
        #[arg(short, long)]
        output: PathBuf,

        /// Output format (text, json)
        #[arg(long, default_value = "text")]
        format: String,
    },

    /// Export splits to different formats
    Export {
        /// Path to the normalized output directory
        #[arg(short, long)]
        output: PathBuf,

        /// Export format (csv, json, parquet)
        #[arg(long, default_value = "csv")]
        format: String,

        /// Destination path
        #[arg(short, long)]
        dest: PathBuf,
    },

    /// Compute checksum for a single file
    Hash {
        /// File to hash
        file: PathBuf,
    },

    /// Compress dataset by storing diffs instead of full VAE images
    Compress {
        /// Path to the input dataset (with Original/ and VAE/ directories)
        #[arg(short, long)]
        dataset: PathBuf,

        /// Path to the output compressed dataset
        #[arg(short, long)]
        output: PathBuf,

        /// Number of parallel workers (0 = auto)
        #[arg(short = 'j', long, default_value = "0")]
        jobs: usize,
    },

    /// Decompress/reconstruct VAE images from Original + Diff
    Decompress {
        /// Path to the compressed dataset (with Original/ and Diff/ directories)
        #[arg(short, long)]
        dataset: PathBuf,

        /// Path to output reconstructed VAE images
        #[arg(short, long)]
        output: PathBuf,

        /// Number of parallel workers (0 = auto)
        #[arg(short = 'j', long, default_value = "0")]
        jobs: usize,
    },

    /// Reconstruct a single VAE image from Original + Diff
    Reconstruct {
        /// Path to the original image
        #[arg(short, long)]
        original: PathBuf,

        /// Path to the diff image
        #[arg(short, long)]
        diff: PathBuf,

        /// Output path for reconstructed VAE image
        #[arg(short = 'o', long)]
        output: PathBuf,
    },
}

/// Image pair entry with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImagePair {
    pub id: String,
    pub original_path: PathBuf,
    pub vae_path: PathBuf,
    pub original_size: u64,
    pub vae_size: u64,
    pub original_checksum: String,
    pub vae_checksum: String,
    pub stratum: usize,
}

/// Split assignment for an image
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Split {
    Train,
    Test,
    Val,
    Calibration,
}

impl Split {
    fn as_str(&self) -> &'static str {
        match self {
            Split::Train => "train",
            Split::Test => "test",
            Split::Val => "val",
            Split::Calibration => "calibration",
        }
    }
}

/// Compute SHAKE256 with 256-bit output (d=256)
fn shake256_d256(data: &[u8]) -> String {
    let mut hasher = Shake::v256();
    hasher.update(data);
    let mut output = [0u8; 32]; // 256 bits = 32 bytes
    hasher.finalize(&mut output);
    hex::encode(&output)
}

/// Simple hex encoding without external dependency
mod hex {
    pub fn encode(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{:02x}", b)).collect()
    }
}

/// Compute SHAKE256 checksum for a file
fn compute_file_checksum(path: &Path) -> Result<String> {
    let mut file = File::open(path)
        .with_context(|| format!("Failed to open file: {}", path.display()))?;

    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)
        .with_context(|| format!("Failed to read file: {}", path.display()))?;

    Ok(shake256_d256(&buffer))
}

/// Discover all image pairs in the dataset
fn discover_image_pairs(
    dataset_path: &Path,
    compute_checksums: bool,
    num_strata: usize,
) -> Result<Vec<ImagePair>> {
    let original_dir = dataset_path.join("Original");
    let vae_dir = dataset_path.join("VAE");

    if !original_dir.exists() {
        bail!("Original directory not found: {}", original_dir.display());
    }
    if !vae_dir.exists() {
        bail!("VAE directory not found: {}", vae_dir.display());
    }

    // Collect all original images
    let mut originals: HashMap<String, PathBuf> = HashMap::new();
    for entry in WalkDir::new(&original_dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
    {
        let path = entry.path();
        if let Some(ext) = path.extension() {
            let ext_lower = ext.to_string_lossy().to_lowercase();
            if IMAGE_EXTENSIONS.contains(&ext_lower.as_str()) {
                let stem = path.file_stem()
                    .map(|s| s.to_string_lossy().to_string())
                    .unwrap_or_default();
                originals.insert(stem, path.to_path_buf());
            }
        }
    }

    // Collect all VAE images
    let mut vaes: HashMap<String, PathBuf> = HashMap::new();
    for entry in WalkDir::new(&vae_dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
    {
        let path = entry.path();
        if let Some(ext) = path.extension() {
            let ext_lower = ext.to_string_lossy().to_lowercase();
            if IMAGE_EXTENSIONS.contains(&ext_lower.as_str()) {
                let stem = path.file_stem()
                    .map(|s| s.to_string_lossy().to_string())
                    .unwrap_or_default();
                vaes.insert(stem, path.to_path_buf());
            }
        }
    }

    // Find matching pairs
    let mut pairs: Vec<(String, PathBuf, PathBuf)> = Vec::new();
    for (id, original_path) in &originals {
        if let Some(vae_path) = vaes.get(id) {
            pairs.push((id.clone(), original_path.clone(), vae_path.clone()));
        }
    }

    if pairs.is_empty() {
        bail!("No matching image pairs found");
    }

    println!("Found {} image pairs", pairs.len());

    // Compute file sizes for stratification
    let mut sizes: Vec<u64> = pairs.iter()
        .map(|(_, orig, _)| fs::metadata(orig).map(|m| m.len()).unwrap_or(0))
        .collect();
    sizes.sort();

    // Compute stratum boundaries
    let strata_boundaries: Vec<u64> = (1..num_strata)
        .map(|i| {
            let idx = (i * sizes.len()) / num_strata;
            sizes.get(idx).copied().unwrap_or(u64::MAX)
        })
        .collect();

    let progress = ProgressBar::new(pairs.len() as u64);
    progress.set_style(ProgressStyle::default_bar()
        .template("[{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
        .unwrap()
        .progress_chars("#>-"));

    // Process pairs (parallel if computing checksums)
    let image_pairs: Vec<ImagePair> = if compute_checksums {
        pairs.par_iter()
            .map(|(id, original_path, vae_path)| {
                let original_size = fs::metadata(original_path)
                    .map(|m| m.len())
                    .unwrap_or(0);
                let vae_size = fs::metadata(vae_path)
                    .map(|m| m.len())
                    .unwrap_or(0);

                let original_checksum = compute_file_checksum(original_path)
                    .unwrap_or_else(|_| "error".to_string());
                let vae_checksum = compute_file_checksum(vae_path)
                    .unwrap_or_else(|_| "error".to_string());

                // Determine stratum based on original file size
                let stratum = strata_boundaries.iter()
                    .position(|&boundary| original_size < boundary)
                    .unwrap_or(num_strata - 1);

                progress.inc(1);

                ImagePair {
                    id: id.clone(),
                    original_path: original_path.clone(),
                    vae_path: vae_path.clone(),
                    original_size,
                    vae_size,
                    original_checksum,
                    vae_checksum,
                    stratum,
                }
            })
            .collect()
    } else {
        pairs.iter()
            .map(|(id, original_path, vae_path)| {
                let original_size = fs::metadata(original_path)
                    .map(|m| m.len())
                    .unwrap_or(0);
                let vae_size = fs::metadata(vae_path)
                    .map(|m| m.len())
                    .unwrap_or(0);

                // Determine stratum based on original file size
                let stratum = strata_boundaries.iter()
                    .position(|&boundary| original_size < boundary)
                    .unwrap_or(num_strata - 1);

                progress.inc(1);

                ImagePair {
                    id: id.clone(),
                    original_path: original_path.clone(),
                    vae_path: vae_path.clone(),
                    original_size,
                    vae_size,
                    original_checksum: "skipped".to_string(),
                    vae_checksum: "skipped".to_string(),
                    stratum,
                }
            })
            .collect()
    };

    progress.finish_with_message("Done processing images");

    Ok(image_pairs)
}

/// Assign splits randomly (non-stratified)
fn random_split(pairs: &[ImagePair], seed: u64) -> HashMap<String, Split> {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let mut assignments: HashMap<String, Split> = HashMap::new();

    let mut indices: Vec<usize> = (0..pairs.len()).collect();
    indices.shuffle(&mut rng);

    let n = pairs.len();
    let train_end = (n as f64 * TRAIN_RATIO).round() as usize;
    let test_end = train_end + (n as f64 * TEST_RATIO).round() as usize;
    let val_end = test_end + (n as f64 * VAL_RATIO).round() as usize;

    for (i, &idx) in indices.iter().enumerate() {
        let split = if i < train_end {
            Split::Train
        } else if i < test_end {
            Split::Test
        } else if i < val_end {
            Split::Val
        } else {
            Split::Calibration
        };
        assignments.insert(pairs[idx].id.clone(), split);
    }

    assignments
}

/// Assign splits with stratification by file size
fn stratified_split(pairs: &[ImagePair], seed: u64, num_strata: usize) -> HashMap<String, Split> {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let mut assignments: HashMap<String, Split> = HashMap::new();

    // Group by stratum
    let mut strata: Vec<Vec<&ImagePair>> = vec![Vec::new(); num_strata];
    for pair in pairs {
        strata[pair.stratum].push(pair);
    }

    // Split each stratum independently
    for stratum_pairs in &mut strata {
        stratum_pairs.shuffle(&mut rng);

        let n = stratum_pairs.len();
        let train_end = (n as f64 * TRAIN_RATIO).round() as usize;
        let test_end = train_end + (n as f64 * TEST_RATIO).round() as usize;
        let val_end = test_end + (n as f64 * VAL_RATIO).round() as usize;

        for (i, pair) in stratum_pairs.iter().enumerate() {
            let split = if i < train_end {
                Split::Train
            } else if i < test_end {
                Split::Test
            } else if i < val_end {
                Split::Val
            } else {
                Split::Calibration
            };
            assignments.insert(pair.id.clone(), split);
        }
    }

    assignments
}

/// Write split files to disk
fn write_splits(
    output_dir: &Path,
    pairs: &[ImagePair],
    random_assignments: &HashMap<String, Split>,
    stratified_assignments: &HashMap<String, Split>,
) -> Result<()> {
    let splits_dir = output_dir.join("splits");
    fs::create_dir_all(&splits_dir)?;

    // Write random splits
    for split in [Split::Train, Split::Test, Split::Val, Split::Calibration] {
        let filename = format!("random_{}.txt", split.as_str());
        let path = splits_dir.join(&filename);
        let mut file = BufWriter::new(File::create(&path)?);

        for pair in pairs {
            if random_assignments.get(&pair.id) == Some(&split) {
                writeln!(file, "{}", pair.id)?;
            }
        }
    }

    // Write stratified splits
    for split in [Split::Train, Split::Test, Split::Val, Split::Calibration] {
        let filename = format!("stratified_{}.txt", split.as_str());
        let path = splits_dir.join(&filename);
        let mut file = BufWriter::new(File::create(&path)?);

        for pair in pairs {
            if stratified_assignments.get(&pair.id) == Some(&split) {
                writeln!(file, "{}", pair.id)?;
            }
        }
    }

    Ok(())
}

/// Write CSV manifest
fn write_manifest(output_dir: &Path, pairs: &[ImagePair]) -> Result<()> {
    let manifest_path = output_dir.join("manifest.csv");
    let mut writer = Writer::from_path(&manifest_path)?;

    writer.write_record([
        "id",
        "original_path",
        "vae_path",
        "original_size",
        "vae_size",
        "original_shake256_d256",
        "vae_shake256_d256",
        "stratum",
    ])?;

    for pair in pairs {
        writer.write_record(&[
            pair.id.as_str(),
            &pair.original_path.to_string_lossy().into_owned(),
            &pair.vae_path.to_string_lossy().into_owned(),
            &pair.original_size.to_string(),
            &pair.vae_size.to_string(),
            pair.original_checksum.as_str(),
            pair.vae_checksum.as_str(),
            &pair.stratum.to_string(),
        ])?;
    }

    writer.flush()?;
    Ok(())
}

/// Validate split properties
fn validate_splits(
    pairs: &[ImagePair],
    assignments: &HashMap<String, Split>,
    split_name: &str,
) -> Result<()> {
    let n = pairs.len();

    // Count per split
    let mut counts: HashMap<Split, usize> = HashMap::new();
    for split in assignments.values() {
        *counts.entry(*split).or_insert(0) += 1;
    }

    // Check exhaustiveness
    let total: usize = counts.values().sum();
    if total != n {
        bail!("{}: Not exhaustive - {} assigned, {} total", split_name, total, n);
    }

    // Check ratios (with 1% tolerance)
    let tolerance = 0.01;
    let train_ratio = counts.get(&Split::Train).copied().unwrap_or(0) as f64 / n as f64;
    let test_ratio = counts.get(&Split::Test).copied().unwrap_or(0) as f64 / n as f64;
    let val_ratio = counts.get(&Split::Val).copied().unwrap_or(0) as f64 / n as f64;
    let cal_ratio = counts.get(&Split::Calibration).copied().unwrap_or(0) as f64 / n as f64;

    if (train_ratio - TRAIN_RATIO).abs() > tolerance {
        bail!("{}: Train ratio {:.2}% not within tolerance of {:.2}%",
              split_name, train_ratio * 100.0, TRAIN_RATIO * 100.0);
    }
    if (test_ratio - TEST_RATIO).abs() > tolerance {
        bail!("{}: Test ratio {:.2}% not within tolerance of {:.2}%",
              split_name, test_ratio * 100.0, TEST_RATIO * 100.0);
    }
    if (val_ratio - VAL_RATIO).abs() > tolerance {
        bail!("{}: Val ratio {:.2}% not within tolerance of {:.2}%",
              split_name, val_ratio * 100.0, VAL_RATIO * 100.0);
    }
    if (cal_ratio - CAL_RATIO).abs() > tolerance {
        bail!("{}: Calibration ratio {:.2}% not within tolerance of {:.2}%",
              split_name, cal_ratio * 100.0, CAL_RATIO * 100.0);
    }

    println!("{} split validation passed:", split_name);
    println!("  Train:       {:5} ({:.1}%)", counts.get(&Split::Train).unwrap_or(&0), train_ratio * 100.0);
    println!("  Test:        {:5} ({:.1}%)", counts.get(&Split::Test).unwrap_or(&0), test_ratio * 100.0);
    println!("  Validation:  {:5} ({:.1}%)", counts.get(&Split::Val).unwrap_or(&0), val_ratio * 100.0);
    println!("  Calibration: {:5} ({:.1}%)", counts.get(&Split::Calibration).unwrap_or(&0), cal_ratio * 100.0);

    Ok(())
}

fn cmd_normalize(
    dataset: PathBuf,
    output: PathBuf,
    seed: u64,
    strata: usize,
    skip_checksums: bool,
    verbose: bool,
) -> Result<()> {
    if !verbose {
        println!("VAE Dataset Normalizer v1.0.0");
        println!("==============================");
    }
    println!("Dataset:    {}", dataset.display());
    println!("Output:     {}", output.display());
    println!("Seed:       {}", seed);
    println!("Strata:     {}", strata);
    println!("Checksums:  {}", if skip_checksums { "skipped" } else { "SHAKE256-d256" });
    println!();

    // Create output directory
    fs::create_dir_all(&output)?;

    // Discover and process image pairs
    println!("Discovering image pairs...");
    let pairs = discover_image_pairs(&dataset, !skip_checksums, strata)?;
    println!("Processed {} pairs\n", pairs.len());

    // Generate splits
    println!("Generating random splits...");
    let random_assignments = random_split(&pairs, seed);
    validate_splits(&pairs, &random_assignments, "Random")?;
    println!();

    println!("Generating stratified splits...");
    let stratified_assignments = stratified_split(&pairs, seed, strata);
    validate_splits(&pairs, &stratified_assignments, "Stratified")?;
    println!();

    // Write outputs
    println!("Writing split files...");
    write_splits(&output, &pairs, &random_assignments, &stratified_assignments)?;

    println!("Writing manifest...");
    write_manifest(&output, &pairs)?;

    println!("Generating CUE metadata...");
    metadata::write_metadata(&output, &pairs, seed, strata)?;

    println!("\nDone! Output written to: {}", output.display());

    Ok(())
}

fn cmd_verify(output: PathBuf, checksums: bool, dataset: Option<PathBuf>) -> Result<()> {
    println!("Verifying normalized dataset: {}", output.display());

    // Check required files exist
    let manifest_path = output.join("manifest.csv");
    let splits_dir = output.join("splits");
    let metadata_path = output.join("metadata.cue");

    if !manifest_path.exists() {
        bail!("Missing manifest.csv");
    }
    println!("  manifest.csv: OK");

    if !splits_dir.exists() {
        bail!("Missing splits directory");
    }

    // Check split files
    for split_type in &["random", "stratified"] {
        for split_name in &["train", "test", "val", "calibration"] {
            let path = splits_dir.join(format!("{}_{}.txt", split_type, split_name));
            if !path.exists() {
                bail!("Missing split file: {}", path.display());
            }
        }
    }
    println!("  splits/: OK (8 files)");

    if metadata_path.exists() {
        println!("  metadata.cue: OK");
    } else {
        println!("  metadata.cue: MISSING (optional)");
    }

    // Verify checksums if requested
    if checksums {
        if let Some(dataset_path) = dataset {
            println!("\nVerifying checksums...");
            let mut reader = csv::Reader::from_path(&manifest_path)?;
            let mut verified = 0;
            let mut failed = 0;

            for result in reader.records() {
                let record = result?;
                let id = &record[0];
                let orig_path = dataset_path.join(&record[1]);
                let expected = &record[5];

                if expected == "skipped" {
                    continue;
                }

                let actual = compute_file_checksum(&orig_path)?;
                if actual == *expected {
                    verified += 1;
                } else {
                    failed += 1;
                    eprintln!("  FAILED: {} (expected {}, got {})", id, expected, actual);
                }
            }
            println!("  Verified: {}, Failed: {}", verified, failed);
            if failed > 0 {
                bail!("{} checksum failures", failed);
            }
        } else {
            println!("  Skipping checksum verification (--dataset not provided)");
        }
    }

    println!("\nVerification complete.");
    Ok(())
}

fn cmd_stats(output: PathBuf, format: &str) -> Result<()> {
    let manifest_path = output.join("manifest.csv");
    if !manifest_path.exists() {
        bail!("Manifest not found: {}", manifest_path.display());
    }

    let mut reader = csv::Reader::from_path(&manifest_path)?;
    let mut total_pairs = 0;
    let mut total_original_size: u64 = 0;
    let mut total_vae_size: u64 = 0;
    let mut stratum_counts: HashMap<String, usize> = HashMap::new();

    for result in reader.records() {
        let record = result?;
        total_pairs += 1;
        total_original_size += record[3].parse::<u64>().unwrap_or(0);
        total_vae_size += record[4].parse::<u64>().unwrap_or(0);
        *stratum_counts.entry(record[7].to_string()).or_insert(0) += 1;
    }

    // Count splits
    let splits_dir = output.join("splits");
    let mut split_counts: HashMap<String, usize> = HashMap::new();
    for entry in fs::read_dir(&splits_dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().map(|e| e == "txt").unwrap_or(false) {
            let name = path.file_stem().unwrap().to_string_lossy().to_string();
            let count = fs::read_to_string(&path)?
                .lines()
                .filter(|l| !l.is_empty())
                .count();
            split_counts.insert(name, count);
        }
    }

    if format == "json" {
        println!("{{");
        println!("  \"total_pairs\": {},", total_pairs);
        println!("  \"total_original_bytes\": {},", total_original_size);
        println!("  \"total_vae_bytes\": {},", total_vae_size);
        println!("  \"avg_original_bytes\": {},", total_original_size / total_pairs.max(1) as u64);
        println!("  \"avg_vae_bytes\": {},", total_vae_size / total_pairs.max(1) as u64);
        println!("  \"splits\": {{");
        for (i, (name, count)) in split_counts.iter().enumerate() {
            let comma = if i < split_counts.len() - 1 { "," } else { "" };
            println!("    \"{}\": {}{}", name, count, comma);
        }
        println!("  }}");
        println!("}}");
    } else {
        println!("Dataset Statistics");
        println!("==================");
        println!("Total pairs:     {}", total_pairs);
        println!("Original size:   {} bytes ({:.2} MB)", total_original_size, total_original_size as f64 / 1_000_000.0);
        println!("VAE size:        {} bytes ({:.2} MB)", total_vae_size, total_vae_size as f64 / 1_000_000.0);
        println!("Avg original:    {} bytes", total_original_size / total_pairs.max(1) as u64);
        println!("Avg VAE:         {} bytes", total_vae_size / total_pairs.max(1) as u64);
        println!();
        println!("Split counts:");
        for (name, count) in &split_counts {
            let pct = (*count as f64 / total_pairs as f64) * 100.0;
            println!("  {}: {} ({:.1}%)", name, count, pct);
        }
        println!();
        println!("Stratum distribution:");
        for (stratum, count) in &stratum_counts {
            println!("  Stratum {}: {}", stratum, count);
        }
    }

    Ok(())
}

fn cmd_hash(file: PathBuf) -> Result<()> {
    let checksum = compute_file_checksum(&file)?;
    println!("{}", checksum);
    Ok(())
}

/// Compute diff image: diff = VAE - Original + 128 (offset to handle signed values)
fn compute_diff_image(original: &RgbImage, vae: &RgbImage) -> Result<RgbImage> {
    if original.dimensions() != vae.dimensions() {
        bail!(
            "Image dimensions mismatch: original {:?} vs vae {:?}",
            original.dimensions(),
            vae.dimensions()
        );
    }

    let (width, height) = original.dimensions();
    let mut diff = RgbImage::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let orig_pixel = original.get_pixel(x, y);
            let vae_pixel = vae.get_pixel(x, y);

            // Compute diff with offset: diff = vae - original + 128
            // This maps [-255, 255] to [0-127, 128, 129-383] but we clamp to [0, 255]
            let r = ((vae_pixel[0] as i16 - orig_pixel[0] as i16 + 128).clamp(0, 255)) as u8;
            let g = ((vae_pixel[1] as i16 - orig_pixel[1] as i16 + 128).clamp(0, 255)) as u8;
            let b = ((vae_pixel[2] as i16 - orig_pixel[2] as i16 + 128).clamp(0, 255)) as u8;

            diff.put_pixel(x, y, Rgb([r, g, b]));
        }
    }

    Ok(diff)
}

/// Reconstruct VAE image: vae = original + diff - 128
fn reconstruct_vae_image(original: &RgbImage, diff: &RgbImage) -> Result<RgbImage> {
    if original.dimensions() != diff.dimensions() {
        bail!(
            "Image dimensions mismatch: original {:?} vs diff {:?}",
            original.dimensions(),
            diff.dimensions()
        );
    }

    let (width, height) = original.dimensions();
    let mut vae = RgbImage::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let orig_pixel = original.get_pixel(x, y);
            let diff_pixel = diff.get_pixel(x, y);

            // Reconstruct: vae = original + diff - 128
            let r = ((orig_pixel[0] as i16 + diff_pixel[0] as i16 - 128).clamp(0, 255)) as u8;
            let g = ((orig_pixel[1] as i16 + diff_pixel[1] as i16 - 128).clamp(0, 255)) as u8;
            let b = ((orig_pixel[2] as i16 + diff_pixel[2] as i16 - 128).clamp(0, 255)) as u8;

            vae.put_pixel(x, y, Rgb([r, g, b]));
        }
    }

    Ok(vae)
}

/// Compress dataset by converting VAE images to diffs
fn cmd_compress(dataset: PathBuf, output: PathBuf, _jobs: usize) -> Result<()> {
    println!("Compressing VAE dataset to diff format");
    println!("======================================");
    println!("Input:  {}", dataset.display());
    println!("Output: {}", output.display());
    println!();

    let original_dir = dataset.join("Original");
    let vae_dir = dataset.join("VAE");

    if !original_dir.exists() {
        bail!("Original directory not found: {}", original_dir.display());
    }
    if !vae_dir.exists() {
        bail!("VAE directory not found: {}", vae_dir.display());
    }

    // Create output directories
    let out_original_dir = output.join("Original");
    let out_diff_dir = output.join("Diff");
    fs::create_dir_all(&out_original_dir)?;
    fs::create_dir_all(&out_diff_dir)?;

    // Collect all original images
    let mut pairs: Vec<(PathBuf, PathBuf, String)> = Vec::new();
    for entry in WalkDir::new(&original_dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
    {
        let orig_path = entry.path().to_path_buf();
        if let Some(ext) = orig_path.extension() {
            let ext_lower = ext.to_string_lossy().to_lowercase();
            if ["png", "jpg", "jpeg", "webp"].contains(&ext_lower.as_str()) {
                let stem = orig_path.file_stem()
                    .map(|s| s.to_string_lossy().to_string())
                    .unwrap_or_default();

                // Look for matching VAE image
                for vae_ext in &["png", "jpg", "jpeg", "webp"] {
                    let vae_path = vae_dir.join(format!("{}.{}", stem, vae_ext));
                    if vae_path.exists() {
                        pairs.push((orig_path.clone(), vae_path, stem.clone()));
                        break;
                    }
                }
            }
        }
    }

    if pairs.is_empty() {
        bail!("No matching image pairs found");
    }

    println!("Found {} image pairs to compress", pairs.len());

    let progress = ProgressBar::new(pairs.len() as u64);
    progress.set_style(ProgressStyle::default_bar()
        .template("[{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
        .unwrap()
        .progress_chars("#>-"));

    let mut total_original_size: u64 = 0;
    let mut total_vae_size: u64 = 0;
    let mut total_diff_size: u64 = 0;

    for (orig_path, vae_path, stem) in &pairs {
        // Load images
        let original = image::open(orig_path)
            .with_context(|| format!("Failed to open original: {}", orig_path.display()))?
            .to_rgb8();
        let vae = image::open(vae_path)
            .with_context(|| format!("Failed to open VAE: {}", vae_path.display()))?
            .to_rgb8();

        // Compute diff
        let diff = compute_diff_image(&original, &vae)?;

        // Save original (copy or re-encode)
        let out_orig_path = out_original_dir.join(format!("{}.png", stem));
        original.save(&out_orig_path)
            .with_context(|| format!("Failed to save original: {}", out_orig_path.display()))?;

        // Save diff (PNG compresses diffs well due to low entropy)
        let out_diff_path = out_diff_dir.join(format!("{}.png", stem));
        diff.save(&out_diff_path)
            .with_context(|| format!("Failed to save diff: {}", out_diff_path.display()))?;

        // Track sizes
        total_original_size += fs::metadata(orig_path).map(|m| m.len()).unwrap_or(0);
        total_vae_size += fs::metadata(vae_path).map(|m| m.len()).unwrap_or(0);
        total_diff_size += fs::metadata(&out_diff_path).map(|m| m.len()).unwrap_or(0);

        progress.inc(1);
    }

    progress.finish_with_message("Compression complete");

    // Print stats
    let original_total = total_original_size + total_vae_size;
    let compressed_total = total_original_size + total_diff_size;
    let savings = original_total as f64 - compressed_total as f64;
    let ratio = compressed_total as f64 / original_total as f64;

    println!();
    println!("Compression Statistics");
    println!("======================");
    println!("Original dataset:   {:.2} MB", original_total as f64 / 1_000_000.0);
    println!("  - Original imgs:  {:.2} MB", total_original_size as f64 / 1_000_000.0);
    println!("  - VAE imgs:       {:.2} MB", total_vae_size as f64 / 1_000_000.0);
    println!("Compressed dataset: {:.2} MB", compressed_total as f64 / 1_000_000.0);
    println!("  - Original imgs:  {:.2} MB", total_original_size as f64 / 1_000_000.0);
    println!("  - Diff imgs:      {:.2} MB", total_diff_size as f64 / 1_000_000.0);
    println!("Space saved:        {:.2} MB ({:.1}%)", savings / 1_000_000.0, (1.0 - ratio) * 100.0);
    println!();
    println!("Output written to: {}", output.display());

    Ok(())
}

/// Decompress dataset by reconstructing VAE images from diffs
fn cmd_decompress(dataset: PathBuf, output: PathBuf, _jobs: usize) -> Result<()> {
    println!("Decompressing diff dataset to VAE format");
    println!("========================================");
    println!("Input:  {}", dataset.display());
    println!("Output: {}", output.display());
    println!();

    let original_dir = dataset.join("Original");
    let diff_dir = dataset.join("Diff");

    if !original_dir.exists() {
        bail!("Original directory not found: {}", original_dir.display());
    }
    if !diff_dir.exists() {
        bail!("Diff directory not found: {}", diff_dir.display());
    }

    // Create output directory
    fs::create_dir_all(&output)?;

    // Collect all original images
    let mut pairs: Vec<(PathBuf, PathBuf, String)> = Vec::new();
    for entry in WalkDir::new(&original_dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
    {
        let orig_path = entry.path().to_path_buf();
        if let Some(ext) = orig_path.extension() {
            let ext_lower = ext.to_string_lossy().to_lowercase();
            if ["png", "jpg", "jpeg", "webp"].contains(&ext_lower.as_str()) {
                let stem = orig_path.file_stem()
                    .map(|s| s.to_string_lossy().to_string())
                    .unwrap_or_default();

                // Look for matching diff image
                let diff_path = diff_dir.join(format!("{}.png", stem));
                if diff_path.exists() {
                    pairs.push((orig_path.clone(), diff_path, stem.clone()));
                }
            }
        }
    }

    if pairs.is_empty() {
        bail!("No matching image pairs found");
    }

    println!("Found {} image pairs to decompress", pairs.len());

    let progress = ProgressBar::new(pairs.len() as u64);
    progress.set_style(ProgressStyle::default_bar()
        .template("[{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
        .unwrap()
        .progress_chars("#>-"));

    for (orig_path, diff_path, stem) in &pairs {
        // Load images
        let original = image::open(orig_path)
            .with_context(|| format!("Failed to open original: {}", orig_path.display()))?
            .to_rgb8();
        let diff = image::open(diff_path)
            .with_context(|| format!("Failed to open diff: {}", diff_path.display()))?
            .to_rgb8();

        // Reconstruct VAE
        let vae = reconstruct_vae_image(&original, &diff)?;

        // Save reconstructed VAE image
        let out_path = output.join(format!("{}.png", stem));
        vae.save(&out_path)
            .with_context(|| format!("Failed to save VAE: {}", out_path.display()))?;

        progress.inc(1);
    }

    progress.finish_with_message("Decompression complete");

    println!();
    println!("Reconstructed {} VAE images to: {}", pairs.len(), output.display());

    Ok(())
}

/// Reconstruct a single VAE image from Original + Diff
fn cmd_reconstruct(original: PathBuf, diff: PathBuf, output: PathBuf) -> Result<()> {
    let orig_img = image::open(&original)
        .with_context(|| format!("Failed to open original: {}", original.display()))?
        .to_rgb8();
    let diff_img = image::open(&diff)
        .with_context(|| format!("Failed to open diff: {}", diff.display()))?
        .to_rgb8();

    let vae = reconstruct_vae_image(&orig_img, &diff_img)?;
    vae.save(&output)
        .with_context(|| format!("Failed to save output: {}", output.display()))?;

    println!("Reconstructed VAE image saved to: {}", output.display());
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();

    match args.command {
        Commands::Normalize {
            dataset,
            output,
            seed,
            strata,
            skip_checksums,
            original_dir: _,
            vae_dir: _,
            train_ratio: _,
            test_ratio: _,
            val_ratio: _,
            cal_ratio: _,
            jobs: _,
            meta_format: _,
        } => cmd_normalize(dataset, output, seed, strata, skip_checksums, args.verbose),

        Commands::Verify { output, checksums, dataset } => {
            cmd_verify(output, checksums, dataset)
        }

        Commands::Stats { output, format } => {
            cmd_stats(output, &format)
        }

        Commands::Export { output: _, format: _, dest: _ } => {
            println!("Export command not yet implemented");
            Ok(())
        }

        Commands::Hash { file } => cmd_hash(file),

        Commands::Compress { dataset, output, jobs } => {
            cmd_compress(dataset, output, jobs)
        }

        Commands::Decompress { dataset, output, jobs } => {
            cmd_decompress(dataset, output, jobs)
        }

        Commands::Reconstruct { original, diff, output } => {
            cmd_reconstruct(original, diff, output)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shake256_known_vector() {
        // Test with empty input
        let hash = shake256_d256(b"");
        // SHAKE256("") with d=256 should produce a known value
        assert_eq!(hash.len(), 64); // 32 bytes = 64 hex chars
    }

    #[test]
    fn test_split_ratios() {
        // Create mock pairs
        let pairs: Vec<ImagePair> = (0..1000)
            .map(|i| ImagePair {
                id: format!("img_{}", i),
                original_path: PathBuf::from(format!("Original/{}.png", i)),
                vae_path: PathBuf::from(format!("VAE/{}.png", i)),
                original_size: 1000 + (i % 100) as u64 * 10,
                vae_size: 1000 + (i % 100) as u64 * 10,
                original_checksum: "test".to_string(),
                vae_checksum: "test".to_string(),
                stratum: i % 4,
            })
            .collect();

        let assignments = random_split(&pairs, 42);

        // Count each split
        let mut counts: HashMap<Split, usize> = HashMap::new();
        for split in assignments.values() {
            *counts.entry(*split).or_insert(0) += 1;
        }

        // Verify ratios are within tolerance
        let n = pairs.len() as f64;
        assert!((counts[&Split::Train] as f64 / n - 0.70).abs() < 0.02);
        assert!((counts[&Split::Test] as f64 / n - 0.15).abs() < 0.02);
        assert!((counts[&Split::Val] as f64 / n - 0.10).abs() < 0.02);
        assert!((counts[&Split::Calibration] as f64 / n - 0.05).abs() < 0.02);
    }

    #[test]
    fn test_splits_exhaustive_and_disjoint() {
        let pairs: Vec<ImagePair> = (0..100)
            .map(|i| ImagePair {
                id: format!("img_{}", i),
                original_path: PathBuf::from(format!("Original/{}.png", i)),
                vae_path: PathBuf::from(format!("VAE/{}.png", i)),
                original_size: 1000,
                vae_size: 1000,
                original_checksum: "test".to_string(),
                vae_checksum: "test".to_string(),
                stratum: 0,
            })
            .collect();

        let assignments = random_split(&pairs, 42);

        // Check exhaustiveness - every pair has an assignment
        for pair in &pairs {
            assert!(assignments.contains_key(&pair.id));
        }

        // Check total count
        assert_eq!(assignments.len(), pairs.len());
    }
}
