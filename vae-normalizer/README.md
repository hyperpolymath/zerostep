<!-- SPDX-FileCopyrightText: 2024 Joshua Jewell -->
<!-- SPDX-License-Identifier: MIT -->

# VAE Dataset Normalizer

Normalize VAE-decoded image datasets for training AI artifact detection models with formal verification guarantees.

## Features

- **SHAKE256 (d=256)** cryptographic checksums for data integrity
- **Train/Test/Val/Calibration splits** (70/15/10/5) - both random and stratified
- **Dublin Core metadata** via CUE configuration language
- **Nickel schema** for flexible configuration
- **Isabelle/HOL proofs** for split property verification
- **Julia utilities** for Flux.jl training integration

## Installation

### Prerequisites

Required:
- Rust 1.70+ with Cargo

Optional:
- [CUE](https://cuelang.org) - metadata validation
- [Nickel](https://nickel-lang.org) - configuration
- [Julia](https://julialang.org) - training utilities
- [Isabelle](https://isabelle.in.tum.de) - formal proofs
- [just](https://github.com/casey/just) - modern task runner

### Build

```bash
# Using Make
make build

# Using Just
just build

# Direct Cargo
cargo build --release
```

## Usage

### Normalize a Dataset

```bash
# Full normalization with checksums
vae-normalizer normalize -d /path/to/dataset -o /path/to/output

# Fast mode (skip checksums)
vae-normalizer normalize -d /path/to/dataset -o /path/to/output --skip-checksums

# Custom seed and strata
vae-normalizer normalize -d /path/to/dataset -o /path/to/output --seed 12345 --strata 8
```

### Verify Output

```bash
# Basic verification
vae-normalizer verify -o /path/to/output

# With checksum verification
vae-normalizer verify -o /path/to/output --checksums -d /path/to/dataset
```

### Show Statistics

```bash
# Text format
vae-normalizer stats -o /path/to/output

# JSON format
vae-normalizer stats -o /path/to/output --format json
```

### Compute File Hash

```bash
vae-normalizer hash /path/to/image.png
```

### Compress Dataset (Diff Format)

Reduce storage by ~50% by storing VAE images as diffs from originals:

```bash
# Compress dataset (creates Original/ + Diff/ structure)
vae-normalizer compress -d /path/to/dataset -o /path/to/compressed

# Decompress (reconstruct full VAE/ directory)
vae-normalizer decompress -d /path/to/compressed -o /path/to/reconstructed-vae

# Reconstruct single image
vae-normalizer reconstruct -o /path/to/original.png -d /path/to/diff.png -o output.png
```

The diff encoding uses: `diff = VAE - Original + 128` (offset to handle signed values).
Reconstruction: `VAE = Original + Diff - 128`.

## Output Structure

```
output/
├── splits/
│   ├── random_train.txt
│   ├── random_test.txt
│   ├── random_val.txt
│   ├── random_calibration.txt
│   ├── stratified_train.txt
│   ├── stratified_test.txt
│   ├── stratified_val.txt
│   └── stratified_calibration.txt
├── manifest.csv
└── metadata.cue
```

## Dataset Structure

### Standard Format

Expected input:
```
dataset/
├── Original/
│   ├── image001.png
│   ├── image002.png
│   └── ...
└── VAE/
    ├── image001.png
    ├── image002.png
    └── ...
```

Image files are paired by filename stem (e.g., `Original/foo.png` pairs with `VAE/foo.png`).

### Compressed Diff Format

After running `compress`, the dataset uses diff encoding:
```
compressed/
├── Original/
│   ├── image001.png
│   ├── image002.png
│   └── ...
└── Diff/
    ├── image001.png  # Encodes (VAE - Original + 128)
    ├── image002.png
    └── ...
```

Diff images compress well (typically ~50% smaller than VAE originals) because most pixels are near 128 (gray) when VAE and original are similar.

## Formal Verification

The Isabelle/HOL theory `VAEDataset_Splits.thy` proves:

1. **Disjointness**: No overlap between train/test/val/calibration sets
2. **Exhaustiveness**: Every image appears in exactly one split
3. **Ratio correctness**: Split sizes within 1% of targets
4. **Bijection**: 1:1 correspondence between original and VAE images

To verify proofs:
```bash
isabelle build -d . -b VAEDataset_Splits
```

## Julia Integration

### Standard Dataset

```julia
include("julia_utils.jl")
using .VAEDatasetUtils

# Load training split
dataset = VAEDetectorDataset(
    "output/splits/random_train.txt",
    "output/manifest.csv",
    "/path/to/dataset"
)

# Create data loader
loader = DataLoader(dataset, batch_size=32, shuffle=true)

# Train with Flux.jl
for (x, y) in loader
    # x: batch of images
    # y: labels (0=original, 1=VAE)
end
```

### Compressed Dataset

```julia
include("julia_utils.jl")
using .VAEDatasetUtils

# Load compressed dataset (VAE images reconstructed on-the-fly)
dataset = CompressedVAEDataset(
    "output/splits/random_train.txt",
    "/path/to/compressed/Original",
    "/path/to/compressed/Diff"
)

# Same API as standard dataset
loader = DataLoader(dataset, batch_size=32, shuffle=true)

for (x, y) in loader
    # VAE images are reconstructed automatically from diffs
end
```

## Configuration

### CUE Schema

Metadata validated against Dublin Core via `metadata_schema.cue`:

```cue
dublin_core: {
    title:       "VAEDecodedImages-SDXL"
    creator:     "Joshua Jewell"
    type:        "Dataset"
    format:      "image/png"
    // ...
}
```

### Nickel Config

Alternative configuration via `config.ncl`:

```nickel
{
  splits.ratios = {
    train = 0.70,
    test = 0.15,
    validation = 0.10,
    calibration = 0.05,
  },
  checksums.algorithm = 'SHAKE256,
  pipeline.vae_model = "SDXL VAE",
}
```

## Build System

Both Make and Just are supported:

```bash
# Make
make all DATASET_PATH=/data/vae OUTPUT_PATH=/data/output
make verify
make clean

# Just
just all
just verify
just clean
```

## License

See source dataset for licensing terms.

## Related

- [VAEDecodedImages-SDXL](https://huggingface.co/datasets/joshuajewell/VAEDecodedImages-SDXL)
- [nordjylland-news-image-captioning](https://huggingface.co/datasets/alexandrainst/nordjylland-news-image-captioning)
