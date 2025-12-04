<!-- SPDX-FileCopyrightText: 2024 Joshua Jewell -->
<!-- SPDX-License-Identifier: MIT -->

# Quick Start Guide

Get up and running with vae-normalizer in 5 minutes.

## 1. Check Dependencies

```bash
# Required
rustc --version  # Need 1.70+
cargo --version

# Optional
cue version
nickel --version
julia --version
```

## 2. Build

```bash
cd vae-normalizer
cargo build --release
```

Or with just:
```bash
just build
```

## 3. Download Dataset

```bash
# Option A: Hugging Face CLI
huggingface-cli download joshuajewell/VAEDecodedImages-SDXL \
    --local-dir ~/vae-dataset \
    --repo-type dataset

# Option B: Git
git clone https://huggingface.co/datasets/joshuajewell/VAEDecodedImages-SDXL ~/vae-dataset
```

## 4. Normalize

Fast mode (no checksums):
```bash
./target/release/vae-normalizer normalize \
    -d ~/vae-dataset \
    -o ~/vae-normalized \
    --skip-checksums
```

Full mode with SHAKE256:
```bash
./target/release/vae-normalizer normalize \
    -d ~/vae-dataset \
    -o ~/vae-normalized
```

## 5. Verify

```bash
./target/release/vae-normalizer verify -o ~/vae-normalized
```

## 6. View Statistics

```bash
./target/release/vae-normalizer stats -o ~/vae-normalized
```

## 7. Train a Model (Julia)

```julia
# Install dependencies
using Pkg
Pkg.add(["Flux", "CSV", "DataFrames", "Images", "FileIO"])

# Load utilities
include("julia_utils.jl")
using .VAEDatasetUtils
using Flux

# Load data
train_data = VAEDetectorDataset(
    "splits/random_train.txt",
    "manifest.csv",
    expanduser("~/vae-dataset")
)

test_data = VAEDetectorDataset(
    "splits/random_test.txt",
    "manifest.csv",
    expanduser("~/vae-dataset")
)

# Create loaders
train_loader = DataLoader(train_data, batch_size=16)
test_loader = DataLoader(test_data, batch_size=16)

# Simple CNN
model = Chain(
    Conv((3, 3), 3 => 32, relu, pad=1),
    MaxPool((2, 2)),
    Conv((3, 3), 32 => 64, relu, pad=1),
    MaxPool((2, 2)),
    Flux.flatten,
    Dense(64 * 192 * 192, 256, relu),
    Dense(256, 1, sigmoid)
)

# Training
opt = Flux.setup(Adam(1e-4), model)
loss(m, x, y) = Flux.binarycrossentropy(m(x), y)

for epoch in 1:10
    for (x, y) in train_loader
        grads = Flux.gradient(m -> loss(m, x, y), model)
        Flux.update!(opt, model, grads[1])
    end

    # Evaluate
    acc = mean([accuracy(model, x, y) for (x, y) in test_loader])
    println("Epoch $epoch: accuracy = $(round(acc * 100, digits=1))%")
end
```

## Common Tasks

### Compare Random vs Stratified Splits

```julia
include("julia_utils.jl")
using .VAEDatasetUtils

# Load both split types
random_train = VAEDetectorDataset("splits/random_train.txt", "manifest.csv", "~/vae-dataset")
strat_train = VAEDetectorDataset("splits/stratified_train.txt", "manifest.csv", "~/vae-dataset")

println("Random train: $(length(random_train)) samples")
println("Stratified train: $(length(strat_train)) samples")
```

### Validate CUE Schema

```bash
cue vet metadata_schema.cue ~/vae-normalized/metadata.cue
```

### Run Isabelle Proofs

```bash
isabelle build -d . -b VAEDataset_Splits
```

### Hash a Single File

```bash
./target/release/vae-normalizer hash ~/vae-dataset/Original/image001.png
```

## Troubleshooting

**Build fails with "missing feature"**
```bash
rustup update
cargo clean && cargo build --release
```

**"No matching image pairs found"**
- Check dataset structure: needs `Original/` and `VAE/` subdirectories
- Image filenames must match (same stem, any extension)

**Out of memory during checksumming**
- Use `--skip-checksums` for large datasets
- Or process in batches

**Julia: "Images package not found"**
```julia
using Pkg
Pkg.add("Images")
```
