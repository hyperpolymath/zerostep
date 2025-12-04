# SPDX-FileCopyrightText: 2024 Joshua Jewell
# SPDX-License-Identifier: MIT
#
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                     VAE DATASET NORMALIZER                                 ║
# ║                     The Trillion-Recipe Justfile                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
#
# Usage: just <recipe>
# Help:  just --list

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths
dataset := env_var_or_default("DATASET_PATH", "~/vae-dataset")
output := env_var_or_default("OUTPUT_PATH", "~/vae-normalized")
target_dir := "target"
release_dir := target_dir / "release"
debug_dir := target_dir / "debug"

# Build settings
seed := "42"
strata := "4"
jobs := num_cpus()
profile := "release"

# Container settings
container_name := "vae-normalizer"
container_registry := "localhost"
container_tag := "latest"

# Remote settings
remote_host := env_var_or_default("REMOTE_HOST", "")
remote_user := env_var_or_default("REMOTE_USER", "")

# ============================================================================
# DEFAULT & HELP
# ============================================================================

# Show all recipes (default)
default:
    @just --list --unsorted

# Show recipes by category
[group('help')]
help:
    @echo "╔═══════════════════════════════════════════════════════════════╗"
    @echo "║              VAE Dataset Normalizer - Just Cookbook           ║"
    @echo "╚═══════════════════════════════════════════════════════════════╝"
    @echo ""
    @echo "Categories:"
    @echo "  just --list --unsorted | grep -E '^\['"
    @echo ""
    @echo "Quick start:"
    @echo "  just setup          # First-time setup"
    @echo "  just build          # Build release"
    @echo "  just run            # Run normalizer"
    @echo "  just validate       # Full validation"

# Show recipe source
[group('help')]
show recipe:
    @just --show {{recipe}}

# Dump entire justfile
[group('help')]
dump:
    @just --dump

# List recipes matching pattern
[group('help')]
search pattern:
    @just --list | grep -i "{{pattern}}" || echo "No matches for '{{pattern}}'"

# ============================================================================
# SETUP & DEPENDENCIES
# ============================================================================

# First-time setup
[group('setup')]
setup: check-deps deps
    @echo "Setup complete!"

# Check all dependencies
[group('setup')]
check-deps:
    @echo "Checking dependencies..."
    @echo "Required:"
    @command -v cargo >/dev/null 2>&1 && echo "  ✓ cargo $(cargo --version | cut -d' ' -f2)" || echo "  ✗ cargo MISSING"
    @command -v rustc >/dev/null 2>&1 && echo "  ✓ rustc $(rustc --version | cut -d' ' -f2)" || echo "  ✗ rustc MISSING"
    @echo "Optional:"
    @command -v nix >/dev/null 2>&1 && echo "  ✓ nix $(nix --version | cut -d' ' -f3)" || echo "  · nix not found"
    @command -v podman >/dev/null 2>&1 && echo "  ✓ podman $(podman --version | cut -d' ' -f3)" || echo "  · podman not found"
    @command -v docker >/dev/null 2>&1 && echo "  ✓ docker $(docker --version | cut -d' ' -f3 | tr -d ',')" || echo "  · docker not found"
    @command -v julia >/dev/null 2>&1 && echo "  ✓ julia $(julia --version | cut -d' ' -f3)" || echo "  · julia not found"
    @command -v cue >/dev/null 2>&1 && echo "  ✓ cue $(cue version | head -1 | cut -d' ' -f2)" || echo "  · cue not found"
    @command -v nickel >/dev/null 2>&1 && echo "  ✓ nickel" || echo "  · nickel not found"
    @command -v isabelle >/dev/null 2>&1 && echo "  ✓ isabelle" || echo "  · isabelle not found"

# Install Rust dependencies
[group('setup')]
deps:
    cargo fetch

# Update dependencies
[group('setup')]
update:
    cargo update

# Install dev tools
[group('setup')]
install-tools:
    @echo "Installing development tools..."
    cargo install cargo-audit cargo-outdated cargo-watch cargo-expand cargo-bloat
    cargo install cargo-criterion cargo-flamegraph cargo-llvm-cov
    cargo install lychee hyperfine tokei

# ============================================================================
# BUILD
# ============================================================================

# Build release binary
[group('build')]
build:
    @echo "Building vae-normalizer (release)..."
    cargo build --release

# Build debug binary
[group('build')]
build-debug:
    @echo "Building vae-normalizer (debug)..."
    cargo build

# Build with all features
[group('build')]
build-all-features:
    cargo build --release --all-features

# Build with specific target
[group('build')]
build-target target:
    cargo build --release --target {{target}}

# Cross-compile for Linux musl (static binary)
[group('build')]
build-musl:
    cargo build --release --target x86_64-unknown-linux-musl

# Cross-compile for Windows
[group('build')]
build-windows:
    cargo build --release --target x86_64-pc-windows-gnu

# Cross-compile for macOS
[group('build')]
build-macos:
    cargo build --release --target x86_64-apple-darwin

# Cross-compile for ARM64
[group('build')]
build-arm64:
    cargo build --release --target aarch64-unknown-linux-gnu

# Build optimized for size
[group('build')]
build-small:
    CARGO_PROFILE_RELEASE_LTO=true \
    CARGO_PROFILE_RELEASE_OPT_LEVEL=s \
    CARGO_PROFILE_RELEASE_CODEGEN_UNITS=1 \
    cargo build --release

# Check without building
[group('build')]
check:
    cargo check

# Check all targets
[group('build')]
check-all:
    cargo check --all-targets --all-features

# Clean build artifacts
[group('build')]
clean:
    cargo clean

# Clean and rebuild
[group('build')]
rebuild: clean build

# Show binary size
[group('build')]
size: build
    @ls -lh {{release_dir}}/vae-normalizer
    @file {{release_dir}}/vae-normalizer

# Analyze binary bloat
[group('build')]
bloat: build
    @command -v cargo-bloat >/dev/null 2>&1 && cargo bloat --release || echo "Install: cargo install cargo-bloat"

# ============================================================================
# RUN
# ============================================================================

# Run normalizer (full)
[group('run')]
run: build
    @echo "Running normalizer..."
    @mkdir -p {{output}}
    {{release_dir}}/vae-normalizer normalize \
        -d {{dataset}} \
        -o {{output}} \
        --seed {{seed}} \
        --strata {{strata}}

# Run without checksums (fast)
[group('run')]
run-fast: build
    @mkdir -p {{output}}
    {{release_dir}}/vae-normalizer normalize \
        -d {{dataset}} \
        -o {{output}} \
        --seed {{seed}} \
        --strata {{strata}} \
        --skip-checksums

# Run with verbose output
[group('run')]
run-verbose: build
    @mkdir -p {{output}}
    {{release_dir}}/vae-normalizer -v normalize \
        -d {{dataset}} \
        -o {{output}} \
        --seed {{seed}} \
        --strata {{strata}}

# Run in debug mode
[group('run')]
run-debug: build-debug
    @mkdir -p {{output}}
    RUST_BACKTRACE=1 {{debug_dir}}/vae-normalizer normalize \
        -d {{dataset}} \
        -o {{output}} \
        --seed {{seed}} \
        --strata {{strata}}

# Verify output
[group('run')]
verify: build
    {{release_dir}}/vae-normalizer verify -o {{output}}

# Verify with checksums
[group('run')]
verify-checksums: build
    {{release_dir}}/vae-normalizer verify -o {{output}} --checksums -d {{dataset}}

# Show stats
[group('run')]
stats: build
    {{release_dir}}/vae-normalizer stats -o {{output}}

# Show stats as JSON
[group('run')]
stats-json: build
    {{release_dir}}/vae-normalizer stats -o {{output}} --format json

# Hash a single file
[group('run')]
hash file: build
    {{release_dir}}/vae-normalizer hash {{file}}

# Run complete pipeline
[group('run')]
all: setup build run verify
    @echo "Pipeline complete!"

# Dry run (show what would happen)
[group('run')]
dry-run: build
    @echo "Would run:"
    @echo "  {{release_dir}}/vae-normalizer normalize -d {{dataset}} -o {{output}} --seed {{seed}} --strata {{strata}}"

# ============================================================================
# TEST
# ============================================================================

# Run all tests
[group('test')]
test:
    cargo test

# Run tests with output
[group('test')]
test-verbose:
    cargo test -- --nocapture

# Run specific test
[group('test')]
test-one name:
    cargo test {{name}} -- --nocapture

# Run tests matching pattern
[group('test')]
test-filter pattern:
    cargo test {{pattern}}

# Run ignored tests
[group('test')]
test-ignored:
    cargo test -- --ignored

# Run all tests including ignored
[group('test')]
test-all:
    cargo test -- --include-ignored

# Run doc tests
[group('test')]
test-doc:
    cargo test --doc

# Run tests with coverage
[group('test')]
coverage:
    @command -v cargo-llvm-cov >/dev/null 2>&1 && cargo llvm-cov || echo "Install: cargo install cargo-llvm-cov"

# Run tests with coverage report
[group('test')]
coverage-html:
    @command -v cargo-llvm-cov >/dev/null 2>&1 && cargo llvm-cov --html || echo "Install: cargo install cargo-llvm-cov"

# Quick check (no tests)
[group('test')]
quick-check: fmt-check clippy

# ============================================================================
# LINT & FORMAT
# ============================================================================

# Format code
[group('lint')]
fmt:
    cargo fmt

# Check formatting
[group('lint')]
fmt-check:
    cargo fmt --check

# Run clippy
[group('lint')]
clippy:
    cargo clippy -- -D warnings

# Run clippy with fixes
[group('lint')]
clippy-fix:
    cargo clippy --fix --allow-dirty

# Run all clippy lints
[group('lint')]
clippy-pedantic:
    cargo clippy -- -W clippy::pedantic

# Lint everything
[group('lint')]
lint: fmt-check clippy

# Fix everything
[group('lint')]
fix: fmt clippy-fix

# Check for common mistakes
[group('lint')]
check-mistakes:
    @echo "Checking for common mistakes..."
    @grep -rn "unwrap()" src/ && echo "  Warning: unwrap() found" || echo "  ✓ No unwrap()"
    @grep -rn "expect(" src/ && echo "  Warning: expect() found" || echo "  ✓ No expect()"
    @grep -rn "panic!" src/ && echo "  Warning: panic! found" || echo "  ✓ No panic!"
    @grep -rn "todo!" src/ && echo "  Warning: todo! found" || echo "  ✓ No todo!"
    @grep -rn "unimplemented!" src/ && echo "  Warning: unimplemented! found" || echo "  ✓ No unimplemented!"
    @grep -rn "dbg!" src/ && echo "  Warning: dbg! found" || echo "  ✓ No dbg!"

# ============================================================================
# DOCUMENTATION
# ============================================================================

# Build documentation
[group('docs')]
docs:
    cargo doc --no-deps

# Build and open documentation
[group('docs')]
docs-open:
    cargo doc --no-deps --open

# Build docs with private items
[group('docs')]
docs-private:
    cargo doc --no-deps --document-private-items

# Check documentation
[group('docs')]
docs-check:
    RUSTDOCFLAGS="-D warnings" cargo doc --no-deps

# Count lines of code
[group('docs')]
loc:
    @command -v tokei >/dev/null 2>&1 && tokei || find src -name "*.rs" | xargs wc -l

# Generate README from template (if exists)
[group('docs')]
readme:
    @test -f README.tmpl && envsubst < README.tmpl > README.md || echo "No README.tmpl found"

# ============================================================================
# SECURITY & AUDIT
# ============================================================================

# Run security audit
[group('security')]
audit:
    @command -v cargo-audit >/dev/null 2>&1 && cargo audit || echo "Install: cargo install cargo-audit"

# Check for outdated dependencies
[group('security')]
outdated:
    @command -v cargo-outdated >/dev/null 2>&1 && cargo outdated || echo "Install: cargo install cargo-outdated"

# Audit licenses
[group('security')]
audit-licence:
    @echo "Auditing licenses..."
    @grep -r "SPDX-License-Identifier" src/ *.jl *.cue *.ncl *.thy 2>/dev/null | wc -l | xargs -I {} echo "  Files with SPDX headers: {}"
    @echo "  License: MIT (see LICENSE.txt)"

# Check for floating versions
[group('security')]
audit-deps:
    @echo "Checking dependency versions..."
    @grep -E '"\^|"~' Cargo.toml && echo "  WARNING: Floating versions found" || echo "  ✓ No floating versions"

# Generate SBOM
[group('security')]
sbom:
    @command -v cargo-sbom >/dev/null 2>&1 && cargo sbom > sbom.json && echo "SBOM written to sbom.json" || echo "Install: cargo install cargo-sbom"

# Full security check
[group('security')]
security: audit audit-licence audit-deps
    @echo "Security check complete"

# ============================================================================
# BENCHMARKS & PROFILING
# ============================================================================

# Run benchmarks
[group('bench')]
bench:
    cargo bench

# Run benchmarks with criterion
[group('bench')]
bench-criterion:
    @command -v cargo-criterion >/dev/null 2>&1 && cargo criterion || echo "Install: cargo install cargo-criterion"

# Profile with flamegraph
[group('bench')]
flamegraph: build
    @command -v cargo-flamegraph >/dev/null 2>&1 && cargo flamegraph -- normalize -d {{dataset}} -o {{output}} --skip-checksums || echo "Install: cargo install cargo-flamegraph"

# Time a command
[group('bench')]
time cmd:
    @command -v hyperfine >/dev/null 2>&1 && hyperfine "{{cmd}}" || time {{cmd}}

# Benchmark build time
[group('bench')]
bench-build:
    @command -v hyperfine >/dev/null 2>&1 && hyperfine --warmup 1 "cargo build --release" || echo "Install: cargo install hyperfine"

# ============================================================================
# CONTAINER
# ============================================================================

# Build container with Podman
[group('container')]
container-build:
    podman build -t {{container_name}}:{{container_tag}} -f Containerfile .

# Build container with Docker
[group('container')]
docker-build:
    docker build -t {{container_name}}:{{container_tag}} -f Containerfile .

# Run container
[group('container')]
container-run:
    podman run --rm -v {{dataset}}:/data/input:ro -v {{output}}:/data/output:Z \
        {{container_name}}:{{container_tag}} normalize -d /data/input -o /data/output

# Run container interactively
[group('container')]
container-shell:
    podman run --rm -it -v {{dataset}}:/data/input:ro -v {{output}}:/data/output:Z \
        --entrypoint /bin/sh {{container_name}}:{{container_tag}}

# Push container to registry
[group('container')]
container-push:
    podman push {{container_name}}:{{container_tag}} {{container_registry}}/{{container_name}}:{{container_tag}}

# Pull container from registry
[group('container')]
container-pull:
    podman pull {{container_registry}}/{{container_name}}:{{container_tag}}

# Remove container image
[group('container')]
container-rm:
    podman rmi {{container_name}}:{{container_tag}}

# Show container info
[group('container')]
container-info:
    podman images {{container_name}}
    @echo ""
    podman inspect {{container_name}}:{{container_tag}} | head -50

# ============================================================================
# NIX
# ============================================================================

# Build with Nix
[group('nix')]
nix-build:
    nix build

# Enter Nix dev shell
[group('nix')]
nix-shell:
    nix develop

# Run Nix checks
[group('nix')]
nix-check:
    nix flake check

# Update Nix flake
[group('nix')]
nix-update:
    nix flake update

# Show Nix flake info
[group('nix')]
nix-info:
    nix flake show
    @echo ""
    nix flake metadata

# Build Nix container
[group('nix')]
nix-container:
    nix build .#container
    @echo "Container image at: result"

# ============================================================================
# GIT
# ============================================================================

# Git status
[group('git')]
status:
    git status

# Git diff
[group('git')]
diff:
    git diff

# Git log (short)
[group('git')]
log:
    git log --oneline -20

# Git log (graph)
[group('git')]
log-graph:
    git log --oneline --graph --all -20

# Stage all changes
[group('git')]
add:
    git add -A

# Commit with message
[group('git')]
commit msg:
    git commit -m "{{msg}}"

# Amend last commit
[group('git')]
amend:
    git commit --amend

# Push to remote
[group('git')]
push:
    git push

# Pull from remote
[group('git')]
pull:
    git pull

# Create branch
[group('git')]
branch name:
    git checkout -b {{name}}

# Switch branch
[group('git')]
checkout name:
    git checkout {{name}}

# Delete branch
[group('git')]
branch-delete name:
    git branch -d {{name}}

# Show branches
[group('git')]
branches:
    git branch -a

# Show tags
[group('git')]
tags:
    git tag -l

# Create tag
[group('git')]
tag name:
    git tag -a {{name}} -m "Release {{name}}"

# Stash changes
[group('git')]
stash:
    git stash

# Pop stash
[group('git')]
stash-pop:
    git stash pop

# Show stash list
[group('git')]
stash-list:
    git stash list

# Reset to HEAD
[group('git')]
reset:
    git reset HEAD

# Hard reset (dangerous!)
[group('git')]
[confirm("This will lose uncommitted changes. Continue?")]
reset-hard:
    git reset --hard HEAD

# Clean untracked files
[group('git')]
[confirm("This will delete untracked files. Continue?")]
git-clean:
    git clean -fd

# ============================================================================
# RELEASE
# ============================================================================

# Create release build
[group('release')]
release: clean build test
    @echo "Release build complete: {{release_dir}}/vae-normalizer"

# Create release archive
[group('release')]
release-archive: release
    tar -czvf vae-normalizer-$(git describe --tags 2>/dev/null || echo "dev").tar.gz \
        -C {{release_dir}} vae-normalizer

# Bump version (patch)
[group('release')]
bump-patch:
    @echo "TODO: Implement version bumping"

# Bump version (minor)
[group('release')]
bump-minor:
    @echo "TODO: Implement version bumping"

# Bump version (major)
[group('release')]
bump-major:
    @echo "TODO: Implement version bumping"

# ============================================================================
# DOWNLOAD & DATA
# ============================================================================

# Download dataset from Hugging Face
[group('data')]
download-dataset hf_repo="joshuajewell/VAEDecodedImages-SDXL":
    @echo "Downloading dataset from Hugging Face..."
    @if command -v huggingface-cli >/dev/null 2>&1; then \
        huggingface-cli download {{hf_repo}} --local-dir {{dataset}} --repo-type dataset; \
    elif command -v git >/dev/null 2>&1; then \
        git clone https://huggingface.co/datasets/{{hf_repo}} {{dataset}}; \
    else \
        echo "Install huggingface-cli or git to download"; exit 1; \
    fi

# Clean output directory
[group('data')]
clean-output:
    rm -rf {{output}}

# Show dataset info
[group('data')]
data-info:
    @echo "Dataset: {{dataset}}"
    @test -d {{dataset}} && echo "  Status: exists" || echo "  Status: not found"
    @test -d {{dataset}} && du -sh {{dataset}} || true
    @echo ""
    @echo "Output: {{output}}"
    @test -d {{output}} && echo "  Status: exists" || echo "  Status: not found"
    @test -d {{output}} && du -sh {{output}} || true

# Count images in dataset
[group('data')]
count-images:
    @echo "Counting images..."
    @test -d {{dataset}}/Original && echo "  Original: $(find {{dataset}}/Original -type f | wc -l)" || echo "  Original: not found"
    @test -d {{dataset}}/VAE && echo "  VAE: $(find {{dataset}}/VAE -type f | wc -l)" || echo "  VAE: not found"

# ============================================================================
# JULIA
# ============================================================================

# Run Julia REPL with utilities loaded
[group('julia')]
julia:
    julia -i -e 'include("julia_utils.jl"); using .VAEDatasetUtils'

# Run Julia script
[group('julia')]
julia-run script:
    julia {{script}}

# Install Julia dependencies
[group('julia')]
julia-deps:
    julia -e 'using Pkg; Pkg.add(["Flux", "CSV", "DataFrames", "Images", "FileIO", "SHA"])'

# Compare splits in Julia
[group('julia')]
compare-splits:
    julia -e 'include("julia_utils.jl"); using .VAEDatasetUtils; \
        results = compare_splits("{{output}}/manifest.csv", "{{output}}/splits", "{{output}}/splits"); \
        for (k, v) in results; println("$k: ", v); end'

# Install Julia project dependencies
[group('julia')]
julia-setup:
    julia --project=. -e 'using Pkg; Pkg.instantiate()'

# Train contrastive model for VAE artifact detection
[group('julia')]
train dataset_path output_path="output" epochs="50" batch_size="32":
    @echo "Training VAE artifact detection model..."
    julia --project=. contrastive_model.jl train \
        --dataset {{dataset_path}} \
        --output {{output_path}} \
        --epochs {{epochs}} \
        --batch-size {{batch_size}} \
        --loss supervised

# Train with compressed diff dataset
[group('julia')]
train-compressed dataset_path output_path="output" epochs="50":
    @echo "Training on compressed dataset..."
    julia --project=. contrastive_model.jl train \
        --dataset {{dataset_path}} \
        --output {{output_path}} \
        --epochs {{epochs}} \
        --compressed

# Evaluate trained model
[group('julia')]
evaluate model_path dataset_path:
    julia --project=. contrastive_model.jl evaluate \
        --model {{model_path}} \
        --dataset {{dataset_path}}

# Extract embeddings for visualization
[group('julia')]
embed model_path dataset_path output_path="output":
    julia --project=. contrastive_model.jl embed \
        --model {{model_path}} \
        --dataset {{dataset_path}} \
        --output {{output_path}}

# Quick training test with small epochs
[group('julia')]
train-quick dataset_path:
    @echo "Quick training test (5 epochs)..."
    just train {{dataset_path}} output 5 16

# Full training pipeline: normalize -> train -> evaluate
[group('julia')]
train-full dataset_path output_path="output":
    @echo "Running full training pipeline..."
    just run-normalize {{dataset_path}} {{output_path}}
    just train {{dataset_path}} {{output_path}} 50 32
    just evaluate {{output_path}}/vae_classifier.bson {{dataset_path}}

# ============================================================================
# ISABELLE
# ============================================================================

# Check Isabelle proofs
[group('isabelle')]
isabelle-check:
    @command -v isabelle >/dev/null 2>&1 && isabelle build -d . -b VAEDataset_Splits || echo "Isabelle not installed"

# Build Isabelle session
[group('isabelle')]
isabelle-build:
    isabelle build -d . VAEDataset_Splits

# Open Isabelle/jEdit
[group('isabelle')]
isabelle-jedit:
    isabelle jedit -d . VAEDataset_Splits.thy

# ============================================================================
# RSR (Rhodium Standard Repository) COMPLIANCE
# ============================================================================

# Run full RSR validation
[group('rsr')]
validate: rsr-docs rsr-spdx rsr-well-known rsr-security fmt clippy test
    @echo ""
    @echo "╔═══════════════════════════════════════════════════════════════╗"
    @echo "║                    RSR Validation Complete                    ║"
    @echo "╚═══════════════════════════════════════════════════════════════╝"

# Check RSR documentation requirements
[group('rsr')]
rsr-docs:
    @echo "Checking RSR documentation..."
    @test -f README.adoc && echo "  ✓ README.adoc" || echo "  ✗ README.adoc MISSING"
    @test -f LICENSE.txt && echo "  ✓ LICENSE.txt" || echo "  ✗ LICENSE.txt MISSING"
    @test -f SECURITY.md && echo "  ✓ SECURITY.md" || echo "  ✗ SECURITY.md MISSING"
    @test -f CODE_OF_CONDUCT.md && echo "  ✓ CODE_OF_CONDUCT.md" || echo "  ✗ CODE_OF_CONDUCT.md MISSING"
    @test -f CONTRIBUTING.adoc && echo "  ✓ CONTRIBUTING.adoc" || echo "  ✗ CONTRIBUTING.adoc MISSING"
    @test -f FUNDING.yml && echo "  ✓ FUNDING.yml" || echo "  ✗ FUNDING.yml MISSING"
    @test -f GOVERNANCE.adoc && echo "  ✓ GOVERNANCE.adoc" || echo "  ✗ GOVERNANCE.adoc MISSING"
    @test -f MAINTAINERS.md && echo "  ✓ MAINTAINERS.md" || echo "  ✗ MAINTAINERS.md MISSING"
    @test -f ACCOUNTABILITY.adoc && echo "  ✓ ACCOUNTABILITY.adoc" || echo "  ✗ ACCOUNTABILITY.adoc MISSING"
    @test -f .gitignore && echo "  ✓ .gitignore" || echo "  ✗ .gitignore MISSING"
    @test -f .gitattributes && echo "  ✓ .gitattributes" || echo "  ✗ .gitattributes MISSING"
    @test -f CHANGELOG.md && echo "  ✓ CHANGELOG.md" || echo "  ✗ CHANGELOG.md MISSING"
    @test -f ROADMAP.md && echo "  ✓ ROADMAP.md" || echo "  ✗ ROADMAP.md MISSING"
    @test -f REVERSIBILITY.md && echo "  ✓ REVERSIBILITY.md" || echo "  ✗ REVERSIBILITY.md MISSING"

# Check SPDX headers on source files
[group('rsr')]
rsr-spdx:
    @echo "Checking SPDX headers..."
    @grep -l "SPDX-License-Identifier" src/*.rs >/dev/null 2>&1 && echo "  ✓ Rust sources" || echo "  ✗ Rust sources MISSING SPDX"
    @grep -l "SPDX-License-Identifier" *.jl >/dev/null 2>&1 && echo "  ✓ Julia sources" || echo "  ✗ Julia sources MISSING SPDX"
    @grep -l "SPDX-License-Identifier" *.cue >/dev/null 2>&1 && echo "  ✓ CUE schemas" || echo "  ✗ CUE schemas MISSING SPDX"
    @grep -l "SPDX-License-Identifier" *.ncl >/dev/null 2>&1 && echo "  ✓ Nickel configs" || echo "  ✗ Nickel configs MISSING SPDX"
    @grep -l "SPDX-License-Identifier" *.thy >/dev/null 2>&1 && echo "  ✓ Isabelle theories" || echo "  ✗ Isabelle theories MISSING SPDX"
    @grep -l "SPDX-License-Identifier" flake.nix >/dev/null 2>&1 && echo "  ✓ Nix flake" || echo "  ✗ Nix flake MISSING SPDX"
    @grep -l "SPDX-License-Identifier" Containerfile >/dev/null 2>&1 && echo "  ✓ Containerfile" || echo "  ✗ Containerfile MISSING SPDX"

# Check .well-known directory
[group('rsr')]
rsr-well-known:
    @echo "Checking .well-known directory..."
    @test -d .well-known && echo "  ✓ .well-known/" || echo "  ✗ .well-known/ MISSING"
    @test -f .well-known/security.txt && echo "  ✓ security.txt" || echo "  ✗ security.txt MISSING"
    @test -f .well-known/ai.txt && echo "  ✓ ai.txt" || echo "  ✗ ai.txt MISSING"
    @test -f .well-known/consent-required.txt && echo "  ✓ consent-required.txt" || echo "  ✗ consent-required.txt MISSING"
    @test -f .well-known/provenance.json && echo "  ✓ provenance.json" || echo "  ✗ provenance.json MISSING"
    @test -f .well-known/humans.txt && echo "  ✓ humans.txt" || echo "  ✗ humans.txt MISSING"

# Check security requirements
[group('rsr')]
rsr-security:
    @echo "Checking security requirements..."
    @echo "  ✓ Type safety: Rust (compile-time checked)"
    @echo "  ✓ Memory safety: Rust ownership model"
    @grep -q "cgr.dev/chainguard" Containerfile && echo "  ✓ Container base: Chainguard Wolfi" || echo "  ✗ Container base: NOT Chainguard"
    @grep -q "USER" Containerfile && echo "  ✓ Non-root container" || echo "  ✗ Non-root container MISSING"

# Check MAA (Mutually Assured Accountability) requirements
[group('rsr')]
rsr-maa:
    @echo "Checking MAA requirements..."
    @test -f ACCOUNTABILITY.adoc && echo "  ✓ ACCOUNTABILITY.adoc" || echo "  ✗ ACCOUNTABILITY.adoc MISSING"
    @grep -q "RMR\|Reputation.*Merit.*Rights" ACCOUNTABILITY.adoc 2>/dev/null && echo "  ✓ RMR utilities documented" || echo "  ✗ RMR utilities MISSING"
    @grep -q "RMO\|Responsibility.*Monitoring.*Obligations" ACCOUNTABILITY.adoc 2>/dev/null && echo "  ✓ RMO utilities documented" || echo "  ✗ RMO utilities MISSING"
    @grep -q "Perimeter" CONTRIBUTING.adoc 2>/dev/null && echo "  ✓ TPCF in CONTRIBUTING.adoc" || echo "  ✗ TPCF MISSING"
    @test -f .well-known/provenance.json && echo "  ✓ Provenance chain" || echo "  ✗ Provenance chain MISSING"
    @echo "  ✓ Audit trail: Git + SPDX"

# Full RSR compliance report
[group('rsr')]
rsr-report:
    @echo "╔═══════════════════════════════════════════════════════════════╗"
    @echo "║                    RSR Compliance Report                      ║"
    @echo "╚═══════════════════════════════════════════════════════════════╝"
    @echo ""
    @just rsr-docs
    @echo ""
    @just rsr-spdx
    @echo ""
    @just rsr-well-known
    @echo ""
    @just rsr-security
    @echo ""
    @just rsr-maa
    @echo ""
    @echo "Run 'just validate' for full validation with tests"

# Check links in documentation
[group('rsr')]
check-links:
    @command -v lychee >/dev/null 2>&1 && lychee --verbose *.md *.adoc || echo "Install: cargo install lychee"

# ============================================================================
# WATCH & DEVELOPMENT
# ============================================================================

# Watch for changes and rebuild
[group('dev')]
watch:
    cargo watch -x build

# Watch for changes and test
[group('dev')]
watch-test:
    cargo watch -x test

# Watch for changes and run
[group('dev')]
watch-run:
    cargo watch -x "run -- normalize -d {{dataset}} -o {{output}} --skip-checksums"

# Watch for changes and check
[group('dev')]
watch-check:
    cargo watch -x check

# Expand macros
[group('dev')]
expand:
    @command -v cargo-expand >/dev/null 2>&1 && cargo expand || echo "Install: cargo install cargo-expand"

# ============================================================================
# REMOTE / DEPLOY
# ============================================================================

# Copy binary to remote host
[group('deploy')]
deploy-binary: build
    @test -n "{{remote_host}}" || { echo "Set REMOTE_HOST"; exit 1; }
    @test -n "{{remote_user}}" || { echo "Set REMOTE_USER"; exit 1; }
    scp {{release_dir}}/vae-normalizer {{remote_user}}@{{remote_host}}:~/

# Run on remote host
[group('deploy')]
remote-run cmd:
    @test -n "{{remote_host}}" || { echo "Set REMOTE_HOST"; exit 1; }
    @test -n "{{remote_user}}" || { echo "Set REMOTE_USER"; exit 1; }
    ssh {{remote_user}}@{{remote_host}} "{{cmd}}"

# SSH to remote host
[group('deploy')]
ssh:
    @test -n "{{remote_host}}" || { echo "Set REMOTE_HOST"; exit 1; }
    @test -n "{{remote_user}}" || { echo "Set REMOTE_USER"; exit 1; }
    ssh {{remote_user}}@{{remote_host}}

# ============================================================================
# MISC UTILITIES
# ============================================================================

# Print configuration
[group('util')]
config:
    @echo "Configuration:"
    @echo "  dataset:    {{dataset}}"
    @echo "  output:     {{output}}"
    @echo "  seed:       {{seed}}"
    @echo "  strata:     {{strata}}"
    @echo "  jobs:       {{jobs}}"
    @echo "  profile:    {{profile}}"
    @echo "  container:  {{container_name}}:{{container_tag}}"

# Print environment
[group('util')]
env:
    @echo "Environment:"
    @echo "  DATASET_PATH: ${DATASET_PATH:-<not set>}"
    @echo "  OUTPUT_PATH:  ${OUTPUT_PATH:-<not set>}"
    @echo "  REMOTE_HOST:  ${REMOTE_HOST:-<not set>}"
    @echo "  REMOTE_USER:  ${REMOTE_USER:-<not set>}"
    @echo "  RUST_LOG:     ${RUST_LOG:-<not set>}"

# Open project in editor
[group('util')]
edit:
    ${EDITOR:-code} .

# Open in file manager
[group('util')]
open:
    @command -v xdg-open >/dev/null 2>&1 && xdg-open . || open .

# Tree view
[group('util')]
tree:
    @command -v tree >/dev/null 2>&1 && tree -I target -I node_modules || find . -type f | head -50

# Today's date
[group('util')]
today:
    @date +%Y-%m-%d

# Generate UUID
[group('util')]
uuid:
    @cat /proc/sys/kernel/random/uuid 2>/dev/null || uuidgen

# Random seed
[group('util')]
random-seed:
    @shuf -i 1-1000000 -n 1

# Show system info
[group('util')]
sysinfo:
    @echo "System Information:"
    @echo "  OS: $(uname -s)"
    @echo "  Arch: $(uname -m)"
    @echo "  Kernel: $(uname -r)"
    @echo "  CPUs: {{jobs}}"
    @free -h 2>/dev/null | head -2 || vm_stat 2>/dev/null | head -5 || true

# ============================================================================
# ALIASES
# ============================================================================

alias b := build
alias r := run
alias t := test
alias c := check
alias f := fmt
alias v := validate
alias d := docs
alias w := watch
alias s := status
