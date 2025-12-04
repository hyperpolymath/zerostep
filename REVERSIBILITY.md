<!-- SPDX-FileCopyrightText: 2024 Joshua Jewell -->
<!-- SPDX-License-Identifier: MIT -->

# Reversibility Policy

This document describes how operations in vae-normalizer can be undone or reverted.

## Core Principle

**Every operation should be reversible.** This project follows the principle that users should never be trapped by their choices.

## Operation Reversibility

### Dataset Normalization (`normalize`)

| Operation | Reversible? | How to Revert |
|-----------|-------------|---------------|
| Generate splits | Yes | Delete output directory |
| Compute checksums | Yes | Delete manifest.csv |
| Generate metadata | Yes | Delete metadata.cue |
| Create output directory | Yes | `rm -rf output/` |

**Note**: The normalize command never modifies the source dataset. All outputs are written to a separate output directory.

### Verification (`verify`)

| Operation | Reversible? | How to Revert |
|-----------|-------------|---------------|
| Read manifest | Yes (read-only) | N/A |
| Check file existence | Yes (read-only) | N/A |
| Verify checksums | Yes (read-only) | N/A |

**Note**: The verify command is entirely read-only and makes no changes.

### Statistics (`stats`)

| Operation | Reversible? | How to Revert |
|-----------|-------------|---------------|
| Read manifest | Yes (read-only) | N/A |
| Display stats | Yes (read-only) | N/A |

**Note**: The stats command is entirely read-only.

### Hash (`hash`)

| Operation | Reversible? | How to Revert |
|-----------|-------------|---------------|
| Compute hash | Yes (read-only) | N/A |

**Note**: The hash command is entirely read-only.

## Non-Destructive Defaults

1. **No overwrites**: The tool will not overwrite existing output without `--force`
2. **No source modification**: Source dataset is never modified
3. **Explicit confirmation**: Destructive operations require explicit flags
4. **Dry-run available**: Use `--dry-run` to preview operations

## Recovery Procedures

### Lost Output Directory

If the output directory is accidentally deleted:

```bash
# Regenerate everything
vae-normalizer normalize -d /path/to/dataset -o /path/to/output
```

The tool is deterministic with the same seed, so regeneration produces identical results.

### Corrupted Manifest

If manifest.csv becomes corrupted:

```bash
# Regenerate with same parameters
vae-normalizer normalize -d /path/to/dataset -o /path/to/output --seed 42
```

### Version Control

All outputs are text-based and suitable for version control:

- `manifest.csv` - Track in git for audit trail
- `splits/*.txt` - Track for reproducibility
- `metadata.cue` - Track for provenance

## Git Safety

This project integrates with git safely:

- Never force-pushes
- Never rewrites history without explicit request
- Always allows reverting commits
- Maintains full audit trail

## Formal Guarantees

The Isabelle/HOL proofs in `VAEDataset_Splits.thy` guarantee:

1. **Determinism**: Same inputs → same outputs (with same seed)
2. **Completeness**: All images accounted for
3. **Disjointness**: No data leakage between splits
4. **Bijection**: Original ↔ VAE mapping preserved

## Contact

Questions about reversibility? Open an issue with the `reversibility` label.
