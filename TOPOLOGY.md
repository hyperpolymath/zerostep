<!-- SPDX-License-Identifier: PMPL-1.0-or-later -->
<!-- TOPOLOGY.md — Project architecture map and completion dashboard -->
<!-- Last updated: 2026-02-19 -->

# VAE Dataset Normalizer (zerostep) — Project Topology

## System Architecture

```
                        ┌─────────────────────────────────────────┐
                        │              ML RESEARCHER              │
                        │        (CLI / Julia Notebook)           │
                        └───────────────────┬─────────────────────┘
                                            │ Normalize / Train
                                            ▼
                        ┌─────────────────────────────────────────┐
                        │           VAE NORMALIZER CORE           │
                        │  ┌───────────┐  ┌───────────────────┐  │
                        │  │ Splitter  │  │  Checksum Engine  │  │
                        │  │ (Rust)    │  │  (SHAKE256)       │  │
                        │  └─────┬─────┘  └────────┬──────────┘  │
                        │        │                 │              │
                        │  ┌─────▼─────┐  ┌────────▼──────────┐  │
                        │  │ Diff Comp │  │  Metadata Layer   │  │
                        │  │ (Storage) │  │  (CUE / Nickel)   │  │
                        │  └─────┬─────┘  └────────┬──────────┘  │
                        └────────│─────────────────│──────────────┘
                                 │                 │
                                 ▼                 ▼
                        ┌─────────────────────────────────────────┐
                        │           VERIFICATION & ML             │
                        │  ┌───────────┐  ┌───────────────────┐  │
                        │  │ Isabelle  │  │  Contrastive Model│  │
                        │  │ (Proofs)  │  │  (Julia/Flux.jl)  │  │
                        │  └───────────┘  └───────────────────┘  │
                        └───────────────────┬─────────────────────┘
                                            │
                                            ▼
                        ┌─────────────────────────────────────────┐
                        │           OUTPUT DATASET                │
                        │      (Train/Test/Val/Calib Splits)      │
                        └─────────────────────────────────────────┘

                        ┌─────────────────────────────────────────┐
                        │          REPO INFRASTRUCTURE            │
                        │  Justfile Automation  .machine_readable/  │
                        │  Nix / Wolfi          0-AI-MANIFEST.a2ml  │
                        └─────────────────────────────────────────┘
```

## Completion Dashboard

```
COMPONENT                          STATUS              NOTES
─────────────────────────────────  ──────────────────  ─────────────────────────────────
CORE NORMALIZATION
  Dataset Splitter (Rust)           ██████████ 100%    Random & Stratified stable
  SHAKE256 Checksums                ██████████ 100%    FIPS 202 compliant verified
  Diff-based Compression            ██████████ 100%    ~50% storage reduction verified
  Stats & Manifest Gen              ██████████ 100%    JSON/CSV exports active

VERIF & LEARNING
  Isabelle/HOL Proofs               ██████████ 100%    Split properties verified
  Contrastive Model (Julia)         ██████████ 100%    Artifact detection active
  Flux.jl Integration               ██████████ 100%    Data loaders verified

REPO INFRASTRUCTURE
  Justfile Automation               ██████████ 100%    Standard build/train tasks
  .machine_readable/                ██████████ 100%    STATE tracking active
  RSR Compliance (MAA)              ██████████ 100%    Accountability verified

─────────────────────────────────────────────────────────────────────────────
OVERALL:                            ██████████ 100%    Production-ready ML infrastructure
```

## Key Dependencies

```
Raw Dataset ──────► Checksum Engine ────► Diff Comp ──────► Dataset Splits
     │                 │                   │                    │
     ▼                 ▼                   ▼                    ▼
CUE Schema ───────► Nickel Config ─────► Isabelle Proof ───► Julia Model
```

## Update Protocol

This file is maintained by both humans and AI agents. When updating:

1. **After completing a component**: Change its bar and percentage
2. **After adding a component**: Add a new row in the appropriate section
3. **After architectural changes**: Update the ASCII diagram
4. **Date**: Update the `Last updated` comment at the top of this file

Progress bars use: `█` (filled) and `░` (empty), 10 characters wide.
Percentages: 0%, 10%, 20%, ... 100% (in 10% increments).
