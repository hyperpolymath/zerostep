<!--
SPDX-License-Identifier: CC-BY-SA-4.0
SPDX-FileCopyrightText: 2026 Jonathan D.A. Jewell (hyperpolymath)
-->

# Changelog

All notable changes to `zerostep` will be documented in this file.

This file is generated from conventional commits by the
[`changelog-reusable.yml`](https://github.com/hyperpolymath/standards/blob/main/.github/workflows/changelog-reusable.yml)
workflow (`hyperpolymath/standards#206`). Adopt the workflow in this repo's CI to keep this file in sync automatically — see
[`templates/cliff.toml`](https://github.com/hyperpolymath/standards/blob/main/templates/cliff.toml)
for the canonical config.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
this project aims to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- feat(crg): add crg-grade and crg-badge justfile recipes
- feat: add stapeln.toml layer-based container definition\n\nConverted from existing Containerfile to stapeln format.\nIncludes Chainguard base, security hardening, SBOM generation.\n\nCo-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
- feat: deploy UX Manifesto infrastructure
- feat: add CLADE.a2ml — clade taxonomy declaration
- feat(ci): enable Hypatia scanning

### Fixed

- fix(ci): bump a2ml/k9-validate-action pins to canonical (#32)
- fix(ci): sync hypatia-scan.yml to canonical (#31)
- fix(ci): build Hypatia escript from repo root (estate dogfood drift)
- fix(ci): hypatia-scan workdir (${{ env.HOME }} resolves empty) (#30)
- fix(ci): bump erlef/setup-beam SHA for ubuntu24 runner support (#23)
- fix(ci): repair YAML block-scalar in workflow-linter Check Permissions step (#26)
- fix(ci): move secret-scanner Cargo.toml gate from job-level if: to step-level (#28)
- fix(src/metadata.rs): remove unused std::fs::File import (#25)
- fix(codeql): switch language matrix to 'actions' (no JS/TS in repo) (#24)
- fix(ci): Resolve workflow-linter self-matching and metadata issues

### Changed

- refactor: migrate 6SCM → 6A2 (.scm → .a2ml format)

### Documentation

- docs: record tech-debt audit findings (2026-05-26) (#43)
- docs: substantive CRG C annotation (EXPLAINME.adoc)
- docs: add EXPLAINME.adoc — prove-it file backing README claims
- docs: add checkpoint files for state tracking

### CI

- ci(rust): convert rust-ci.yml to thin wrapper (standards#174) (#40)
- ci: redistribute concurrency-cancel guard to read-only check workflows (#34)
- ci(dependabot): restore cargo PR limit so security PRs flow (#16)
- ci(secret-scanner): drop duplicate --fail from trufflehog extra_args (#15)
- ci: bump actions/upload-artifact SHA to current v4 (#14)

## Pre-history

Prior commits to this file's introduction are recorded in git history but not formally classified into Keep-a-Changelog sections. To backfill, run `git cliff -o CHANGELOG.md` locally using the canonical [`cliff.toml`](https://github.com/hyperpolymath/standards/blob/main/templates/cliff.toml) — this is one-shot mechanical work.

---

<!-- This file was seeded by the 2026-05-26 estate tech-debt audit follow-up (Row-2 Phase 3); see [`hyperpolymath/standards/docs/audits/2026-05-26-estate-documentation-debt.md`](https://github.com/hyperpolymath/standards/blob/main/docs/audits/2026-05-26-estate-documentation-debt.md). -->
