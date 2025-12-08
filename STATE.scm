;; SPDX-FileCopyrightText: 2024 Joshua Jewell
;; SPDX-License-Identifier: MIT
;;
;; STATE.scm - Project State Checkpoint
;; Format: Guile Scheme (declarative, human-readable)
;; Reference: https://github.com/hyperpolymath/state.scm

;; ============================================================================
;; METADATA
;; ============================================================================

(define-module (zerostep state)
  #:export (state))

(define state
  '((metadata
     (format-version . "1.0.0")
     (created . "2024-12-08")
     (updated . "2024-12-08")
     (project . "ZeroStep / VAE Dataset Normalizer")
     (repository . "https://github.com/hyperpolymath/ZeroStep"))

;; ============================================================================
;; CURRENT POSITION
;; ============================================================================

    (current-position
     (version . "1.0.0")
     (status . "released")
     (completion . 100)
     (phase . "maintenance")

     (summary . "Core VAE dataset normalization tool is complete and production-ready.
All v1.0.0 features implemented: SHAKE256 checksums, train/test/val/cal splits,
Dublin Core metadata, diff compression, Isabelle proofs, Julia/Flux training,
and full RSR compliance.")

     (implemented-features
      ("SHAKE256 (d=256) cryptographic checksums - FIPS 202 compliant")
      ("Train/Test/Val/Calibration splits - 70/15/10/5 ratio")
      ("Random and stratified split generation")
      ("Dublin Core metadata via CUE configuration")
      ("Nickel schema for flexible configuration")
      ("Diff-based compression - ~50% storage reduction")
      ("Isabelle/HOL formal proofs for split correctness")
      ("Julia/Flux.jl training utilities")
      ("Contrastive learning model for VAE artifact detection")
      ("RSR (Rhodium Standard Repository) compliance")
      ("Podman containerization with Chainguard Wolfi")
      ("Nix flakes for reproducible builds"))

     (tech-stack
      (language . "Rust 1.70+")
      (cryptography . "SHAKE256 (FIPS 202)")
      (rng . "ChaCha20 deterministic")
      (parallelism . "Rayon")
      (ml-framework . "Flux.jl (Julia)")
      (configuration . "CUE + Nickel")
      (formal-verification . "Isabelle/HOL")
      (task-runner . "Justfile")
      (build-system . "Nix Flakes")
      (containers . "Podman (Chainguard Wolfi)")
      (licenses . "MIT OR GPL-3.0-or-later")))

;; ============================================================================
;; ROUTE TO MVP v1.1.0
;; ============================================================================

    (route-to-next-milestone
     (target . "v1.1.0")
     (theme . "Multi-VAE support and export formats")
     (estimated-completion . "unspecified")

     (tasks
      ((id . "multi-vae")
       (title . "Multi-VAE Support")
       (status . "planned")
       (priority . "high")
       (description . "Process datasets through different VAE models")
       (subtasks
        ("SD 1.5 VAE support")
        ("SDXL VAE support")
        ("Flux VAE support")
        ("Custom VAE path configuration")))

      ((id . "parallel-processing")
       (title . "Parallel Processing Enhancement")
       (status . "planned")
       (priority . "high")
       (description . "Configurable worker threads for large datasets")
       (subtasks
        ("Implement --jobs N flag")
        ("Rayon thread pool optimization")
        ("Memory-mapped file I/O for large datasets")))

      ((id . "export-formats")
       (title . "Additional Export Formats")
       (status . "planned")
       (priority . "medium")
       (description . "Support more output formats beyond CSV")
       (subtasks
        ("Parquet export")
        ("HuggingFace datasets format")
        ("TFRecord format")))

      ((id . "incremental-processing")
       (title . "Incremental Processing")
       (status . "planned")
       (priority . "medium")
       (description . "Resume interrupted normalization jobs"))

      ((id . "progress-reporting")
       (title . "Enhanced Progress Reporting")
       (status . "planned")
       (priority . "low")
       (description . "Better ETA and speed metrics for large datasets"))))

;; ============================================================================
;; KNOWN ISSUES & GAPS
;; ============================================================================

    (issues
     (blockers
      ;; No critical blockers - v1.0.0 is stable
      )

     (observations
      ((id . "cargo-lock")
       (severity . "minor")
       (description . "Cargo.lock not committed to version control")
       (impact . "May affect build reproducibility for exact dependency versions")
       (recommendation . "Consider adding Cargo.lock to git for pinned versions"))

      ((id . "version-bump-recipes")
       (severity . "minor")
       (description . "Version bump recipes in justfile contain TODO placeholders")
       (impact . "Manual version bumping required")
       (location . "justfile: bump-patch, bump-minor, bump-major"))

      ((id . "filename-matching")
       (severity . "limitation")
       (description . "Requires exact filename stem matching between Original/ and VAE/")
       (impact . "Datasets must have identical naming in both directories")
       (recommendation . "Document clearly; consider fuzzy matching in v1.2+"))

      ((id . "stratification-basis")
       (severity . "design-choice")
       (description . "Stratification based on file size, not content characteristics")
       (impact . "May not perfectly balance by visual complexity")
       (recommendation . "Consider content-based stratification in v2.0"))

      ((id . "julia-integration")
       (severity . "minor")
       (description . "Julia dependencies require manual setup outside Nix")
       (impact . "Training pipeline setup is not fully reproducible via Nix alone")
       (recommendation . "Add Julia2Nix integration in future version")))

     (technical-debt
      ;; Minimal - codebase is clean and well-documented
      ("No unsafe Rust code - memory safety verified")
      ("No TODOs/FIXMEs in core implementation")
      ("Comprehensive test coverage via `just test`")))

;; ============================================================================
;; QUESTIONS FOR USER/MAINTAINER
;; ============================================================================

    (questions
     ((id . "q1")
      (topic . "Prioritization")
      (question . "Which v1.1.0 feature should be prioritized first: Multi-VAE support, parallel processing, or export formats?"))

     ((id . "q2")
      (topic . "VAE Models")
      (question . "Are there specific VAE models beyond SD 1.5/SDXL/Flux that should be supported?"))

     ((id . "q3")
      (topic . "Export Formats")
      (question . "Is HuggingFace datasets format the highest priority export, or would Parquet be more useful for your workflows?"))

     ((id . "q4")
      (topic . "Performance")
      (question . "What is the typical dataset size you work with? This helps prioritize memory-mapped I/O and parallel processing."))

     ((id . "q5")
      (topic . "Metrics")
      (question . "For v1.2.0 metrics (PSNR/SSIM), should these be computed at normalization time or as a separate post-processing command?"))

     ((id . "q6")
      (topic . "Distribution")
      (question . "Would pre-built binaries (Homebrew, apt) be more valuable than the current Nix/container distribution?"))

     ((id . "q7")
      (topic . "Research Direction")
      (question . "Is there interest in expanding beyond VAE to GAN/diffusion model artifacts for v2.0?"))

     ((id . "q8")
      (topic . "Community")
      (question . "Any external contributors or institutions showing interest in collaboration?")))

;; ============================================================================
;; LONG-TERM ROADMAP
;; ============================================================================

    (roadmap
     ((version . "1.1.0")
      (theme . "Multi-VAE & Performance")
      (status . "planned")
      (features
       ("Multi-VAE support (SD 1.5, SDXL, Flux, custom)")
       ("--jobs N parallel processing flag")
       ("Rayon thread pool optimization")
       ("Parquet export format")
       ("HuggingFace datasets format")
       ("TFRecord format")
       ("Memory-mapped file I/O")
       ("Incremental/resumable processing")
       ("Enhanced progress reporting")))

     ((version . "1.2.0")
      (theme . "Preprocessing & Metrics")
      (status . "planned")
      (features
       ("Automatic image resizing")
       ("Format conversion utilities")
       ("Quality filtering")
       ("Augmentation impact documentation")
       ("Augmentation-aware split generation")
       ("PSNR/SSIM computation between original and VAE")
       ("Artifact intensity scoring")
       ("Statistical summaries")))

     ((version . "1.2.0-infra")
      (theme . "Distribution & CI")
      (status . "planned")
      (features
       ("GitHub Actions / GitLab CI templates")
       ("Pre-built binaries for major platforms")
       ("Homebrew formula")
       ("APT/RPM packages")))

     ((version . "2.0.0")
      (theme . "Multi-Model & Federation")
      (status . "vision")
      (features
       ("Non-VAE generative model support")
       ("GAN artifact datasets")
       ("Autoregressive model artifacts")
       ("Distributed split generation")
       ("Cross-institution dataset pooling")
       ("Privacy-preserving checksums")
       ("Active learning integration")
       ("Uncertainty-based sample selection")
       ("Human-in-the-loop verification")))

     ((version . "2.0.0-research")
      (theme . "Research Directions")
      (status . "exploratory")
      (features
       ("VAE artifact taxonomy development")
       ("Detection model benchmarks")
       ("Adversarial robustness testing")
       ("Cross-model generalization studies"))))

;; ============================================================================
;; MAINTENANCE COMMITMENTS
;; ============================================================================

    (maintenance
     (active-development . "ongoing")
     (security-fixes . "minimum 2 years from v1.0.0 (until 2026)")
     (critical-bugs . "minimum 3 years from v1.0.0 (until 2027)")

     (succession-plan
      ("Repository remains MIT licensed (always forkable)")
      ("Archive on Software Heritage")
      ("Transfer to community organization if interest exists")
      ("Data export always available"))

     (archive-strategy
      ("Full source history preserved")
      ("Binary releases archived")
      ("Documentation snapshots")
      ("Dataset compatibility notes")))

;; ============================================================================
;; SESSION NOTES
;; ============================================================================

    (session-notes
     (last-session . "2024-12-08")
     (context . "Initial STATE.scm creation - comprehensive project state capture")
     (accomplishments
      ("Created STATE.scm checkpoint file")
      ("Documented current position at v1.0.0")
      ("Mapped route to v1.1.0 with prioritized tasks")
      ("Identified minor issues and technical observations")
      ("Formulated questions for maintainer input")
      ("Documented complete roadmap through v2.0.0"))

     (next-session-priorities
      ("Address any questions answered by maintainer")
      ("Begin implementation of highest-priority v1.1.0 feature")
      ("Update STATE.scm with progress")))))

;; ============================================================================
;; USAGE
;; ============================================================================
;;
;; This file serves as a checkpoint for AI-assisted development sessions.
;;
;; At session start:
;;   - Load this file to restore full project context
;;   - Review current-position and route-to-next-milestone
;;   - Check questions for any pending decisions
;;
;; At session end:
;;   - Update completion percentages
;;   - Add new issues discovered
;;   - Document session accomplishments
;;   - Update next-session-priorities
;;
;; Format chosen: Guile Scheme
;;   - Minimal syntax, obvious structure
;;   - Human-readable and AI-parseable
;;   - Self-documenting with comments
;;   - Easily diffable in version control
;;
;; ============================================================================
