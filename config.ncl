# SPDX-FileCopyrightText: 2024 Joshua Jewell
# SPDX-License-Identifier: MIT
#
# VAEDecodedImages-SDXL Configuration Schema
# Nickel configuration for dataset normalization
# Reference: https://nickel-lang.org/

# Dublin Core Metadata contract
let DublinCore = {
  title
    | String
    | doc "Dataset title"
    | default = "VAEDecodedImages-SDXL",

  creator
    | String
    | doc "Dataset creator"
    | default = "Joshua Jewell",

  subject
    | Array String
    | doc "Subject keywords"
    | default = [
        "VAE artifacts",
        "diffusion models",
        "AI image detection",
        "SDXL",
        "latent diffusion"
      ],

  description
    | String
    | doc "Dataset description"
    | default = "Paired dataset of original images and VAE-encoded/decoded counterparts",

  publisher
    | String
    | default = "Hugging Face",

  contributor
    | Array String
    | default = ["alexandrainst"],

  date
    | String
    | doc "ISO 8601 date"
    | default = "2024-01-01",

  type
    | [| 'Dataset, 'Image, 'Collection |]
    | default = 'Dataset,

  format
    | String
    | default = "image/png",

  identifier
    | String
    | default = "joshuajewell/VAEDecodedImages-SDXL",

  source
    | String
    | default = "alexandrainst/nordjylland-news-image-captioning",

  language
    | String
    | doc "ISO 639-1 language code"
    | default = "en",

  relation
    | Array String
    | default = [],

  coverage
    | String
    | default = "Danish newspaper TV2 Nord images",

  rights
    | String
    | default = "See source dataset license",

  # Extended Dublin Core
  audience
    | String
    | optional,

  provenance
    | String
    | optional,

  rights_holder
    | String
    | optional,

  license
    | String
    | optional,
}
in

# Split ratio contract with sum validation
let SplitRatio = {
  train
    | Number
    | doc "Training set ratio"
    | default = 0.70,

  test
    | Number
    | doc "Test set ratio"
    | default = 0.15,

  validation
    | Number
    | doc "Validation set ratio"
    | default = 0.10,

  calibration
    | Number
    | doc "Calibration set ratio"
    | default = 0.05,
}
in

# Checksum configuration
let ChecksumConfig = {
  algorithm
    | [| 'SHAKE256, 'SHA256, 'SHA3_256, 'BLAKE3 |]
    | default = 'SHAKE256,

  output_bits
    | Number
    | doc "Output size in bits"
    | default = 256,

  notation
    | String
    | default = "d=256",

  reference
    | String
    | default = "FIPS 202",
}
in

# Pipeline configuration
let PipelineConfig = {
  vae_model
    | String
    | default = "SDXL VAE",

  max_dimension
    | Number
    | doc "Maximum image dimension"
    | default = 768,

  process
    | String
    | default = "encode -> latent -> decode",

  output_format
    | [| 'PNG, 'JPEG, 'WEBP |]
    | default = 'PNG,
}
in

# Verification configuration
let VerificationConfig = {
  proof_assistant
    | [| 'Isabelle, 'Coq, 'Lean, 'Agda |]
    | default = 'Isabelle,

  theory_file
    | String
    | default = "VAEDataset_Splits.thy",

  proven_properties
    | Array String
    | default = [
        "disjoint_splits",
        "exhaustive_splits",
        "ratio_correctness",
        "original_vae_bijection"
      ],
}
in

# Main configuration
{
  dublin_core
    | DublinCore
    | default = {},

  splits = {
    ratios
      | SplitRatio
      | default = {},

    seed
      | Number
      | doc "Random seed for reproducibility"
      | default = 42,

    num_strata
      | Number
      | doc "Number of stratification bins"
      | default = 4,
  },

  checksums
    | ChecksumConfig
    | default = {},

  pipeline
    | PipelineConfig
    | default = {},

  verification
    | VerificationConfig
    | default = {},

  # Paths configuration
  paths = {
    dataset
      | String
      | doc "Input dataset path"
      | optional,

    output
      | String
      | doc "Output directory path"
      | optional,

    original_subdir
      | String
      | default = "Original",

    vae_subdir
      | String
      | default = "VAE",
  },
}
