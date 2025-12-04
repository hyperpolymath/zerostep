// SPDX-FileCopyrightText: 2024 Joshua Jewell
// SPDX-License-Identifier: MIT

// VAEDecodedImages-SDXL Metadata Schema
// Validates dataset metadata against Dublin Core Metadata Initiative (DCMI) terms
// Reference: https://www.dublincore.org/specifications/dublin-core/dcmi-terms/

package vae_dataset

import "strings"

// Dublin Core Metadata Element Set validation
#DublinCore: {
	// Core DCMI elements (required)
	title:       string & strings.MinRunes(1)
	creator:     string & strings.MinRunes(1)
	subject:     [...string] & [_, ...]  // non-empty list
	description: string & strings.MinRunes(10)
	publisher:   string
	contributor: [...string]
	date:        =~"^\\d{4}-\\d{2}-\\d{2}$"  // ISO 8601 date
	type:        "Dataset" | "Image" | "Collection"
	format:      =~"^(image|application)/.*$"  // MIME type
	identifier:  string & strings.MinRunes(1)
	source:      string
	language:    =~"^[a-z]{2}(-[A-Z]{2})?$"  // ISO 639-1
	relation:    [...string]
	coverage:    string
	rights:      string

	// Extended DCMI Terms
	audience?:      string
	provenance?:    string
	rights_holder?: string
	license?:       string
}

// Statistics must have positive values
#Statistics: {
	total_pairs:         int & >0
	total_original_bytes: int & >=0
	total_vae_bytes:     int & >=0
	avg_original_bytes:  int & >=0
	avg_vae_bytes:       int & >=0
}

// Split ratios must sum to 1.0 (with tolerance)
#SplitRatios: {
	train:       float & >=0 & <=1
	test:        float & >=0 & <=1
	validation:  float & >=0 & <=1
	calibration: float & >=0 & <=1

	// Constraint: ratios must sum to approximately 1.0
	_sum: train + test + validation + calibration
	_sum: >=0.99 & <=1.01
}

#StratumCount: {
	stratum: int & >=0
	count:   int & >=0
}

#Splits: {
	ratios:         #SplitRatios
	seed:           int & >=0
	num_strata:     int & >=1 & <=16
	stratum_counts: [...#StratumCount]
}

// Checksum configuration
#Checksums: {
	algorithm:   "SHAKE256" | "SHA256" | "SHA3-256" | "BLAKE3"
	output_bits: 256 | 512
	notation:    string
	reference:   string
}

// VAE processing pipeline
#Pipeline: {
	vae_model:     string & strings.MinRunes(1)
	max_dimension: int & >=64 & <=4096
	process:       string
	output_format: "PNG" | "JPEG" | "WEBP"
}

// Formal verification
#Verification: {
	proof_assistant:   "Isabelle/HOL" | "Coq" | "Lean" | "Agda"
	theory_file:       =~".*\\.(thy|v|lean|agda)$"
	proven_properties: [...string] & [_, ...]  // non-empty
}

// Top-level schema
dublin_core:  #DublinCore
statistics:   #Statistics
splits:       #Splits
checksums:    #Checksums
pipeline:     #Pipeline
verification: #Verification
