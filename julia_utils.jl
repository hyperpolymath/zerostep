# SPDX-FileCopyrightText: 2024 Joshua Jewell
# SPDX-License-Identifier: MIT

#=
VAEDatasetUtils Module: VAE Dataset Utilities for Julia/Flux
============================================================

This module provides a comprehensive set of utilities for efficiently loading,
processing, and managing image datasets specifically tailored for VAE (Variational
Autoencoder) artifact detection tasks within Julia and Flux.jl. It supports
both standard dataset formats and a specialized compressed diff format,
along with robust data integrity checks.

Purpose and Rationale:
----------------------
The primary goal of `VAEDatasetUtils` is to streamline the data pipeline for
training machine learning models (such as those in `VAEContrastive`) to
distinguish between original images and images that have been processed (and potentially
degraded) by VAEs. Key features address common challenges in large-scale image datasets:

1.  **Flexible Data Loading:** Supports loading image data from structured directories
    and manifest files.
2.  **Data Integrity:** Implements SHA-256 checksum verification to ensure that
    image files are not corrupted or tampered with during storage or transfer.
    This is crucial for reproducible research and reliable model training.
3.  **Space Efficiency:** Introduces a `CompressedVAEDataset` format where VAE-decoded
    images are stored as "diffs" (differences) from their corresponding original images.
    This significantly reduces storage requirements by leveraging the high similarity
    between original and VAE-decoded images, allowing VAE images to be reconstructed
    on-the-fly.
4.  **Batching and Shuffling:** Provides a generic `DataLoader` for efficient
    batching and shuffling of datasets, which is essential for training deep learning models.

Workflow Support:
-----------------
This module is designed to work in conjunction with model training modules (e.g., `VAEContrastive`)
by providing data in a format ready for consumption by Flux.jl models. It handles the
complexity of file I/O, image loading, preprocessing (via transforms), and batch creation,
allowing researchers to focus on model development.

Usage:
------
To use this module, include it and then import its functions:
```julia
include("julia_utils.jl")
using .VAEDatasetUtils

# Example: Load a standard VAE artifact detection dataset
dataset = VAEDetectorDataset(
    "splits/train.txt",          # Path to a file listing image IDs for the split
    "manifest.csv",              # Path to the manifest file describing all image pairs
    "/path/to/base_dataset_dir"  # Base directory where original and VAE images are stored
)
x_orig, x_vae = dataset[1] # Get first original and VAE image pair

# Example: Create a DataLoader
train_loader = DataLoader(dataset, batch_size=32, shuffle=true)
first_batch_images, first_batch_labels = first(train_loader)
```
See the "Example usage with Flux.jl" section at the end of this file for more complete examples.

Requires: SHA.jl, CSV.jl, DataFrames.jl, FileIO.jl, Images.jl, Random, Statistics.jl, Flux.jl (for examples).
=#

module VAEDatasetUtils


using SHA
using CSV
using DataFrames
using FileIO
using Images
using Random
using Statistics

export VAEDetectorDataset, load_split, verify_checksum, compute_shake256
export load_manifest, get_image_pair, DataLoader, accuracy, train_epoch!
export CompressedVAEDataset, reconstruct_vae_from_diff

"""
    compute_shake256(data::Vector{UInt8})::Vector{UInt8}
    compute_shake256(filepath::AbstractString)::Vector{UInt8}

Computes the SHAKE256 hash (an Extendable Output Function - XOF) with an output
digest length of 256 bits (32 bytes). SHAKE256 is part of the SHA-3 family of
cryptographic hash functions and is suitable for generating variable-length
digests, providing strong cryptographic integrity checks.

This function is overloaded to accept either raw `UInt8` byte data or a file path:

1.  `compute_shake256(data::Vector{UInt8})`:
    Calculates the SHAKE256 hash of the provided byte `data`.

    Arguments:
    - `data`: A `Vector{UInt8}` representing the raw data to be hashed.

    Returns:
    - A `Vector{UInt8}` of length 32 bytes, representing the SHAKE256 hash digest.

2.  `compute_shake256(filepath::AbstractString)`:
    Reads the content of the file at `filepath` and then calculates its SHAKE256 hash.

    Arguments:
    - `filepath`: An `AbstractString` representing the path to the file whose
                  content is to be hashed.

    Returns:
    - A `Vector{UInt8}` of length 32 bytes, representing the SHAKE256 hash digest
      of the file's content.
"""
function compute_shake256(data::Vector{UInt8})::Vector{UInt8}
    ctx = SHA.SHA3_256_CTX()
    SHA.update!(ctx, data)
    return SHA.digest!(ctx)
end

function compute_shake256(filepath::AbstractString)::Vector{UInt8}
    data = read(filepath)
    return compute_shake256(data)
end

"""
    bytes_to_hex(bytes::Vector{UInt8})::String

Converts a `Vector{UInt8}` (raw byte data, typically a hash digest) into
its hexadecimal string representation. Each byte is converted to a two-character
hexadecimal string, and these are then joined together.

Arguments:
- `bytes`: A `Vector{UInt8}` to be converted to hexadecimal.

Returns:
- A `String` representing the hexadecimal form of the input bytes.
"""
function bytes_to_hex(bytes::Vector{UInt8})::String
    return join(string(b, base=16, pad=2) for b in bytes)
end

"""
    verify_checksum(filepath::AbstractString, expected_hex::AbstractString)::Bool

Verifies the integrity of a file by comparing its computed SHAKE256 checksum
with an expected hexadecimal checksum string. This function is crucial for
ensuring data reliability in datasets, detecting accidental corruption or
unauthorized modifications.

Arguments:
- `filepath`: An `AbstractString` representing the path to the file whose
              checksum is to be verified.
- `expected_hex`: An `AbstractString` representing the expected SHAKE256 checksum
                  in hexadecimal format. A special value of `"skipped"` will
                  cause the function to return `true` without performing the
                  checksum computation, effectively bypassing verification
                  for that file.

Returns:
- `true` if the computed checksum matches `expected_hex` or if `expected_hex`
  is `"skipped"`.
- `false` otherwise (checksum mismatch).
"""
function verify_checksum(filepath::AbstractString, expected_hex::AbstractString)::Bool
    if expected_hex == "skipped"
        return true
    end
    actual = bytes_to_hex(compute_shake256(filepath))
    return actual == expected_hex
end

"""
    load_manifest(manifest_path::AbstractString)::DataFrame

Loads the dataset manifest from a CSV file into a `DataFrame`. The manifest
file is expected to contain metadata for each image pair, including their IDs,
paths, sizes, checksums, and a stratum identifier (for stratified sampling).

Arguments:
- `manifest_path`: An `AbstractString` representing the path to the manifest CSV file.

Expected CSV Columns:
- `id`: Unique identifier for the image pair (String).
- `original_path`: Relative path to the original image (String).
- `vae_path`: Relative path to the VAE-decoded image (String).
- `original_size`: Size of the original image file in bytes (Int64).
- `vae_size`: Size of the VAE-decoded image file in bytes (Int64).
- `original_shake256_d256`: SHAKE256 checksum of the original image (hex String).
- `vae_shake256_d256`: SHAKE256 checksum of the VAE-decoded image (hex String).
- `stratum`: Stratum identifier for the image pair (Int) - useful for ensuring
             balanced splits during dataset creation.

Returns:
- A `DataFrame` containing the parsed manifest data.
"""
function load_manifest(manifest_path::AbstractString)::DataFrame
    return CSV.read(manifest_path, DataFrame)
end

"""
    load_split(split_path::AbstractString)::Vector{String}

Loads a list of image identifiers (IDs) from a text file, where each ID
is expected to be on a new line. This function is used to define specific
subsets of the dataset, such as training, validation, or testing splits.

Arguments:
- `split_path`: An `AbstractString` representing the path to the split file.

Expected File Format:
- A plain text file where each line contains a unique image ID.
- Empty lines are filtered out.

Returns:
- A `Vector{String}` containing the image IDs belonging to the specified split.
"""
function load_split(split_path::AbstractString)::Vector{String}
    return filter(!isempty, readlines(split_path))
end

"""
    ImagePair

A structure to represent a single pair of images (an original image and its
corresponding VAE-decoded version) along with associated metadata. This
struct is used internally by dataset loaders to manage image information.

Fields:
- `id::String`: A unique identifier for this image pair.
- `original_path::String`: The relative path to the original, unprocessed image file.
- `vae_path::String`: The relative path to the VAE-decoded image file.
- `original_size::Int64`: The size of the original image file in bytes.
- `vae_size::Int64`: The size of the VAE-decoded image file in bytes.
- `original_checksum::String`: The SHAKE256 checksum (hexadecimal string) of the original image.
- `vae_checksum::String`: The SHAKE256 checksum (hexadecimal string) of the VAE-decoded image.
- `stratum::Int`: An integer identifier for the stratum this image pair belongs to,
                  useful for stratified sampling during dataset splitting.
"""
struct ImagePair
    id::String
    original_path::String
    vae_path::String
    original_size::Int64
    vae_size::Int64
    original_checksum::String
    vae_checksum::String
    stratum::Int
end

"""
    VAEDetectorDataset

A structure representing a dataset for VAE artifact detection, where original
and VAE-decoded images are loaded directly from disk. This dataset handles
the organization of image pairs based on a manifest file and a split definition.

Fields:
- `pairs::Vector{ImagePair}`: A vector of `ImagePair` structs, each containing
                               metadata for an original/VAE image pair relevant
                               to the current dataset split.
- `base_path::String`: The base directory from which all relative image paths
                       (specified in `ImagePair`) are resolved.
- `transform::Function`: An optional function applied to each image after loading
                         (e.g., resizing, normalization, augmentation).
                         Defaults to `identity` (no transformation).
- `verify_checksums::Bool`: A flag indicating whether SHAKE256 checksums should
                            be verified for each image during loading.
                            Defaults to `false`.
"""
struct VAEDetectorDataset
    pairs::Vector{ImagePair}
    base_path::String
    transform::Function
    verify_checksums::Bool
end

"""
    VAEDetectorDataset(
        split_path::AbstractString,
        manifest_path::AbstractString,
        base_path::AbstractString;
        transform::Function = identity,
        verify_checksums::Bool = false
    )

Constructs a `VAEDetectorDataset` instance by loading and filtering image
metadata based on a manifest file and a specified dataset split.

Arguments:
- `split_path`: An `AbstractString` path to a text file containing the IDs
                of image pairs to include in this dataset split.
- `manifest_path`: An `AbstractString` path to the CSV manifest file containing
                   metadata for all possible image pairs.
- `base_path`: An `AbstractString` representing the root directory where all
               image files (original and VAE) are stored. Relative paths in the
               manifest are resolved against this base path.

Keyword Arguments:
- `transform::Function`: A function to apply to each loaded image. This can be
                         used for preprocessing steps like resizing, normalization,
                         or data augmentation. Defaults to `identity`.
- `verify_checksums::Bool`: If `true`, the SHAKE256 checksum of each image file
                            will be computed and compared against the values
                            in the manifest during `getindex` operations.
                            Errors will be thrown on mismatch. Defaults to `false`.

Returns:
- A `VAEDetectorDataset` instance, ready for iteration or indexing.

Usage:
The constructor first loads the full manifest, then filters it down to only
those `ImagePair`s whose IDs are present in the `split_path` file.
"""
function VAEDetectorDataset(
    split_path::AbstractString,
    manifest_path::AbstractString,
    base_path::AbstractString;
    transform::Function = identity,
    verify_checksums::Bool = false
)
    # Load manifest and split
    manifest = load_manifest(manifest_path)
    split_ids = Set(load_split(split_path))

    # Filter manifest rows to create ImagePair objects for the current split
    pairs = ImagePair[]
    for row in eachrow(manifest)
        if row.id in split_ids
            # Directly use row values as they match ImagePair fields
            push!(pairs, ImagePair(
                row.id,
                row.original_path,
                row.vae_path,
                row.original_size,
                row.vae_size,
                row.original_shake256_d256,
                row.vae_shake256_d256,
                row.stratum
            ))
        end
    end

    return VAEDetectorDataset(pairs, base_path, transform, verify_checksums)
end

"""
    Base.length(d::VAEDetectorDataset)

Returns the total number of image pairs in the `VAEDetectorDataset`.
This makes the dataset iterable and compatible with functions that rely
on `length`.
"""
Base.length(d::VAEDetectorDataset) = length(d.pairs)

"""
    Base.getindex(d::VAEDetectorDataset, i::Int)

Retrieves a single image pair (original and VAE-decoded) from the dataset
at the specified index `i`. This method handles loading the images from disk,
optionally verifying their checksums, applying any specified transformations,
and returning them as a tuple.

Arguments:
- `d`: The `VAEDetectorDataset` instance.
- `i`: An integer index corresponding to the desired image pair within the dataset.

Returns:
- A `Tuple` containing two `Array{Float32}` images:
    1. The original image, processed by `d.transform`.
    2. The VAE-decoded image, processed by `d.transform`.

Each image is loaded as a `Float32` array with pixel values normalized to the `[0, 1]` range.
If `d.verify_checksums` is `true`, a checksum mismatch will throw an `error`.
"""
function Base.getindex(d::VAEDetectorDataset, i::Int)
    pair = d.pairs[i]

    # Construct absolute paths to images
    orig_path = joinpath(d.base_path, pair.original_path)
    vae_path = joinpath(d.base_path, pair.vae_path)

    # Verify checksums if enabled for this dataset instance
    if d.verify_checksums
        if !verify_checksum(orig_path, pair.original_checksum)
            error("Checksum mismatch for original image ID: $(pair.id) at path: $orig_path")
        end
        if !verify_checksum(vae_path, pair.vae_checksum)
            error("Checksum mismatch for VAE image ID: $(pair.id) at path: $vae_path")
        end
    end

    # Load images using FileIO and Images.jl, then convert to Float32 channelview
    # Images are expected to be 3D arrays (Height x Width x Channels) or 2D (H x W)
    # channelview transforms to (Channels x Height x Width)
    orig_img = Float32.(channelview(load(orig_path)))
    vae_img = Float32.(channelview(load(vae_path)))

    # Apply any user-defined transformations
    orig_img = d.transform(orig_img)
    vae_img = d.transform(vae_img)

    # Return the image pair. Labels (0=original, 1=VAE) are typically added by the DataLoader
    return (orig_img, vae_img)
end

#=
Compressed Dataset Support
==========================

For datasets stored in compressed diff format (Original/ + Diff/ instead of Original/ + VAE/).
The diff encoding is: diff = VAE - Original + 128 (offset to handle signed values)
Reconstruction is: VAE = Original + Diff - 128
=#

"""
    reconstruct_vae_from_diff(original::Array{Float32}, diff::Array{Float32})::Array{Float32}

Reconstructs a VAE-decoded image from an original image and a compressed
difference image. This method is central to the `CompressedVAEDataset`
strategy, significantly reducing storage requirements by not storing
full VAE images.

The diff encoding assumes an offset of 128 was applied during compression
to handle signed differences. The reconstruction formula in the `[0, 255]`
pixel value range is:
`VAE = Original + Diff - 128`

The input `original` and `diff` images are expected to be `Float32` arrays
with pixel values normalized to the `[0, 1]` range. These are first scaled
to `[0, 255]`, the reconstruction arithmetic is performed, and then the result
is clamped to `[0, 255]` and normalized back to `[0, 1]`.

Arguments:
- `original`: The original image as a `Float32` array (pixel values in `[0, 1]`).
- `diff`: The difference image as a `Float32` array (pixel values in `[0, 1]`),
          representing `(VAE - Original + 128) / 255`.

Returns:
- A `Float32` array representing the reconstructed VAE-decoded image,
  with pixel values clamped to the `[0, 1]` range.
"""
function reconstruct_vae_from_diff(original::Array{Float32}, diff::Array{Float32})::Array{Float32}
    # Convert from normalized [0,1] to [0,255] range for arithmetic
    orig_255 = original .* 255.0f0
    diff_255 = diff .* 255.0f0

    # Reconstruct: VAE = Original + Diff - 128
    vae_255 = orig_255 .+ diff_255 .- 128.0f0

    # Clamp and normalize back to [0, 1]
    return clamp.(vae_255 ./ 255.0f0, 0.0f0, 1.0f0)
end

"""
    CompressedVAEDataset

A structure representing a dataset for VAE artifact detection, specifically
designed to work with a compressed diff format. In this format, VAE-decoded
images are not stored directly but are reconstructed on-the-fly from
an original image and a "diff" image, significantly reducing storage requirements.

Fields:
- `ids::Vector{String}`: A vector of image IDs that constitute this dataset split.
- `original_dir::String`: The path to the directory containing the original images.
- `diff_dir::String`: The path to the directory containing the diff images.
- `transform::Function`: An optional function applied to each image after loading
                         and reconstruction. Defaults to `identity` (no transformation).
"""
struct CompressedVAEDataset
    ids::Vector{String}
    original_dir::String
    diff_dir::String
    transform::Function
end

"""
    CompressedVAEDataset(
        split_path::AbstractString,
        original_dir::AbstractString,
        diff_dir::AbstractString;
        transform::Function = identity
    )

Constructs a `CompressedVAEDataset` instance. This dataset reads image IDs
from a split file and then, for each ID, expects to find an original image
in `original_dir` and a corresponding diff image in `diff_dir`. VAE-decoded
images are reconstructed from these two components at retrieval time.

Arguments:
- `split_path`: An `AbstractString` path to a text file containing the IDs
                of image pairs to include in this dataset split.
- `original_dir`: An `AbstractString` path to the directory where original
                  image files are stored.
- `diff_dir`: An `AbstractString` path to the directory where diff image
              files are stored.

Keyword Arguments:
- `transform::Function`: A function to apply to each loaded and reconstructed
                         image (both original and VAE). This can be used for
                         preprocessing steps like resizing, normalization, or
                         data augmentation. Defaults to `identity`.

Returns:
- A `CompressedVAEDataset` instance, ready for iteration or indexing.

Storage Efficiency:
This dataset format is particularly useful for large VAE datasets where the
VAE-decoded images are very similar to their originals. Storing the diffs
(differences) typically requires less space than storing the full VAE images,
as diff images often have lower information entropy.
"""
function CompressedVAEDataset(
    split_path::AbstractString,
    original_dir::AbstractString,
    diff_dir::AbstractString;
    transform::Function = identity
)
    ids = load_split(split_path)
    return CompressedVAEDataset(ids, original_dir, diff_dir, transform)
end

"""
    Base.length(d::CompressedVAEDataset)

Returns the total number of image pairs (defined by IDs) in the `CompressedVAEDataset`.
This makes the dataset iterable and compatible with functions that rely on `length`.
"""
Base.length(d::CompressedVAEDataset) = length(d.ids)

"""
    Base.getindex(d::CompressedVAEDataset, i::Int)

Retrieves a single image pair (original and reconstructed VAE-decoded) from the dataset
at the specified index `i`. This method handles loading the original and diff images
from their respective directories, reconstructs the VAE image on-the-fly, and
applies any specified transformations.

Arguments:
- `d`: The `CompressedVAEDataset` instance.
- `i`: An integer index corresponding to the desired image pair within the dataset.

Returns:
- A `Tuple` containing two `Array{Float32}` images:
    1. The original image, processed by `d.transform`.
    2. The reconstructed VAE-decoded image, processed by `d.transform`.

Each image is loaded or reconstructed as a `Float32` array with pixel values
normalized to the `[0, 1]` range.

Image Loading Process:
1.  The `id` for the given index `i` is retrieved.
2.  The corresponding original image is searched for in `d.original_dir`,
    trying common image file extensions (`.png`, `.jpg`, `.jpeg`, `.webp`).
    An `error` is thrown if the original image cannot be found.
3.  The diff image is loaded from `d.diff_dir` (expected to be `.png`). An `error`
    is thrown if the diff image cannot be found.
4.  `reconstruct_vae_from_diff` is used to create the VAE-decoded image.
5.  Finally, `d.transform` is applied to both the original and reconstructed VAE images.
"""
function Base.getindex(d::CompressedVAEDataset, i::Int)
    id = d.ids[i]

    # Find the original image (try common extensions)
    orig_path = nothing
    # Diff image is always expected to be a PNG
    diff_path = joinpath(d.diff_dir, "$(id).png")

    for ext in ["png", "jpg", "jpeg", "webp"]
        candidate = joinpath(d.original_dir, "$(id).$(ext)")
        if isfile(candidate)
            orig_path = candidate
            break
        end
    end

    if orig_path === nothing
        error("Original image not found for ID: $id in $(d.original_dir) with common extensions.")
    end
    if !isfile(diff_path)
        error("Diff image not found for ID: $id at path: $diff_path.")
    end

    # Load images and convert to Float32 arrays
    orig_img = Float32.(channelview(load(orig_path)))
    diff_img = Float32.(channelview(load(diff_path)))

    # Reconstruct VAE image from original + diff
    vae_img = reconstruct_vae_from_diff(orig_img, diff_img)

    # Apply transform
    orig_img = d.transform(orig_img)
    vae_img = d.transform(vae_img)

    # Return (original, vae) pair
    return (orig_img, vae_img)
end

"""
    DataLoader{T}

A generic data loader structure that wraps any Julia dataset (`T`) and provides
an iterable interface for generating mini-batches. This is essential for
training deep learning models efficiently.

Fields:
- `dataset::T`: The underlying dataset from which to load samples.
                 This can be `VAEDetectorDataset` or `CompressedVAEDataset`
                 (or any custom type that implements `Base.length` and `Base.getindex`).
- `batch_size::Int`: The number of samples to include in each mini-batch.
- `shuffle::Bool`: A boolean indicating whether to shuffle the dataset indices
                   at the beginning of each epoch.
- `indices::Vector{Int}`: An internal vector of indices used to track the order
                           of samples in the dataset for batching.
"""
struct DataLoader{T}
    dataset::T
    batch_size::Int
    shuffle::Bool
    indices::Vector{Int}
end

"""
    DataLoader(dataset; batch_size::Int=32, shuffle::Bool=true)

Constructs a `DataLoader` instance from a given dataset.

Arguments:
- `dataset`: The dataset to be loaded. It must implement `Base.length` and `Base.getindex`.

Keyword Arguments:
- `batch_size::Int`: The number of samples per batch. Defaults to `32`.
- `shuffle::Bool`: If `true`, the dataset indices will be shuffled at the start
                   of each iteration (epoch). Defaults to `true`.

Returns:
- A `DataLoader` instance that can be iterated over to yield batches of data.

Batch Format:
Each iteration of the `DataLoader` yields a tuple `(x, y)`, where:
- `x`: A concatenated `Array{Float32}` of original and VAE images from the batch,
       typically of shape `(channels, height, width, 2 * batch_size)`.
- `y`: A `Vector{Float32}` of binary labels, with `0.0f0` for original images
       and `1.0f0` for VAE images. The length of `y` will be `2 * batch_size`.
"""
function DataLoader(dataset; batch_size::Int=32, shuffle::Bool=true)
    indices = collect(1:length(dataset))
    if shuffle
        shuffle!(indices)
    end
    return DataLoader(dataset, batch_size, shuffle, indices)
end

"""
    Base.iterate(dl::DataLoader, state::Int=1)

Implements the iteration protocol for `DataLoader`, allowing it to be used
in `for` loops to yield mini-batches of data.

Arguments:
- `dl`: The `DataLoader` instance.
- `state`: An integer representing the current starting index for the batch.
           Defaults to `1` for the first iteration.

Returns:
- A `Tuple` `((x, y), next_state)` where:
    - `x`: An `Array{Float32}` containing concatenated original and VAE images
           for the current batch. The dimensions will be
           `(channels, height, width, 2 * batch_size)`.
    - `y`: A `Vector{Float32}` containing binary labels for `x`, where
           `0.0f0` corresponds to original images and `1.0f0` to VAE images.
    - `next_state`: The starting index for the next batch.
- `nothing` if all batches have been yielded.

Batch Creation Process:
1.  For each `ImagePair` in the current batch:
    a.  The original image is loaded/reconstructed and transformed.
    b.  The VAE image is loaded/reconstructed and transformed.
2.  All original images in the batch are concatenated, and all VAE images
    in the batch are concatenated.
3.  These two concatenated arrays are then joined (concatenated) along the
    batch dimension (typically the last dimension), effectively creating a
    batch where half the samples are original and half are VAE.
4.  Corresponding labels (`0.0f0` for original, `1.0f0` for VAE) are generated.
"""
function Base.iterate(dl::DataLoader, state::Int=1)
    if state > length(dl.indices)
        # If all indices have been processed, reset for the next epoch if shuffling
        if dl.shuffle
            shuffle!(dl.indices)
        end
        return nothing
    end

    batch_end = min(state + dl.batch_size - 1, length(dl.indices))
    batch_indices = dl.indices[state:batch_end]

    # Collect batch of (original, vae) pairs from the dataset
    orig_batch = []
    vae_batch = []
    for i in batch_indices
        orig, vae = dl.dataset[i]
        push!(orig_batch, orig)
        push!(vae_batch, vae)
    end

    # Stack into arrays - combine originals and VAEs with labels
    # x: images (channels, height, width, 2*n)
    # y: labels (0 for original, 1 for VAE)
    n = length(batch_indices) # Actual number of pairs in this batch
    x = cat(orig_batch..., vae_batch..., dims=4) # Concatenate along a new 4th dimension
    y = vcat(zeros(Float32, n), ones(Float32, n)) # Labels for the concatenated batch

    return ((x, y), batch_end + 1)
end

"""
    Base.length(dl::DataLoader)

Returns the total number of batches that the `DataLoader` will yield
per full iteration (epoch).

Arguments:
- `dl`: The `DataLoader` instance.

Returns:
- An `Int` representing the number of batches.
"""
Base.length(dl::DataLoader) = ceil(Int, length(dl.dataset) / dl.batch_size)

"""
    accuracy(model, x, y)

Calculates the accuracy of a binary classification model's predictions
against true labels. This is a utility function typically used during
training or evaluation to monitor model performance.

Arguments:
- `model`: A trained classification model (e.g., `VAEArtifactClassifier`)
           that can take input `x` and produce a prediction (probability score).
- `x`: The input data batch (e.g., image batch).
- `y`: The true binary labels corresponding to the input `x`.

Returns:
- A `Float32` value representing the accuracy, which is the proportion of
  correct predictions (where predicted label matches true label).
  Predictions are converted to binary labels by thresholding at `0.5f0`.
"""
function accuracy(model, x, y)
    predictions = model(x)
    pred_labels = predictions .> 0.5f0
    return mean(pred_labels .== y)
end

"""
    train_epoch!(model, opt_state, train_loader, loss_fn)

Performs a single training epoch for a Flux.jl model. This function iterates
through all batches in the `train_loader`, computes the loss, calculates
gradients, and updates the model's parameters using the provided optimizer state.

Arguments:
- `model`: The Flux.jl model to be trained.
- `opt_state`: The state of the optimizer (e.g., from `Flux.setup(Adam(...), model)`).
- `train_loader`: A `DataLoader` instance yielding batches of `(x, y)` for training.
- `loss_fn`: A loss function (e.g., `Flux.binarycrossentropy`, `NTXentLoss`)
             that takes the model's predictions and true labels as input
             and returns a scalar loss value.

Returns:
- A `Float32` value representing the average loss over all batches in the epoch.

Process:
1.  Iterates over each `(x, y)` batch from the `train_loader`.
2.  Computes the loss for the current batch using `loss_fn`.
3.  Calculates gradients of the loss with respect to the model's parameters
    using `Flux.gradient`.
4.  Updates the model's parameters using `Flux.update!` and the `opt_state`.
5.  Accumulates the loss and returns the average loss for the epoch.
"""
function train_epoch!(model, opt_state, train_loader, loss_fn)
    total_loss = 0.0f0
    n_batches = 0

    for (x, y) in train_loader
        # Compute gradients and update
        grads = Flux.gradient(model) do m
            loss_fn(m(x), y)
        end
        Flux.update!(opt_state, model, grads[1])

        total_loss += loss_fn(model(x), y)
        n_batches += 1
    end

    return total_loss / n_batches
end

"""
    compare_splits(
        manifest_path::AbstractString,
        random_split_dir::AbstractString,
        stratified_split_dir::AbstractString
    )

Compares two different dataset splitting strategies (e.g., random vs. stratified)
to assess their characteristics and potential biases. This function is useful
for understanding how different splitting methods distribute samples and
strata across training, testing, validation, and calibration sets.

Arguments:
- `manifest_path`: An `AbstractString` path to the CSV manifest file, which
                   should contain the `stratum` information for each image ID.
- `random_split_dir`: An `AbstractString` path to the directory containing
                      split files generated by a random splitting strategy.
                      Expected files are `random_train.txt`, `random_test.txt`,
                      `random_val.txt`, `random_calibration.txt`.
- `stratified_split_dir`: An `AbstractString` path to the directory containing
                          split files generated by a stratified splitting strategy.
                          Expected files are `stratified_train.txt`, `stratified_test.txt`,
                          `stratified_val.txt`, `stratified_calibration.txt`.

Returns:
- `results::Dict{String, Dict{String, Any}}`: A nested dictionary where the
  top-level keys are the split names (`"train"`, `"test"`, `"val"`, `"calibration"`).
  Each inner dictionary contains:
    - `"random_count"`: Number of samples in the random split.
    - `"stratified_count"`: Number of samples in the stratified split.
    - `"random_stratum_dist"`: A `Dict` mapping stratum ID to count for the random split.
    - `"stratified_stratum_dist"`: A `Dict` mapping stratum ID to count for the stratified split.
    - `"overlap"`: Number of common image IDs between the random and stratified versions of that split.
"""
function compare_splits(
    manifest_path::AbstractString,
    random_split_dir::AbstractString,
    stratified_split_dir::AbstractString
)
    manifest = load_manifest(manifest_path)

    results = Dict{String, Dict{String, Any}}()

    for split_name in ["train", "test", "val", "calibration"]
        # Load IDs for both random and stratified splits
        random_ids = Set(load_split(joinpath(random_split_dir, "random_$(split_name).txt")))
        strat_ids = Set(load_split(joinpath(stratified_split_dir, "stratified_$(split_name).txt")))

        # Calculate stratum distribution for each split type
        random_strata = [row.stratum for row in eachrow(manifest) if row.id in random_ids]
        strat_strata = [row.stratum for row in eachrow(manifest) if row.id in strat_ids]

        results[split_name] = Dict(
            "random_count" => length(random_ids),
            "stratified_count" => length(strat_ids),
            "random_stratum_dist" => countmap(random_strata),
            "stratified_stratum_dist" => countmap(strat_strata),
            "overlap" => length(intersect(random_ids, strat_ids)) # Overlap in IDs between the two split types
        )
    end

    return results
end

"""
    countmap(xs)

A simple utility function that counts the occurrences of each unique element
in a collection `xs`. It returns a dictionary where keys are the unique
elements and values are their respective counts. This provides similar
functionality to `StatsBase.countmap` but is implemented here directly
to avoid an additional dependency or for simplicity in this context.

Arguments:
- `xs`: A collection (e.g., `Vector`, `Array`) of elements.

Returns:
- `counts::Dict{Any, Int}`: A dictionary where keys are the unique elements
                            from `xs` and values are their corresponding counts.
"""
function countmap(xs)
    counts = Dict{Any, Int}()
    for x in xs
        counts[x] = get(counts, x, 0) + 1
    end
    return counts
end

end # module

#=
Example usage with Flux.jl:

using Flux
include("julia_utils.jl")
using .VAEDatasetUtils

# Load dataset
train_data = VAEDetectorDataset(
    "splits/random_train.txt",
    "manifest.csv",
    "/path/to/vae-dataset"
)

# Create data loader
train_loader = DataLoader(train_data, batch_size=32, shuffle=true)

# Simple binary classifier
model = Chain(
    Conv((3, 3), 3 => 32, relu),
    MaxPool((2, 2)),
    Conv((3, 3), 32 => 64, relu),
    MaxPool((2, 2)),
    Flux.flatten,
    Dense(64 * 190 * 190, 128, relu),
    Dense(128, 1, sigmoid)
)

# Training
opt = Flux.setup(Adam(1e-4), model)
loss_fn = Flux.binarycrossentropy

for epoch in 1:10
    loss = train_epoch!(model, opt, train_loader, loss_fn)
    println("Epoch $epoch: loss = $loss")
end


# ============================================
# Example usage with COMPRESSED diff format:
# ============================================

# Load compressed dataset (Original/ + Diff/ directories)
# VAE images are reconstructed on-the-fly from diffs
compressed_data = CompressedVAEDataset(
    "splits/random_train.txt",
    "/path/to/compressed-dataset/Original",
    "/path/to/compressed-dataset/Diff"
)

# Create data loader (same API as regular dataset)
compressed_loader = DataLoader(compressed_data, batch_size=32, shuffle=true)

# Training works identically - VAE images are reconstructed automatically
for epoch in 1:10
    loss = train_epoch!(model, opt, compressed_loader, loss_fn)
    println("Epoch $epoch: loss = $loss")
end
=#
