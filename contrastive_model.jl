# SPDX-FileCopyrightText: 2024 Joshua Jewell
# SPDX-License-Identifier: MIT

#=
VAEContrastive Module: Contrastive Learning for VAE Artifact Detection
=====================================================================

This module implements a contrastive learning framework to detect and analyze
artifacts introduced by Variational Autoencoders (VAEs) in images. The core idea
is to train a neural network (an encoder) to learn robust representations of
images such that it can effectively distinguish between original images and their
VAE-decoded counterparts. This approach leverages contrastive loss functions to
maximize the similarity between "positive" pairs (e.g., original and VAE-decoded images
derived from the same source, or different augmentations of the same image) and
minimize similarity between "negative" pairs (images from different sources or
different VAE processing).

The `zerostep` project aims to test whether VAE artifacts are
detectable by neural networks, providing insights into the quality and fidelity
of VAE generative processes.

Theoretical Foundations:
------------------------
The system is built upon modern contrastive learning principles, drawing inspiration
from methods like SimCLR and supervised contrastive learning. It employs an encoder-projector
architecture where the encoder extracts high-level features, and the projector maps these
features to a lower-dimensional space optimized for contrastive loss.

Methodology for VAE Artifact Detection:
--------------------------------------
1.  **Data Preparation:** Datasets typically consist of pairs of original images
    and their VAE-decoded versions. The `VAEDatasetUtils` module (from `julia_utils.jl`)
    is used to efficiently load and preprocess these image pairs, including support
    for compressed diff formats and checksum verification.
2.  **Contrastive Pre-training:** A `ContrastiveModel` (comprising an `ContrastiveEncoder`
    and `ProjectionHead`) is trained using one of several contrastive loss functions
    (`NTXentLoss`, `SupervisedContrastiveLoss`, `TripletLoss`, `ContrastiveLoss`).
    During this phase, the model learns a feature space where original and VAE-decoded
    images from the same source are considered similar (positive pairs), while images
    from different sources or of different types (original vs. VAE from unrelated source)
    are dissimilar (negative pairs).
3.  **Classifier Fine-tuning:** After pre-training, the `ContrastiveEncoder`'s learned
    representations are used as a backbone for a `VAEArtifactClassifier`. A small
    classification head is fine-tuned on top of the encoder (optionally freezing the
    encoder) to perform binary classification: predicting whether an input image is
    an original or a VAE-decoded artifact.
4.  **Evaluation:** The performance of the classifier is rigorously evaluated using
    metrics such as accuracy, precision, recall, F1 score, and confusion matrices.
5.  **Embedding Analysis:** Embeddings extracted from the `ContrastiveEncoder` can be
    analyzed (e.g., using t-SNE or UMAP) to visualize the learned feature space and
    gain insights into how the model distinguishes between image types.

Usage:
------
This module can be executed directly from the command line for training, evaluation,
or embedding extraction, as shown in the `main` script block at the end of this file.

Examples:
```julia
# Train a model
julia --project contrastive_model.jl train --dataset /path/to/dataset --output /path/to/output

# Evaluate a pre-trained model
julia --project contrastive_model.jl evaluate --model /path/to/model.bson --dataset /path/to/dataset

# Extract embeddings for visualization
julia --project contrastive_model.jl embed --model /path/to/model.bson --dataset /path/to/dataset
```

Requires: Flux.jl, CUDA.jl (optional), Images.jl, BSON.jl, ArgParse.jl (for CLI)
=#

module VAEContrastive


using Flux
using Flux: onehotbatch, onecold, crossentropy, logitcrossentropy
using Flux.Losses: mse
using Statistics
using LinearAlgebra
using Random
using Printf
using Dates

# Try to load CUDA for GPU support
const USE_CUDA = try
    using CUDA
    CUDA.functional()
catch
    false
end

export ContrastiveEncoder, ProjectionHead, ContrastiveModel
export NTXentLoss, TripletLoss, ContrastiveLoss
export train_contrastive!, evaluate_model, extract_embeddings
export VAEArtifactClassifier, train_classifier!, predict

#=============================================================================
                           Model Architecture
=============================================================================#

"""
    ContrastiveEncoder(; in_channels::Int=3, embed_dim::Int=256)

Constructs a Convolutional Neural Network (CNN) encoder designed to extract
rich image features for contrastive learning. The architecture is inspired
by ResNet, incorporating residual connections to facilitate learning in
deeper networks and improve gradient flow.

The encoder processes an input image through a series of convolutional layers,
batch normalization, activation functions (ReLU), and pooling operations,
gradually reducing spatial dimensions while increasing feature depth.

Architecture Breakdown:
1.  **Initial Block:**
    *   `Conv((7, 7), in_channels => 64, pad=3, stride=2)`: A large initial
        convolutional layer to capture broad features, with a stride of 2 for
        initial downsampling.
    *   `BatchNorm(64)`: Normalizes activations for stability.
    *   `relu`: Rectified Linear Unit activation.
    *   `MaxPool((3, 3), pad=1, stride=2)`: Further downsamples the spatial
        dimensions.
2.  **Residual Blocks (3 stages):** Each stage consists of:
    *   `SkipConnection`: Implements a residual connection, adding the input
        of the block to its output, helping to mitigate vanishing gradients.
    *   Two `Conv((3, 3), ...)` layers: Standard 3x3 convolutions for feature
        extraction.
    *   `BatchNorm`: After each convolution.
    *   `relu`: After each batch normalization.
    *   The second and third residual blocks include an initial `Conv` with
        `stride=2` to perform downsampling, increasing the feature map depth
        (e.g., 64 -> 128, 128 -> 256 channels).
3.  **Global Average Pooling:** `GlobalMeanPool()`: Reduces each feature map
    to a single value, effectively summarizing the features across spatial
    dimensions.
4.  **Flatten:** `Flux.flatten`: Converts the pooled feature vector into a
    1D array.
5.  **Embedding Layer:** `Dense(256, embed_dim)`: A fully connected layer that
    projects the extracted features into the final `embed_dim`-dimensional
    embedding space.
    *   `relu`: Activation function for the embedding layer.

Arguments:
- `in_channels`: The number of input channels in the image (e.g., `3` for RGB,
                 `1` for grayscale). Defaults to `3`.
- `embed_dim`: The dimensionality of the final feature embedding vector.
               Defaults to `256`.

Returns:
- A `Flux.Chain` object representing the convolutional encoder.
"""
function ContrastiveEncoder(; in_channels::Int=3, embed_dim::Int=256)
    return Chain(
        # Initial convolution
        Conv((7, 7), in_channels => 64, pad=3, stride=2),
        BatchNorm(64),
        x -> relu.(x),
        MaxPool((3, 3), pad=1, stride=2),

        # Residual block 1
        SkipConnection(
            Chain(
                Conv((3, 3), 64 => 64, pad=1),
                BatchNorm(64),
                x -> relu.(x),
                Conv((3, 3), 64 => 64, pad=1),
                BatchNorm(64)
            ),
            +
        ),
        x -> relu.(x),

        # Residual block 2 with downsampling
        Conv((3, 3), 64 => 128, pad=1, stride=2),
        BatchNorm(128),
        x -> relu.(x),
        SkipConnection(
            Chain(
                Conv((3, 3), 128 => 128, pad=1),
                BatchNorm(128),
                x -> relu.(x),
                Conv((3, 3), 128 => 128, pad=1),
                BatchNorm(128)
            ),
            +
        ),
        x -> relu.(x),

        # Residual block 3 with downsampling
        Conv((3, 3), 128 => 256, pad=1, stride=2),
        BatchNorm(256),
        x -> relu.(x),
        SkipConnection(
            Chain(
                Conv((3, 3), 256 => 256, pad=1),
                BatchNorm(256),
                x -> relu.(x),
                Conv((3, 3), 256 => 256, pad=1),
                BatchNorm(256)
            ),
            +
        ),
        x -> relu.(x),

        # Global average pooling
        GlobalMeanPool(),
        Flux.flatten,

        # Embedding layer
        Dense(256, embed_dim),
        x -> relu.(x)
    )
end

"""
    ProjectionHead(; embed_dim::Int=256, proj_dim::Int=128)

Constructs a projection head, a critical component in many contrastive learning
frameworks (e.g., SimCLR). Its purpose is to map the high-dimensional embeddings
produced by the `ContrastiveEncoder` into a lower-dimensional latent space
where the contrastive loss is actually computed.

This separation is often beneficial because:
1.  The `encoder` learns robust feature representations for various downstream tasks.
2.  The `projection head` transforms these representations into a space
    specifically optimized for the contrastive objective, which might discard
    some information unnecessary for the contrastive task but valuable for
    other tasks.

Architecture Breakdown:
The projection head typically consists of a small, non-linear multi-layer
perceptron (MLP). In this implementation, it is a 2-layer MLP:
1.  `Dense(embed_dim, embed_dim)`: A fully connected layer that maps the
    input embeddings to a hidden layer of the same `embed_dim` size.
2.  `relu`: A Rectified Linear Unit activation function introduces non-linearity.
3.  `Dense(embed_dim, proj_dim)`: A final fully connected layer that projects
    the hidden representation into the desired `proj_dim`-dimensional output space.

Arguments:
- `embed_dim`: The dimensionality of the input embeddings from the `ContrastiveEncoder`.
               Defaults to `256`.
- `proj_dim`: The dimensionality of the output projection space where contrastive
              loss is applied. Defaults to `128`.

Returns:
- A `Flux.Chain` object representing the projection head.
"""
function ProjectionHead(; embed_dim::Int=256, proj_dim::Int=128)
    return Chain(
        Dense(embed_dim, embed_dim),
        x -> relu.(x),
        Dense(embed_dim, proj_dim)
    )
end

"""
    ContrastiveModel

A composite model that combines the `ContrastiveEncoder` and `ProjectionHead`
into a single structure for contrastive learning. This model takes an image
as input and outputs both the raw embeddings (from the encoder) and the
projected representations (from the projection head).

Fields:
- `encoder::Chain`: The `ContrastiveEncoder` backbone that extracts robust
                     feature representations from input images.
- `projector::Chain`: The `ProjectionHead` that maps the encoder's output
                      into a lower-dimensional space optimized for contrastive
                      loss calculations.
"""
struct ContrastiveModel
    encoder::Chain
    projector::Chain
end

Flux.@functor ContrastiveModel

"""
    ContrastiveModel(; in_channels::Int=3, embed_dim::Int=256, proj_dim::Int=128)

Constructs a `ContrastiveModel` by initializing its `ContrastiveEncoder` and
`ProjectionHead` components.

Arguments:
- `in_channels`: The number of input channels for the images, passed directly
                 to the `ContrastiveEncoder`. Defaults to `3`.
- `embed_dim`: The dimensionality of the feature embeddings from the encoder,
               used by both `ContrastiveEncoder` and `ProjectionHead`.
               Defaults to `256`.
- `proj_dim`: The dimensionality of the projection head's output, passed
              directly to the `ProjectionHead`. Defaults to `128`.

Returns:
- A `ContrastiveModel` instance ready for use in contrastive learning.
"""
function ContrastiveModel(; in_channels::Int=3, embed_dim::Int=256, proj_dim::Int=128)
    encoder = ContrastiveEncoder(in_channels=in_channels, embed_dim=embed_dim)
    projector = ProjectionHead(embed_dim=embed_dim, proj_dim=proj_dim)
    return ContrastiveModel(encoder, projector)
end

"""
    (model::ContrastiveModel)(x)

Performs a forward pass through the complete `ContrastiveModel`.
This function first passes the input `x` through the `encoder` to obtain
feature embeddings, and then passes these embeddings through the `projector`
to obtain projected representations.

Arguments:
- `model`: The `ContrastiveModel` instance.
- `x`: The input data (e.g., an image batch) to be processed by the model.

Returns:
- A `Tuple` containing two elements:
    1. `embeddings`: The feature embeddings from the `ContrastiveEncoder`.
                      These are typically higher-dimensional and suitable for
                      various downstream tasks.
    2. `projections`: The lower-dimensional projected representations from the
                      `ProjectionHead`. These are specifically optimized for
                      contrastive loss calculation.
"""
function (model::ContrastiveModel)(x)
    embeddings = model.encoder(x)
    projections = model.projector(embeddings)
    return embeddings, projections
end

"""
    get_embeddings(model::ContrastiveModel, x)

Extracts only the feature embeddings from the `ContrastiveEncoder` of the
`ContrastiveModel`. This function is useful when the raw, high-dimensional
features are needed for downstream tasks (e.g., classification, clustering)
after the contrastive model has been pre-trained.

Arguments:
- `model`: The `ContrastiveModel` instance.
- `x`: The input data (e.g., an image batch) to be processed by the encoder.

Returns:
- `embeddings`: The feature embeddings from the `ContrastiveEncoder`.
"""
function get_embeddings(model::ContrastiveModel, x)
    return model.encoder(x)
end

#=============================================================================
                           Loss Functions
=============================================================================#

"""
    NTXentLoss(z1, z2; temperature::Float32=0.5f0)

Calculates the Normalized Temperature-scaled Cross Entropy (NT-Xent) Loss,
a widely used contrastive loss in self-supervised learning, particularly
popularized by SimCLR. This loss function aims to pull positive pairs (similar
samples) closer together in the embedding space while pushing negative pairs
(dissimilar samples) further apart.

In the context of VAE artifact detection, positive pairs are typically derived
from the same image source (e.g., two augmented views of an original image, or
two augmented views of a VAE-decoded image). Negative pairs are samples from
different image sources or different augmentations that are not considered
similar.

For a given anchor sample `i`, its positive pair is `i+batch_size` (and vice-versa).
All other `2*batch_size - 2` samples in the concatenated batch are considered
negative samples.

The loss for an anchor `i` with its positive `p` is defined as:
```
-log( exp(sim(z_i, z_p) / τ) / Σ_{k≠i} exp(sim(z_i, z_k) / τ) )
```
where:
- `sim(u, v)` is the cosine similarity between vectors `u` and `v`.
- `τ` (temperature) is a scalar hyperparameter that scales the logits before
  applying the softmax, influencing the sharpness of the probability distribution
  and how strongly negative samples are penalized. A smaller `τ` makes the model
  focus more on distinguishing hard negatives.
- The sum `Σ_{k≠i}` is over all other `2*batch_size - 2` samples in the batch,
  excluding the anchor itself.

The total NT-Xent loss is the average of this negative log-likelihood across
all samples in both `z1` and `z2` (treating each as an anchor once).

Arguments:
- `z1`: A `proj_dim x batch_size` matrix of normalized projection vectors
        from the first set of augmented samples.
- `z2`: A `proj_dim x batch_size` matrix of normalized projection vectors
        from the second set of augmented samples.
- `temperature`: A `Float32` scalar controlling the temperature scaling.
                 Defaults to `0.5f0`.

Returns:
- A `Float32` scalar representing the average NT-Xent loss for the batch.
"""
function NTXentLoss(z1, z2; temperature::Float32=0.5f0)

    # L2 normalize
    z1_norm = z1 ./ sqrt.(sum(z1.^2, dims=1) .+ 1f-8)
    z2_norm = z2 ./ sqrt.(sum(z2.^2, dims=1) .+ 1f-8)

    batch_size = size(z1, 2)

    # Concatenate all projections
    z = hcat(z1_norm, z2_norm)  # [proj_dim, 2*batch_size]

    # Compute similarity matrix
    sim_matrix = z' * z ./ temperature  # [2*batch_size, 2*batch_size]

    # Mask out self-similarity (diagonal)
    mask = 1f0 .- Float32.(I(2 * batch_size))
    sim_matrix = sim_matrix .* mask .- 1f10 .* (1f0 .- mask)

    # Create positive pair indices
    # For i in 1:batch_size, positive is at i+batch_size and vice versa
    labels = vcat(collect(batch_size+1:2*batch_size), collect(1:batch_size))

    # Cross entropy loss
    loss = 0f0
    for i in 1:2*batch_size
        exp_sims = exp.(sim_matrix[i, :])
        loss -= log(exp_sims[labels[i]] / sum(exp_sims))
    end

    return loss / (2 * batch_size)
end

"""
    SupervisedContrastiveLoss(z, labels; temperature::Float32=0.1f0)

Calculates the Supervised Contrastive Loss, which extends the idea of
contrastive learning to a supervised setting. Instead of relying on data
augmentations to form positive pairs, it uses class labels: samples from the
same class are treated as positive pairs, while samples from different classes
are negative pairs. This loss function encourages samples from the same class
to cluster together in the embedding space, and samples from different classes
to be far apart.

The loss for an anchor sample `i` is defined as:
```
L_i = - Σ_{p ∈ P(i)} [log( exp(sim(z_i, z_p) / τ) / (Σ_{k ≠ i} exp(sim(z_i, z_k) / τ)) )] / |P(i)|
```
where:
- `sim(u, v)` is the cosine similarity between vectors `u` and `v`.
- `τ` (temperature) is a scalar hyperparameter, similar to NT-Xent Loss.
- `P(i)` is the set of indices of all other samples in the batch that belong
  to the same class as sample `i`.
- `|P(i)|` is the cardinality of the set `P(i)`.
- The sum `Σ_{k ≠ i}` is over all other samples in the batch, excluding the anchor itself.

The total supervised contrastive loss is the average of `L_i` for all samples
`i` in the batch that have at least one positive pair.

Arguments:
- `z`: A `proj_dim x batch_size` matrix of normalized projection vectors.
- `labels`: A vector of binary labels (e.g., `0` for original, `1` for VAE)
            corresponding to the samples in `z`.
- `temperature`: A `Float32` scalar controlling the temperature scaling.
                 Defaults to `0.1f0`.

Returns:
- A `Float32` scalar representing the average Supervised Contrastive Loss for the batch.
"""
function SupervisedContrastiveLoss(z, labels; temperature::Float32=0.1f0)
    # L2 normalize
    z_norm = z ./ sqrt.(sum(z.^2, dims=1) .+ 1f-8)

    batch_size = size(z, 2)

    # Similarity matrix
    sim_matrix = z_norm' * z_norm ./ temperature

    # Create mask for positive pairs (same label)
    label_matrix = labels .== labels'

    # Remove diagonal (self-similarity)
    diag_mask = .!Bool.(I(batch_size))
    positive_mask = label_matrix .& diag_mask

    loss = 0f0
    for i in 1:batch_size
        pos_indices = findall(positive_mask[i, :])
        if !isempty(pos_indices)
            exp_sims = exp.(sim_matrix[i, :])
            # Exclude self
            exp_sims[i] = 0f0
            denominator = sum(exp_sims)

            for j in pos_indices
                loss -= log(exp.(sim_matrix[i, j]) / (denominator + 1f-8))
            end
            loss /= length(pos_indices)
        end
    end

    return loss / batch_size
end

"""
    TripletLoss(embeddings, labels; margin::Float32=1.0f0)

Calculates the Triplet Loss, a metric learning loss function that aims to
ensure that an anchor sample (`a`) is closer to a positive sample (`p`) than
it is to a negative sample (`n`) by at least a certain `margin`. This is
typically expressed as:

`distance(a, p) + margin < distance(a, n)`

The objective of the loss function for a given triplet (anchor, positive, negative) is:
```
L(a, p, n) = max(0, distance(a, p) - distance(a, n) + margin)
```
where `distance` is a similarity metric (e.g., Euclidean distance).

This implementation incorporates **hard negative mining**, which means for each
(anchor, positive) pair, it selects the "hardest" negative example (the one
that is closest to the anchor but belongs to a different class) to compute the loss.
This strategy helps the model learn more effectively from challenging examples.

Arguments:
- `embeddings`: A `embed_dim x batch_size` matrix of embedding vectors. These
                embeddings are typically the output of the encoder before the
                projection head.
- `labels`: A vector of binary labels (e.g., `0` for original, `1` for VAE)
            corresponding to the samples in `embeddings`.
- `margin`: A `Float32` scalar that defines the minimum distance required
            between an anchor-negative pair and an anchor-positive pair.
            Defaults to `1.0f0`.

Returns:
- A `Float32` scalar representing the average Triplet Loss for the batch,
  calculated only for triplets where positive and negative samples could be found.
"""
function TripletLoss(embeddings, labels; margin::Float32=1.0f0)
    batch_size = size(embeddings, 2)

    # Compute pairwise distances
    # dist[i,j] = ||emb_i - emb_j||^2
    sq_norms = sum(embeddings.^2, dims=1)
    distances = sq_norms .+ sq_norms' .- 2 .* (embeddings' * embeddings)
    distances = sqrt.(max.(distances, 0f0))

    loss = 0f0
    n_triplets = 0

    for i in 1:batch_size
        # Find positive (same class) and negative (different class)
        pos_indices = findall(labels .== labels[i])
        neg_indices = findall(labels .!= labels[i])

        filter!(x -> x != i, pos_indices)

        if !isempty(pos_indices) && !isempty(neg_indices)
            # Hard positive: furthest same-class sample
            pos_dists = [distances[i, j] for j in pos_indices]
            hard_pos_idx = pos_indices[argmax(pos_dists)]

            # Hard negative: closest different-class sample
            neg_dists = [distances[i, j] for j in neg_indices]
            hard_neg_idx = neg_indices[argmin(neg_dists)]

            # Triplet loss
            loss += max(0f0, distances[i, hard_pos_idx] - distances[i, hard_neg_idx] + margin)
            n_triplets += 1
        end
    end

    return n_triplets > 0 ? loss / n_triplets : 0f0
end

"""
    ContrastiveLoss(embeddings, labels; margin::Float32=2.0f0)

Calculates a simple binary contrastive loss. This loss function aims to:
1. Minimize the distance between pairs of samples that belong to the same class.
2. Maximize the distance between pairs of samples that belong to different classes,
   up to a certain `margin`.

The loss for a pair of samples (`i`, `j`) is defined as:
```
L(i, j) =
  (1/2) * distance(i, j)^2                     if labels[i] == labels[j] (same class)
  (1/2) * max(0, margin - distance(i, j))^2    if labels[i] != labels[j] (different class)
```
where `distance` is typically the Euclidean distance.

This loss is applied to all possible pairs within a batch. The `margin` parameter
is crucial for dissimilar pairs; if the distance between them is greater than
the `margin`, no penalty is applied, allowing the model to focus on harder
negative examples.

Arguments:
- `embeddings`: A `embed_dim x batch_size` matrix of embedding vectors.
- `labels`: A vector of binary labels (e.g., `0` for original, `1` for VAE)
            corresponding to the samples in `embeddings`.
- `margin`: A `Float32` scalar that defines the desired minimum distance between
            dissimilar samples. Defaults to `2.0f0`.

Returns:
- A `Float32` scalar representing the average Contrastive Loss for the batch.
"""
function ContrastiveLoss(embeddings, labels; margin::Float32=2.0f0)
    batch_size = size(embeddings, 2)

    # Compute pairwise distances
    sq_norms = sum(embeddings.^2, dims=1)
    distances = sq_norms .+ sq_norms' .- 2 .* (embeddings' * embeddings)
    distances = sqrt.(max.(distances, 0f0))

    loss = 0f0
    n_pairs = 0

    for i in 1:batch_size
        for j in (i+1):batch_size
            d = distances[i, j]
            if labels[i] == labels[j]
                # Same class: minimize distance
                loss += d^2
            else
                # Different class: maximize distance (up to margin)
                loss += max(0f0, margin - d)^2
            end
            n_pairs += 1
        end
    end

    return loss / max(n_pairs, 1)
end

#=============================================================================
                           Classifier Head
=============================================================================#

"""
    VAEArtifactClassifier

A specialized binary classifier designed for detecting VAE artifacts.
This model leverages a pre-trained `ContrastiveEncoder` as its feature
extraction backbone and adds a trainable classification head on top.

Fields:
- `encoder::Chain`: The `ContrastiveEncoder` (or any `Flux.Chain` serving as an encoder)
                     that extracts feature embeddings from input images.
                     This encoder is typically pre-trained via contrastive learning.
- `classifier::Chain`: A small, fully connected neural network that takes
                       the encoder's embeddings as input and outputs a
                       binary classification score (e.g., probability of being a VAE artifact).
"""
struct VAEArtifactClassifier
    encoder::Chain
    classifier::Chain
end

Flux.@functor VAEArtifactClassifier

"""
    VAEArtifactClassifier(encoder::Chain; embed_dim::Int=256, freeze_encoder::Bool=false)

Constructs a `VAEArtifactClassifier` with a given `encoder` backbone and
initializes a new classification head.

The classification head is a multi-layer perceptron (MLP) with dropout for
regularization, designed to map the `embed_dim` features from the encoder
to a single binary output (via sigmoid activation).

Classification Head Architecture:
1.  `Dense(embed_dim, 128)`, `relu`: First hidden layer.
2.  `Dropout(0.5)`: Dropout layer for regularization.
3.  `Dense(128, 64)`, `relu`: Second hidden layer.
4.  `Dropout(0.3)`: Dropout layer.
5.  `Dense(64, 1)`, `sigmoid`: Output layer, producing a probability score between 0 and 1.

Arguments:
- `encoder`: The `Flux.Chain` instance to be used as the feature extractor.
- `embed_dim`: The dimensionality of the embeddings produced by the `encoder`.
               Defaults to `256`.
- `freeze_encoder`: A `Bool` flag. If `true`, the `encoder`'s parameters
                    will be marked as non-trainable, allowing only the
                    `classifier` head to be updated during training. This is
                    common in transfer learning scenarios. Defaults to `false`.

Returns:
- A `VAEArtifactClassifier` instance.
"""
function VAEArtifactClassifier(encoder::Chain; embed_dim::Int=256, freeze_encoder::Bool=false)
    classifier = Chain(
        Dense(embed_dim, 128),
        x -> relu.(x),
        Dropout(0.5),
        Dense(128, 64),
        x -> relu.(x),
        Dropout(0.3),
        Dense(64, 1),
        x -> sigmoid.(x)
    )

    if freeze_encoder
        # Mark encoder as non-trainable using Flux.freeze
        encoder = Flux.freeze(encoder)
    end

    return VAEArtifactClassifier(encoder, classifier)
end

"""
    (model::VAEArtifactClassifier)(x)

Performs a forward pass through the `VAEArtifactClassifier`.
The input `x` is first processed by the `encoder` to extract embeddings,
which are then passed to the `classifier` head to produce a binary prediction.

Arguments:
- `model`: The `VAEArtifactClassifier` instance.
- `x`: The input data (e.g., an image batch) to be classified.

Returns:
- A `Flux.AbstractArray` containing the binary prediction scores (probabilities)
  for each sample in the batch.
"""
function (model::VAEArtifactClassifier)(x)
    embeddings = model.encoder(x)
    return model.classifier(embeddings)
end

"""
    VAEArtifactClassifier(contrastive_model::ContrastiveModel; freeze_encoder::Bool=true)

Convenience constructor to create a `VAEArtifactClassifier` directly from a
pre-trained `ContrastiveModel`. It extracts the `encoder` from the contrastive model
and initializes the classification head.

Arguments:
- `contrastive_model`: An instance of `ContrastiveModel` which has been
                       pre-trained (or is ready for use).
- `freeze_encoder`: A `Bool` flag, passed to the primary constructor. If `true`,
                    the encoder from the contrastive model will be frozen.
                    Defaults to `true` as this is a common practice when
                    fine-tuning a classifier on top of a pre-trained backbone.

Returns:
- A `VAEArtifactClassifier` instance.

# TODO: The `embed_dim` is currently hardcoded to 256. It would be more robust
#       to derive this value dynamically from the `contrastive_model.encoder`
#       (e.g., by inspecting the output dimension of its last layer) to ensure
#       flexibility and prevent potential errors if the encoder's architecture changes.
"""
function VAEArtifactClassifier(contrastive_model::ContrastiveModel; freeze_encoder::Bool=true)
    return VAEArtifactClassifier(
        contrastive_model.encoder,
        embed_dim=256, # The embed_dim is hardcoded here, should be flexible
        freeze_encoder=freeze_encoder
    )
end

#=============================================================================
                           Training Functions
=============================================================================#

"""
    train_contrastive!(
        model::ContrastiveModel,
        train_loader;
        epochs::Int=50,
        lr::Float64=1e-4,
        loss_fn::Symbol=:supervised,
        temperature::Float32=0.1f0,
        device=cpu,
        verbose::Bool=true
    )

Trains a `ContrastiveModel` using one of several contrastive loss functions.
This pre-training phase is crucial for learning robust feature embeddings
from the input data without direct supervision, by encouraging similar samples
to be close and dissimilar samples to be far apart in the embedding space.

The function iterates through the `train_loader` for a specified number of `epochs`,
computes the chosen contrastive loss, and updates the model's parameters using
the Adam optimizer.

Arguments:
- `model`: An instance of `ContrastiveModel` to be trained.
- `train_loader`: A `DataLoader` instance (typically from `VAEDatasetUtils`)
                  that yields batches of input data (`x`) and corresponding
                  labels (`y`). The structure of `x` and `y` depends on the
                  chosen `loss_fn`.

Keyword Arguments:
- `epochs`: The total number of training epochs to run. Defaults to `50`.
- `lr`: The learning rate for the Adam optimizer. Defaults to `1e-4`.
- `loss_fn`: A `Symbol` indicating which contrastive loss function to use:
    - `:ntxent`: Normalized Temperature-scaled Cross Entropy Loss. Requires
                  `train_loader` to yield batches where `x` contains two
                  augmented views of each sample.
    - `:supervised`: Supervised Contrastive Loss. Uses `y` labels to define
                     positive and negative pairs.
    - `:triplet`: Triplet Loss. Uses `y` labels to form triplets and performs
                  hard negative mining on embeddings.
    - `:contrastive`: Simple binary Contrastive Loss. Uses `y` labels to
                      minimize/maximize distances.
                  Defaults to `:supervised`.
- `temperature`: A `Float32` scalar used by `NTXentLoss` and `SupervisedContrastiveLoss`
                 to scale the similarity logits. Defaults to `0.1f0`.
- `device`: The computational device (`cpu` or `gpu`) on which to perform
            training. Defaults to `cpu`.
- `verbose`: A `Bool` indicating whether to print epoch-wise training progress
             (loss and time). Defaults to `true`.

Returns:
- `history`: A `Dict` containing training metrics, specifically:
    - `"loss"`: A `Vector{Float32}` of average batch losses per epoch.
    - `"epoch_time"`: A `Vector{Float64}` of the time taken for each epoch in seconds.
"""
function train_contrastive!(
    model::ContrastiveModel,
    train_loader;
    epochs::Int=50,
    lr::Float64=1e-4,
    loss_fn::Symbol=:supervised,
    temperature::Float32=0.1f0,
    device=cpu,
    verbose::Bool=true
)
    model = model |> device
    opt_state = Flux.setup(Adam(lr), model)

    history = Dict(
        "loss" => Float32[],
        "epoch_time" => Float64[]
    )

    for epoch in 1:epochs
        epoch_start = time()
        epoch_loss = 0f0
        n_batches = 0

        for (x, y) in train_loader
            x = x |> device
            y = y |> device

            loss, grads = Flux.withgradient(model) do m
                embeddings, projections = m(x)

                if loss_fn == :ntxent
                    # Split batch in half for NT-Xent
                    mid = size(projections, 2) ÷ 2
                    NTXentLoss(projections[:, 1:mid], projections[:, mid+1:end], temperature=temperature)
                elseif loss_fn == :supervised
                    SupervisedContrastiveLoss(projections, y, temperature=temperature)
                elseif loss_fn == :triplet
                    TripletLoss(embeddings, y)
                else
                    ContrastiveLoss(embeddings, y)
                end
            end

            Flux.update!(opt_state, model, grads[1])

            epoch_loss += loss
            n_batches += 1
        end

        avg_loss = epoch_loss / n_batches
        epoch_time = time() - epoch_start

        push!(history["loss"], avg_loss)
        push!(history["epoch_time"], epoch_time)

        if verbose
            @printf("Epoch %3d/%d | Loss: %.4f | Time: %.1fs\n",
                    epoch, epochs, avg_loss, epoch_time)
        end
    end

    return history
end

"""
    train_classifier!(
        model::VAEArtifactClassifier,
        train_loader,
        val_loader=nothing;
        epochs::Int=30,
        lr::Float64=1e-3,
        device=cpu,
        verbose::Bool=true
    )

Trains a `VAEArtifactClassifier` for the binary classification task of
detecting VAE artifacts. This function typically follows the contrastive
pre-training phase. The model's classification head is fine-tuned, and
optionally the encoder as well (if not frozen), to perform the specific
detection task.

The function iterates through the `train_loader` for a specified number of `epochs`,
computes the binary cross-entropy loss, and updates the model's parameters using
the Adam optimizer. If a `val_loader` is provided, the model's performance on
a validation set is also evaluated at the end of each epoch.

Arguments:
- `model`: An instance of `VAEArtifactClassifier` to be trained.
- `train_loader`: A `DataLoader` instance yielding batches of input data (`x`)
                  and corresponding binary labels (`y`) for training.
- `val_loader`: An optional `DataLoader` instance for validation data. If provided,
                the model's performance on this data will be reported. Defaults to `nothing`.

Keyword Arguments:
- `epochs`: The total number of training epochs to run. Defaults to `30`.
- `lr`: The learning rate for the Adam optimizer. Defaults to `1e-3`.
- `device`: The computational device (`cpu` or `gpu`) on which to perform
            training. Defaults to `cpu`.
- `verbose`: A `Bool` indicating whether to print epoch-wise training progress
             (loss and accuracy for train and validation sets). Defaults to `true`.

Returns:
- `history`: A `Dict` containing training and validation metrics, specifically:
    - `"train_loss"`: A `Vector{Float32}` of average batch training losses per epoch.
    - `"train_acc"`: A `Vector{Float32}` of training accuracies per epoch.
    - `"val_loss"`: A `Vector{Float32}` of average batch validation losses per epoch (if `val_loader` is provided).
    - `"val_acc"`: A `Vector{Float32}` of validation accuracies per epoch (if `val_loader` is provided).
"""
function train_classifier!(
    model::VAEArtifactClassifier,
    train_loader,
    val_loader=nothing;
    epochs::Int=30,
    lr::Float64=1e-3,
    device=cpu,
    verbose::Bool=true
)
    model = model |> device
    opt_state = Flux.setup(Adam(lr), model)

    history = Dict(
        "train_loss" => Float32[],
        "train_acc" => Float32[],
        "val_loss" => Float32[],
        "val_acc" => Float32[]
    )

    for epoch in 1:epochs
        # Training
        train_loss = 0f0
        train_correct = 0
        train_total = 0

        for (x, y) in train_loader
            x = x |> device
            y = reshape(y, 1, :) |> device

            loss, grads = Flux.withgradient(model) do m
                pred = m(x)
                Flux.binarycrossentropy(pred, y)
            end

            Flux.update!(opt_state, model, grads[1])

            train_loss += loss
            pred = model(x)
            train_correct += sum((pred .> 0.5f0) .== y)
            train_total += length(y)
        end

        avg_train_loss = train_loss / length(train_loader)
        train_acc = train_correct / train_total

        push!(history["train_loss"], avg_train_loss)
        push!(history["train_acc"], train_acc)

        # Validation
        if val_loader !== nothing
            val_loss = 0f0
            val_correct = 0
            val_total = 0

            Flux.testmode!(model)
            for (x, y) in val_loader
                x = x |> device
                y = reshape(y, 1, :) |> device

                pred = model(x)
                val_loss += Flux.binarycrossentropy(pred, y)
                val_correct += sum((pred .> 0.5f0) .== y)
                val_total += length(y)
            end
            Flux.trainmode!(model)

            avg_val_loss = val_loss / length(val_loader)
            val_acc = val_correct / val_total

            push!(history["val_loss"], avg_val_loss)
            push!(history["val_acc"], val_acc)

            if verbose
                @printf("Epoch %3d/%d | Train Loss: %.4f Acc: %.2f%% | Val Loss: %.4f Acc: %.2f%%\n",
                        epoch, epochs, avg_train_loss, train_acc*100, avg_val_loss, val_acc*100)
            end
        else
            if verbose
                @printf("Epoch %3d/%d | Train Loss: %.4f Acc: %.2f%%\n",
                        epoch, epochs, avg_train_loss, train_acc*100)
            end
        end
    end

    return history
end

#=============================================================================
                           Evaluation Functions
=============================================================================#

"""
    evaluate_model(model, test_loader; device=cpu)

Evaluates the performance of a trained classification model on a given test dataset.
This function computes common classification metrics such as accuracy, precision,
recall, F1 score, and the confusion matrix. These metrics provide a comprehensive
understanding of the model's ability to distinguish between original and VAE-generated
images.

Arguments:
- `model`: The trained classification model (e.g., `VAEArtifactClassifier` or any
           model that takes an input `x` and produces a probability score).
- `test_loader`: A `DataLoader` instance yielding batches of input data (`x`)
                 and corresponding binary labels (`y`) for testing.

Keyword Arguments:
- `device`: The computational device (`cpu` or `gpu`) on which to perform
            evaluation. Defaults to `cpu`.

Returns:
- `results::Dict`: A dictionary containing the evaluation metrics:
    - `"accuracy"`: `Float64` - Overall accuracy of the model.
    - `"precision"`: `Float64` - The proportion of positive identifications that were actually correct.
                     (Here, positive means predicting VAE artifact).
    - `"recall"`: `Float64` - The proportion of actual positives that were identified correctly.
                  (Here, positive means actual VAE artifact).
    - `"f1"`: `Float64` - The harmonic mean of precision and recall, providing a single
            score that balances both.
    - `"confusion_matrix"`: `Matrix{Int}` - A 2x2 matrix representing:
        ```
        [ True Negatives (TN)  False Positives (FP) ]
        [ False Negatives (FN) True Positives (TP)  ]
        ```
        Where:
        - TN: Correctly predicted Original (negative class).
        - FP: Incorrectly predicted VAE (positive class, when it was Original).
        - FN: Incorrectly predicted Original (negative class, when it was VAE).
        - TP: Correctly predicted VAE (positive class).
    - `"predictions"`: `Vector{Float32}` - Raw prediction scores (probabilities) for all samples.
    - `"labels"`: `Vector{Float32}` - Ground truth binary labels for all samples.
"""
function evaluate_model(model, test_loader; device=cpu)
    model = model |> device
    Flux.testmode!(model)

    all_preds = Float32[]
    all_labels = Float32[]

    for (x, y) in test_loader
        x = x |> device
        pred = model(x) |> cpu
        append!(all_preds, vec(pred))
        append!(all_labels, vec(y))
    end

    # Binary predictions
    pred_labels = all_preds .> 0.5f0

    # Confusion matrix
    tp = sum(pred_labels .& (all_labels .== 1))
    tn = sum(.!pred_labels .& (all_labels .== 0))
    fp = sum(pred_labels .& (all_labels .== 0))
    fn = sum(.!pred_labels .& (all_labels .== 1))

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return Dict(
        "accuracy" => accuracy,
        "precision" => precision,
        "recall" => recall,
        "f1" => f1,
        "confusion_matrix" => [tn fp; fn tp],
        "predictions" => all_preds,
        "labels" => all_labels
    )
end

"""
    extract_embeddings(model, data_loader; device=cpu)

Extracts feature embeddings from the encoder of a given model for all samples
in a `DataLoader`. These embeddings are useful for post-training analysis,
such as visualizing the learned feature space using dimensionality reduction
techniques like t-SNE or UMAP, or for feeding into other machine learning
models.

Arguments:
- `model`: The model from which to extract embeddings. This can be a `ContrastiveModel`
           (in which case its `encoder` is used), a `VAEArtifactClassifier`
           (its `encoder` is used), or directly a `Flux.Chain` representing an encoder.
- `data_loader`: A `DataLoader` instance that yields batches of input data (`x`)
                 and corresponding labels (`y`). The `x` data will be fed to the
                 encoder, and `y` labels will be returned alongside the embeddings.

Keyword Arguments:
- `device`: The computational device (`cpu` or `gpu`) on which to perform
            embedding extraction. Defaults to `cpu`.

Returns:
- A `Tuple` containing two elements:
    1. `embeddings`: A `Matrix{Float32}` where each column is an embedding vector
                      for a sample. The shape will be `(embed_dim, num_samples)`.
    2. `labels`: A `Vector{Float32}` containing the ground truth labels for
                 each extracted embedding.
"""
function extract_embeddings(model, data_loader; device=cpu)
    encoder = if model isa ContrastiveModel
        model.encoder
    elseif model isa VAEArtifactClassifier
        model.encoder
    else
        model
    end

    encoder = encoder |> device
    Flux.testmode!(encoder)

    all_embeddings = []
    all_labels = Float32[]

    for (x, y) in data_loader
        x = x |> device
        emb = encoder(x) |> cpu
        push!(all_embeddings, emb)
        append!(all_labels, vec(y))
    end

    embeddings = hcat(all_embeddings...)

    return embeddings, all_labels
end

#=============================================================================
                           Utility Functions
=============================================================================#

"""
    save_model(model, path::String)

Saves a trained Flux.jl model to a file using the BSON (Binary JSON) format.
BSON is a suitable format for serializing Julia data structures, including
Flux models, for later loading and inference.

Before saving, the model is moved to `cpu` to ensure compatibility across
different environments (e.g., loading a GPU-trained model on a CPU-only machine).

Arguments:
- `model`: The Flux.jl model instance to be saved. This can be a `ContrastiveModel`,
           `VAEArtifactClassifier`, or any other Flux `Chain` or custom model struct.
- `path`: A `String` specifying the full file path where the model should be saved
          (e.g., `"output/my_model.bson"`). The file extension `.bson` is
          conventionally used.

Side effects:
- Writes the serialized model to the specified `path`.
- Prints a confirmation message to the console.
"""
function save_model(model, path::String)
    model_cpu = model |> cpu
    BSON.@save path model_cpu
    println("Model saved to: $path")
end

"""
    load_model(path::String)

Loads a trained Flux.jl model from a specified BSON file.
This function uses BSON's `@load` macro to deserialize the model object.

Arguments:
- `path`: A `String` specifying the full file path from which the model should be loaded.

Returns:
- `model_cpu`: The loaded Flux.jl model instance. The model is loaded onto the CPU,
               regardless of whether it was originally saved from a GPU-enabled context.
               It can be subsequently moved to a GPU using `model |> gpu` if `CUDA.jl`
               is functional.
"""
function load_model(path::String)
    BSON.@load path model_cpu
    return model_cpu
end

"""
    model_summary(model)

Prints a summary of a Flux.jl model, including the total number of trainable
parameters. This provides a quick overview of the model's complexity.

Arguments:
- `model`: The Flux.jl model instance for which to print the summary. This can be
           a `ContrastiveModel`, `VAEArtifactClassifier`, `Flux.Chain`, or any
           other Flux model struct that `Flux.params` can operate on.

Output:
- Prints a formatted summary to the console, including a header, a count of
  total trainable parameters, and a footer.
"""
function model_summary(model)
    println("="^60)
    println("Model Summary")
    println("="^60)

    total_params = 0
    # Iterate through each parameter group in the model
    # Flux.params(model) returns an iterable collection of trainable parameters
    # The 'name' in pairs(Flux.params(model)) typically refers to the path within the model
    # to the parameter, but for a simple count, we just need the 'layer' (parameter array itself).
    for (name, layer) in pairs(Flux.params(model))
        n = length(layer) # Get the number of elements (parameters) in the current layer/parameter array
        total_params += n
    end

    println("Total trainable parameters: $(total_params)")
    println("="^60)
end

"""
    cosine_annealing(epoch, max_epochs, lr_max, lr_min=1e-6)

Implements a cosine annealing learning rate schedule. This scheduler varies
the learning rate over `max_epochs` following a cosine curve, starting
from `lr_max` and decreasing to `lr_min`. This approach is often used
in deep learning to allow for large learning rates at the beginning of
training (for faster convergence) and small learning rates towards the end
(for fine-tuning and better generalization).

The learning rate `lr(epoch)` is calculated using the formula:
```
lr(epoch) = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(π * epoch / max_epochs))
```
where `epoch` is 0-indexed.

Arguments:
- `epoch`: The current epoch number (0-indexed).
- `max_epochs`: The total number of epochs for the annealing cycle.
- `lr_max`: The maximum learning rate (initial learning rate).
- `lr_min`: The minimum learning rate (final learning rate). Defaults to `1e-6`.

Returns:
- A `Float64` value representing the calculated learning rate for the current `epoch`.
"""
function cosine_annealing(epoch, max_epochs, lr_max, lr_min=1e-6)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(π * epoch / max_epochs))
end

end # module VAEContrastive


#=============================================================================
                           Main Training Script
=============================================================================#

# Only run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__

    using BSON
    using ArgParse

    include("julia_utils.jl")
    using .VAEDatasetUtils
    using .VAEContrastive

    function parse_args()
        s = ArgParseSettings(
            description="Train contrastive model for VAE artifact detection"
        )

        @add_arg_table! s begin
            "command"
                help = "Command: train, evaluate, embed"
                required = true
            "--dataset", "-d"
                help = "Path to dataset directory"
                required = true
            "--output", "-o"
                help = "Output directory for models/results"
                default = "output"
            "--splits"
                help = "Path to splits directory"
                default = nothing
            "--manifest"
                help = "Path to manifest.csv"
                default = nothing
            "--compressed"
                help = "Use compressed diff format"
                action = :store_true
            "--epochs"
                help = "Number of training epochs"
                arg_type = Int
                default = 50
            "--batch-size"
                help = "Batch size"
                arg_type = Int
                default = 32
            "--lr"
                help = "Learning rate"
                arg_type = Float64
                default = 1e-4
            "--loss"
                help = "Loss function: supervised, ntxent, triplet, contrastive"
                default = "supervised"
            "--model"
                help = "Path to pre-trained model (for evaluate/embed)"
                default = nothing
        end

        return ArgParse.parse_args(s)
    end

    function main()
        args = parse_args()

        println("="^60)
        println("VAE Artifact Detection - Contrastive Learning")
        println("="^60)
        println("Command:    $(args["command"])")
        println("Dataset:    $(args["dataset"])")
        println("Compressed: $(args["compressed"])")
        println()

        # Setup paths
        dataset_path = args["dataset"]
        output_dir = args["output"]
        mkpath(output_dir)

        splits_dir = something(args["splits"], joinpath(output_dir, "splits"))
        manifest_path = something(args["manifest"], joinpath(output_dir, "manifest.csv"))

        if args["command"] == "train"
            println("Loading training data...")

            # Load dataset
            if args["compressed"]
                train_data = CompressedVAEDataset(
                    joinpath(splits_dir, "random_train.txt"),
                    joinpath(dataset_path, "Original"),
                    joinpath(dataset_path, "Diff")
                )
                val_data = CompressedVAEDataset(
                    joinpath(splits_dir, "random_val.txt"),
                    joinpath(dataset_path, "Original"),
                    joinpath(dataset_path, "Diff")
                )
            else
                train_data = VAEDetectorDataset(
                    joinpath(splits_dir, "random_train.txt"),
                    manifest_path,
                    dataset_path
                )
                val_data = VAEDetectorDataset(
                    joinpath(splits_dir, "random_val.txt"),
                    manifest_path,
                    dataset_path
                )
            end

            train_loader = DataLoader(train_data, batch_size=args["batch-size"], shuffle=true)
            val_loader = DataLoader(val_data, batch_size=args["batch-size"], shuffle=false)

            println("Training samples: $(length(train_data))")
            println("Validation samples: $(length(val_data))")
            println()

            # Phase 1: Contrastive pre-training
            println("="^40)
            println("Phase 1: Contrastive Pre-training")
            println("="^40)

            contrastive_model = ContrastiveModel(in_channels=3, embed_dim=256, proj_dim=128)
            loss_sym = Symbol(args["loss"])

            contrastive_history = train_contrastive!(
                contrastive_model,
                train_loader,
                epochs=args["epochs"],
                lr=args["lr"],
                loss_fn=loss_sym
            )

            # Save contrastive model
            contrastive_path = joinpath(output_dir, "contrastive_model.bson")
            save_model(contrastive_model, contrastive_path)

            # Phase 2: Fine-tune classifier
            println()
            println("="^40)
            println("Phase 2: Classifier Fine-tuning")
            println("="^40)

            classifier = VAEArtifactClassifier(contrastive_model, freeze_encoder=false)

            classifier_history = train_classifier!(
                classifier,
                train_loader,
                val_loader,
                epochs=30,
                lr=1e-3
            )

            # Save classifier
            classifier_path = joinpath(output_dir, "vae_classifier.bson")
            save_model(classifier, classifier_path)

            # Final evaluation
            println()
            println("="^40)
            println("Final Evaluation")
            println("="^40)

            results = evaluate_model(classifier, val_loader)

            println("Accuracy:  $(round(results["accuracy"]*100, digits=2))%")
            println("Precision: $(round(results["precision"]*100, digits=2))%")
            println("Recall:    $(round(results["recall"]*100, digits=2))%")
            println("F1 Score:  $(round(results["f1"]*100, digits=2))%")
            println()
            println("Confusion Matrix:")
            println("              Predicted")
            println("              Orig   VAE")
            println("Actual Orig  $(Int(results["confusion_matrix"][1,1]))  $(Int(results["confusion_matrix"][1,2]))")
            println("       VAE   $(Int(results["confusion_matrix"][2,1]))  $(Int(results["confusion_matrix"][2,2]))")

            # Save results
            results_path = joinpath(output_dir, "training_results.txt")
            open(results_path, "w") do f
                println(f, "VAE Artifact Detection Results")
                println(f, "==============================")
                println(f, "Date: $(Dates.now())")
                println(f, "Dataset: $(dataset_path)")
                println(f, "Epochs: $(args["epochs"])")
                println(f, "Batch size: $(args["batch-size"])")
                println(f, "Loss function: $(args["loss"])")
                println(f, "")
                println(f, "Results:")
                println(f, "  Accuracy:  $(round(results["accuracy"]*100, digits=2))%")
                println(f, "  Precision: $(round(results["precision"]*100, digits=2))%")
                println(f, "  Recall:    $(round(results["recall"]*100, digits=2))%")
                println(f, "  F1 Score:  $(round(results["f1"]*100, digits=2))%")
            end
            println("Results saved to: $results_path")

        elseif args["command"] == "evaluate"
            model_path = args["model"]
            if model_path === nothing
                error("--model required for evaluate command")
            end

            println("Loading model: $model_path")
            classifier = load_model(model_path)

            # Load test data
            if args["compressed"]
                test_data = CompressedVAEDataset(
                    joinpath(splits_dir, "random_test.txt"),
                    joinpath(dataset_path, "Original"),
                    joinpath(dataset_path, "Diff")
                )
            else
                test_data = VAEDetectorDataset(
                    joinpath(splits_dir, "random_test.txt"),
                    manifest_path,
                    dataset_path
                )
            end

            test_loader = DataLoader(test_data, batch_size=args["batch-size"], shuffle=false)

            results = evaluate_model(classifier, test_loader)

            println("Test Results:")
            println("  Accuracy:  $(round(results["accuracy"]*100, digits=2))%")
            println("  Precision: $(round(results["precision"]*100, digits=2))%")
            println("  Recall:    $(round(results["recall"]*100, digits=2))%")
            println("  F1 Score:  $(round(results["f1"]*100, digits=2))%")

        elseif args["command"] == "embed"
            model_path = args["model"]
            if model_path === nothing
                error("--model required for embed command")
            end

            println("Loading model: $model_path")
            model = load_model(model_path)

            # Load all data
            if args["compressed"]
                all_data = CompressedVAEDataset(
                    joinpath(splits_dir, "random_train.txt"),
                    joinpath(dataset_path, "Original"),
                    joinpath(dataset_path, "Diff")
                )
            else
                all_data = VAEDetectorDataset(
                    joinpath(splits_dir, "random_train.txt"),
                    manifest_path,
                    dataset_path
                )
            end

            data_loader = DataLoader(all_data, batch_size=args["batch-size"], shuffle=false)

            embeddings, labels = extract_embeddings(model, data_loader)

            # Save embeddings for visualization
            embed_path = joinpath(output_dir, "embeddings.bson")
            BSON.@save embed_path embeddings labels
            println("Embeddings saved to: $embed_path")
            println("Shape: $(size(embeddings))")
            println("Use t-SNE or UMAP to visualize the embedding space")

        else
            error("Unknown command: $(args["command"])")
        end
    end

    main()
end
