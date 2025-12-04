# SPDX-FileCopyrightText: 2024 Joshua Jewell
# SPDX-License-Identifier: MIT

#=
Contrastive Learning for VAE Artifact Detection
================================================

Train a model to distinguish original images from VAE-decoded images
using contrastive learning. This tests whether VAE artifacts are
detectable by neural networks.

Usage:
    julia --project contrastive_model.jl train /path/to/dataset
    julia --project contrastive_model.jl evaluate /path/to/model.bson /path/to/dataset

Requires: Flux.jl, CUDA.jl (optional), Images.jl, BSON.jl
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
CNN Encoder backbone for extracting image features.
Uses a ResNet-style architecture with residual connections.
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
Projection head for contrastive learning.
Maps embeddings to a lower-dimensional space where contrastive loss is applied.
"""
function ProjectionHead(; embed_dim::Int=256, proj_dim::Int=128)
    return Chain(
        Dense(embed_dim, embed_dim),
        x -> relu.(x),
        Dense(embed_dim, proj_dim)
    )
end

"""
Complete contrastive model with encoder and projection head.
"""
struct ContrastiveModel
    encoder::Chain
    projector::Chain
end

Flux.@functor ContrastiveModel

function ContrastiveModel(; in_channels::Int=3, embed_dim::Int=256, proj_dim::Int=128)
    encoder = ContrastiveEncoder(in_channels=in_channels, embed_dim=embed_dim)
    projector = ProjectionHead(embed_dim=embed_dim, proj_dim=proj_dim)
    return ContrastiveModel(encoder, projector)
end

"""
Forward pass: returns both embeddings and projections.
"""
function (model::ContrastiveModel)(x)
    embeddings = model.encoder(x)
    projections = model.projector(embeddings)
    return embeddings, projections
end

"""
Get just the embeddings (for downstream tasks).
"""
function get_embeddings(model::ContrastiveModel, x)
    return model.encoder(x)
end

#=============================================================================
                           Loss Functions
=============================================================================#

"""
NT-Xent (Normalized Temperature-scaled Cross Entropy) Loss
Used in SimCLR for contrastive learning.

For VAE detection, we treat:
- Positive pairs: same image type (both original or both VAE)
- Negative pairs: different image types
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
Supervised Contrastive Loss for binary classification.
Pulls same-class samples together, pushes different-class samples apart.

Arguments:
- z: Projections [proj_dim, batch_size]
- labels: Binary labels (0=original, 1=VAE)
- temperature: Scaling factor
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
Triplet Loss with hard negative mining.
Anchor-Positive-Negative triplets based on labels.
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
Simple binary contrastive loss.
Minimizes distance for same-class pairs, maximizes for different-class pairs.
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
Binary classifier for VAE artifact detection.
Uses pre-trained contrastive encoder + classification head.
"""
struct VAEArtifactClassifier
    encoder::Chain
    classifier::Chain
end

Flux.@functor VAEArtifactClassifier

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
        # Mark encoder as non-trainable
        encoder = Flux.freeze(encoder)
    end

    return VAEArtifactClassifier(encoder, classifier)
end

function (model::VAEArtifactClassifier)(x)
    embeddings = model.encoder(x)
    return model.classifier(embeddings)
end

"""
Create classifier from pre-trained contrastive model.
"""
function VAEArtifactClassifier(contrastive_model::ContrastiveModel; freeze_encoder::Bool=true)
    return VAEArtifactClassifier(
        contrastive_model.encoder,
        embed_dim=256,
        freeze_encoder=freeze_encoder
    )
end

#=============================================================================
                           Training Functions
=============================================================================#

"""
Train contrastive model on VAE dataset.

Arguments:
- model: ContrastiveModel
- train_loader: DataLoader yielding (images, labels) batches
- epochs: Number of training epochs
- lr: Learning rate
- loss_fn: Loss function (:ntxent, :supervised, :triplet, :contrastive)
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
Train binary classifier on VAE detection task.
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
Evaluate classifier on test set.
Returns accuracy, precision, recall, F1 score, and confusion matrix.
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
Extract embeddings for visualization (t-SNE, UMAP, etc.)
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
Save model to BSON file.
"""
function save_model(model, path::String)
    model_cpu = model |> cpu
    BSON.@save path model_cpu
    println("Model saved to: $path")
end

"""
Load model from BSON file.
"""
function load_model(path::String)
    BSON.@load path model_cpu
    return model_cpu
end

"""
Print model summary.
"""
function model_summary(model)
    println("="^60)
    println("Model Summary")
    println("="^60)

    total_params = 0
    for (name, layer) in pairs(Flux.params(model))
        n = length(layer)
        total_params += n
    end

    println("Total parameters: $(total_params)")
    println("="^60)
end

"""
Learning rate scheduler - cosine annealing.
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
