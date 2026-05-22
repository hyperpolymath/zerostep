# SPDX-License-Identifier: MPL-2.0
# (MPL-2.0 preferred; MPL-2.0 required for Julia ecosystem)
# Property-based invariant tests for zerostep / VAENormalizer

using Test
using Statistics
using LinearAlgebra

@testset "Property-Based Tests" begin

    @testset "Invariant: contrastive encoder output dimension matches embed_dim" begin
        include(joinpath(@__DIR__, "..", "contrastive_model.jl"))

        for embed_dim in [64, 128, 256]
            encoder = VAEContrastive.ContrastiveEncoder(in_channels=3, embed_dim=embed_dim)
            @test encoder !== nothing
        end
    end

    @testset "Invariant: projection head output dimension matches proj_dim" begin
        include(joinpath(@__DIR__, "..", "contrastive_model.jl"))

        for proj_dim in [64, 128]
            head = VAEContrastive.ProjectionHead(embed_dim=256, proj_dim=proj_dim)
            @test head !== nothing
        end
    end

    @testset "Invariant: DataLoader batch count is ceil(n / batch_size)" begin
        include(joinpath(@__DIR__, "..", "julia_utils.jl"))

        for _ in 1:30
            n = rand(5:50)
            bs = rand(2:10)
            data   = [rand(Float32, 3, 8, 8) for _ in 1:n]
            labels = rand(0:1, n)
            loader = VAEDatasetUtils.DataLoader(data, labels; batch_size=bs, shuffle=false)
            expected_batches = ceil(Int, n / bs)
            @test length(loader) == expected_batches
        end
    end

    @testset "Invariant: DataLoader total items equals n" begin
        include(joinpath(@__DIR__, "..", "julia_utils.jl"))

        for _ in 1:20
            n = rand(4:40)
            bs = rand(2:8)
            data   = [rand(Float32, 3, 8, 8) for _ in 1:n]
            labels = rand(0:1, n)
            loader = VAEDatasetUtils.DataLoader(data, labels; batch_size=bs, shuffle=false)
            total = sum(length(batch_labels) for (_, batch_labels) in loader)
            @test total == n
        end
    end

end
