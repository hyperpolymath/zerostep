# SPDX-License-Identifier: MPL-2.0
# (MPL-2.0 preferred; MPL-2.0 required for Julia ecosystem)
# E2E pipeline tests for zerostep / VAENormalizer

using Test

@testset "E2E Pipeline Tests" begin

    @testset "ContrastiveEncoder construction pipeline" begin
        # Load the contrastive model module directly since VAEContrastive is a
        # standalone Julia file not packaged as a Pkg module
        include(joinpath(@__DIR__, "..", "contrastive_model.jl"))

        # Encoder should construct without error
        encoder = VAEContrastive.ContrastiveEncoder(in_channels=3, embed_dim=256)
        @test encoder !== nothing
    end

    @testset "ProjectionHead construction pipeline" begin
        include(joinpath(@__DIR__, "..", "contrastive_model.jl"))

        head = VAEContrastive.ProjectionHead(embed_dim=256, proj_dim=128)
        @test head !== nothing
    end

    @testset "DataLoader construction pipeline" begin
        include(joinpath(@__DIR__, "..", "julia_utils.jl"))

        # Build a minimal DataLoader from in-memory arrays
        data = [rand(Float32, 3, 8, 8) for _ in 1:20]
        labels = rand(0:1, 20)
        loader = VAEDatasetUtils.DataLoader(data, labels; batch_size=4, shuffle=false)
        @test loader !== nothing
    end

    @testset "Error handling: mismatched data and label lengths" begin
        include(joinpath(@__DIR__, "..", "julia_utils.jl"))

        data   = [rand(Float32, 3, 8, 8) for _ in 1:10]
        labels = rand(0:1, 5)   # wrong length

        @test_throws Exception VAEDatasetUtils.DataLoader(data, labels; batch_size=2)
    end

end
