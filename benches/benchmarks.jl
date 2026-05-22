# SPDX-License-Identifier: MPL-2.0
# (MPL-2.0 preferred; MPL-2.0 required for Julia ecosystem)
# BenchmarkTools benchmarks for zerostep / VAENormalizer

using BenchmarkTools

const SUITE = BenchmarkGroup()

SUITE["model_construction"] = BenchmarkGroup()

SUITE["model_construction"]["encoder_default"] = @benchmarkable begin
    include(joinpath(@__DIR__, "..", "contrastive_model.jl"))
    VAEContrastive.ContrastiveEncoder()
end

SUITE["model_construction"]["encoder_custom"] = @benchmarkable begin
    include(joinpath(@__DIR__, "..", "contrastive_model.jl"))
    VAEContrastive.ContrastiveEncoder(in_channels=1, embed_dim=128)
end

SUITE["model_construction"]["projection_head"] = @benchmarkable begin
    include(joinpath(@__DIR__, "..", "contrastive_model.jl"))
    VAEContrastive.ProjectionHead(embed_dim=128, proj_dim=64)
end

SUITE["data_loader"] = BenchmarkGroup()

SUITE["data_loader"]["create_loader_20"] = @benchmarkable begin
    include(joinpath(@__DIR__, "..", "julia_utils.jl"))
    data   = [rand(Float32, 3, 8, 8) for _ in 1:20]
    labels = rand(0:1, 20)
    VAEDatasetUtils.DataLoader(data, labels; batch_size=4, shuffle=false)
end

SUITE["data_loader"]["create_loader_100"] = @benchmarkable begin
    include(joinpath(@__DIR__, "..", "julia_utils.jl"))
    data   = [rand(Float32, 3, 8, 8) for _ in 1:100]
    labels = rand(0:1, 100)
    VAEDatasetUtils.DataLoader(data, labels; batch_size=16, shuffle=false)
end

if abspath(PROGRAM_FILE) == @__FILE__
    tune!(SUITE)
    results = run(SUITE, verbose=true)
    BenchmarkTools.save("benchmarks_results.json", results)
end
