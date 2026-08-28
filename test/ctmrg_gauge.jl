using Test
using LinearAlgebra
using Random
using GrassmannTensorNetworks

function ctmrg_gauge_test_env(; T=ComplexF64)
    Random.seed!(0x43544d5247)
    return CTMRGEnv(2, 1, 2, 1, 3, 2, 1, 1, T)
end

function phase_rotate_env(env::CTMRGEnv, phase)
    return CTMRGEnv{eltype(env)}(
        phase .* env.El,
        phase .* env.Er,
        phase .* env.Eu,
        phase .* env.Ed,
        phase .* env.Clu,
        phase .* env.Cru,
        phase .* env.Cld,
        phase .* env.Crd,
    )
end

@testset "CTMRG PEPSKit-style spectral diagnostics" begin
    env = ctmrg_gauge_test_env()
    scaled = phase_rotate_env(env, 3.0 - 4.0im)
    metrics = ctmrg_spectrum_distance(scaled, env)

    @test metrics.max ≤ 1e-12
    @test metrics.corner ≤ 1e-12
    @test metrics.edge ≤ 1e-12

    changed = deepcopy(env)
    changed.Clu[1, 1] = changed.Clu[1, 1] + 0.2 * Grassmann(
        size(changed.Clu[1, 1]),
        even(changed.Clu[1, 1]),
        index_type(changed.Clu[1, 1]),
        eltype(changed.Clu[1, 1]);
        init=:random,
    )
    changed_metrics = ctmrg_spectrum_distance(changed, env)
    @test changed_metrics.corner > 1e-8
end

@testset "CTMRG environment gauge fixing aligns tensor phases" begin
    env = ctmrg_gauge_test_env()
    rotated = phase_rotate_env(env, cis(0.37))
    fixed = fixgauge(rotated, env)

    @test ctmrg_elementwise_distance(fixed, env) ≤ 1e-12

    fixgauge!(rotated, env)
    @test ctmrg_elementwise_distance(rotated, env) ≤ 1e-12
end

@testset "CTMRG convergence metric selection" begin
    @test GrassmannTensorNetworks._ctmrg_convergence_error(:spectrum, 0.1, 0.2) == 0.1
    @test GrassmannTensorNetworks._ctmrg_convergence_error(:elementwise, 0.1, 0.2) == 0.2
    @test GrassmannTensorNetworks._ctmrg_convergence_error(:both, 0.1, 0.2) == 0.2
    @test_throws ArgumentError GrassmannTensorNetworks._ctmrg_convergence_error(:projector, 0.1, 0.2)
end
