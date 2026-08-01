using Test

const EXAMPLE_ROOT = normpath(joinpath(@__DIR__, ".."))
let gtn_root = normpath(joinpath(EXAMPLE_ROOT, "..", ".."))
    gtn_root in LOAD_PATH || push!(LOAD_PATH, gtn_root)
end

include(joinpath(
    EXAMPLE_ROOT, "Spinless_Fermion_2D_Square_AD_nested.jl"
))

@testset "Nested spinless exact energy" begin
    @test spinless_exact_energy(1.0, 1.0, 3.0; nk=512) ≈
        -6.170521774015 atol=2e-6
    @test_throws ArgumentError spinless_exact_energy(
        1.0, 1.0, 3.0; nk=0
    )
end

@testset "Normalized gradient candidate" begin
    params = normalized_params([1.0, -0.5, 0.75, 2.0])
    target = normalized_params([-0.5, 1.5, -1.0, 0.25])
    objective = x -> sum(abs2, x - target)

    first_result = _normalized_gradient_candidate(
        objective, params;
        inner_optim_iter=2,
        max_step_norm=0.25,
    )
    second_result = _normalized_gradient_candidate(
        objective, params;
        inner_optim_iter=2,
        max_step_norm=0.25,
    )

    @test first_result.candidate == second_result.candidate
    @test first_result.gradient == second_result.gradient
    @test first_result.candidate_energy < first_result.energy
    @test norm(first_result.candidate) / sqrt(length(params)) ≈ 1.0
    @test norm(first_result.candidate - params) <=
        0.25 + 64eps(Float64)
    @test first_result.optim_status == :accepted
    @test first_result.accepted_inner_steps > 0
    @test first_result.backtracking_trials >=
        first_result.accepted_inner_steps

    constant_result = _normalized_gradient_candidate(
        _ -> 2.0, params;
        inner_optim_iter=2,
        max_step_norm=0.25,
    )
    @test constant_result.candidate == params
    @test iszero(norm(constant_result.gradient))
    @test constant_result.optim_status == :zero_gradient
    @test constant_result.optim_converged

    nonfinite_result = _normalized_gradient_candidate(
        _ -> NaN, params;
        inner_optim_iter=2,
        max_step_norm=0.25,
    )
    @test nonfinite_result.candidate == params
    @test nonfinite_result.optim_status == :nonfinite_value
    @test !nonfinite_result.optim_converged

    no_step_result = _normalized_gradient_candidate(
        objective, params;
        inner_optim_iter=0,
        max_step_norm=0.25,
    )
    @test no_step_result.candidate == params
    @test no_step_result.optim_status == :no_step
    @test_throws ArgumentError _normalized_gradient_candidate(
        objective, params; inner_optim_iter=-1
    )
    @test_throws ArgumentError _normalized_gradient_candidate(
        objective, params; max_step_norm=-0.25
    )
    @test_throws ArgumentError _normalized_gradient_candidate(
        objective, params; max_step_norm=Inf
    )
    @test_throws ArgumentError _normalized_gradient_candidate(
        objective, params; max_step_norm=NaN
    )
end

@testset "Nested AD line-search validation" begin
    function captured_error(; kwargs...)
        try
            run_Square_SpinlessFermion_AD_nested(
                1.0,
                1.0,
                3.0,
                1,
                1,
                1,
                4,
                1;
                ad_iter=1,
                kwargs...,
            )
        catch error
            return error
        end
        return nothing
    end

    for step_shrink in (0.0, 1.0, Inf, NaN)
        error = captured_error(; step_shrink)
        @test error isa ArgumentError
        @test occursin("step_shrink", sprint(showerror, error))
    end
    for min_step in (0.0, -1.0, Inf, NaN)
        error = captured_error(; min_step)
        @test error isa ArgumentError
        @test occursin("min_step", sprint(showerror, error))
    end
end

@testset "Nested AD example smoke" begin
    @eval GrassmannTensorNetworks global global_sign = auto_sign
    smoke = run_Square_SpinlessFermion_AD_nested(
        1.0,
        1.0,
        3.0,
        2,
        1,
        1,
        4,
        1;
        ad_iter=1,
        inner_optim_iter=1,
        seed=1234,
        verbosity=0,
    )
    @test length(smoke.history) == 1
    @test isfinite(smoke.report.energy)
    @test isfinite(smoke.report.final_gradient_norm)
    @test smoke.history[1].optim_status in (
        :accepted, :zero_gradient, :line_search_failed
    )
    @test smoke.history[1].backtracking_trials >=
        smoke.history[1].accepted_inner_steps
end
