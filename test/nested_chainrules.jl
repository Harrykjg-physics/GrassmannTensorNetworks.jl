using ChainRulesCore
using FiniteDifferences
using LinearAlgebra
using Random
using Test
using Zygote

@eval GrassmannTensorNetworks global global_sign = auto_sign

import GrassmannTensorNetworks: _nested_ket_for_network
import GrassmannTensorNetworks:
    _graded_pair_sign,
    _nested_ket, _nested_bra,
    _nested_x, _nested_y, _physical_identity,
    _nested_bra_for_network,
    _nested_x_for_network, _nested_y_for_network

function directional_error(f, x, direction; step=1e-6)
    value, pullback = Zygote.pullback(f, x)
    gradient = only(pullback(one(value)))
    finite = (f(x + step * direction) - f(x - step * direction)) / (2step)
    analytic = real(dot(gradient, direction))
    relative =
        abs(analytic - finite) /
        max(abs(analytic), abs(finite), eps(Float64))
    return relative, analytic, finite
end

function grassmann_directional_error(transform, x, direction; step=1e-6)
    objective(input) = sum(abs2, transform(input))
    value, pullback = Zygote.pullback(objective, x)
    gradient = only(pullback(one(value)))
    finite =
        (objective(x + step * direction) - objective(x - step * direction)) /
        (2step)
    analytic = real(dot(gradient, direction))
    relative =
        abs(analytic - finite) /
        max(abs(analytic), abs(finite), eps(Float64))
    return relative, analytic, finite
end

@testset "Nested reverse rules" begin
    Random.seed!(0x4e455354)
    count = square_gpeps_parameter_count(2, 1, 2, 1, 1)
    params = randn(count)
    direction = randn(count)
    direction ./= norm(direction)

    forward_peps = Square_GPEPS(2, 1, 2, 1, 1, params, false)
    rule_config = Zygote.ZygoteRuleConfig(Zygote.Context())
    @test _nested_ket_for_network(forward_peps.A[1, 1]) isa Grassmann

    tensor = forward_peps.A[1, 1]
    tensor_direction = Square_GPEPS(
        2, 1, 2, 1, 1, randn(count), false
    ).A[1, 1]
    tensor_direction = tensor_direction / norm(tensor_direction)
    _, pair_sign_pullback = rrule(_graded_pair_sign, tensor, 1, 3)
    pair_sign_zero = pair_sign_pullback(ZeroTangent())
    @test pair_sign_zero[1] isa NoTangent
    @test pair_sign_zero[2] isa ZeroTangent
    @test pair_sign_zero[3] isa NoTangent
    @test pair_sign_zero[4] isa NoTangent

    for (name, transform) in (
        ("graded pair sign", input -> _graded_pair_sign(input, 1, 3)),
        ("nested ket", _nested_ket),
        ("nested bra", _nested_bra),
        ("placed nested ket", _nested_ket_for_network),
        ("placed nested bra", _nested_bra_for_network),
    )
        relative, analytic, finite = grassmann_directional_error(
            transform, tensor, tensor_direction
        )
        @info "nested tensor directional derivative" name relative analytic finite
        @test relative <= 1e-4
    end
    for transform in (
        _nested_ket,
        _nested_bra,
        _nested_ket_for_network,
        _nested_bra_for_network,
    )
        _, transform_pullback = rrule(rule_config, transform, tensor)
        transform_zero = transform_pullback(ZeroTangent())
        @test transform_zero[1] isa NoTangent
        @test transform_zero[2] isa ZeroTangent
    end

    xraw = _nested_x(2, 1, 2, 1, Float64)
    xdirection = Grassmann(
        size(xraw), even(xraw), index_type(xraw), Float64;
        init=:random, parity=:even,
    )
    xdirection = xdirection / norm(xdirection)
    relative, analytic, finite = grassmann_directional_error(
        _nested_x_for_network, xraw, xdirection
    )
    @info "nested X placement directional derivative" relative analytic finite
    @test relative <= 1e-4
    _, x_pullback = rrule(_nested_x_for_network, xraw)
    x_zero = x_pullback(ZeroTangent())
    @test x_zero[1] isa NoTangent
    @test x_zero[2] isa ZeroTangent

    yraw = _nested_y(_physical_identity(tensor), 2, 1, 2, 1)
    ydirection = Grassmann(
        size(yraw), even(yraw), index_type(yraw), Float64;
        init=:random, parity=:even,
    )
    ydirection = ydirection / norm(ydirection)
    relative, analytic, finite = grassmann_directional_error(
        _nested_y_for_network, yraw, ydirection
    )
    @info "nested Y placement directional derivative" relative analytic finite
    @test relative <= 1e-4
    _, y_pullback = rrule(_nested_y_for_network, yraw)
    y_zero = y_pullback(ZeroTangent())
    @test y_zero[1] isa NoTangent
    @test y_zero[2] isa ZeroTangent

    nested_norm2(x) = begin
        peps = Square_GPEPS(2, 1, 2, 1, 1, x, false)
        nested = nested_network(peps)
        sum(t -> sum(abs2, t), nested.network)
    end
    relative, analytic, finite =
        directional_error(nested_norm2, params, direction)
    @info "nested network directional derivative" relative analytic finite
    @test relative <= 1e-4

    network_primal, network_pullback = rrule(
        rule_config,
        nested_network,
        forward_peps,
        NestedLayout(forward_peps),
    )
    @test network_pullback(ZeroTangent())[2] isa ZeroTangent
    structured_network_zero = Tangent{typeof(network_primal)}(
        ; network=ZeroTangent(),
          layout=NoTangent(),
          x_crossings=NoTangent(),
    )
    @test network_pullback(structured_network_zero)[2] isa ZeroTangent

    crossing_norm2(x) = begin
        peps = Square_GPEPS(2, 1, 2, 1, 1, x, false)
        nested = nested_network(peps)
        sum(t -> sum(abs2, t), nested.x_crossings)
    end
    crossing_gradient = only(Zygote.gradient(crossing_norm2, params))
    @test crossing_gradient === nothing || iszero(norm(crossing_gradient))

    initial_nested = nested_network(forward_peps)
    model = SpinlessFermionModel(1.0, 1.0, 3.0)
    number = n_site(model)
    operator_direction = Grassmann(
        size(number), even(number), index_type(number), Float64;
        init=:random, parity=:even,
    )
    operator_direction = operator_direction / norm(operator_direction)
    relative, analytic, finite = grassmann_directional_error(
        operator -> nested_y_operator(
            initial_nested, forward_peps, (1, 1), operator
        ),
        number,
        operator_direction,
    )
    @info "nested operator-dressed Y directional derivative" relative analytic finite
    @test relative <= 1e-4

    operator_primal, operator_pullback = rrule(
        rule_config,
        nested_y_operator,
        initial_nested,
        forward_peps,
        (1, 1),
        number,
    )
    @test operator_primal ≈ nested_y_operator(
        initial_nested, forward_peps, (1, 1), number
    )
    operator_zero = operator_pullback(ZeroTangent())
    @test length(operator_zero) == 5
    @test all(tangent -> tangent isa NoTangent, operator_zero[1:4])
    @test operator_zero[5] isa ZeroTangent

    invalid_operator = Grassmann(
        Matrix{Float64}(I, 2, 2),
        (2, 2),
        (1, 1),
        (:in, :out),
    )
    @test_throws ArgumentError nested_y_operator(
        initial_nested, forward_peps, (1, 1), invalid_operator
    )
    @test_throws ArgumentError rrule(
        rule_config,
        nested_y_operator,
        initial_nested,
        forward_peps,
        (1, 1),
        invalid_operator,
    )

    function convert_nested_env(env::CTMRGEnv, ::Type{T}) where {T}
        convert_grid(grid) = Matrix{Grassmann{T, tensor_rank(first(grid))}}(
            reshape([convert(tensor, T) for tensor in grid], size(grid))
        )
        return CTMRGEnv{T}(
            convert_grid(env.El), convert_grid(env.Er),
            convert_grid(env.Eu), convert_grid(env.Ed),
            convert_grid(env.Clu), convert_grid(env.Cru),
            convert_grid(env.Cld), convert_grid(env.Crd),
        )
    end

    frozen_env = initialize_nested_environment(initial_nested, 4)
    run_nested_GCTMRG!(
        initial_nested, frozen_env, 4;
        ctmrg_iter=1, verbosity=0, save_iter=0,
    )
    bond = nn_bond(model)

    function nested_inputs(x)
        peps = Square_GPEPS(2, 1, 2, 1, 1, x, false)
        nested = nested_network(peps)
        return peps, nested, convert_nested_env(frozen_env, eltype(x))
    end

    site_energy(operator) = begin
        _, value =
            compute_nested_exp_site(
                initial_nested,
                forward_peps,
                operator,
                frozen_env,
                (1, 1),
            )
        real(value)
    end
    horizontal_energy(x) = begin
        peps, nested, env = nested_inputs(x)
        _, value =
            compute_nested_exp_hbond(nested, peps, bond, env, (1, 1))
        real(value)
    end
    vertical_energy(x) = begin
        peps, nested, env = nested_inputs(x)
        _, value =
            compute_nested_exp_vbond(nested, peps, bond, env, (1, 1))
        real(value)
    end

    for (name, measurement) in (
        ("horizontal", compute_nested_exp_hbond),
        ("vertical", compute_nested_exp_vbond),
    )
        primal, measurement_pullback = rrule(
            rule_config,
            measurement,
            initial_nested,
            forward_peps,
            bond,
            frozen_env,
            (1, 1),
        )
        tangents = measurement_pullback(
            (ZeroTangent(), one(last(primal)))
        )
        @info "nested measurement pullback types" name types=map(typeof, tangents)
        @test length(tangents) == 6
        @test !(tangents[2] isa AbstractZero)
        @test tangents[3] isa NoTangent
        @test tangents[4] isa NoTangent
        @test !(tangents[5] isa AbstractZero)
        @test tangents[6] isa NoTangent
    end

    site_direction = _physical_identity(forward_peps.A[1, 1])
    site_direction = site_direction / norm(site_direction)
    for (name, objective, input, input_direction) in (
        ("site", site_energy, number, site_direction),
        ("horizontal", horizontal_energy, params, direction),
        ("vertical", vertical_energy, params, direction),
    )
        relative, analytic, finite =
            directional_error(objective, input, input_direction)
        @info "nested measurement directional derivative" name relative analytic finite
        @test max(abs(analytic), abs(finite)) > 1e-8
        @test relative <= 1e-4
    end
end
