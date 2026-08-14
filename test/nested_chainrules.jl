using ChainRulesCore
using GrassmannTensorNetworks
using LinearAlgebra
using Random
using Test
using Zygote

@eval GrassmannTensorNetworks global global_sign = auto_sign

import GrassmannTensorNetworks: _nested_ket, _nested_bra

function nested_directional_error(f, x, direction; step=1e-6)
    value, pullback = Zygote.pullback(f, x)
    gradient = only(pullback(one(value)))
    finite = (f(x + step * direction) - f(x - step * direction)) / (2step)
    analytic = real(dot(gradient, direction))
    relative = abs(analytic - finite) /
        max(abs(analytic), abs(finite), eps(Float64))
    return relative, analytic, finite
end

function nested_grassmann_directional_error(transform, x, direction; step=1e-6)
    objective(input) = sum(abs2, transform(input))
    return nested_directional_error(objective, x, direction; step)
end

@testset "Simplified nested reverse rules without CTMRG" begin
    Random.seed!(0x4e455354)
    count = square_gpeps_parameter_count(2, 1, 2, 1, 1)
    params = randn(count)
    direction = randn(count)
    direction ./= norm(direction)

    peps = Square_GPEPS(2, 1, 2, 1, 1, params, false)
    tensor = peps.A[1, 1]
    tensor_direction = Square_GPEPS(
        2, 1, 2, 1, 1, randn(count), false
    ).A[1, 1]
    tensor_direction = tensor_direction / norm(tensor_direction)

    for (name, transform) in (
        ("nested ket", _nested_ket),
        ("nested bra", _nested_bra),
    )
        relative, analytic, finite = nested_grassmann_directional_error(
            transform, tensor, tensor_direction
        )
        @info "simplified nested tensor derivative" name relative analytic finite
        @test relative <= 1e-4
    end

    nested_norm2(x) = begin
        state = Square_GPEPS(2, 1, 2, 1, 1, x, false)
        network = nested_network(state)
        sum(tensor -> sum(abs2, tensor), network.network)
    end
    relative, analytic, finite = nested_directional_error(
        nested_norm2, params, direction
    )
    @info "simplified nested network derivative" relative analytic finite
    @test relative <= 1e-4

    nested = nested_network(peps)
    number = n_site(SpinlessFermionModel(1.0, 1.0, 3.0))
    odd_endpoint = Grassmann(
        [0.0 1.0; 0.0 0.0],
        (2, 2), (1, 1), (:out, :in);
        parity=:odd,
    )
    operator_transform(operator) =
        nested_x_operator(nested, peps, (1, 1), operator)
    for operator in (number, odd_endpoint)
        operator_direction = Grassmann(
            size(operator), even(operator), index_type(operator), Float64;
            init=:random,
            parity=tensor_parity(operator) == 0 ? :even : :odd,
        )
        operator_direction = operator_direction / norm(operator_direction)
        relative, analytic, finite = nested_grassmann_directional_error(
            operator_transform, operator, operator_direction
        )
        parity = tensor_parity(operator)
        @info "operator-dressed X derivative" parity relative analytic finite
        @test relative <= 1e-4
    end
    @test nested_y_operator(nested, peps, (1, 1), number) ≈
        nested_x_operator(nested, peps, (1, 1), number)

    rule_config = Zygote.ZygoteRuleConfig(Zygote.Context())
    network_primal, network_pullback = rrule(
        rule_config,
        nested_network,
        peps,
        NestedLayout(peps),
    )
    @test network_pullback(ZeroTangent())[2] isa ZeroTangent
    structured_zero = Tangent{typeof(network_primal)}(
        ; network=ZeroTangent(),
          layout=NoTangent(),
          x_crossings=NoTangent(),
    )
    @test network_pullback(structured_zero)[2] isa ZeroTangent

    operator_primal, operator_pullback = rrule(
        rule_config,
        nested_x_operator,
        nested,
        peps,
        (1, 1),
        number,
    )
    @test operator_primal ≈ operator_transform(number)
    operator_zero = operator_pullback(ZeroTangent())
    @test length(operator_zero) == 5
    @test all(tangent -> tangent isa NoTangent, operator_zero[1:4])
    @test operator_zero[5] isa ZeroTangent
end
