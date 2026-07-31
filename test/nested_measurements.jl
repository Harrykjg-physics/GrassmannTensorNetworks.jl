using Test
using Random
using GrassmannTensorNetworks

import GrassmannTensorNetworks: _check_nested_operator_unit_cell

function physical_identity(::Type{T}=Float64) where {T}
    return Grassmann(
        Matrix{T}(I, 2, 2), (2, 2), (1, 1), (:out, :in)
    )
end

const MEASUREMENT_SPIN_STRUCTURES =
    ((false, false), (true, false), (false, true), (true, true))

function exact_site_networks(
    peps,
    operator,
    source::CartesianIndex{2},
)
    nested = nested_network(peps)
    nested_impurity = copy(nested.network)
    nested_impurity[nested.layout.y_sites[source]] =
        nested_y_operator(nested, peps, source, operator)

    T = eltype(peps)
    reduced_bulk =
        Matrix{Grassmann{T, 4}}(reduced_tensor.(peps.A))
    reduced_impurity = copy(reduced_bulk)
    reduced_impurity[source] =
        reduced_tensor(peps.A[source], operator)

    return (
        nested=nested,
        nested_bulk=nested.network,
        nested_impurity=nested_impurity,
        reduced_bulk=reduced_bulk,
        reduced_impurity=reduced_impurity,
    )
end

function exact_site_scalars(
    networks;
    twist_x::Bool,
    twist_y::Bool,
)
    close(tensors) = nested_test_torus_scalar(
        tensors; twist_x, twist_y
    )
    return (
        nested_denominator=close(networks.nested_bulk),
        nested_numerator=close(networks.nested_impurity),
        reduced_denominator=close(networks.reduced_bulk),
        reduced_numerator=close(networks.reduced_impurity),
    )
end

@testset "Exact nested one-site measurements" begin
    Random.seed!(0x4e455354)
    peps = Square_GPEPS(2, 1, 2, 1, 2, Float64, false)
    source = CartesianIndex(1, 1)
    nested = nested_network(peps)
    identity = physical_identity()
    number = n_site(SpinlessFermionModel(1.0, 1.0, 3.0))

    @test nested_y_operator(nested, peps, source, identity) ≈
        nested[nested.layout.y_sites[source]]

    wrong_size = Grassmann(
        Matrix{Float64}(I, 4, 4),
        (4, 4), (2, 2), (:out, :in),
    )
    wrong_even = Grassmann(
        Matrix{Float64}(I, 2, 2),
        (2, 2), (2, 2), (:out, :in),
    )
    wrong_arrows = Grassmann(
        Matrix{Float64}(I, 2, 2),
        (2, 2), (1, 1), (:in, :out),
    )
    @test_throws ArgumentError nested_y_operator(
        nested, peps, (2, 1), number
    )
    @test_throws DimensionMismatch nested_y_operator(
        nested, peps, source, wrong_size
    )
    @test_throws DimensionMismatch nested_y_operator(
        nested, peps, source, wrong_even
    )
    @test_throws ArgumentError nested_y_operator(
        nested, peps, source, wrong_arrows
    )

    operators = fill(number, size(peps))
    @test _check_nested_operator_unit_cell(peps, operators) === nothing
    @test_throws DimensionMismatch _check_nested_operator_unit_cell(
        peps, fill(number, 1, 3)
    )

    identity_networks = exact_site_networks(peps, identity, source)
    number_networks = exact_site_networks(peps, number, source)
    number_numerators = Float64[]
    for (twist_x, twist_y) in MEASUREMENT_SPIN_STRUCTURES
        identity_data = exact_site_scalars(
            identity_networks; twist_x, twist_y
        )
        @test identity_data.nested_denominator ≈
            identity_data.reduced_denominator rtol=5e-12 atol=1e-12
        @test identity_data.nested_numerator ≈
            identity_data.reduced_numerator rtol=5e-12 atol=1e-12
        @test identity_data.nested_numerator /
            identity_data.nested_denominator ≈ 1 atol=1e-12

        number_data = exact_site_scalars(
            number_networks; twist_x, twist_y
        )
        @test number_data.nested_denominator ≈
            number_data.reduced_denominator rtol=5e-12 atol=1e-12
        @test number_data.nested_numerator ≈
            number_data.reduced_numerator rtol=5e-12 atol=1e-12
        @test number_data.nested_numerator /
            number_data.nested_denominator ≈
            number_data.reduced_numerator /
            number_data.reduced_denominator rtol=5e-12 atol=1e-12
        push!(number_numerators, abs(number_data.nested_numerator))
    end
    @test maximum(number_numerators) > 1e-12
end
