using Test
using Random
using GrassmannTensorNetworks

import GrassmannTensorNetworks: _check_nested_operator_unit_cell

function physical_identity(::Type{T}=Float64) where {T}
    return Grassmann(
        Matrix{T}(I, 2, 2), (2, 2), (1, 1), (:out, :in)
    )
end

function two_site_identity(::Type{T}=Float64) where {T}
    identity = physical_identity(T)
    product = contract(
        identity, identity;
        sign_function=GrassmannTensorNetworks.global_sign,
    )
    return permutedims(
        product, (1, 3, 2, 4);
        sign_function=GrassmannTensorNetworks.global_sign,
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

function close_reduced_hbond(
    tensor::Grassmann;
    twist_x::Bool,
    twist_y::Bool,
)
    closed = trace(
        tensor,
        ((1, 3, 4), (2, 5, 6));
        pbc=(!twist_x, !twist_y, !twist_y),
        sign_function=GrassmannTensorNetworks.global_sign,
    )
    return scalar(closed)
end

function close_reduced_vbond(
    tensor::Grassmann;
    twist_x::Bool,
    twist_y::Bool,
)
    closed = trace(
        tensor,
        ((1, 2, 5), (3, 4, 6));
        pbc=(!twist_x, !twist_x, !twist_y),
        sign_function=GrassmannTensorNetworks.global_sign,
    )
    return scalar(closed)
end

function exact_nested_bond_numerator(
    nested::NestedNetwork,
    peps::Square_GPEPS,
    operator::Grassmann,
    source::CartesianIndex{2},
    orientation::Symbol;
    twist_x::Bool,
    twist_y::Bool,
)
    neighbor = orientation === :horizontal ?
        CartesianIndex(
            source[1], Nmod(source[2] + 1, size(peps)[2])
        ) :
        CartesianIndex(
            Nmod(source[1] + 1, size(peps)[1]), source[2]
        )
    total = zero(nested_test_torus_scalar(
        nested.network; twist_x, twist_y
    ))
    for (left_operator, right_operator) in
        GrassmannTensorNetworks._operator_schmidt(operator)
        impurity = copy(nested.network)
        impurity[nested.layout.y_sites[source]] =
            nested_y_operator(
                nested, peps, source, left_operator
            )
        impurity[nested.layout.y_sites[neighbor]] =
            nested_y_operator(
                nested, peps, neighbor, right_operator
            )
        term_sign = orientation === :horizontal ?
            (-one(eltype(operator)))^tensor_parity(left_operator) :
            one(eltype(operator))
        total += term_sign * nested_test_torus_scalar(
            impurity; twist_x, twist_y
        )
    end
    return total
end

@testset "Exact nested nearest-neighbor measurements" begin
    identity2 = two_site_identity()
    hamiltonian =
        nn_bond(SpinlessFermionModel(1.0, 1.0, 3.0))

    terms = GrassmannTensorNetworks._operator_schmidt(hamiltonian)
    reconstructed = sum(
        contract(
            left, right;
            sign_function=GrassmannTensorNetworks.global_sign,
        ) for (left, right) in terms
    )
    ordered = permutedims(
        reconstructed, (1, 3, 2, 4);
        sign_function=GrassmannTensorNetworks.global_sign,
    )
    @test ordered ≈ hamiltonian rtol=1e-12 atol=1e-12

    Random.seed!(0x48424f4e44)
    hpeps = Square_GPEPS(2, 1, 2, 1, 2, Float64, false)
    hnested = nested_network(hpeps)
    hsource = CartesianIndex(1, 1)
    hdenominator_reference = reduced_tensor_hbond(
        hpeps.A[1, 1], hpeps.A[1, 2], identity2
    )
    hnumerator_reference = reduced_tensor_hbond(
        hpeps.A[1, 1], hpeps.A[1, 2], hamiltonian
    )

    Random.seed!(0x56424f4e44)
    vpeps = Square_GPEPS(2, 1, 2, 2, 1, Float64, false)
    vnested = nested_network(vpeps)
    vsource = CartesianIndex(1, 1)
    vdenominator_reference = reduced_tensor_vbond(
        vpeps.A[1, 1], vpeps.A[2, 1], identity2
    )
    vnumerator_reference = reduced_tensor_vbond(
        vpeps.A[1, 1], vpeps.A[2, 1], hamiltonian
    )

    @test GrassmannTensorNetworks._check_nested_bond_operator(
        hpeps, hamiltonian,
        CartesianIndex(1, 1), CartesianIndex(1, 2),
    ) === nothing
    wrong_size = Grassmann(
        (3, 2, 3, 2), (1, 1, 1, 1),
        (:out, :out, :in, :in), Float64;
        init=:zeros,
    )
    wrong_even = Grassmann(
        (2, 2, 2, 2), (2, 1, 2, 1),
        (:out, :out, :in, :in), Float64;
        init=:zeros,
    )
    wrong_arrows = Grassmann(
        (2, 2, 2, 2), (1, 1, 1, 1),
        (:in, :out, :out, :in), Float64;
        init=:zeros,
    )
    odd_operator = Grassmann(
        (2, 2, 2, 2), (1, 1, 1, 1),
        (:out, :out, :in, :in), Float64;
        init=:zeros, parity=:odd,
    )
    @test_throws ArgumentError GrassmannTensorNetworks._check_nested_bond_operator(
        hpeps, hamiltonian,
        CartesianIndex(2, 1), CartesianIndex(1, 2),
    )
    @test_throws DimensionMismatch GrassmannTensorNetworks._check_nested_bond_operator(
        hpeps, wrong_size,
        CartesianIndex(1, 1), CartesianIndex(1, 2),
    )
    @test_throws DimensionMismatch GrassmannTensorNetworks._check_nested_bond_operator(
        hpeps, wrong_even,
        CartesianIndex(1, 1), CartesianIndex(1, 2),
    )
    @test_throws ArgumentError GrassmannTensorNetworks._check_nested_bond_operator(
        hpeps, wrong_arrows,
        CartesianIndex(1, 1), CartesianIndex(1, 2),
    )
    @test_throws ArgumentError GrassmannTensorNetworks._check_nested_bond_operator(
        hpeps, odd_operator,
        CartesianIndex(1, 1), CartesianIndex(1, 2),
    )

    bond_numerators = Float64[]
    for (twist_x, twist_y) in MEASUREMENT_SPIN_STRUCTURES
        hdenominator = nested_test_torus_scalar(
            hnested.network; twist_x, twist_y
        )
        hnumerator = exact_nested_bond_numerator(
            hnested, hpeps, hamiltonian, hsource, :horizontal;
            twist_x, twist_y
        )
        hdenominator_reduced = close_reduced_hbond(
            hdenominator_reference; twist_x, twist_y
        )
        hnumerator_reduced = close_reduced_hbond(
            hnumerator_reference; twist_x, twist_y
        )
        @test hdenominator ≈ hdenominator_reduced rtol=5e-12 atol=1e-12
        @test hnumerator ≈ hnumerator_reduced rtol=5e-12 atol=1e-12
        @test hnumerator / hdenominator ≈
            hnumerator_reduced / hdenominator_reduced rtol=5e-12 atol=1e-12

        vdenominator = nested_test_torus_scalar(
            vnested.network; twist_x, twist_y
        )
        vnumerator = exact_nested_bond_numerator(
            vnested, vpeps, hamiltonian, vsource, :vertical;
            twist_x, twist_y
        )
        vdenominator_reduced = close_reduced_vbond(
            vdenominator_reference; twist_x, twist_y
        )
        vnumerator_reduced = close_reduced_vbond(
            vnumerator_reference; twist_x, twist_y
        )
        @test vdenominator ≈ vdenominator_reduced rtol=5e-12 atol=1e-12
        @test vnumerator ≈ vnumerator_reduced rtol=5e-12 atol=1e-12
        @test vnumerator / vdenominator ≈
            vnumerator_reduced / vdenominator_reduced rtol=5e-12 atol=1e-12

        push!(bond_numerators, abs(hnumerator), abs(vnumerator))
    end
    @test all(value -> value > 1e-12, bond_numerators)
end

@testset "Public nested bond measurements sum multiple Schmidt terms" begin
    Random.seed!(0x5055424c4943)
    peps = Square_GPEPS(2, 1, 2, 1, 1, Float64, false)
    nested = nested_network(peps)
    env = initialize_nested_environment(nested, 4)
    source = CartesianIndex(1, 1)
    identity = physical_identity()
    bond = nn_bond(SpinlessFermionModel(1.0, 1.0, 3.0))
    terms = GrassmannTensorNetworks._operator_schmidt(bond)
    @test length(terms) > 1
    zero_complex_bond = Grassmann(
        size(bond), even(bond), index_type(bond), ComplexF64;
        init=:zeros, parity=:even,
    )
    @test isempty(
        GrassmannTensorNetworks._operator_schmidt(zero_complex_bond)
    )

    closed = nested_y_operator(nested, peps, source, identity)
    scalar_or_zero(value) = isempty(nonzero_keys(value)) ?
        zero(eltype(value)) : scalar(value)

    horizontal_denominator =
        GrassmannTensorNetworks._contract_nested_hpatch3(
            nested, env, source, closed, closed
        )
    horizontal_numerator = sum(terms; init=zero(Float64)) do term
        left_operator, right_operator = term
        left_y = nested_y_operator(
            nested, peps, source, left_operator
        )
        right_y = nested_y_operator(
            nested, peps, source, right_operator
        )
        sign = (-one(eltype(bond)))^tensor_parity(left_operator)
        term = GrassmannTensorNetworks._contract_nested_hpatch3(
            nested, env, source, left_y, right_y
        )
        sign * scalar_or_zero(term)
    end

    vertical_denominator =
        GrassmannTensorNetworks._contract_nested_vpatch3(
            nested, env, source, closed, closed
        )
    vertical_numerator = sum(terms; init=zero(Float64)) do term
        top_operator, bottom_operator = term
        top_y = nested_y_operator(
            nested, peps, source, top_operator
        )
        bottom_y = nested_y_operator(
            nested, peps, source, bottom_operator
        )
        term = GrassmannTensorNetworks._contract_nested_vpatch3(
            nested, env, source, top_y, bottom_y
        )
        scalar_or_zero(term)
    end

    @testset "horizontal" begin
        denominator, value = compute_nested_exp_hbond(
            nested, peps, bond, env, source
        )
        @test scalar(denominator) ≈ scalar(horizontal_denominator)
        @test value ≈
            horizontal_numerator / scalar(horizontal_denominator)
    end

    @testset "vertical" begin
        denominator, value = compute_nested_exp_vbond(
            nested, peps, bond, env, source
        )
        @test scalar(denominator) ≈ scalar(vertical_denominator)
        @test value ≈ vertical_numerator / scalar(vertical_denominator)
    end

    @testset "zero complex operator" begin
        for compute_bond in (
            compute_nested_exp_hbond,
            compute_nested_exp_vbond,
        )
            _, value = compute_bond(
                nested, peps, zero_complex_bond, env, source
            )
            @test value isa ComplexF64
            @test iszero(value)
        end
    end
end
