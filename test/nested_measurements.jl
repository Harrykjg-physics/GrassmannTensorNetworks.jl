using Test
using Random
using LinearAlgebra
using GrassmannTensorNetworks

import GrassmannTensorNetworks:
    _check_nested_operator_unit_cell,
    _bond_operator_gsvd,
    _contract_nested_hpatch3,
    _contract_nested_hpatch3_alpha,
    _contract_nested_vpatch3,
    _contract_nested_vpatch3_alpha,
    _nested_x_bond_operator,
    _nested_scalar_or_zero,
    _match_nested_site_arrows,
    _nested_bra,
    _nested_ket,
    _nested_x,
    _nested_y,
    _layout_bra_site,
    _layout_down_source,
    _layout_ket_site,
    _layout_right_source,
    _layout_y_site,
    _layout_x_site

const CTMRG_INDEX_TYPE = (:out, :in, :in, :out)

function reference_adapt_CTMRG_tensor(t::Grassmann{T, 4}) where {T}
    adapted = t
    for ind in 1:4
        if index_type(adapted)[ind] != CTMRG_INDEX_TYPE[ind]
            if ind in (1, 3)
                # A_ctm[l, r, u, d] = (-1)^i * conjugation_i(A[l, r, u, d])
                adapted = add_parity_sign(
                    index_conjugation(adapted, ind),
                    ind;
                    sign_function=GrassmannTensorNetworks.global_sign,
                )
            else
                # A_ctm[l, r, u, d] = conjugation_i(A[l, r, u, d])
                adapted = index_conjugation(adapted, ind)
            end
        end
    end
    return adapted
end

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

function simplified_nested_impurity(A, operator)
    B = _nested_bra(A)
    X = _nested_x(
        operator,
        size(A)[3], even(A)[3],
        size(A)[4], even(A)[4],
    )
    Y = _nested_y(
        size(A)[2], even(A)[2],
        size(A)[5], even(A)[5],
        eltype(A),
    )
    K = _nested_ket(A)
    return contract_simplified_nested_cell(B, X, Y, K)
end

function bond_factor_slices(left::Grassmann, right::Grassmann)
    total_size = size(left)[3]
    even_size = even(left)[3]
    dense_left = convert(Array, left)
    dense_right = convert(Array, right)
    terms = Tuple{Grassmann, Grassmann}[]
    for alpha in 1:total_size
        parity_symbol = alpha <= even_size ? :even : :odd
        left_slice = Grassmann(
            dense_left[:, :, alpha],
            (size(left)[1], size(left)[2]),
            (even(left)[1], even(left)[2]),
            (index_type(left)[1], index_type(left)[2]);
            parity=parity_symbol,
        )
        right_slice = Grassmann(
            dense_right[:, :, alpha],
            (size(right)[1], size(right)[2]),
            (even(right)[1], even(right)[2]),
            (index_type(right)[1], index_type(right)[2]);
            parity=parity_symbol,
        )
        if norm(left_slice) > 1e-14 && norm(right_slice) > 1e-14
            push!(terms, (left_slice, right_slice))
        end
    end
    return terms
end

function contract_simplified_nested_cell(B, X, Y, K)
    sign = GrassmannTensorNetworks.global_sign

    # BX[lB, uB, dB, rX, uX, dX] = B[lB, R, uB, dB] * X[R, rX, uX, dX]
    BX = contract(B, X, (2, 1); sign_function=sign)
    # BXY[lB, uB, rX, uX, dX, lY, rY, dY] = BX[lB, uB, U, rX, uX, dX] * Y[lY, rY, U, dY]
    BXY = contract(BX, Y, (3, 3); sign_function=sign)
    # cell[lB, uB, rX, uX, lY, dY, rK, dK] = BXY[lB, uB, rX, uX, U, lY, L, dY] * K[L, rK, U, dK]
    cell = contract(BXY, K, ((5, 7), (3, 1)); sign_function=sign)
    # ordered[lB, lY, rX, rK, uB, uX, dY, dK] = cell[lB, uB, rX, uX, lY, dY, rK, dK]
    ordered = permutedims(cell, (1, 5, 3, 7, 2, 4, 6, 8); sign_function=sign)
    # signed[lB, lK, rB, rK, uB, uK, dB, dK] = (-1)^lB ordered[lB, lK, rB, rK, uB, uK, dB, dK]
    signed = add_parity_sign(ordered, 1; sign_function=sign)
    # left[L, rX, rK, uB, uX, dY, dK] = ordered[(lB, lY), rX, rK, uB, uX, dY, dK]
    left = fuse(signed, (1, 2); index_type_fused=:out)
    # left_ordered[L, rB, rK, uB, uK, dB, dK] = (-1)^perm left[L, rB, rK, uB, uK, dB, dK]
    left_ordered = add_perm_sign(left, (1, 3, 2, 4, 5, 6, 7); sign_function=sign)
    # right[L, R, uB, uX, dY, dK] = left[L, (rX, rK), uB, uX, dY, dK]
    right = fuse(left_ordered, (2, 3); index_type_fused=:in)
    # right_ordered[L, R, uB, uK, dB, dK] = (-1)^perm right[L, R, uB, uK, dB, dK]
    right_ordered = add_perm_sign(right, (1, 2, 4, 3, 5, 6); sign_function=sign)
    # up[L, R, U, dY, dK] = right[L, R, (uB, uX), dY, dK]
    up = fuse(right_ordered, (3, 4); index_type_fused=:in)
    # up_signed[L, R, U, dB, dK] = (-1)^dB up[L, R, U, dB, dK]
    up_signed = add_parity_sign(up, 4; sign_function=sign)
    # reduced[L, R, U, D] = up[L, R, U, (dY, dK)]
    return fuse(up_signed, (4, 5); index_type_fused=:out)
end

function close_nested_test_row(row; twist_x=false)
    sign = GrassmannTensorNetworks.global_sign
    cols = length(row)
    current = permutedims(row[1], (1, 3, 4, 2); sign_function=sign)
    for c in 2:cols
        j = c - 1
        rank = 2j + 2
        permutation = (
            1,
            (2:(j + 1))...,
            2j + 3,
            ((j + 2):(2j + 1))...,
            2j + 4,
            2j + 2,
        )
        # current[...] = current[..., east] * row[c][west, ...]
        current = contract(current, row[c], (rank, 1); perm=permutation, sign_function=sign)
    end
    # closed[...] = current[west, ..., east]
    return trace(current, (1, 2cols + 2); pbc=!twist_x, sign_function=sign)
end

function nested_test_torus_scalar(tensors; twist_x=false, twist_y=false)
    sign = GrassmannTensorNetworks.global_sign
    rows, cols = size(tensors)
    row_tensors = [
        close_nested_test_row(collect(tensors[r, :]); twist_x)
        for r in 1:rows
    ]
    current = row_tensors[1]
    for r in 2:rows
        # current[...] = current[..., south] * row_tensors[r][north, ...]
        current = contract(
            current,
            row_tensors[r],
            (ntuple(i -> cols + i, cols), ntuple(identity, cols));
            sign_function=sign,
        )
    end
    # closed[] = current[north..., south...]
    closed = trace(
        current,
        (ntuple(identity, cols), ntuple(i -> cols + i, cols));
        pbc=ntuple(_ -> !twist_y, cols),
        sign_function=sign,
    )
    return scalar(closed)
end

function reblocked_nested_network(
    nested::NestedNetwork,
    tensors::AbstractMatrix=nested.network)

    T = eltype(first(nested.network))
    reblocked = Matrix{Grassmann{T, 4}}(undef, nested.layout.source_size...)
    for source in CartesianIndices(reblocked)
        B = tensors[_layout_bra_site(nested.layout, source)]
        X = tensors[_layout_x_site(nested.layout, source)]
        Y = tensors[_layout_y_site(nested.layout, source)]
        K = tensors[_layout_ket_site(nested.layout, source)]
        # reduced[source] = contract([B X; Y K])
        reblocked[source] = contract_simplified_nested_cell(B, X, Y, K)
    end
    return reblocked
end

function deterministic_nested_tensor(::Type{T}, sizes, evens) where {T}
    A = Grassmann(
        sizes,
        evens,
        (:out, :out, :in, :in, :out),
        T;
        init=:zeros,
    )
    for (block_number, sector) in enumerate(sort!(collect(nonzero_keys(A))))
        block = A[sector]
        for (coordinate_number, coordinate) in enumerate(CartesianIndices(block))
            serial = 10_000 * block_number + coordinate_number
            block[coordinate] = T <: Complex ? T(serial, -2 * serial - 1) : T(serial)
        end
    end
    return A
end

function tensor_comparison(candidate, target)
    candidate_dense = convert(Array, candidate)
    target_dense = convert(Array, target)
    delta = candidate_dense - target_dense
    target_norm = norm(target_dense)
    return (
        max_element_error=maximum(abs, delta; init=0.0),
        norm_error=norm(delta),
        relative_norm_error=norm(delta) / max(target_norm, eps(Float64)),
    )
end

function test_strict_tensor_equal(candidate, target; atol, rtol)
    @test size(candidate) == size(target)
    @test even(candidate) == even(target)
    @test index_type(candidate) == index_type(target)
    candidate_keys = sort!(collect(nonzero_keys(candidate)))
    target_keys = sort!(collect(nonzero_keys(target)))
    @test candidate_keys == target_keys
    for key in candidate_keys
        @test candidate[key] ≈ target[key] atol=atol rtol=rtol
    end
end

@testset "Nested measurement impurities match the supplied network arrows" begin
    Random.seed!(0x43544d5247)
    peps = Square_GPEPS(2, 1, 2, 2, 2, Float64, false)
    source = CartesianIndex(1, 1)
    number = n_site(SpinlessFermionModel(1.0, 1.0, 3.0))
    nested = nested_network(peps)
    adapted = adapt_CTMRG(nested)
    xsite = _layout_x_site(nested.layout, source)

    raw_impurity = nested_x_operator(nested, peps, source, number)
    @test _match_nested_site_arrows(raw_impurity, nested[xsite]) ≈
        raw_impurity

    adapted_impurity = _match_nested_site_arrows(
        raw_impurity, adapted[xsite]
    )
    @test index_type(adapted_impurity) == index_type(adapted[xsite])
    @test size(adapted_impurity) == size(raw_impurity)
    @test even(adapted_impurity) == even(raw_impurity)
    @test adapted_impurity ≈ reference_adapt_CTMRG_tensor(raw_impurity)
end

@testset "Rank-3 GSVD bond factors reconstruct bond operators" begin
    for bond in (
        two_site_identity(),
        nn_bond(SpinlessFermionModel(1.0, 1.0, 3.0)),
    )
        left, right = _bond_operator_gsvd(bond)
        @test length(size(left)) == 3
        @test length(size(right)) == 3
        @test (index_type(left)[1], index_type(left)[2]) == (:out, :in)
        @test (index_type(right)[1], index_type(right)[2]) == (:out, :in)
        @test index_type(left)[3] != index_type(right)[3]

        # reconstructed[po1, pi1, po2, pi2] = L[po1, pi1, a] * R[po2, pi2, a]
        reconstructed = contract(
            left, right, (3, 3);
            sign_function=GrassmannTensorNetworks.global_sign,
        )
        # ordered[po1, po2, pi1, pi2] = reconstructed[po1, pi1, po2, pi2]
        ordered = permutedims(
            reconstructed, (1, 3, 2, 4);
            sign_function=GrassmannTensorNetworks.global_sign,
        )
        @test ordered ≈ bond rtol=1e-12 atol=1e-12
    end
    @test !isdefined(GrassmannTensorNetworks, :_operator_schmidt)
end

@testset "Strict local X impurities versus reduced tensors" begin
    A = deterministic_nested_tensor(
        Float64,
        (2, 2, 2, 2, 2),
        (1, 1, 1, 1, 1),
    )
    bond = nn_bond(SpinlessFermionModel(1.0, 1.0, 3.0))
    left, right = _bond_operator_gsvd(bond)
    factors = reduce(vcat, collect(term) for term in
        bond_factor_slices(left, right))
    @test any(operator -> tensor_parity(operator) == 0, factors)
    @test any(operator -> tensor_parity(operator) == 1, factors)
    for operator in factors
        candidate = simplified_nested_impurity(A, operator)
        target = reduced_tensor(A, operator)
        data = tensor_comparison(candidate, target)
        @info "strict local X impurity" parity=tensor_parity(operator) data
        test_strict_tensor_equal(
            candidate,
            target;
            atol=5e-12,
            rtol=5e-12,
        )
    end
end

@testset "Alpha-connected bond patches match explicit factor sums" begin
    hamiltonian = nn_bond(SpinlessFermionModel(1.0, 1.0, 3.0))

    Random.seed!(0x48414c5048)
    hpeps = Square_GPEPS(2, 1, 2, 1, 2, Float64, false)
    hnested = adapt_CTMRG(nested_network(hpeps))
    henv = initialize_nested_environment(hnested, 4, 2)
    hsource = CartesianIndex(1, 1)
    hneighbor = _layout_right_source(hnested.layout, hsource)
    hleft, hright = _bond_operator_gsvd(hamiltonian)
    hleft_x = _nested_x_bond_operator(hnested, hpeps, hsource, hleft)
    hright_x = _nested_x_bond_operator(hnested, hpeps, hneighbor, hright)
    hright_x = add_parity_sign(
        hright_x, 5; sign_function=GrassmannTensorNetworks.global_sign
    )
    halpha = _contract_nested_hpatch3_alpha(
        hnested, henv, hsource, hleft_x, hright_x
    )
    hsum = zero(eltype(hamiltonian))
    for (left_slice, right_slice) in bond_factor_slices(hleft, hright)
        left_x = nested_x_operator(hnested, hpeps, hsource, left_slice)
        right_x = nested_x_operator(hnested, hpeps, hneighbor, right_slice)
        term = _contract_nested_hpatch3(
            hnested, henv, hsource, left_x, right_x
        )
        hsum += _nested_scalar_or_zero(term)
    end
    @test _nested_scalar_or_zero(halpha) ≈ hsum rtol=1e-12 atol=1e-12

    Random.seed!(0x56414c5056)
    vpeps = Square_GPEPS(2, 1, 2, 2, 1, Float64, false)
    vnested = adapt_CTMRG(nested_network(vpeps))
    venv = initialize_nested_environment(vnested, 4, 2)
    vsource = CartesianIndex(1, 1)
    vneighbor = _layout_down_source(vnested.layout, vsource)
    vtop, vbottom = _bond_operator_gsvd(hamiltonian)
    vtop_x = _nested_x_bond_operator(vnested, vpeps, vsource, vtop)
    vbottom_x = _nested_x_bond_operator(vnested, vpeps, vneighbor, vbottom)
    valpha = _contract_nested_vpatch3_alpha(
        vnested,
        venv,
        vsource,
        add_parity_sign(vtop_x, 5; sign_function=GrassmannTensorNetworks.global_sign),
        vbottom_x,
    )
    vsum = zero(eltype(hamiltonian))
    for (top_slice, bottom_slice) in bond_factor_slices(vtop, vbottom)
        top_x = nested_x_operator(vnested, vpeps, vsource, top_slice)
        bottom_x = nested_x_operator(vnested, vpeps, vneighbor, bottom_slice)
        term = _contract_nested_vpatch3(
            vnested, venv, vsource, top_x, bottom_x
        )
        vsum += _nested_scalar_or_zero(term)
    end
    @test _nested_scalar_or_zero(valpha) ≈ vsum rtol=1e-12 atol=1e-12
end

function exact_site_networks(
    peps,
    operator,
    source::CartesianIndex{2},
)
    nested = nested_network(peps)
    nested_impurity = copy(nested.network)
    nested_impurity[_layout_x_site(nested.layout, source)] =
        nested_x_operator(nested, peps, source, operator)

    T = eltype(peps)
    reduced_bulk =
        Matrix{Grassmann{T, 4}}(reduced_tensor.(peps.A))
    reduced_impurity = copy(reduced_bulk)
    reduced_impurity[source] =
        reduced_tensor(peps.A[source], operator)

    return (
        nested=nested,
        nested_bulk=reblocked_nested_network(nested),
        nested_impurity=reblocked_nested_network(
            nested, nested_impurity
        ),
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

    @test nested_x_operator(nested, peps, source, identity) ≈
        nested[_layout_x_site(nested.layout, source)]
    @test nested_y_operator(nested, peps, source, identity) ≈
        nested_x_operator(nested, peps, source, identity)

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
    @test_throws ArgumentError nested_x_operator(
        nested, peps, (2, 1), number
    )
    @test_throws DimensionMismatch nested_x_operator(
        nested, peps, source, wrong_size
    )
    @test_throws DimensionMismatch nested_x_operator(
        nested, peps, source, wrong_even
    )
    @test_throws ArgumentError nested_x_operator(
        nested, peps, source, wrong_arrows
    )

    operators = fill(number, size(peps))
    @test _check_nested_operator_unit_cell(peps, operators) === nothing
    @test_throws DimensionMismatch _check_nested_operator_unit_cell(
        peps, fill(number, 1, 3)
    )

    identity_networks = exact_site_networks(peps, identity, source)
    number_networks = exact_site_networks(peps, number, source)
    for networks in (identity_networks, number_networks)
        test_strict_tensor_equal(
            networks.nested_impurity[source],
            networks.reduced_impurity[source];
            atol=1e-12,
            rtol=5e-12,
        )
    end
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
        _layout_right_source(nested.layout, source) :
        _layout_down_source(nested.layout, source)
    total = zero(nested_test_torus_scalar(
        nested.network; twist_x, twist_y
    ))
    left_factor, right_factor = _bond_operator_gsvd(operator)
    for (left_operator, right_operator) in
        bond_factor_slices(left_factor, right_factor)
        impurity = copy(nested.network)
        impurity[_layout_x_site(nested.layout, source)] =
            nested_x_operator(
                nested, peps, source, left_operator
            )
        impurity[_layout_x_site(nested.layout, neighbor)] =
            nested_x_operator(
                nested, peps, neighbor, right_operator
            )
        term_sign = orientation === :horizontal ?
            (-one(eltype(operator)))^tensor_parity(left_operator) :
            one(eltype(operator))
        raw_term = nested_test_torus_scalar(
            reblocked_nested_network(nested, impurity);
            twist_x,
            twist_y,
        )
        total += term_sign * raw_term
    end
    return total
end

@testset "Exact nested nearest-neighbor measurements" begin
    identity2 = two_site_identity()
    hamiltonian =
        nn_bond(SpinlessFermionModel(1.0, 1.0, 3.0))

    left_factor, right_factor = _bond_operator_gsvd(hamiltonian)
    reconstructed = contract(
        left_factor, right_factor, (3, 3);
        sign_function=GrassmannTensorNetworks.global_sign,
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
            reblocked_nested_network(hnested); twist_x, twist_y
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
            reblocked_nested_network(vnested); twist_x, twist_y
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
