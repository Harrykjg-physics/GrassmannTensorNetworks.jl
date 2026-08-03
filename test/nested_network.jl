using Test
using Random
using LinearAlgebra
using GrassmannTensorNetworks

@eval GrassmannTensorNetworks global global_sign = auto_sign

@test isdefined(GrassmannTensorNetworks, :nested_network)
@test isdefined(GrassmannTensorNetworks, :initialize_nested_environment)
@test isdefined(GrassmannTensorNetworks, :run_nested_GCTMRG!)

import GrassmannTensorNetworks:
    _nested_ket,
    _nested_bra,
    _nested_x,
    _nested_y,
    _layout_ket_site,
    _layout_bra_site,
    _layout_x_site,
    _layout_y_site

@testset "Nested layout follows the CTMRG coordinate convention" begin
    layout = NestedLayout((2, 2))
    @test size(layout) == (4, 4)
    @test layout.source_size == (2, 2)

    expected_bra = [
        CartesianIndex(3, 1) CartesianIndex(3, 3)
        CartesianIndex(1, 1) CartesianIndex(1, 3)
    ]
    expected_x = [
        CartesianIndex(3, 4) CartesianIndex(3, 2)
        CartesianIndex(1, 4) CartesianIndex(1, 2)
    ]
    expected_y = [
        CartesianIndex(2, 1) CartesianIndex(2, 3)
        CartesianIndex(4, 1) CartesianIndex(4, 3)
    ]
    expected_ket = [
        CartesianIndex(2, 4) CartesianIndex(2, 2)
        CartesianIndex(4, 4) CartesianIndex(4, 2)
    ]

    @test layout.bra_sites == expected_bra
    @test layout.x_sites == expected_x
    @test layout.y_sites == expected_y
    @test layout.ket_sites == expected_ket
    @test_throws ArgumentError NestedLayout((0, 3))

    peps = Square_GPEPS(2, 1, 2, 1, 2, Float64, false)
    @test NestedLayout(peps).source_size == (1, 2)
end

@testset "Nested network wrapper" begin
    tensor = Grassmann(
        (1, 1, 1, 1),
        (1, 1, 1, 1),
        (:in, :out, :in, :out),
        Float64;
        init=:zeros,
    )
    network = Matrix{Grassmann{Float64, 4}}(undef, 2, 3)
    fill!(network, tensor)
    nested = NestedNetwork(network, NestedLayout((1, 1)), trues(1, 1))

    @test size(nested) == (2, 3)
    @test size(nested, 2) == 3
    @test axes(nested) == axes(network)
    @test nested[2, 3] === network[2, 3]
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
        for (coordinate_number, coordinate) in
            enumerate(CartesianIndices(block))
            serial = 10_000 * block_number + coordinate_number
            block[coordinate] =
                T <: Complex ? T(serial, -2 * serial - 1) : T(serial)
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
        relative_norm_error=
            norm(delta) / max(target_norm, eps(Float64)),
    )
end

function test_strict_tensor_equal(candidate, target; atol, rtol)
    @test size(candidate) == size(target)
    @test even(candidate) == even(target)
    @test index_type(candidate) == index_type(target)
    candidate_keys = sort!(collect(nonzero_keys(candidate)))
    target_keys = sort!(collect(nonzero_keys(target)))
    @test candidate_keys == target_keys
    for sector in intersect(candidate_keys, target_keys)
        @test candidate[sector] ≈ target[sector] atol=atol rtol=rtol
    end
    data = tensor_comparison(candidate, target)
    target_norm = norm(convert(Array, target))
    @test data.max_element_error <= atol + rtol * target_norm
    @test data.norm_error <= atol + rtol * target_norm
    @test data.relative_norm_error <= rtol
    @test convert(Array, candidate) ≈
        convert(Array, target) atol=atol rtol=rtol
    return data
end

function contract_simplified_nested_cell(B, X, Y, K)
    sign = GrassmannTensorNetworks.global_sign

    # BX[lB, uB, dB, rX, uX, dX] = B[lB, R, uB, dB] * X[R, rX, uX, dX]
    BX = contract(B, X, (2, 1); sign_function=sign)
    # BXY[lB, uB, rX, uX, dX, lY, rY, dY] = BX[lB, uB, U, rX, uX, dX] * Y[lY, rY, U, dY]
    BXY = contract(BX, Y, (3, 3); sign_function=sign)
    # cell[lB, uB, rX, uX, lY, dY, rK, dK] = BXY[lB, uB, rX, uX, U, lY, L, dY] * K[L, rK, U, dK]
    cell = contract(
        BXY,
        K,
        ((5, 7), (3, 1));
        sign_function=sign,
    )
    # ordered[lB, lY, rX, rK, uB, uX, dY, dK] = cell[lB, uB, rX, uX, lY, dY, rK, dK]
    ordered = permutedims(
        cell,
        (1, 5, 3, 7, 2, 4, 6, 8);
        sign_function=sign,
    )
    # signed[lB, lK, rB, rK, uB, uK, dB, dK] = (-1)^lB ordered[lB, lK, rB, rK, uB, uK, dB, dK]
    signed = add_parity_sign(ordered, 1; sign_function=sign)
    # left[L, rX, rK, uB, uX, dY, dK] = ordered[(lB, lY), rX, rK, uB, uX, dY, dK]
    left = fuse(signed, (1, 2); index_type_fused=:out)
    # left_ordered[L, rB, rK, uB, uK, dB, dK] = (-1)^perm left[L, rB, rK, uB, uK, dB, dK]
    left_ordered = add_perm_sign(
        left,
        (1, 3, 2, 4, 5, 6, 7);
        sign_function=sign,
    )
    # right[L, R, uB, uX, dY, dK] = left[L, (rX, rK), uB, uX, dY, dK]
    right = fuse(left_ordered, (2, 3); index_type_fused=:in)
    # right_ordered[L, R, uB, uK, dB, dK] = (-1)^perm right[L, R, uB, uK, dB, dK]
    right_ordered = add_perm_sign(
        right,
        (1, 2, 4, 3, 5, 6);
        sign_function=sign,
    )
    # up[L, R, U, dY, dK] = right[L, R, (uB, uX), dY, dK]
    up = fuse(right_ordered, (3, 4); index_type_fused=:in)
    # up_signed[L, R, U, dB, dK] = (-1)^dB up[L, R, U, dB, dK]
    up_signed = add_parity_sign(up, 4; sign_function=sign)
    # reduced[L, R, U, D] = up[L, R, U, (dY, dK)]
    return fuse(up_signed, (4, 5); index_type_fused=:out)
end

function simplified_nested_cell(A)
    K = _nested_ket(A)
    B = _nested_bra(A)
    X = _nested_x(
        size(A)[3],
        even(A)[3],
        size(A)[4],
        even(A)[4],
        size(A)[1],
        even(A)[1],
        eltype(A),
    )
    Y = _nested_y(
        size(A)[2],
        even(A)[2],
        size(A)[5],
        even(A)[5],
        eltype(A),
    )
    return contract_simplified_nested_cell(B, X, Y, K)
end

@testset "Simplified local tensors" begin
    A = deterministic_nested_tensor(
        Float64,
        (2, 2, 2, 2, 2),
        (1, 1, 1, 1, 1),
    )
    K = _nested_ket(A)
    B = _nested_bra(A)
    X = _nested_x(2, 1, 2, 1, 2, 1, Float64)
    Y = _nested_y(2, 1, 2, 1, Float64)

    @test size(K) == (2, 2, 4, 2)
    @test size(B) == (2, 4, 2, 2)
    @test size(X) == (4, 2, 2, 4)
    @test size(Y) == (2, 2, 2, 2)
    @test index_type(K) == (:out, :in, :in, :out)
    @test index_type(B) == (:out, :in, :in, :out)
    @test index_type(X) == (:out, :in, :in, :out)
    @test index_type(Y) == (:out, :in, :in, :out)

    Xdense = convert(Array, X)
    Ydense = convert(Array, Y)
    @test count(!iszero, Xdense) == 8
    @test count(!iszero, Ydense) == 4
end

@testset "Strict local nested versus reduced tensor" begin
    cases = (
        ((2, 2, 2, 2, 2), (1, 1, 1, 1, 1)),
        ((3, 3, 4, 3, 4), (2, 1, 3, 2, 1)),
    )
    for T in (Float64, ComplexF64), (sizes, evens) in cases
        A = deterministic_nested_tensor(T, sizes, evens)
        candidate = simplified_nested_cell(A)
        target = reduced_tensor(A)
        data = test_strict_tensor_equal(
            candidate,
            target;
            atol=5e-12,
            rtol=5e-12,
        )
        @info "strict local nested comparison" T sizes data
    end
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
        current = contract(
            current,
            row[c],
            (rank, 1);
            perm=permutation,
            sign_function=sign,
        )
    end
    # closed[...] = current[west, ..., east]
    return trace(
        current,
        (1, 2cols + 2);
        pbc=!twist_x,
        sign_function=sign,
    )
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

function mixed_periodic_peps(::Type{T}) where {T}
    horizontal = [(2, 1) (3, 2); (4, 3) (2, 1)]
    vertical = [(3, 2) (2, 1); (2, 1) (4, 3)]
    tensors = Matrix{Grassmann{T, 5}}(undef, 2, 2)
    for r in 1:2, c in 1:2
        left = horizontal[r, Nmod(c - 1, 2)]
        right = horizontal[r, c]
        up = vertical[Nmod(r - 1, 2), c]
        down = vertical[r, c]
        sizes = (3, left[1], right[1], up[1], down[1])
        evens = (2, left[2], right[2], up[2], down[2])
        tensors[r, c] = deterministic_nested_tensor(T, sizes, evens)
    end
    return Square_GPEPS{T}(tensors, missing, missing)
end

function reblocked_nested_network(
    nested::NestedNetwork,
    tensors::AbstractMatrix=nested.network,
)
    T = eltype(first(nested.network))
    reblocked = Matrix{Grassmann{T, 4}}(
        undef, nested.layout.source_size...
    )
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

@testset "Strict periodic nested versus reduced network" begin
    spin_structures =
        ((false, false), (true, false), (false, true), (true, true))
    for (rows, cols) in ((1, 1), (1, 2), (2, 1), (2, 2))
        Random.seed!(10_000rows + cols)
        peps = Square_GPEPS(2, 1, 2, rows, cols, ComplexF64, false)
        nested = nested_network(peps)
        reblocked = reblocked_nested_network(nested)
        reduced = reduced_tensor(peps)
        for source in CartesianIndices(reduced)
            test_strict_tensor_equal(
                reblocked[source], reduced[source];
                atol=5e-12,
                rtol=5e-12,
            )
        end
        for (twist_x, twist_y) in spin_structures
            nested_value = nested_test_torus_scalar(
                reblocked;
                twist_x,
                twist_y,
            )
            reduced_value = nested_test_torus_scalar(
                reduced;
                twist_x,
                twist_y,
            )
            difference = nested_value - reduced_value
            relative = abs(difference) /
                max(abs(reduced_value), eps(Float64))
            @info "strict periodic nested comparison" rows cols twist_x twist_y difference relative
            @test nested_value ≈ reduced_value rtol=5e-12 atol=1e-12
        end
    end

    for T in (Float64, ComplexF64)
        peps = mixed_periodic_peps(T)
        nested = nested_network(peps)
        reblocked = reblocked_nested_network(nested)
        reduced = reduced_tensor(peps)
        for source in CartesianIndices(reduced)
            test_strict_tensor_equal(
                reblocked[source], reduced[source];
                atol=5e-12,
                rtol=5e-12,
            )
        end
        for (twist_x, twist_y) in spin_structures
            nested_value = nested_test_torus_scalar(
                reblocked;
                twist_x,
                twist_y,
            )
            reduced_value = nested_test_torus_scalar(
                reduced;
                twist_x,
                twist_y,
            )
            @test nested_value ≈ reduced_value rtol=5e-12 atol=1e-12
        end
    end
end

@testset "Direct simplified network assembly" begin
    peps = mixed_periodic_peps(Float64)
    nested = nested_network(peps)
    @test size(nested) == (4, 4)
    for source in CartesianIndices(peps.A)
        @test nested[_layout_ket_site(nested.layout, source)] ≈
            _nested_ket(peps.A[source])
        @test nested[_layout_bra_site(nested.layout, source)] ≈
            _nested_bra(peps.A[source])
        @test nested[_layout_x_site(nested.layout, source)] ≈
            nested.x_crossings[source]
        @test nested[_layout_y_site(nested.layout, source)] ≈
            _nested_y(
                size(peps.A[source])[2], even(peps.A[source])[2],
                size(peps.A[source])[5], even(peps.A[source])[5],
                eltype(peps.A[source]),
            )
    end

    for r in axes(nested, 1), c in axes(nested, 2)
        below = Nmod(r + 1, size(nested, 1))
        right = Nmod(c + 1, size(nested, 2))
        @test size(nested[r, c])[2] == size(nested[r, right])[1]
        @test even(nested[r, c])[2] == even(nested[r, right])[1]
        @test size(nested[r, c])[4] == size(nested[below, c])[3]
        @test even(nested[r, c])[4] == even(nested[below, c])[3]
        @test index_type(nested[r, c])[2] !=
            index_type(nested[r, right])[1]
        @test index_type(nested[r, c])[4] !=
            index_type(nested[below, c])[3]

        # horizontal_flow[right, left] = T_left[l, r, u, d] * T_right[l, r, u, d]
        @test index_type(nested[r, c])[1] == :out
        @test index_type(nested[r, c])[2] == :in
        # vertical_flow[top, bottom] = T_top[l, r, u, d] * T_bottom[l, r, u, d]
        @test index_type(nested[r, c])[3] == :in
        @test index_type(nested[r, c])[4] == :out
    end

    @test_throws ArgumentError nested_network(peps, NestedLayout((1, 1)))
end
