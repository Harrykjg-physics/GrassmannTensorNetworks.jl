using Test
using GrassmannTensorNetworks

@eval GrassmannTensorNetworks global global_sign = auto_sign

import GrassmannTensorNetworks:
    _graded_pair_sign,
    _nested_ket,
    _nested_bra,
    _nested_x,
    _placed_nested_x,
    _nested_y,
    _physical_identity,
    _nested_reduced_basis

@testset "Nested layout" begin
    layout = NestedLayout((2, 3))
    @test size(layout) == (4, 6)
    @test layout.source_size == (2, 3)
    @test layout.ket_sites[2, 3] == CartesianIndex(3, 5)
    @test layout.y_sites[2, 3] == CartesianIndex(3, 6)
    @test layout.x_sites[2, 3] == CartesianIndex(4, 5)
    @test layout.bra_sites[2, 3] == CartesianIndex(4, 6)
    @test_throws ArgumentError NestedLayout((0, 3))

    peps = Square_GPEPS(2, 1, 2, 1, 2, Float64, false)
    @test NestedLayout(peps).source_size == (1, 2)
end

@testset "Nested network wrapper" begin
    tensor = Grassmann((1, 1, 1, 1), (1, 1, 1, 1), (:in, :out, :in, :out), Float64; init=:zeros)
    network = Matrix{Grassmann{Float64, 4}}(undef, 2, 3)
    fill!(network, tensor)
    nested = NestedNetwork(network, NestedLayout((1, 1)), trues(1, 1))

    @test size(nested) == (2, 3)
    @test size(nested, 2) == 3
    @test axes(nested) == axes(network)
    @test nested[2, 3] === network[2, 3]
end

@testset "Nested graded primitives" begin
    odd_crossing = _nested_x(1, 0, 1, 0, Float64)
    @test !haskey(odd_crossing, (0, 0, 0, 0))
    @test only(odd_crossing[(1, 1, 1, 1)]) == -1

    mixed_crossing = _nested_x(2, 1, 2, 1, Float64)
    dense = convert(Array, mixed_crossing)
    @test dense[1, 1, 1, 1] == 1
    @test dense[2, 2, 1, 1] == 1
    @test dense[1, 1, 2, 2] == 1
    @test dense[2, 2, 2, 2] == -1

    peps = Square_GPEPS(2, 1, 2, 1, 1, Float64, false)
    A = peps.A[1, 1]
    K = _nested_ket(A)
    B = _nested_bra(A)
    @test size(K) == (2, 4, 2, 2)
    @test size(B) == (2, 2, 4, 2)
    @test index_type(K) == (:out, :in, :in, :out)
    @test index_type(B) == (:out, :in, :in, :out)

    identity = Grassmann(
        Matrix{Float64}(I, 2, 2), (2, 2), (1, 1), (:out, :in)
    )
    Y = _nested_y(identity, 2, 1, 2, 1)
    @test size(Y) == (4, 2, 2, 4)
    @test index_type(Y) == (:out, :in, :in, :out)
end

function contract_nested_tile(K, Y, X, B)
    sign = GrassmannTensorNetworks.global_sign
    ky = contract(K, Y, (2, 1); sign_function=sign)
    kx = contract(ky, X, (3, 3); sign_function=sign)
    tile = contract(kx, B, ((5, 7), (3, 1)); sign_function=sign)
    ordered = permutedims(
        tile, (5, 1, 7, 3, 4, 2, 8, 6); sign_function=sign
    )
    ordered = _nested_reduced_basis(ordered)
    ordered = add_parity_sign(
        ordered, 1; sign_function=sign
    )
    left = fuse(ordered, (1, 2); index_type_fused=:out)
    left = add_perm_sign(
        left, (1, 3, 2, 4, 5, 6, 7); sign_function=sign
    )
    right = fuse(left, (2, 3); index_type_fused=:in)
    right = add_perm_sign(
        right, (1, 2, 4, 3, 5, 6); sign_function=sign
    )
    up = fuse(right, (3, 4); index_type_fused=:in)
    up = add_parity_sign(up, 4; sign_function=sign)
    return fuse(up, (4, 5); index_type_fused=:out)
end

function deterministic_nested_tensor(::Type{T}, sizes, evens) where {T}
    A = Grassmann(
        sizes, evens, (:out, :out, :in, :in, :out), T; init=:zeros
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

function closed_nested_tile(A)
    K, B = _nested_ket(A), _nested_bra(A)
    Xraw = _nested_x(
        size(A)[2], even(A)[2], size(A)[5], even(A)[5], eltype(A)
    )
    Y = _nested_y(
        _physical_identity(A),
        size(A)[3],
        even(A)[3],
        size(A)[4],
        even(A)[4],
    )
    return contract_nested_tile(K, Y, _placed_nested_x(Xraw), B)
end

const NESTED_PAIR_TO_FUSED_INDEX = Dict(
    (0, 0) => 1,
    (1, 1) => 2,
    (1, 0) => 3,
    (0, 1) => 4,
)

@testset "Native bra-first pair fusion basis" begin
    pair_fixture = Grassmann(
        (2, 2, 2), (1, 1, 1), (:out, :out, :in), ComplexF64; init=:zeros
    )
    pair_fixture[(0, 0, 0)][1] = 11 + 2im
    pair_fixture[(1, 1, 0)][1] = 22 + 3im
    pair_fixture[(1, 0, 1)][1] = 31 + 5im
    pair_fixture[(0, 1, 1)][1] = 41 + 7im

    fused = convert(
        Array, fuse(pair_fixture, (1, 2); index_type_fused=:out)
    )
    @test fused[NESTED_PAIR_TO_FUSED_INDEX[(0, 0)], 1] == 11 + 2im
    @test fused[NESTED_PAIR_TO_FUSED_INDEX[(1, 1)], 1] == 22 + 3im
    @test fused[NESTED_PAIR_TO_FUSED_INDEX[(1, 0)], 2] == 31 + 5im
    @test fused[NESTED_PAIR_TO_FUSED_INDEX[(0, 1)], 2] == 41 + 7im
end

@testset "Local nested factorization" begin
    for T in (Float64, ComplexF64)
        A = deterministic_nested_tensor(T, (2, 2, 2, 2, 2), (1, 1, 1, 1, 1))
        @test closed_nested_tile(A) ≈ reduced_tensor(A) rtol = 1e-12
    end
end

@testset "Nested native basis covers all compatible parity sectors" begin
    A = deterministic_nested_tensor(
        Float64, (2, 2, 2, 2, 2), (1, 1, 1, 1, 1)
    )
    candidate = convert(Array, closed_nested_tile(A))
    target = convert(Array, reduced_tensor(A))
    compatible = 0
    for bra in Iterators.product(ntuple(_ -> 0:1, 4)...)
        for ket in Iterators.product(ntuple(_ -> 0:1, 4)...)
            mod(sum(bra), 2) == mod(sum(ket), 2) || continue
            compatible += 1
            output_index = ntuple(
                direction -> NESTED_PAIR_TO_FUSED_INDEX[
                    (bra[direction], ket[direction])
                ],
                4,
            )
            @test candidate[output_index...] == target[output_index...]
        end
    end
    @test compatible == 128
    @test count(!iszero, target) == 128
end

@testset "Nested factorization preserves anisotropic link roles" begin
    sizes = (3, 3, 4, 3, 4)
    evens = (2, 1, 3, 2, 1)
    for T in (Float64, ComplexF64)
        A = deterministic_nested_tensor(T, sizes, evens)
        entries = reduce(vcat, vec(block) for block in nonzero_vals(A))
        @test length(unique(entries)) == length(entries)

        Xraw = _nested_x(sizes[2], evens[2], sizes[5], evens[5], T)
        Y = _nested_y(
            _physical_identity(A), sizes[3], evens[3], sizes[4], evens[4]
        )
        @test size(Xraw) == (sizes[2], sizes[2], sizes[5], sizes[5])
        @test size(Y) == (
            sizes[1] * sizes[3],
            sizes[3],
            sizes[4],
            sizes[1] * sizes[4],
        )
        @test closed_nested_tile(A) ≈ reduced_tensor(A) rtol = 5e-13
    end
end
