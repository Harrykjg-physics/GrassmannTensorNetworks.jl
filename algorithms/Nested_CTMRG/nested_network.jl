struct NestedLayout
    source_size::Tuple{Int, Int}
    nested_size::Tuple{Int, Int}
    ket_sites::Matrix{CartesianIndex{2}}
    y_sites::Matrix{CartesianIndex{2}}
    x_sites::Matrix{CartesianIndex{2}}
    bra_sites::Matrix{CartesianIndex{2}}
end

function NestedLayout(source_size::Tuple{Int, Int})
    rows, cols = source_size
    rows > 0 && cols > 0 ||
        throw(ArgumentError("source unit-cell dimensions must be positive"))
    ket = [CartesianIndex(2r - 1, 2c - 1) for r in 1:rows, c in 1:cols]
    y = [CartesianIndex(2r - 1, 2c) for r in 1:rows, c in 1:cols]
    x = [CartesianIndex(2r, 2c - 1) for r in 1:rows, c in 1:cols]
    bra = [CartesianIndex(2r, 2c) for r in 1:rows, c in 1:cols]
    return NestedLayout(source_size, (2rows, 2cols), ket, y, x, bra)
end

NestedLayout(peps::Square_GPEPS) = NestedLayout(size(peps))
Base.size(layout::NestedLayout) = layout.nested_size
Base.size(layout::NestedLayout, dim::Integer) = layout.nested_size[dim]

struct NestedNetwork{T<:Number, X<:AbstractMatrix}
    network::Matrix{Grassmann{T, 4}}
    layout::NestedLayout
    x_crossings::X
end

Base.size(nested::NestedNetwork, args...) = size(nested.network, args...)
Base.axes(nested::NestedNetwork, args...) = axes(nested.network, args...)
Base.getindex(nested::NestedNetwork, inds...) = getindex(nested.network, inds...)

function _graded_pair_sign(
    t::Grassmann{T, N, AT}, i::Int, j::Int
) where {T, N, AT}
    1 <= i <= N && 1 <= j <= N && i != j ||
        throw(ArgumentError("graded-pair indices must be distinct and in bounds"))
    blocks = Dict{NTuple{N, Int}, AT}()
    for (sector, block) in nonzero_pairs(t)
        blocks[sector] = (-1)^(sector[i] * sector[j]) .* block
    end
    return Grassmann(size(t), even(t), index_type(t), blocks)
end

function _nested_x(
    horizontal_size::Int,
    horizontal_even::Int,
    vertical_size::Int,
    vertical_even::Int,
    ::Type{T},
) where {T}
    crossing = Grassmann(
        (horizontal_size, horizontal_size, vertical_size, vertical_size),
        (horizontal_even, horizontal_even, vertical_even, vertical_even),
        (:out, :in, :in, :out),
        T;
        init=:zeros,
    )
    for horizontal_parity in 0:1, vertical_parity in 0:1
        sector = (
            horizontal_parity,
            horizontal_parity,
            vertical_parity,
            vertical_parity,
        )
        haskey(crossing, sector) || continue
        block = crossing[sector]
        coefficient =
            (-one(T))^(horizontal_parity * vertical_parity)
        for horizontal in axes(block, 1), vertical in axes(block, 3)
            block[horizontal, horizontal, vertical, vertical] = coefficient
        end
    end
    return crossing
end

_placed_nested_x(x::Grassmann) =
    add_parity_sign(x, 1; sign_function=global_sign)

function _bend_index(t::Grassmann, index::Int)
    return add_parity_sign(
        index_conjugation(t, index), index; sign_function=global_sign
    )
end

function _nested_ket_raw(A::Grassmann{T, 5}) where {T}
    signed = _graded_pair_sign(A, 1, 3)
    routed = permutedims(signed, (2, 1, 3, 4, 5); sign_function=global_sign)
    return fuse(routed, (2, 3); index_type_fused=:in)
end

_nested_ket(A::Grassmann{T, 5}) where {T} = _nested_ket_raw(A)

function _nested_bra_raw(A::Grassmann{T, 5}) where {T}
    bra = conj(A; sign_function=global_sign)
    signed = _graded_pair_sign(bra, 1, 4)
    routed = permutedims(signed, (2, 3, 1, 4, 5); sign_function=global_sign)
    fused = fuse(routed, (3, 4); index_type_fused=:in)
    return foldl(_bend_index, (1, 2, 4); init=fused)
end

_nested_bra(A::Grassmann{T, 5}) where {T} = _nested_bra_raw(A)

function _physical_identity(A::Grassmann{T, 5}) where {T}
    p, pe = size(A)[1], even(A)[1]
    return Grassmann(Matrix{T}(I, p, p), (p, p), (pe, pe), (:out, :in))
end

function _nested_y(
    operator::Grassmann{T, 2},
    horizontal_size::Int,
    horizontal_even::Int,
    vertical_size::Int,
    vertical_even::Int,
) where {T}
    crossing = _nested_x(
        horizontal_size, horizontal_even, vertical_size, vertical_even, T
    )
    product = contract(operator, crossing; sign_function=global_sign)
    routed = permutedims(
        product, (1, 3, 4, 5, 2, 6); sign_function=global_sign
    )
    west_fused = fuse(routed, (1, 2); index_type_fused=:out)
    return fuse(west_fused, (4, 5); index_type_fused=:out)
end

function _nested_reduced_basis(ordered::Grassmann{T, 8}) where {T}
    corrected = add_perm_sign(
        ordered,
        (2, 3, 5, 4, 6, 7, 1, 8);
        sign_function=global_sign,
    )
    for index in (2, 4, 6)
        corrected = add_parity_sign(
            corrected, index; sign_function=global_sign
        )
    end
    return corrected
end
