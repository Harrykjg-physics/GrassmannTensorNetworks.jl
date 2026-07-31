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

function _nested_input_north_twist(A::Grassmann)
    return add_parity_sign(A, 4; sign_function=global_sign)
end

_nested_ket_for_network(A::Grassmann{T, 5}) where {T} =
    _nested_ket(_nested_input_north_twist(A))

_nested_bra_for_network(A::Grassmann{T, 5}) where {T} =
    _nested_bra(_nested_input_north_twist(A))

function _nested_x_for_network(xraw::Grassmann{T, 4}) where {T}
    placed = _placed_nested_x(xraw)
    return add_perm_sign(
        placed, (1, 3, 2, 4); sign_function=global_sign
    )
end

function _nested_y_for_network(yraw::Grassmann{T, 4}) where {T}
    return add_perm_sign(
        yraw, (1, 3, 2, 4); sign_function=global_sign
    )
end

function _nested_network_reduced_basis(
    ordered::Grassmann{T, 8}
) where {T}
    corrected = ordered
    for axis in (1, 2, 4, 5)
        corrected = add_parity_sign(
            corrected, axis; sign_function=global_sign
        )
    end
    return corrected
end

function nested_network(
    peps::Square_GPEPS{T},
    layout::NestedLayout=NestedLayout(peps),
) where {T}
    layout.source_size == size(peps) ||
        throw(ArgumentError("layout source size does not match PEPS unit cell"))

    rows, cols = size(peps)
    ket = [
        _nested_ket_for_network(peps.A[r, c])
        for r in 1:rows, c in 1:cols
    ]
    bra = [
        _nested_bra_for_network(peps.A[r, c])
        for r in 1:rows, c in 1:cols
    ]
    xraw = [
        _nested_x(
            size(bra[r, c])[1], even(bra[r, c])[1],
            size(ket[r, c])[4], even(ket[r, c])[4], T,
        )
        for r in 1:rows, c in 1:cols
    ]

    tensors = Matrix{Grassmann{T, 4}}(undef, size(layout)...)
    for r in 1:rows, c in 1:cols
        north_bra = bra[Nmod(r - 1, rows), c]
        east_ket = ket[r, Nmod(c + 1, cols)]
        yraw = _nested_y(
            _physical_identity(peps.A[r, c]),
            size(east_ket)[1], even(east_ket)[1],
            size(north_bra)[4], even(north_bra)[4],
        )

        tensors[layout.ket_sites[r, c]] = ket[r, c]
        tensors[layout.bra_sites[r, c]] = bra[r, c]
        tensors[layout.x_sites[r, c]] =
            _nested_x_for_network(xraw[r, c])
        tensors[layout.y_sites[r, c]] =
            _nested_y_for_network(yraw)
    end

    nested = NestedNetwork(tensors, layout, xraw)
    _check_nested_links(nested)
    return nested
end

function _check_nested_link(
    left::Grassmann,
    left_axis::Int,
    right::Grassmann,
    right_axis::Int,
    left_site::CartesianIndex{2},
    right_site::CartesianIndex{2},
)
    size(left)[left_axis] == size(right)[right_axis] ||
        throw(DimensionMismatch(
            "nested link $left_site[$left_axis] -> " *
            "$right_site[$right_axis] has unequal dimensions",
        ))
    even(left)[left_axis] == even(right)[right_axis] ||
        throw(DimensionMismatch(
            "nested link $left_site[$left_axis] -> " *
            "$right_site[$right_axis] has unequal even sectors",
        ))
    index_type(left)[left_axis] != index_type(right)[right_axis] ||
        throw(DimensionMismatch(
            "nested link $left_site[$left_axis] -> " *
            "$right_site[$right_axis] has equal arrow directions",
        ))
    return nothing
end

function _check_nested_links(nested::NestedNetwork)
    for site in CartesianIndices(nested.network)
        right = CartesianIndex(
            site[1], Nmod(site[2] + 1, size(nested, 2))
        )
        below = CartesianIndex(
            Nmod(site[1] + 1, size(nested, 1)), site[2]
        )
        _check_nested_link(
            nested[site], 2, nested[right], 1, site, right
        )
        _check_nested_link(
            nested[site], 4, nested[below], 3, site, below
        )
    end
    return nested
end

initialize_nested_environment(
    nested::NestedNetwork,
    chi::Int,
    chi_even::Int=div(chi, 2),
) = CTMRGEnv(nested.network, chi, chi_even)

function run_nested_GCTMRG!(
    nested::NestedNetwork,
    env::CTMRGEnv,
    chi::Int;
    kwargs...,
)
    size(env) == size(nested) ||
        throw(DimensionMismatch(
            "nested environment and network sizes differ"
        ))
    run_GCTMRG!(nested.network, nested.network, env, chi; kwargs...)
    return env
end
