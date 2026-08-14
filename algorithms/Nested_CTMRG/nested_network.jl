
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
    rows > 0 && cols > 0 || throw(ArgumentError("source unit-cell dimensions must be positive"))

    bra = [CartesianIndex(2Nmod(r + 1, rows) - 1, 2c - 1) for r in 1:rows, c in 1:cols]
    x = [CartesianIndex(2Nmod(r + 1, rows) - 1, 2Nmod(c - 1, cols)) for r in 1:rows, c in 1:cols]
    y = [CartesianIndex(2r, 2c - 1) for r in 1:rows, c in 1:cols]
    ket = [CartesianIndex(2r, 2Nmod(c - 1, cols)) for r in 1:rows, c in 1:cols]

    return NestedLayout(source_size, (2rows, 2cols), ket, y, x, bra)
end

NestedLayout(peps::Square_GPEPS) = NestedLayout(size(peps))
Base.size(layout::NestedLayout) = layout.nested_size
Base.size(layout::NestedLayout, dim::Integer) = layout.nested_size[dim]

_layout_right_source(layout::NestedLayout, source::CartesianIndex{2}) =
    CartesianIndex(source[1], Nmod(source[2] + 1, layout.source_size[2]))

_layout_down_source(layout::NestedLayout, source::CartesianIndex{2}) =
    CartesianIndex(Nmod(source[1] + 1, layout.source_size[1]), source[2])

_layout_bra_site(layout::NestedLayout, source::CartesianIndex{2}) =
    layout.bra_sites[source]

_layout_x_site(layout::NestedLayout, source::CartesianIndex{2}) =
    layout.x_sites[_layout_right_source(layout, source)]

_layout_y_site(layout::NestedLayout, source::CartesianIndex{2}) =
    layout.y_sites[_layout_down_source(layout, source)]

_layout_ket_site(layout::NestedLayout, source::CartesianIndex{2}) =
    layout.ket_sites[_layout_down_source(layout, _layout_right_source(layout, source))]

struct NestedNetwork{T<:Number, X<:AbstractMatrix}
    network::Matrix{Grassmann{T, 4}}
    layout::NestedLayout
    x_crossings::X
end

Base.size(nested::NestedNetwork, args...) = size(nested.network, args...)
Base.axes(nested::NestedNetwork, args...) = axes(nested.network, args...)
Base.getindex(nested::NestedNetwork, inds...) = getindex(nested.network, inds...)

const _CTMRG_INDEX_TYPE = (:out, :in, :in, :out)

function _adapt_CTMRG_tensor(t::Grassmann{T, 4}) where {T}

    adapted = t
    for ind in 1:4
        if index_type(adapted)[ind] != _CTMRG_INDEX_TYPE[ind]
            if ind in (1, 3)
                adapted = add_parity_sign(index_conjugation(adapted, ind), ind; sign_function=global_sign)
            else
                adapted = index_conjugation(adapted, ind)
            end

        end
    end

    return adapted
end

function adapt_CTMRG(nested::NestedNetwork)

    T = eltype(first(nested.network))
    tensors = Matrix{Grassmann{T, 4}}(undef, size(nested)...)

    for site in CartesianIndices(nested.network)
        # T_ctm[l, r, u, d] = adapt_CTMRG(T_nested[l, r, u, d])
        tensors[site] = _adapt_CTMRG_tensor(nested[site])
    end

    adapted = NestedNetwork(tensors, nested.layout, nested.x_crossings)
    _check_nested_links(adapted)

    return adapted
end

function _nested_ket(K::Grassmann{T, 5}) where {T}

    # K_perm[l, r, p, u, d] <-- K[p, l, r, u, d]
    K_perm = permutedims(K, (2, 3, 1, 4, 5); sign_function=global_sign)
    # Ko1[l, r, p, u, d] = (-1)^p * K_perm[l, r, p, u, d]
    Ko1 = add_parity_sign(K_perm, 3; sign_function=global_sign)
    # Ko2[l, r, U, d] = Ko1[l, r, (p, u), d]
    Ko2 = fuse(Ko1, (3, 4); index_type_fused=:in)

    return Ko2
end

function _nested_bra(K::Grassmann{T, 5}) where {T}

    B = conj(K; sign_function=global_sign)
    # B_perm[l, p, r, u, d] <-- B[p, l, r, u, d]
    B_perm = permutedims(B, (2, 1, 3, 4, 5); sign_function=global_sign)
    # Bo1[l, p, r, u, d] = (-1)^r * B_perm[l, p, r, u, d]
    Bo1 = add_parity_sign(B_perm, 3; sign_function=global_sign)
    # Bo2[l, R, u, d] = Bo1[l, (p, r), u, d]
    Bo2 = fuse(Bo1, (2, 3); index_type_fused=:in)

    return Bo2
end

function _nested_x(
    operator::Grassmann{T, 2},
    hor_total_size::Int, hor_even_size::Int,
    ver_total_size::Int, ver_even_size::Int) where {T}

    identity_hor = Grassmann(
        Matrix(Diagonal(ones(T, hor_total_size))),
        (hor_total_size, hor_total_size),
        (hor_even_size, hor_even_size),
        (:in, :out))

    identity_ver = Grassmann(
        Matrix(Diagonal(ones(T, ver_total_size))),
        (ver_total_size, ver_total_size),
        (ver_even_size, ver_even_size),
        (:in, :out))

    # X0[l, r, u, d] = identity_hor[l, r] * identity_ver[u, d]
    X0 = contract(identity_hor, identity_ver; sign_function=global_sign)
    # X1[ph, l, r, u, pv, d] <-- X1[ph, pv, l, r, u, d] = operator[ph, pv] * X0[l, r, u, d]
    X1 = contract(operator, X0; perm=(1, 3, 4, 5, 2, 6), sign_function=global_sign)
    # X2[ph, l, r, u, pv, d] = (-1)^(ph*l) * X1[ph, l, r, u, pv, d]
    X2 = add_perm_sign(X1, (2, 1, 3, 4, 5, 6); sign_function=global_sign)
    # X3[ph, l, r, u, pv, d] = (-1)^(pv*d) * X2[ph, l, r, u, pv, d]
    X3 = add_perm_sign(X2, (1, 2, 3, 4, 6, 5); sign_function=global_sign)
    # X4[L, r, u, pv, d] = X3[(ph, l), r, u, pv, d]
    X4 = fuse(X3, (1, 2); index_type_fused=:out)
    # X5[L, r, u, D] = X4[L, r, u, (pv, d)]
    X5 = fuse(X4, (4, 5); index_type_fused=:out)
    # X6[L, r, u, D] = (-1)^r * X5[L, r, u, D]
    X6 = add_parity_sign(X5, 2; sign_function=global_sign)
    # X7[L, r, u, D] = (-1)^u * X6[L, r, u, D]
    X7 = add_parity_sign(X6, 3; sign_function=global_sign)

    return X7
end

function _nested_x(
    hor_total_size::Int, hor_even_size::Int,
    ver_total_size::Int, ver_even_size::Int,
    phy_total_size::Int, phy_even_size::Int,
    ::Type{T}) where {T}

    identity_phy = Grassmann(
        Matrix(Diagonal(ones(T, phy_total_size))),
        (phy_total_size, phy_total_size),
        (phy_even_size, phy_even_size),
        (:out, :in))

    return _nested_x(
        identity_phy,
        hor_total_size,
        hor_even_size,
        ver_total_size,
        ver_even_size)
end

function _nested_y(
    hor_total_size::Int, hor_even_size::Int,
    ver_total_size::Int, ver_even_size::Int,
    ::Type{T}) where {T}

    identity_hor = Grassmann(
        Matrix(Diagonal(ones(T, hor_total_size))),
        (hor_total_size, hor_total_size),
        (hor_even_size, hor_even_size),
        (:out, :in))

    identity_ver = Grassmann(
        Matrix(Diagonal(ones(T, ver_total_size))),
        (ver_total_size, ver_total_size),
        (ver_even_size, ver_even_size),
        (:out, :in))

    # Y0[l, r, u, d] = identity_hor[l, r] * identity_ver[u, d]
    Y0 = contract(identity_hor, identity_ver; sign_function=global_sign)

    return Y0
end

function nested_network(
    peps::Square_GPEPS{T},
    layout::NestedLayout=NestedLayout(peps)) where {T}

    layout.source_size == size(peps) ||
    throw(ArgumentError("layout source size does not match PEPS unit cell"))

    rows, cols = size(peps)

    ket = [_nested_ket(peps.A[r, c]) for r in 1:rows, c in 1:cols]
    bra = [_nested_bra(peps.A[r, c]) for r in 1:rows, c in 1:cols]
    x = [_nested_x(
        size(peps.A[r, c])[3], even(peps.A[r, c])[3],
        size(peps.A[r, c])[4], even(peps.A[r, c])[4],
        size(peps.A[r, c])[1], even(peps.A[r, c])[1],
        T) for r in 1:rows, c in 1:cols]
    y = [_nested_y(
        size(peps.A[r, c])[2], even(peps.A[r, c])[2],
        size(peps.A[r, c])[5], even(peps.A[r, c])[5],
        T) for r in 1:rows, c in 1:cols]

    tensors = Matrix{Grassmann{T, 4}}(undef, size(layout)...)

    for r in 1:rows, c in 1:cols

        source = CartesianIndex(r, c)
        # network[B_s] = B(A_s)
        tensors[_layout_bra_site(layout, source)] = bra[source]
        # network[X_s] = X(A_s), placed to the right of B_s.
        tensors[_layout_x_site(layout, source)] = x[source]
        # network[Y_s] = Y(A_s), placed below B_s.
        tensors[_layout_y_site(layout, source)] = y[source]
        # network[K_s] = K(A_s), placed below X_s.
        tensors[_layout_ket_site(layout, source)] = ket[source]
    end

    nested = NestedNetwork(tensors, layout, x)
    _check_nested_links(nested)

    return nested
end

function _check_nested_link(
    left::Grassmann,
    left_axis::Int,
    right::Grassmann,
    right_axis::Int,
    left_site::CartesianIndex{2},
    right_site::CartesianIndex{2})

    size(left)[left_axis] == size(right)[right_axis] || throw(DimensionMismatch(
        "nested link $left_site[$left_axis] -> " *
        "$right_site[$right_axis] has unequal dimensions"))

    even(left)[left_axis] == even(right)[right_axis] || throw(DimensionMismatch(
        "nested link $left_site[$left_axis] -> " *
        "$right_site[$right_axis] has unequal even sectors"))

    index_type(left)[left_axis] != index_type(right)[right_axis] || throw(DimensionMismatch(
        "nested link $left_site[$left_axis] -> " *
        "$right_site[$right_axis] has equal arrow directions"))

    return nothing
end

function _check_nested_links(nested::NestedNetwork)

    for site in CartesianIndices(nested.network)

        right = CartesianIndex(site[1], Nmod(site[2] + 1, size(nested, 2)))
        below = CartesianIndex(Nmod(site[1] + 1, size(nested, 1)), site[2])

        _check_nested_link(nested[site], 2, nested[right], 1, site, right)
        _check_nested_link(nested[site], 4, nested[below], 3, site, below)
    end

    return nested
end
