
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
    # X[x, y] = (2 * mod(x + 1, Lx) - 1, 2 * mod(y - 1, Ly))
    x = [
        CartesianIndex(
            2Nmod(r + 1, rows) - 1,
            2Nmod(c - 1, cols),
        )
        for r in 1:rows, c in 1:cols
    ]
    # Y[x, y] = (2x, 2y - 1)
    y = [CartesianIndex(2r, 2c - 1) for r in 1:rows, c in 1:cols]
    # K[x, y] = (2x, 2 * mod(y - 1, Ly))
    ket = [
        CartesianIndex(2r, 2Nmod(c - 1, cols))
        for r in 1:rows, c in 1:cols
    ]

    return NestedLayout(source_size, (2rows, 2cols), ket, y, x, bra)
end

NestedLayout(peps::Square_GPEPS) = NestedLayout(size(peps))
Base.size(layout::NestedLayout) = layout.nested_size
Base.size(layout::NestedLayout, dim::Integer) = layout.nested_size[dim]

_layout_right_source(layout::NestedLayout, source::CartesianIndex{2}) =
    CartesianIndex(source[1], Nmod(source[2] + 1, layout.source_size[2]))

_layout_down_source(layout::NestedLayout, source::CartesianIndex{2}) =
    CartesianIndex(Nmod(source[1] + 1, layout.source_size[1]), source[2])

_layout_x_site(layout::NestedLayout, source::CartesianIndex{2}) =
    layout.x_sites[_layout_right_source(layout, source)]

_layout_y_site(layout::NestedLayout, source::CartesianIndex{2}) =
    layout.y_sites[_layout_down_source(layout, source)]

_layout_ket_site(layout::NestedLayout, source::CartesianIndex{2}) =
    layout.ket_sites[
        _layout_down_source(layout, _layout_right_source(layout, source))
    ]

_layout_bra_site(layout::NestedLayout, source::CartesianIndex{2}) =
    layout.bra_sites[source]

struct NestedNetwork{T<:Number, X<:AbstractMatrix}
    network::Matrix{Grassmann{T, 4}}
    layout::NestedLayout
    x_crossings::X
end

Base.size(nested::NestedNetwork, args...) = size(nested.network, args...)
Base.axes(nested::NestedNetwork, args...) = axes(nested.network, args...)
Base.getindex(nested::NestedNetwork, inds...) = getindex(nested.network, inds...)

# K[l, r, U, d] carries the ket tensor and the fused physical-up index U = (p, u).
function _nested_ket(A::Grassmann{T, 5}) where {T}
    # A_perm[l, r, p, u, d] <-- A[p, l, r, u, d]
    A_perm = permutedims(A, (2, 3, 1, 4, 5); sign_function=global_sign)
    # Ao1[l, r, p, u, d] = (-1)^p A_perm[l, r, p, u, d]
    Ao1 = add_parity_sign(A_perm, 3; sign_function=global_sign)
    # Ao2[l, r, U, d] = Ao1[l, r, (p, u), d]
    Ao2 = fuse(Ao1, (3, 4); index_type_fused=:in)
    # Ao3[l, r, U, d] = (-1)^l Ao2[l, r, U, d]
    Ao3 = add_parity_sign(Ao2, 1; sign_function=global_sign)
    # Ao4[l, r, U, d] = (-1)^r Ao3[l, r, U, d]
    Ao4 = add_parity_sign(Ao3, 2; sign_function=global_sign)
    # Ao5[l, r, U, d] = (-1)^d Ao4[l, r, U, d]
    return add_parity_sign(Ao4, 4; sign_function=global_sign)
end

# B[l, R, u, d] carries the bra tensor and the fused physical-right index R = (p, r).
function _nested_bra(A::Grassmann{T, 5}) where {T}
    A_conj = conj(A; sign_function=global_sign)
    # A_perm[l, p, r, u, d] <-- A_conj[p, l, r, u, d]
    A_perm = permutedims(A_conj, (2, 1, 3, 4, 5); sign_function=global_sign)
    # Ao1[l, p, r, u, d] = (-1)^d A_perm[l, p, r, u, d]
    Ao1 = add_parity_sign(A_perm, 5; sign_function=global_sign)
    # Ao2[l, R, u, d] = Ao1[l, (p, r), u, d]
    Ao2 = fuse(Ao1, (2, 3); index_type_fused=:in)
    # Ao3[l, R, u, d] = conjugation(Ao2[l, R, u, d], (l, u, d))
    Ao3 = index_conjugation(Ao2, (1, 3, 4))
    # Ao4[l, R, u, d] = (-1)^l Ao3[l, R, u, d]
    Ao4 = add_parity_sign(Ao3, 1; sign_function=global_sign)
    # Ao5[l, R, u, d] = (-1)^u Ao4[l, R, u, d]
    return add_parity_sign(Ao4, 3; sign_function=global_sign)
end

# X is at the right of B and carries the physical identity or local operator.
function _nested_x(
    operator::Grassmann{T, 2},
    hor_total_size::Int,
    hor_even_size::Int,
    ver_total_size::Int,
    ver_even_size::Int,
) where {T}
    identity_hor = Grassmann(
        Matrix{T}(I, hor_total_size, hor_total_size),
        (hor_total_size, hor_total_size),
        (hor_even_size, hor_even_size),
        (:out, :in),
    )
    identity_ver = Grassmann(
        Matrix{T}(I, ver_total_size, ver_total_size),
        (ver_total_size, ver_total_size),
        (ver_even_size, ver_even_size),
        (:in, :out),
    )

    # X0[l, r, u, d] = identity_hor[l, r] * identity_ver[u, d]
    X0 = contract(identity_hor, identity_ver; sign_function=global_sign)
    # X1[po, l, r, u, pi, d] = operator[po, pi] * X0[l, r, u, d]
    X1 = contract(
        X0,
        operator;
        perm=(5, 1, 2, 3, 6, 4),
        sign_function=global_sign,
    )
    # X2[po, l, r, u, pi, d] = (-1)^l X1[po, l, r, u, pi, d]
    X2 = add_parity_sign(X1, 2; sign_function=global_sign)
    # X3[po, l, r, u, pi, d] = (-1)^(po*l) X2[po, l, r, u, pi, d]
    X3 = add_perm_sign(
        X2, (2, 1, 3, 4, 5, 6);
        sign_function=global_sign,
    )
    # X4[po, l, r, u, pi, d] = (-1)^(po*u) X3[po, l, r, u, pi, d]
    X4 = add_perm_sign(
        X3, (2, 3, 4, 1, 5, 6);
        sign_function=global_sign,
    )
    # X5[po, l, r, u, pi, d] = (-1)^(|O|*(1+u)) X4[po, l, r, u, pi, d]
    X5 = tensor_parity(operator) == 0 ? X4 :
        add_parity_sign(X4, 4; sign_function=global_sign) * (-one(T))
    # X6[L, r, u, pi, d] = X5[(po, l), r, u, pi, d]
    X6 = fuse(X5, (1, 2); index_type_fused=:out)
    # X7[L, r, u, D] = X6[L, r, u, (pi, d)]
    return fuse(X6, (4, 5); index_type_fused=:out)
end

function _nested_x(
    hor_total_size::Int,
    hor_even_size::Int,
    ver_total_size::Int,
    ver_even_size::Int,
    phy_total_size::Int,
    phy_even_size::Int,
    ::Type{T},
) where {T}
    identity_phy = Grassmann(
        Matrix{T}(I, phy_total_size, phy_total_size),
        (phy_total_size, phy_total_size),
        (phy_even_size, phy_even_size),
        (:out, :in),
    )
    return _nested_x(
        identity_phy,
        hor_total_size,
        hor_even_size,
        ver_total_size,
        ver_even_size,
    )
end

# Y is at the left of K and contains only the virtual Grassmann crossing.
function _nested_y(
    hor_total_size::Int,
    hor_even_size::Int,
    ver_total_size::Int,
    ver_even_size::Int,
    ::Type{T},
) where {T}
    identity_hor = Grassmann(
        Matrix{T}(I, hor_total_size, hor_total_size),
        (hor_total_size, hor_total_size),
        (hor_even_size, hor_even_size),
        (:out, :in),
    )
    identity_ver = Grassmann(
        Matrix{T}(I, ver_total_size, ver_total_size),
        (ver_total_size, ver_total_size),
        (ver_even_size, ver_even_size),
        (:out, :in),
    )

    # Y0[l, r, u, d] = identity_hor[l, r] * identity_ver[u, d]
    Y0 = contract(identity_hor, identity_ver; sign_function=global_sign)
    # Y1[l, r, u, d] = conjugation(Y0[l, r, u, d], (u, d))
    Y1 = index_conjugation(Y0, (3, 4))
    # Y2[l, r, u, d] = (-1)^u Y1[l, r, u, d]
    return add_parity_sign(Y1, 3; sign_function=global_sign)
end

function nested_network(
    peps::Square_GPEPS{T},
    layout::NestedLayout=NestedLayout(peps),
) where {T}
    layout.source_size == size(peps) ||
        throw(ArgumentError("layout source size does not match PEPS unit cell"))

    rows, cols = size(peps)
    ket = [
        _nested_ket(peps.A[r, c])
        for r in 1:rows, c in 1:cols
    ]
    bra = [
        _nested_bra(peps.A[r, c])
        for r in 1:rows, c in 1:cols
    ]
    x = [
        _nested_x(
            size(peps.A[r, c])[3], even(peps.A[r, c])[3],
            size(peps.A[r, c])[4], even(peps.A[r, c])[4],
            size(peps.A[r, c])[1], even(peps.A[r, c])[1],
            T,
        )
        for r in 1:rows, c in 1:cols
    ]
    y = [
        _nested_y(
            size(peps.A[r, c])[2], even(peps.A[r, c])[2],
            size(peps.A[r, c])[5], even(peps.A[r, c])[5],
            T,
        )
        for r in 1:rows, c in 1:cols
    ]

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
