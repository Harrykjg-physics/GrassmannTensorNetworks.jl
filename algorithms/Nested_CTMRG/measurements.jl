_source_site(site::CartesianIndex{2}) = site
_source_site(site::Tuple{Int, Int}) = CartesianIndex(site)

function _nested_x_operator_raw(
    peps::Square_GPEPS,
    source::CartesianIndex{2},
    operator::Grassmann{T, 2}) where {T}

    As = peps.A[source]
    return _nested_x(operator, size(As)[3], even(As)[3], size(As)[4], even(As)[4])
end

function _nested_x_with_alpha(
    operator::Grassmann{T, 3},
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
    # X1[ph, l, r, u, pv, d, a] <-- X1[ph, pv, a, l, r, u, d] = operator[ph, pv, a] * X0[l, r, u, d]
    X1 = contract(operator, X0; perm=(1, 4, 5, 6, 2, 7, 3), sign_function=global_sign)
    # X2[ph, l, r, u, pv, d, a] = (-1)^(ph*l) * X1[ph, l, r, u, pv, d, a]
    X2 = add_perm_sign(X1, (2, 1, 3, 4, 5, 6, 7); sign_function=global_sign)
    # X3[ph, l, r, u, pv, d, a] = (-1)^(pv*d) * X2[ph, l, r, u, pv, d, a]
    X3 = add_perm_sign(X2, (1, 2, 3, 4, 6, 5, 7); sign_function=global_sign)
    # X4[L, r, u, pv, d, a] = X3[(ph, l), r, u, pv, d, a]
    X4 = fuse(X3, (1, 2); index_type_fused=:out)
    # X5[L, r, u, D, a] = X4[L, r, u, (pv, d), a]
    X5 = fuse(X4, (4, 5); index_type_fused=:out)
    # X6[L, r, u, D, a] = (-1)^r * X5[L, r, u, D, a]
    X6 = add_parity_sign(X5, 2; sign_function=global_sign)
    # X7[L, r, u, D, a] = (-1)^u * X6[L, r, u, D, a]
    X7 = add_parity_sign(X6, 3; sign_function=global_sign)

    return X7
end

function _nested_x_operator_raw(
    peps::Square_GPEPS,
    source::CartesianIndex{2},
    operator::Grassmann{T, 3}) where {T}

    As = peps.A[source]
    return _nested_x_with_alpha(operator, size(As)[3], even(As)[3], size(As)[4], even(As)[4])
end

function _match_nested_site_arrows(
    tensor::Grassmann{T, N},
    reference::Grassmann{Q, 4}) where {T, Q, N}

    N >= 4 || throw(ArgumentError("nested measurement tensor must have at least four lattice legs"))
    tensor_size = ntuple(i -> size(tensor)[i], 4)
    tensor_even = ntuple(i -> even(tensor)[i], 4)
    tensor_size == size(reference) ||
    throw(DimensionMismatch("nested measurement tensor and reference site have unequal dimensions"))
    tensor_even == even(reference) ||
    throw(DimensionMismatch("nested measurement tensor and reference site have unequal even sectors"))

    matched = tensor
    for ind in 1:4
        if index_type(matched)[ind] != index_type(reference)[ind]
            if ind in (1, 3)
                matched = add_parity_sign(index_conjugation(matched, ind), ind; sign_function=global_sign)
            else
                matched = index_conjugation(matched, ind)
            end
        end
    end

    ntuple(i -> index_type(matched)[i], 4) == index_type(reference) ||
    throw(ArgumentError("nested measurement tensor arrows cannot be matched to the reference site"))

    return matched
end

function nested_x_operator(
    peps::Square_GPEPS,
    site,
    operator::Grassmann{T, 2}) where {T}

    source = _source_site(site)
    checkbounds(Bool, peps.A, source) || throw(ArgumentError("source site $source is outside the unit cell"))

    physical_size = size(peps.A[source])[1]
    physical_even = even(peps.A[source])[1]

    size(operator) == (physical_size, physical_size) || throw(DimensionMismatch("operator physical dimensions do not match PEPS"))
    even(operator) == (physical_even, physical_even) || throw(DimensionMismatch("operator physical parity split does not match PEPS"))
    index_type(operator) == (:out, :in) || throw(ArgumentError("operator arrows must be (:out, :in)"))

    return _nested_x_operator_raw(peps, source, operator)
end

function nested_x_operator(
    nested::NestedNetwork,
    peps::Square_GPEPS,
    site,
    operator::Grassmann{T, 2}) where {T}

    source = _source_site(site)
    raw = nested_x_operator(peps, source, operator)
    return _match_nested_site_arrows(raw, nested[_layout_x_site(nested.layout, source)])
end

function _nested_x_bond_operator(
    nested::NestedNetwork,
    peps::Square_GPEPS,
    site,
    operator::Grassmann{T, 3}) where {T}

    source = _source_site(site)
    checkbounds(Bool, peps.A, source) || throw(ArgumentError("source site $source is outside the unit cell"))

    physical_size = size(peps.A[source])[1]
    physical_even = even(peps.A[source])[1]

    size(operator)[1:2] == (physical_size, physical_size) || throw(DimensionMismatch("operator physical dimensions do not match PEPS"))
    even(operator)[1:2] == (physical_even, physical_even) || throw(DimensionMismatch("operator physical parity split does not match PEPS"))
    index_type(operator)[1:2] == (:out, :in) || throw(ArgumentError("operator physical arrows must be (:out, :in)"))

    raw = _nested_x_operator_raw(peps, source, operator)
    return _match_nested_site_arrows(raw, nested[_layout_x_site(nested.layout, source)])
end

nested_y_operator(args...) = nested_x_operator(args...)

function compute_nested_exp_site(
    nested::NestedNetwork,
    peps::Square_GPEPS,
    operator::Grassmann{<:Number, 2},
    env::CTMRGEnv,
    site)

    source = _source_site(site)
    xsite = _layout_x_site(nested.layout, source)
    impurity = nested_x_operator(nested, peps, source, operator)
    den, expval = compute_exp_site(
        nested[xsite], impurity,
        env.El[xsite], env.Er[xsite],
        env.Eu[xsite], env.Ed[xsite],
        env.Clu[xsite], env.Cru[xsite],
        env.Cld[xsite], env.Crd[xsite])

    return den, expval
end

function compute_nested_exp_site(
    nested::NestedNetwork,
    peps::Square_GPEPS,
    operators::AbstractMatrix{<:Grassmann{Q, 2}},
    env::CTMRGEnv) where {Q}

    _check_nested_operator_unit_cell(peps, operators)
    denominator = Matrix{promote_type(eltype(peps), Q)}(undef, size(peps)...)
    values = similar(denominator)
    for site in CartesianIndices(peps.A)
        denominator[site], values[site] =
        compute_nested_exp_site(nested, peps, operators[site], env, site)
    end

    return denominator, values
end

function _check_nested_operator_unit_cell(
    peps::Square_GPEPS,
    operators::AbstractMatrix)

    size(operators) == size(peps) ||
    throw(DimensionMismatch("operator and PEPS unit cells differ"))
    return nothing
end

function _check_nested_bond_operator(
    peps::Square_GPEPS,
    operator::Grassmann{<:Number, 4},
    source::CartesianIndex{2},
    neighbor::CartesianIndex{2})

    checkbounds(Bool, peps.A, source) ||
    throw(ArgumentError("source site $source is outside the unit cell"))

    checkbounds(Bool, peps.A, neighbor) ||
    throw(ArgumentError("neighbor site $neighbor is outside the unit cell"))

    source_physical = size(peps.A[source])[1]
    neighbor_physical = size(peps.A[neighbor])[1]
    expected_size = (source_physical, neighbor_physical, source_physical, neighbor_physical)

    size(operator) == expected_size ||
    throw(DimensionMismatch("bond-operator physical dimensions do not match PEPS"))

    source_even = even(peps.A[source])[1]
    neighbor_even = even(peps.A[neighbor])[1]
    expected_even = (source_even, neighbor_even, source_even, neighbor_even)

    even(operator) == expected_even ||
    throw(DimensionMismatch("bond-operator parity splits do not match PEPS"))

    index_type(operator) == (:out, :out, :in, :in) ||
    throw(ArgumentError("bond-operator arrows must be (:out, :out, :in, :in)"))

    tensor_parity(operator) == 0 ||
    throw(ArgumentError("bond operator must have even total parity"))

    return nothing
end

function _contract_horizontal_strip(
    bulks::NTuple{N, <:Grassmann},
    env::CTMRGEnv,
    sites::NTuple{N, CartesianIndex{2}}) where {N}

    N > 0 || throw(ArgumentError("a strip must contain at least one tensor"))
    left_site = first(sites)
    right_site = last(sites)

    left_top = contract(env.Clu[left_site], env.El[left_site], (2, 1); sign_function=global_sign)
    state = contract(left_top, env.Cld[left_site], (2, 2); sign_function=global_sign)

    for (bulk, site) in zip(bulks, sites)
        state, _ = left_move(state, env.Ed[site], bulk, env.Eu[site])
    end

    right_top = contract(env.Cru[right_site], env.Er[right_site], (2, 1); sign_function=global_sign)
    right = contract(right_top, env.Crd[right_site], (2, 2); sign_function=global_sign)

    out = contract(state, right, ((1, 2, 3), (1, 2, 3)); sign_function=global_sign)

    return out
end

function _contract_vertical_strip(
    bulks::NTuple{N, <:Grassmann},
    env::CTMRGEnv,
    sites::NTuple{N, CartesianIndex{2}}) where {N}

    N > 0 || throw(ArgumentError("a strip must contain at least one tensor"))
    top_site = first(sites)
    bottom_site = last(sites)

    upper_left = contract(env.Clu[top_site], env.Eu[top_site], (1, 1); sign_function=global_sign)
    state = contract(upper_left, env.Cru[top_site], (2, 1); sign_function=global_sign)

    for (bulk, site) in zip(bulks, sites)
        state, _ = up_move(state, env.El[site], bulk, env.Er[site])
    end

    lower_left = contract(env.Cld[bottom_site], env.Ed[bottom_site], (1, 1);  sign_function=global_sign)

    lower = contract(lower_left, env.Crd[bottom_site], (2, 1); sign_function=global_sign)

    out = contract(state, lower, ((1, 2, 3), (1, 2, 3)); sign_function=global_sign)

    return out
end

function _contract_nested_hpatch3(
    nested::NestedNetwork,
    env::CTMRGEnv,
    source::CartesianIndex{2},
    left_x::Grassmann,
    right_x::Grassmann)

    x1 = _layout_x_site(nested.layout, source)
    next_source = _layout_right_source(nested.layout, source)
    x2 = _layout_x_site(nested.layout, next_source)
    middle = CartesianIndex(x1[1], Nmod(x1[2] + 1, size(nested, 2)))
    out = _contract_horizontal_strip((left_x, nested[middle], right_x), env, (x1, middle, x2))

    return out
end

function _contract_nested_hpatch3_alpha(
    nested::NestedNetwork,
    env::CTMRGEnv,
    source::CartesianIndex{2},
    left_x::Grassmann,
    right_x::Grassmann)

    x1 = _layout_x_site(nested.layout, source)
    next_source = _layout_right_source(nested.layout, source)
    x2 = _layout_x_site(nested.layout, next_source)
    middle = CartesianIndex(x1[1], Nmod(x1[2] + 1, size(nested, 2)))

    left_top = contract(env.Clu[x1], env.El[x1], (2, 1); sign_function=global_sign)
    state = contract(left_top, env.Cld[x1], (2, 2); sign_function=global_sign)

    # state[h1, h2, h3, aL] = C_left[X_left(aL)]
    state, _ = _left_move_keep_open(state, env.Ed[x1], left_x, env.Eu[x1])
    # state[h1, h2, h3, aL] = C_left[X_left(aL), B]
    state, _ = _left_move_keep_open(state, env.Ed[middle], nested[middle], env.Eu[middle])
    # state[h1, h2, h3, aL, aR] = C_left[X_left(aL), B, X_right(aR)]
    state, _ = _left_move_keep_open(state, env.Ed[x2], right_x, env.Eu[x2])
    # state[h1, h2, h3] = sum_a state[h1, h2, h3, a, a]
    state = trace(state, (4, 5); sign_function=global_sign)

    right_top = contract(env.Cru[x2], env.Er[x2], (2, 1); sign_function=global_sign)
    right = contract(right_top, env.Crd[x2], (2, 2); sign_function=global_sign)
    out = contract(state, right, ((1, 2, 3), (1, 2, 3)); sign_function=global_sign)

    return out
end

function _contract_nested_vpatch3(
    nested::NestedNetwork,
    env::CTMRGEnv,
    source::CartesianIndex{2},
    top_x::Grassmann,
    bottom_x::Grassmann)

    x1 = _layout_x_site(nested.layout, source)
    next_source = _layout_down_source(nested.layout, source)
    x2 = _layout_x_site(nested.layout, next_source)
    middle = CartesianIndex(Nmod(x1[1] + 1, size(nested, 1)), x1[2])

    out = _contract_vertical_strip((top_x, nested[middle], bottom_x), env, (x1, middle, x2))

    return out
end

function _contract_nested_vpatch3_alpha(
    nested::NestedNetwork,
    env::CTMRGEnv,
    source::CartesianIndex{2},
    top_x::Grassmann,
    bottom_x::Grassmann)

    x1 = _layout_x_site(nested.layout, source)
    next_source = _layout_down_source(nested.layout, source)
    x2 = _layout_x_site(nested.layout, next_source)
    middle = CartesianIndex(Nmod(x1[1] + 1, size(nested, 1)), x1[2])

    upper_left = contract(env.Clu[x1], env.Eu[x1], (1, 1); sign_function=global_sign)
    state = contract(upper_left, env.Cru[x1], (2, 1); sign_function=global_sign)

    # state[v1, v2, v3, aT] = C_top[X_top(aT)]
    state, _ = _up_move_keep_open(state, env.El[x1], top_x, env.Er[x1])
    # state[v1, v2, v3, aT] = C_top[X_top(aT), K]
    state, _ = _up_move_keep_open(state, env.El[middle], nested[middle], env.Er[middle])
    # state[v1, v2, v3, aT, aB] = C_top[X_top(aT), K, X_bottom(aB)]
    state, _ = _up_move_keep_open(state, env.El[x2], bottom_x, env.Er[x2])
    # state[v1, v2, v3] = sum_a state[v1, v2, v3, a, a]
    state = trace(state, (4, 5); sign_function=global_sign)

    lower_left = contract(env.Cld[x2], env.Ed[x2], (1, 1); sign_function=global_sign)
    lower = contract(lower_left, env.Crd[x2], (2, 1); sign_function=global_sign)

    out = contract(state, lower, ((1, 2, 3), (1, 2, 3)); sign_function=global_sign)

    return out
end

_nested_scalar_or_zero(value::GrassmannScalar) =
isempty(nonzero_keys(value)) ? zero(eltype(value)) : scalar(value)

function compute_nested_exp_hbond(
    nested::NestedNetwork,
    peps::Square_GPEPS,
    operator::Grassmann{<:Number, 4},
    env::CTMRGEnv,
    site)

    source = _source_site(site)
    neighbor = _layout_right_source(nested.layout, source)
    _check_nested_bond_operator(peps, operator, source, neighbor)
    closed_left = nested[_layout_x_site(nested.layout, source)]
    closed_right = nested[_layout_x_site(nested.layout, neighbor)]
    # denominator = C_h[X_source(I), B, X_neighbor(I)]
    denominator = _contract_nested_hpatch3(nested, env, source, closed_left, closed_right)
    denominator_value = _nested_scalar_or_zero(denominator)
    left_op, right_op = _bond_operator_gsvd(operator)
    left_x = _nested_x_bond_operator(nested, peps, source, left_op)
    right_x = _nested_x_bond_operator(nested, peps, neighbor, right_op)
    # numerator = C_h[X_left(a), B, X_right(a)]
    numerator_tensor = _contract_nested_hpatch3_alpha(nested, env, source, left_x, right_x)
    numerator = _nested_scalar_or_zero(numerator_tensor)

    return denominator, numerator/denominator_value
end

function compute_nested_exp_hbond(
    nested::NestedNetwork,
    peps::Square_GPEPS,
    operators::AbstractMatrix{<:Grassmann},
    env::CTMRGEnv)

    size(operators) == size(peps) ||
    throw(DimensionMismatch("operator and PEPS unit cells differ"))
    results = map(CartesianIndices(peps.A)) do site
        compute_nested_exp_hbond(nested, peps, operators[site], env, site)
    end

    return first.(results), last.(results)
end

function compute_nested_exp_vbond(
    nested::NestedNetwork,
    peps::Square_GPEPS,
    operator::Grassmann{<:Number, 4},
    env::CTMRGEnv,
    site)

    source = _source_site(site)
    neighbor = _layout_down_source(nested.layout, source)
    _check_nested_bond_operator(peps, operator, source, neighbor)
    closed_top = nested[_layout_x_site(nested.layout, source)]
    closed_bottom = nested[_layout_x_site(nested.layout, neighbor)]
    # denominator = C_v[X_source(I), K, X_neighbor(I)]
    denominator = _contract_nested_vpatch3(nested, env, source, closed_top, closed_bottom)
    denominator_value = _nested_scalar_or_zero(denominator)
    bottom_op, top_op = _bond_operator_gsvd(operator)
    top_x = _nested_x_bond_operator(nested, peps, source, top_op)
    bottom_x = _nested_x_bond_operator(nested, peps, neighbor, bottom_op)
    # numerator = C_v[X_top(a), K, X_bottom(a)]
    numerator_tensor = _contract_nested_vpatch3_alpha(nested, env, source, top_x, bottom_x)
    numerator = _nested_scalar_or_zero(numerator_tensor)

    return denominator, numerator/denominator_value
end

function compute_nested_exp_vbond(
    nested::NestedNetwork,
    peps::Square_GPEPS,
    operators::AbstractMatrix{<:Grassmann},
    env::CTMRGEnv)

    size(operators) == size(peps) ||
    throw(DimensionMismatch("operator and PEPS unit cells differ"))
    results = map(CartesianIndices(peps.A)) do site
        compute_nested_exp_vbond(nested, peps, operators[site], env, site)
    end

    return first.(results), last.(results)
end
