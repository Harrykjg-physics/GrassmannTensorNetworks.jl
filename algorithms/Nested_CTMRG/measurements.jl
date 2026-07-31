_source_site(site::CartesianIndex{2}) = site
_source_site(site::Tuple{Int, Int}) = CartesianIndex(site)

function _nested_y_operator_raw(
    nested::NestedNetwork,
    peps::Square_GPEPS,
    source::CartesianIndex{2},
    operator::Grassmann{T, 2},
) where {T}
    rows, cols = size(peps)
    east_source =
        CartesianIndex(source[1], Nmod(source[2] + 1, cols))
    north_source =
        CartesianIndex(Nmod(source[1] - 1, rows), source[2])
    east_ket = nested[nested.layout.ket_sites[east_source]]
    north_bra = nested[nested.layout.bra_sites[north_source]]
    raw = _nested_y(
        operator,
        size(east_ket)[1], even(east_ket)[1],
        size(north_bra)[4], even(north_bra)[4],
    )
    return _nested_y_for_network(raw)
end

function nested_y_operator(
    nested::NestedNetwork,
    peps::Square_GPEPS,
    site,
    operator::Grassmann{T, 2},
) where {T}
    source = _source_site(site)
    checkbounds(Bool, peps.A, source) ||
        throw(ArgumentError("source site $source is outside the unit cell"))
    physical_size = size(peps.A[source])[1]
    physical_even = even(peps.A[source])[1]
    size(operator) == (physical_size, physical_size) ||
        throw(DimensionMismatch(
            "operator physical dimensions do not match PEPS"
        ))
    even(operator) == (physical_even, physical_even) ||
        throw(DimensionMismatch(
            "operator physical parity split does not match PEPS"
        ))
    index_type(operator) == (:out, :in) ||
        throw(ArgumentError("operator arrows must be (:out, :in)"))
    return _nested_y_operator_raw(nested, peps, source, operator)
end

function compute_nested_exp_site(
    nested::NestedNetwork,
    peps::Square_GPEPS,
    operator::Grassmann{<:Number, 2},
    env::CTMRGEnv,
    site,
)
    source = _source_site(site)
    ysite = nested.layout.y_sites[source]
    impurity = nested_y_operator(nested, peps, source, operator)
    return compute_exp_site(
        nested[ysite], impurity,
        env.El[ysite], env.Er[ysite], env.Eu[ysite], env.Ed[ysite],
        env.Clu[ysite], env.Cru[ysite], env.Cld[ysite], env.Crd[ysite],
    )
end

function _check_nested_operator_unit_cell(
    peps::Square_GPEPS,
    operators::AbstractMatrix,
)
    size(operators) == size(peps) ||
        throw(DimensionMismatch("operator and PEPS unit cells differ"))
    return nothing
end

function compute_nested_exp_site(
    nested::NestedNetwork,
    peps::Square_GPEPS,
    operators::AbstractMatrix{<:Grassmann{Q, 2}},
    env::CTMRGEnv,
) where {Q}
    _check_nested_operator_unit_cell(peps, operators)
    denominator =
        Matrix{promote_type(eltype(peps), Q)}(undef, size(peps)...)
    values = similar(denominator)
    for site in CartesianIndices(peps.A)
        denominator[site], values[site] =
            compute_nested_exp_site(
                nested, peps, operators[site], env, site
            )
    end
    return denominator, values
end

function _check_nested_bond_operator(
    peps::Square_GPEPS,
    operator::Grassmann{<:Number, 4},
    source::CartesianIndex{2},
    neighbor::CartesianIndex{2},
)
    checkbounds(Bool, peps.A, source) ||
        throw(ArgumentError("source site $source is outside the unit cell"))
    checkbounds(Bool, peps.A, neighbor) ||
        throw(ArgumentError("neighbor site $neighbor is outside the unit cell"))
    source_physical = size(peps.A[source])[1]
    neighbor_physical = size(peps.A[neighbor])[1]
    expected_size = (
        source_physical,
        neighbor_physical,
        source_physical,
        neighbor_physical,
    )
    size(operator) == expected_size ||
        throw(DimensionMismatch(
            "bond-operator physical dimensions do not match PEPS"
        ))
    source_even = even(peps.A[source])[1]
    neighbor_even = even(peps.A[neighbor])[1]
    expected_even = (
        source_even,
        neighbor_even,
        source_even,
        neighbor_even,
    )
    even(operator) == expected_even ||
        throw(DimensionMismatch(
            "bond-operator parity splits do not match PEPS"
        ))
    index_type(operator) == (:out, :out, :in, :in) ||
        throw(ArgumentError(
            "bond-operator arrows must be (:out, :out, :in, :in)"
        ))
    tensor_parity(operator) == 0 ||
        throw(ArgumentError("bond operator must have even total parity"))
    return nothing
end

function _operator_schmidt(operator::Grassmann{T, 4}) where {T}
    matrix_order = permutedims(
        operator, (1, 3, 2, 4); sign_function=global_sign
    )
    dense = convert(Array, matrix_order)
    dout1, dout2, din1, din2 = size(operator)
    matrix = reshape(
        dense,
        dout1 * din1, dout2 * din2,
    )
    parity(index, even_dim) = index <= even_dim ? 0 : 1
    row_parity = [
        mod(
            parity(out, even(operator)[1]) +
            parity(input, even(operator)[3]),
            2,
        ) for out in 1:dout1, input in 1:din1
    ][:]
    col_parity = [
        mod(
            parity(out, even(operator)[2]) +
            parity(input, even(operator)[4]),
            2,
        ) for out in 1:dout2, input in 1:din2
    ][:]

    terms = Tuple{Grassmann, Grassmann}[]
    for sector in 0:1
        rows = findall(==(sector), row_parity)
        cols = findall(==(sector), col_parity)
        factor = svd(matrix[rows, cols])
        for alpha in eachindex(factor.S)
            factor.S[alpha] > eps(real(float(one(T)))) || continue
            left_vector = zeros(eltype(factor.U), dout1 * din1)
            right_vector = zeros(eltype(factor.Vt), dout2 * din2)
            left_vector[rows] =
                factor.U[:, alpha] * sqrt(factor.S[alpha])
            right_vector[cols] =
                factor.Vt[alpha, :] * sqrt(factor.S[alpha])
            parity_symbol = sector == 0 ? :even : :odd
            left = Grassmann(
                reshape(left_vector, dout1, din1),
                (dout1, din1),
                (even(operator)[1], even(operator)[3]),
                (:out, :in);
                parity=parity_symbol,
            )
            right = Grassmann(
                reshape(right_vector, dout2, din2),
                (dout2, din2),
                (even(operator)[2], even(operator)[4]),
                (:out, :in);
                parity=parity_symbol,
            )
            push!(terms, (left, right))
        end
    end
    return terms
end

function _contract_horizontal_strip(
    bulks::NTuple{N, <:Grassmann},
    env::CTMRGEnv,
    sites::NTuple{N, CartesianIndex{2}},
) where {N}
    N > 0 || throw(ArgumentError("a strip must contain at least one tensor"))
    left_site = first(sites)
    right_site = last(sites)

    left_top = contract(
        env.Clu[left_site], env.El[left_site], (2, 1);
        sign_function=global_sign,
    )
    state = contract(
        left_top, env.Cld[left_site], (2, 2);
        sign_function=global_sign,
    )
    for (bulk, site) in zip(bulks, sites)
        state, _ = left_move(
            state, env.Ed[site], bulk, env.Eu[site]
        )
    end

    right_top = contract(
        env.Cru[right_site], env.Er[right_site], (2, 1);
        sign_function=global_sign,
    )
    right = contract(
        right_top, env.Crd[right_site], (2, 2);
        sign_function=global_sign,
    )
    return contract(
        state, right, ((1, 2, 3), (1, 2, 3));
        sign_function=global_sign,
    )
end

function _contract_vertical_strip(
    bulks::NTuple{N, <:Grassmann},
    env::CTMRGEnv,
    sites::NTuple{N, CartesianIndex{2}},
) where {N}
    N > 0 || throw(ArgumentError("a strip must contain at least one tensor"))
    top_site = first(sites)
    bottom_site = last(sites)

    upper_left = contract(
        env.Clu[top_site], env.Eu[top_site], (1, 1);
        sign_function=global_sign,
    )
    state = contract(
        upper_left, env.Cru[top_site], (2, 1);
        sign_function=global_sign,
    )
    for (bulk, site) in zip(bulks, sites)
        state, _ = up_move(
            state, env.El[site], bulk, env.Er[site]
        )
    end

    lower_left = contract(
        env.Cld[bottom_site], env.Ed[bottom_site], (1, 1);
        sign_function=global_sign,
    )
    lower = contract(
        lower_left, env.Crd[bottom_site], (2, 1);
        sign_function=global_sign,
    )
    return contract(
        state, lower, ((1, 2, 3), (1, 2, 3));
        sign_function=global_sign,
    )
end

function _contract_nested_hpatch3(
    nested::NestedNetwork,
    env::CTMRGEnv,
    source::CartesianIndex{2},
    left_y::Grassmann,
    right_y::Grassmann,
)
    y1 = nested.layout.y_sites[source]
    next_source = CartesianIndex(
        source[1],
        Nmod(source[2] + 1, nested.layout.source_size[2]),
    )
    y2 = nested.layout.y_sites[next_source]
    middle = CartesianIndex(
        y1[1], Nmod(y1[2] + 1, size(nested, 2))
    )
    return _contract_horizontal_strip(
        (left_y, nested[middle], right_y),
        env,
        (y1, middle, y2),
    )
end

function _contract_nested_vpatch3(
    nested::NestedNetwork,
    env::CTMRGEnv,
    source::CartesianIndex{2},
    top_y::Grassmann,
    bottom_y::Grassmann,
)
    y1 = nested.layout.y_sites[source]
    next_source = CartesianIndex(
        Nmod(source[1] + 1, nested.layout.source_size[1]),
        source[2],
    )
    y2 = nested.layout.y_sites[next_source]
    middle = CartesianIndex(
        Nmod(y1[1] + 1, size(nested, 1)), y1[2]
    )
    return _contract_vertical_strip(
        (top_y, nested[middle], bottom_y),
        env,
        (y1, middle, y2),
    )
end

function compute_nested_exp_hbond(
    nested::NestedNetwork,
    peps::Square_GPEPS,
    operator::Grassmann{<:Number, 4},
    env::CTMRGEnv,
    site,
)
    source = _source_site(site)
    neighbor = CartesianIndex(
        source[1], Nmod(source[2] + 1, size(peps)[2])
    )
    _check_nested_bond_operator(peps, operator, source, neighbor)
    identity_left = _physical_identity(peps.A[source])
    identity_right = _physical_identity(peps.A[neighbor])
    closed_left = nested_y_operator(nested, peps, source, identity_left)
    closed_right = nested_y_operator(nested, peps, neighbor, identity_right)
    denominator = _contract_nested_hpatch3(
        nested, env, source, closed_left, closed_right
    )
    terms = _operator_schmidt(operator)
    numerator = isempty(terms) ? zero(denominator) : sum(terms) do (left_op, right_op)
        left_y = nested_y_operator(nested, peps, source, left_op)
        right_y = nested_y_operator(nested, peps, neighbor, right_op)
        term_sign =
            (-one(eltype(operator)))^tensor_parity(left_op)
        term_sign * _contract_nested_hpatch3(
            nested, env, source, left_y, right_y
        )
    end
    return denominator, scalar(numerator) / scalar(denominator)
end

function compute_nested_exp_vbond(
    nested::NestedNetwork,
    peps::Square_GPEPS,
    operator::Grassmann{<:Number, 4},
    env::CTMRGEnv,
    site,
)
    source = _source_site(site)
    neighbor = CartesianIndex(
        Nmod(source[1] + 1, size(peps)[1]), source[2]
    )
    _check_nested_bond_operator(peps, operator, source, neighbor)
    identity_top = _physical_identity(peps.A[source])
    identity_bottom = _physical_identity(peps.A[neighbor])
    closed_top = nested_y_operator(nested, peps, source, identity_top)
    closed_bottom = nested_y_operator(nested, peps, neighbor, identity_bottom)
    denominator = _contract_nested_vpatch3(
        nested, env, source, closed_top, closed_bottom
    )
    terms = _operator_schmidt(operator)
    numerator = isempty(terms) ? zero(denominator) : sum(terms) do (top_op, bottom_op)
        top_y = nested_y_operator(nested, peps, source, top_op)
        bottom_y = nested_y_operator(nested, peps, neighbor, bottom_op)
        _contract_nested_vpatch3(nested, env, source, top_y, bottom_y)
    end
    return denominator, scalar(numerator) / scalar(denominator)
end

function compute_nested_exp_hbond(
    nested::NestedNetwork,
    peps::Square_GPEPS,
    operators::AbstractMatrix{<:Grassmann},
    env::CTMRGEnv,
)
    size(operators) == size(peps) ||
        throw(DimensionMismatch("operator and PEPS unit cells differ"))
    results = map(CartesianIndices(peps.A)) do site
        compute_nested_exp_hbond(
            nested, peps, operators[site], env, site
        )
    end
    return first.(results), last.(results)
end

function compute_nested_exp_vbond(
    nested::NestedNetwork,
    peps::Square_GPEPS,
    operators::AbstractMatrix{<:Grassmann},
    env::CTMRGEnv,
)
    size(operators) == size(peps) ||
        throw(DimensionMismatch("operator and PEPS unit cells differ"))
    results = map(CartesianIndices(peps.A)) do site
        compute_nested_exp_vbond(
            nested, peps, operators[site], env, site
        )
    end
    return first.(results), last.(results)
end
