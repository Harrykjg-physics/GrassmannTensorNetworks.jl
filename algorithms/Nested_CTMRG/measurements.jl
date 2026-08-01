_source_site(site::CartesianIndex{2}) = site
_source_site(site::Tuple{Int, Int}) = CartesianIndex(site)

function _nested_x_operator_raw(
    nested::NestedNetwork,
    peps::Square_GPEPS,
    source::CartesianIndex{2},
    operator::Grassmann{T, 2},
) where {T}
    A = peps.A[source]
    # X_source(O)[L, r, u, D] = _nested_x(O, right(A), up(A))
    return _nested_x(
        operator,
        size(A)[3], even(A)[3],
        size(A)[4], even(A)[4],
    )
end

function nested_x_operator(
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
    return _nested_x_operator_raw(nested, peps, source, operator)
end

# nested_y_operator(...) = nested_x_operator(...) preserves the former public API name.
nested_y_operator(args...) = nested_x_operator(args...)

function compute_nested_exp_site(
    nested::NestedNetwork,
    peps::Square_GPEPS,
    operator::Grassmann{<:Number, 2},
    env::CTMRGEnv,
    site,
)
    source = _source_site(site)
    xsite = nested.layout.x_sites[source]
    # impurity[xsite] = X_source(operator)
    impurity = nested_x_operator(nested, peps, source, operator)
    return compute_exp_site(
        nested[xsite], impurity,
        env.El[xsite], env.Er[xsite], env.Eu[xsite], env.Ed[xsite],
        env.Clu[xsite], env.Cru[xsite], env.Cld[xsite], env.Crd[xsite],
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
    # matrix_order[po1, pi1, po2, pi2] <-- operator[po1, po2, pi1, pi2]
    matrix_order = permutedims(
        operator, (1, 3, 2, 4); sign_function=global_sign
    )
    dense = convert(Array, matrix_order)
    dout1, dout2, din1, din2 = size(operator)
    # matrix[(po1, pi1), (po2, pi2)] = matrix_order[po1, pi1, po2, pi2]
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
            # operator[po1, po2, pi1, pi2] += left[po1, pi1] * right[po2, pi2]
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
    left_x::Grassmann,
    right_x::Grassmann,
)
    x1 = nested.layout.x_sites[source]
    next_source = CartesianIndex(
        source[1],
        Nmod(source[2] + 1, nested.layout.source_size[2]),
    )
    x2 = nested.layout.x_sites[next_source]
    middle = CartesianIndex(
        x1[1], Nmod(x1[2] + 1, size(nested, 2))
    )
    return _contract_horizontal_strip(
        (left_x, nested[middle], right_x),
        env,
        (x1, middle, x2),
    )
end

function _contract_nested_vpatch3(
    nested::NestedNetwork,
    env::CTMRGEnv,
    source::CartesianIndex{2},
    top_x::Grassmann,
    bottom_x::Grassmann,
)
    x1 = nested.layout.x_sites[source]
    next_source = CartesianIndex(
        Nmod(source[1] + 1, nested.layout.source_size[1]),
        source[2],
    )
    x2 = nested.layout.x_sites[next_source]
    middle = CartesianIndex(
        Nmod(x1[1] + 1, size(nested, 1)), x1[2]
    )
    return _contract_vertical_strip(
        (top_x, nested[middle], bottom_x),
        env,
        (x1, middle, x2),
    )
end

_nested_scalar_or_zero(value::GrassmannScalar) =
    isempty(nonzero_keys(value)) ? zero(eltype(value)) : scalar(value)

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
    closed_left = nested[nested.layout.x_sites[source]]
    closed_right = nested[nested.layout.x_sites[neighbor]]
    # denominator = C_h[X_source(I), B, X_neighbor(I)]
    denominator = _contract_nested_hpatch3(
        nested, env, source, closed_left, closed_right
    )
    denominator_value = _nested_scalar_or_zero(denominator)
    terms = _operator_schmidt(operator)
    numerator_type =
        promote_type(typeof(denominator_value), eltype(operator))
    numerator = zero(numerator_type)
    for (left_op, right_op) in terms
        left_x = nested_x_operator(nested, peps, source, left_op)
        right_x = nested_x_operator(nested, peps, neighbor, right_op)
        term = _contract_nested_hpatch3(
            nested, env, source, left_x, right_x
        )
        # numerator += C_h[X(left_op), B, X(right_op)]
        numerator += _nested_scalar_or_zero(term)
    end
    return denominator, numerator / denominator_value
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
    closed_top = nested[nested.layout.x_sites[source]]
    closed_bottom = nested[nested.layout.x_sites[neighbor]]
    # denominator = C_v[X_source(I), K, X_neighbor(I)]
    denominator = _contract_nested_vpatch3(
        nested, env, source, closed_top, closed_bottom
    )
    denominator_value = _nested_scalar_or_zero(denominator)
    terms = _operator_schmidt(operator)
    numerator_type =
        promote_type(typeof(denominator_value), eltype(operator))
    numerator = zero(numerator_type)
    for (top_op, bottom_op) in terms
        top_x = nested_x_operator(nested, peps, source, top_op)
        bottom_x = nested_x_operator(nested, peps, neighbor, bottom_op)
        term = _contract_nested_vpatch3(
            nested, env, source, top_x, bottom_x
        )
        term_sign =
            (-one(eltype(operator)))^tensor_parity(top_op)
        # numerator += (-1)^|top_op| * C_v[X(top_op), K, X(bottom_op)]
        numerator += term_sign * _nested_scalar_or_zero(term)
    end
    return denominator, numerator / denominator_value
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
