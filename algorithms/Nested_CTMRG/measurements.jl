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
