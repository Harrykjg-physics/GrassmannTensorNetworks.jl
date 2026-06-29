"""
    log(t::GrassmannMatrix)

Compute the principal matrix logarithm independently in every nonzero sector.
"""

function LinearAlgebra.log(t::GrassmannMatrix)

    data_pairs = collect(nonzero_pairs(t))
    isempty(data_pairs) && throw(ArgumentError("cannot take the logarithm of an empty Grassmann matrix"))

    first_log = log(first(data_pairs)[2])
    OAT = typeof(first_log)
    data_dict = Dict{NTuple{2, Int}, OAT}()
    sizehint!(data_dict, length(data_pairs))

    for (i, (sector, block)) in enumerate(data_pairs)
        data_dict[sector] = i == 1 ? first_log : log(block)
    end

    return Grassmann(size(t), even(t), index_type(t), data_dict)
end

"""
    norm(t::Grassmann, p::Real=2)

Compute the entrywise `p`-norm across all nonzero sectors of `t`.
"""

function LinearAlgebra.norm(t::Grassmann, p::Real=2)

    isempty(data(t)) && return zero(real(eltype(t)))

    if p == 2
        return sqrt(sum(sum(abs2, block) for block in nonzero_vals(t)))
    elseif p == Inf
        return maximum(maximum(abs, block) for block in nonzero_vals(t))
    elseif p == -Inf
        return minimum(minimum(abs, block) for block in nonzero_vals(t))
    elseif p == 0
        return sum(count(!iszero, block) for block in nonzero_vals(t))
    elseif p > 0
        return sum(sum(x -> abs(x)^p, block) for block in nonzero_vals(t))^inv(p)
    end

    throw(ArgumentError("p must be nonnegative, Inf, or -Inf"))
end

"""
    diag(t::GrassmannMatrix)

Return the concatenated diagonals of the nonzero even sectors.
"""

function LinearAlgebra.diag(t::GrassmannMatrix)

    has_even = haskey(t, (0, 0))
    has_odd = haskey(t, (1, 1))

    has_even && has_odd && return vcat(diag(t[(0, 0)]), diag(t[(1, 1)]))
    has_even && return diag(t[(0, 0)])
    has_odd && return diag(t[(1, 1)])
    return Vector{eltype(t)}()
end

"""
    transpose(t::GrassmannMatrix; sign_function=trivial_sign)

Transpose every sector and exchange the two Grassmann indices.
"""

function LinearAlgebra.transpose(t::GrassmannMatrix{T}; sign_function::Function=trivial_sign) where {T}
    
    permutation = (2, 1)
    total_size_out = permute(size(t), permutation)
    even_size_out = permute(even(t), permutation)
    index_type_out = permute(index_type(t), permutation)

    data_pairs = nonzero_pairs(t)
    data_dict = Dict{NTuple{2, Int}, Matrix{T}}()
    sizehint!(data_dict, length(data_pairs))

    for (sector, block) in data_pairs
        permuted_sector = permute(sector, permutation)
        data_dict[permuted_sector] = sign_function(sector, permutation) * transpose(block)
    end

    return Grassmann(total_size_out, even_size_out, index_type_out, data_dict)
end

"""
    inv(t::GrassmannMatrix)

Invert every nonzero sector independently.
"""

function LinearAlgebra.inv(t::GrassmannMatrix{T}) where {T}
    
    data_pairs = nonzero_pairs(t)
    data_dict = Dict{NTuple{2, Int}, Matrix{T}}()
    sizehint!(data_dict, length(data_pairs))

    for (sector, block) in data_pairs
        data_dict[sector] = inv(block)
    end

    return Grassmann(size(t), even(t), index_type(t), data_dict)
end

# Basic vector-space methods backed by VectorInterface.
function LinearAlgebra.rmul!(t::Grassmann, alpha::Number)
    return iszero(alpha) ? zerovector!(t) : scale!(t, alpha)
end

function LinearAlgebra.lmul!(alpha::Number, t::Grassmann)
    return iszero(alpha) ? zerovector!(t) : scale!(t, alpha)
end

function LinearAlgebra.dot(t1::Grassmann, t2::Grassmann)
    size(t1) == size(t2) || throw(DimensionMismatch("Grassmann tensor sizes must match"))
    even(t1) == even(t2) || throw(DimensionMismatch("Grassmann sector sizes must match"))
    nonzero_keys(t1) == nonzero_keys(t2) || throw(DimensionMismatch("Grassmann sectors must match"))
    return sum(dot(t1[sector], t2[sector]) for sector in nonzero_keys(t1))
end

function LinearAlgebra.mul!(t1::Grassmann, t2::Grassmann, alpha::Number)
    return scale!(t1, t2, alpha)
end

function LinearAlgebra.mul!(t1::Grassmann, alpha::Number, t2::Grassmann)
    return scale!(t1, t2, alpha)
end

function LinearAlgebra.axpy!(alpha::Number, t1::Grassmann, t2::Grassmann)
    return VectorInterface.add!(t2, t1, alpha)
end

function LinearAlgebra.axpby!(alpha::Number, t1::Grassmann, beta::Number, t2::Grassmann)
    return VectorInterface.add!(t2, t1, alpha, beta)
end