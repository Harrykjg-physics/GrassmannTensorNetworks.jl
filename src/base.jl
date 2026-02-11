################################ Helper for AbstractArray backend support ################################

"""
    _similar_arraytype(::Type{AT}, ::Type{Q}) -> Type

Compute the output array type when changing element type from T to Q,
preserving the array backend. Specialized in extensions (e.g., CuArray).
"""

@inline _similar_arraytype(::Type{AT}, ::Type{Q}) where {T, N, AT<:Array{T, N}, Q} = Array{Q, N}
@inline _similar_arraytype(::Type{AT}, ::Type{Q}) where {T, N, AT<:AbstractArray{T, N}, Q} = Array{Q, N}

################################ Base operations ################################

"""
    Base.similar(t::Grassmann{T, N, AT}) where {T, N, AT}

Create a new Grassmann tensor with the same structure as `t`.
"""

function Base.similar(t::Grassmann{T, N, AT}) where {T, N, AT}
    data_dict = Dict{NTuple{N, Int}, AT}()
    data_pairs = nonzero_pairs(t)
    sizehint!(data_dict, length(data_pairs))
    @inbounds for (key, block) in data_pairs
        data_dict[key] = similar(block)
    end
    return Grassmann(size(t), even(t), index_type(t), data_dict)
end

function Base.similar(t::Grassmann{T, N, AT}, Q::Type) where {T, N, AT}
    OAT = _similar_arraytype(AT, Q)
    data_dict = Dict{NTuple{N, Int}, OAT}()
    data_pairs = nonzero_pairs(t)
    sizehint!(data_dict, length(data_pairs))
    @inbounds for (key, block) in data_pairs
        data_dict[key] = similar(block, Q)
    end
    return Grassmann(size(t), even(t), index_type(t), data_dict)
end

"""
    Base.isapprox(t1::Grassmann, t2::Grassmann; kwargs...)

Check approximate equality of two Grassmann tensors.
Returns `false` for mismatched structures instead of throwing.
Supports all keyword arguments of `Base.isapprox` (e.g., `atol`, `rtol`).
"""

function Base.isapprox(t1::Grassmann, t2::Grassmann; kwargs...)
    size(t1) == size(t2) || return false
    even(t1) == even(t2) || return false
    nonzero_keys(t1) == nonzero_keys(t2) || return false
    @inbounds for k in nonzero_keys(t1)
        isapprox(t1[k], t2[k]; kwargs...) || return false
    end
    return true
end

"""
    Base.copy(t::Grassmann{T, N, AT}) where {T, N, AT}

Create a deep copy of the given Grassmann tensor, preserving the array backend.
"""

function Base.copy(t::Grassmann{T, N, AT}) where {T, N, AT}
    data_dict = Dict{NTuple{N, Int}, AT}()
    data_pairs = nonzero_pairs(t)
    sizehint!(data_dict, length(data_pairs))
    @inbounds for (key, block) in data_pairs
        data_dict[key] = copy(block)
    end
    return Grassmann(size(t), even(t), index_type(t), data_dict)
end

"""
    Base.:+(t1::Grassmann, t2::Grassmann)

Addition of two Grassmann tensors. Supports mixed element types via promotion.
"""

function Base.:+(t1::Grassmann{T1, N, AT1}, t2::Grassmann{T2, N, AT2}) where {T1, T2, N, AT1, AT2}

    (size(t1) == size(t2)) || throw(ArgumentError("Total dimensions mismatch"))
    (even(t1) == even(t2)) || throw(ArgumentError("Even-parity dimensions mismatch"))
    (tensor_parity(t1) == tensor_parity(t2)) || throw(ArgumentError("Tensor parity mismatch"))
    (index_type(t1) == index_type(t2)) || throw(ArgumentError("Index type mismatch"))

    T3 = promote_type(T1, T2)
    OAT = _similar_arraytype(AT1, T3)
    data_keys = nonzero_keys(t1)
    data_dict = Dict{NTuple{N, Int}, OAT}()
    sizehint!(data_dict, length(data_keys))
    @inbounds for key in data_keys
        data_dict[key] = t1[key] + t2[key]
    end
    return Grassmann(size(t1), even(t1), index_type(t1), data_dict)
end

"""
    Base.:-(t1::Grassmann, t2::Grassmann)

Subtraction of two Grassmann tensors. Supports mixed element types via promotion.
"""

function Base.:-(t1::Grassmann{T1, N, AT1}, t2::Grassmann{T2, N, AT2}) where {T1, T2, N, AT1, AT2}
    
    (size(t1) == size(t2)) || throw(ArgumentError("Total dimensions mismatch"))
    (even(t1) == even(t2)) || throw(ArgumentError("Even-parity dimensions mismatch"))
    (tensor_parity(t1) == tensor_parity(t2)) || throw(ArgumentError("Tensor parity mismatch"))
    (index_type(t1) == index_type(t2)) || throw(ArgumentError("Index type mismatch"))

    T3 = promote_type(T1, T2)
    OAT = _similar_arraytype(AT1, T3)
    data_keys = nonzero_keys(t1)
    data_dict = Dict{NTuple{N, Int}, OAT}()
    sizehint!(data_dict, length(data_keys))
    @inbounds for key in data_keys
        data_dict[key] = t1[key] - t2[key]
    end
    return Grassmann(size(t1), even(t1), index_type(t1), data_dict)
end

"""
    Base.:*(t::Grassmann{T, N, AT}, val::Number) where {T, N, AT}

Right scalar multiplication of a Grassmann tensor.
"""

function Base.:*(t::Grassmann{T, N, AT}, val::Number) where {T, N, AT}
    Q = promote_type(T, typeof(val))
    OAT = _similar_arraytype(AT, Q)
    data_dict = Dict{NTuple{N, Int}, OAT}()
    data_pairs = nonzero_pairs(t)
    sizehint!(data_dict, length(data_pairs))
    @inbounds for (key, block) in data_pairs
        data_dict[key] = block * val
    end
    return Grassmann(size(t), even(t), index_type(t), data_dict)
end

"""
    Base.:*(val::Number, t::Grassmann)

Left scalar multiplication of a Grassmann tensor.
"""

@inline Base.:*(val::Number, t::Grassmann) = t * val

"""
    Base.:/(t::Grassmann{T, N, AT}, val::Number) where {T, N, AT}

Scalar division of a Grassmann tensor.
"""

function Base.:/(t::Grassmann{T, N, AT}, val::Number) where {T, N, AT}
    Q = promote_type(T, typeof(val))
    OAT = _similar_arraytype(AT, Q)
    data_dict = Dict{NTuple{N, Int}, OAT}()
    data_pairs = nonzero_pairs(t)
    sizehint!(data_dict, length(data_pairs))
    @inbounds for (key, block) in data_pairs
        data_dict[key] = block / val
    end
    return Grassmann(size(t), even(t), index_type(t), data_dict)
end

"""
    Base.maximum(t::Grassmann{T}) where {T}

Maximum value across all blocks of the given Grassmann tensor.
"""

function Base.maximum(t::Grassmann{T}) where {T}
    isempty(data(t)) && return typemin(T)
    max_val = typemin(T)
    @inbounds for block in nonzero_vals(t)
        max_val = max(max_val, maximum(block))
    end
    return max_val
end

"""
    Base.sum(t::Grassmann)

Sum of all elements across all blocks of the Grassmann tensor.
"""

function Base.sum(t::Grassmann)
    return sum(sum(v) for (k, v) in nonzero_pairs(t))
end

"""
    Base.sum(f, t::Grassmann)

Sum of f(x) over all elements x in the Grassmann tensor.
"""

function Base.sum(f::F, t::Grassmann) where F
    return sum(sum(f.(v)) for (k, v) in nonzero_pairs(t))
end

"""
    Base.real(t::Grassmann{T, N, AT}) where {T, N, AT}

Real part of the given Grassmann tensor, preserving the array backend.
"""

function Base.real(t::Grassmann{T, N, AT}) where {T, N, AT}
    T_real = real(T)
    OAT = _similar_arraytype(AT, T_real)
    data_dict = Dict{NTuple{N, Int}, OAT}()
    data_pairs = nonzero_pairs(t)
    sizehint!(data_dict, length(data_pairs))
    @inbounds for (key, block) in data_pairs
        data_dict[key] = real(block)
    end
    return Grassmann(size(t), even(t), index_type(t), data_dict)
end

"""
    Base.conj(t::Grassmann{T, N, AT}; sign_function=trivial_sign) where {T, N, AT}

Conjugate the index types and tensor values of the given Grassmann tensor.

A conjugation sign factor is incorporated if `sign_function = auto_sign` is enabled.
"""

function Base.conj(
    t::Grassmann{T, N, AT};
    sign_function::F=trivial_sign) where {T, N, AT, F}

    data_dict = Dict{NTuple{N, Int}, AT}()
    data_pairs = nonzero_pairs(t)
    sizehint!(data_dict, length(data_pairs))
    perm_dst = ntuple(i -> N - i + 1, Val(N))
    @inbounds for (key, block) in data_pairs
        sign_num = sign_function(key, perm_dst)
        data_dict[key] = conj(block) * sign_num
    end
    new_index_types = map(conjugate, index_type(t))
    return Grassmann(size(t), even(t), new_index_types, data_dict)
end

"""
    Base.permutedims(t::Grassmann{T, N, AT}, dst; sign_function=trivial_sign)

Permute the given Grassmann tensor.

A permutation sign factor is incorporated if `sign_function = auto_sign` is enabled.
"""

function Base.permutedims(
    t::Grassmann{T, N, AT},
    dst::NTuple{N, Int};
    sign_function::F=trivial_sign) where {T, N, AT, F}

    data_dict = Dict{NTuple{N, Int}, AT}()
    data_pairs = nonzero_pairs(t)
    sizehint!(data_dict, length(data_pairs))
    total_size = permute(size(t), dst)
    even_size = permute(even(t), dst)
    index_types = permute(index_type(t), dst)
    @inbounds for (key, block) in data_pairs
        sign_num = sign_function(key, dst)
        key_permute = permute(key, dst)
        data_dict[key_permute] = permutedims(block, dst) * sign_num
    end
    return Grassmann(total_size, even_size, index_types, data_dict)
end

"""
    Base.getindex(t::Grassmann, target_idx::Int, dim_trunc::Tuple{Int, Int})

Truncate the dimension of a specified Z2 index (`target_idx`) of a Grassmann tensor.

    dim_trunc[1] : The retained even-parity dimension
    dim_trunc[2] : The retained odd-parity dimension
"""

function Base.getindex(
    t::Grassmann{T, N, AT},
    target_idx::Int,
    dim_trunc::Tuple{Int, Int}) where {T, N, AT}

    target_idx ∈ 1:N || throw(ArgumentError("Index $target_idx is out of range!"))

    total_size = size(t)
    even_size = even(t)
    odd_size = odd(t)

    even_dim_trunc, odd_dim_trunc = dim_trunc

    (even_dim_trunc + odd_dim_trunc > 0) || throw(ArgumentError("The total index dimension after truncation cannot be zero!"))
    even_dim_trunc <= even_size[target_idx] || throw(ArgumentError(
        "The retained even-parity dimension $even_dim_trunc is larger than the original dimension!"))
    odd_dim_trunc <= odd_size[target_idx] || throw(ArgumentError(
        "The retained odd-parity dimension $odd_dim_trunc is larger than the original dimension!"))

    range_dict = prepare_range_dict(t, total_size, even_size)

    total_size1 = deleteat(total_size, target_idx)
    even_size1 = deleteat(even_size, target_idx)
    total_size_new = insertafter(total_size1, target_idx - 1, (even_dim_trunc + odd_dim_trunc,))
    even_size_new = insertafter(even_size1, target_idx - 1, (even_dim_trunc,))

    data_dict = Dict{NTuple{N, Int}, AT}()
    data_pairs = nonzero_pairs(t)
    sizehint!(data_dict, length(data_pairs))

    @inbounds for (key, block) in data_pairs
        indices_range = range_dict[key]
        if key[target_idx] == 0 && even_dim_trunc != 0
            indices_range[target_idx] = 1:even_dim_trunc
            data_dict[key] = block[indices_range...]
        elseif key[target_idx] == 1 && odd_dim_trunc != 0
            indices_range[target_idx] = 1:odd_dim_trunc
            data_dict[key] = block[indices_range...]
        end
    end

    return Grassmann(total_size_new, even_size_new, index_type(t), data_dict)
end

"""
    prepare_range_dict(t::Grassmann, total_size, even_size)

Compute the index ranges of each sector for usage in dimension truncation.

e.g. For a tensor with total_size=(4, 4, 6) and even_parity_size=(2, 2, 3), return a Dict:

        (0, 0, 0) => [1:2, 1:2, 1:3], (1, 1, 0) => [1:2, 1:2, 1:3]
        (0, 1, 1) => [1:2, 1:2, 1:3], (1, 0, 1) => [1:2, 1:2, 1:3]
"""

function prepare_range_dict(
    t::Grassmann{T, N},
    total_size::NTuple{N, Int},
    even_size::NTuple{N, Int}) where {T, N}

    _, mask_range = _parity_mask(total_size, even_size)
    range_dict = Dict{NTuple{N, Int}, Vector{UnitRange{Int}}}()
    data_keys = nonzero_keys(t)
    sizehint!(range_dict, length(data_keys))
    indices_range = Vector{UnitRange{Int}}(undef, N)
    @inbounds for key in data_keys
        for idx in 1:N
            qn = key[idx]
            indices_range[idx] = 1:length(mask_range[idx][qn + 1])
        end
        range_dict[key] = copy(indices_range)
    end
    return range_dict
end

"""
    Base.abs(t::Grassmann{T, N, AT}) where {T, N, AT}

Element-wise absolute value of the Grassmann tensor, preserving the array backend.
"""

function Base.abs(t::Grassmann{T, N, AT}) where {T, N, AT}
    T_real = real(T)
    OAT = _similar_arraytype(AT, T_real)
    data_dict = Dict{NTuple{N, Int}, OAT}()
    data_pairs = nonzero_pairs(t)
    sizehint!(data_dict, length(data_pairs))
    @inbounds for (key, block) in data_pairs
        data_dict[key] = abs.(block)
    end
    return Grassmann(size(t), even(t), index_type(t), data_dict)
end

"""
    Base.abs2(t::Grassmann{T, N, AT}) where {T, N, AT}

Element-wise squared absolute value (|z|²) of the Grassmann tensor, preserving the array backend.
"""

function Base.abs2(t::Grassmann{T, N, AT}) where {T, N, AT}
    T_real = real(T)
    OAT = _similar_arraytype(AT, T_real)
    data_dict = Dict{NTuple{N, Int}, OAT}()
    data_pairs = nonzero_pairs(t)
    sizehint!(data_dict, length(data_pairs))
    @inbounds for (key, block) in data_pairs
        data_dict[key] = abs2.(block)
    end
    return Grassmann(size(t), even(t), index_type(t), data_dict)
end

"""
    Base.sqrt(t::Grassmann{T, N, AT}) where {T, N, AT}

Element-wise square root of the Grassmann tensor, preserving the array backend.
"""

function Base.sqrt(t::Grassmann{T, N, AT}) where {T, N, AT}
    data_dict = Dict{NTuple{N, Int}, AT}()
    data_pairs = nonzero_pairs(t)
    sizehint!(data_dict, length(data_pairs))
    @inbounds for (key, block) in data_pairs
        data_dict[key] = sqrt.(block)
    end
    return Grassmann(size(t), even(t), index_type(t), data_dict)
end

################################# Iteration Interface #########################################

# Simplified iterator: delegates to Dict iterator, avoiding custom state allocation.
# Returns (key, block) tuples for backward compatibility.
# https://docs.julialang.org/en/v1/manual/interfaces/#man-interface-iteration

@inline function Base.iterate(t::Grassmann)
    next = iterate(t.data)
    next === nothing && return nothing
    (p, s) = next
    return (p.first, p.second), s
end

@inline function Base.iterate(t::Grassmann, state)
    next = iterate(t.data, state)
    next === nothing && return nothing
    (p, s) = next
    return (p.first, p.second), s
end

Base.IteratorSize(::Type{<:Grassmann}) = Base.HasLength()
Base.length(t::Grassmann) = length(t.data)
Base.eltype(::Type{<:Grassmann{T, N}}) where {T, N} = Tuple{NTuple{N, Int}, AbstractArray{T, N}}
