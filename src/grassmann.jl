
################################ Define custom type of Grassmann tensor ###############################

# AbstractGrassmann <: Any
# Grassmann <:AbstractGrassmann
abstract type AbstractGrassmann{T, N} end

"""
A custom datatype to represent Grassmann tensors with Z2-symmetry

    the tensor can be either Grassmann-even or Grassmann-odd 
    Grassmann-even(odd) tensor means the tensor takes zero entries if the total quantum number(QN) is odd(even)
    e.g. if T[i, j, k, l] is Grassmann-even, then T[i, j, k, l] = 0 as long as mod(sum(p(i)+p(j)+p(k)+p(l)), 2) = 1
    where p(i) = 0, 1 computes the Z2 QN of the index state i

Fields:
    `total_size` : the total size of the tensor
    `even_parity_size` : the even-parity size of the tensor
    `index_type`: the type of the indices, 
     ---- :in (Symbol object) is associated with an ordinary Grassmann number 
     ---- :out (Symbol object) is associated with a dual Grassmann number
    `data` : a Dict object which stores non-trivial blocks of the given Z2-symmetric tensor 
    ----  key : Z2 QNs of the given block
    ----  value : CPU/GPU array 
"""

struct Grassmann{T, N, AT<:AbstractArray{T, N}} <: AbstractGrassmann{T, N}
    total_size::NTuple{N, Int}
    even_parity_size::NTuple{N, Int}
    index_type::NTuple{N, Symbol}
    data::Dict{NTuple{N, Int}, AT}
end

"""
Outer constructor of the Grassmann object, with dense array and specified initialization schemes.

Arguments:
    `total_size` : the total size of the tensor
    `even_parity_size` : the even-parity size of the tensor
    `index_type`: the type of the indices
    `T`: the element type of the tensor
    `init`: the initialization scheme with init=:undef, :random and :zeros (the default is init=:undef)
    `parity`: Specify the Z2-symmetry of the tensor with :even(Grassmann-even) and :odd(Grassmann-odd)
"""

function Grassmann(
    total_size::NTuple{N, Int},
    even_parity_size::NTuple{N, Int},
    index_types::NTuple{N, Symbol},
    T::Type;
    init::Symbol=:random,
    parity::Symbol=:even) where {N}
    
    block_dict = _fixed_parity_blocks(total_size, even_parity_size; parity)
    data_dict = Dict{NTuple{N, Int}, Array{T, N}}()
    sizehint!(data_dict, length(block_dict))

    @inbounds for (key, inds_range) in block_dict
        inds_size = map(length, inds_range)  
        val = if init == :undef
            Array{T, N}(undef, inds_size...)
        elseif init == :random
            N == 0 ? fill(rand(T)) : rand(T, inds_size...)
        elseif init == :zeros
            N == 0 ? fill(zero(T)) : zeros(T, inds_size...)
        else
            throw(ArgumentError("Unsupported initialization scheme: $init"))
        end
        data_dict[key] = val
    end

    return Grassmann(total_size, even_parity_size, index_types, data_dict)
end

"""
Outer constructor of the Grassmann object, given a dense array.

Note that the block structure of the given dense tensor should match the target Grassmann tensor!
"""

function Grassmann(
    tensor::Array{T, N},
    total_size::NTuple{N, Int},
    even_parity_size::NTuple{N, Int},
    index_types::NTuple{N, Symbol};
    parity::Symbol=:even) where {T, N}

    block_dict = _fixed_parity_blocks(total_size, even_parity_size; parity)
    data_dict = Dict{NTuple{N, Int}, Array{T, N}}()
    sizehint!(data_dict, length(block_dict))

    @inbounds for (key, inds_range) in block_dict
        block_view = view(tensor, inds_range...)
        data_dict[key] = copy(block_view)
    end

    return Grassmann(total_size, even_parity_size, index_types, data_dict)
end

################################ helper functions ###############################

"""
Indicate the parity structure of the indices of a given tensor (type stable function)

    e.g. a tensor with total_size = (6, 6, 6) and even_parity_size = (4, 6, 0), return

    mask_even_blocks = (
                        (0, 1), (the first index contains both even and odd-parity states)
                        (0, 0), (the second index contains only even-parity states)
                        (1, 1), (the third index contains only odd-parity states)
                        )

    mask_range = ( 
                  (1:4, 5:6),  (the range of even and odd-parity states for the first index)
                  (1:6, 1:6),  (the range of even-parity states for the second index)
                  (1:6, 1:6),  (the range of odd-parity states for the third index)
                  )
"""

function _parity_mask(
    total_size::NTuple{N, Int}, 
    even_parity_size::NTuple{N, Int}) where {N}

    all(total_size .>= even_parity_size) || throw(ArgumentError("The total dimensions 
    should not be smaller than the even-parity dimensions"))

    odd_parity_size = total_size .- even_parity_size
    
    combined = ntuple(Val(N)) do i
        @inbounds begin
            total_dim = total_size[i]
            even_dim = even_parity_size[i]
            odd_dim = odd_parity_size[i]
            
            if even_dim == 0
                # This index contains only odd-parity states
                ((1, 1), (1:total_dim, 1:total_dim))
            elseif odd_dim == 0
                # This index contains only even-parity states
                ((0, 0), (1:total_dim, 1:total_dim))
            else
                # This index contains both even-parity and odd-parity states
                ((0, 1), (1:even_dim, (even_dim + 1):total_dim))
            end
        end
    end

    mask_even_blocks = map(first, combined)
    mask_range = map(last, combined)
    
    return mask_even_blocks, mask_range
end

"""
Compute all the Z2 blocks (type stable function)

    Construct a Dict object containing key => value pairs

    ---- The key represents the Z2 QNs of the corresponding block
    ---- The value represents index range associated with the the QNs of a given block

    e.g. For a tensor with total_size=(4, 4, 6) and even_parity_size=(2, 2, 3), return a Dict :

        (0, 0, 0) => [1:2, 1:2, 1:3], (1, 1, 0) => [3:4, 3:4, 1:3]
        (0, 1, 1) => [1:2, 3:4, 4:6], (1, 0, 1) => [3:4, 1:2, 4:6]
"""

function _fixed_parity_blocks(
    total_size::NTuple{N, Int}, 
    even_parity_size::NTuple{N, Int}; 
    parity::Symbol=:even) where {N}

    parity ∈ (:even, :odd) || throw(ArgumentError("Tensor parity must be either Grassmann-even or Grassmann-odd"))

    mask_even_blocks, mask_range = _parity_mask(total_size, even_parity_size)

    dict = Dict{NTuple{N, Int}, Vector{UnitRange{Int}}}()
    isempty(mask_even_blocks) && isempty(mask_range) && (dict[()] = UnitRange{Int}[]; return dict)

    estimated_size = min(2^N, prod(i -> length(mask_even_blocks[i][1]:mask_even_blocks[i][2]), 1:N))
    sizehint!(dict, estimated_size)

    iter = ntuple(i -> mask_even_blocks[i][1]:mask_even_blocks[i][2], Val(N))
    target_parity = (parity === :even ? 0 : 1)

    sector_range = Vector{UnitRange{Int}}(undef, N)

    @inbounds for sector in Iterators.product(iter...)
        
        sum_parity = sum(sector) % 2
        sum_parity == target_parity || continue
        
        for (ind, p) in enumerate(sector)
            sector_range[ind] = mask_range[ind][p + 1]
        end
        dict[sector] = copy(sector_range)
    end

    return dict
end

################################ Basic methods for the Grassmann type ###############################

const GrassmannScalar{T} = Grassmann{T, 0}
const GrassmannVector{T} = Grassmann{T, 1}
const GrassmannMatrix{T} = Grassmann{T, 2}

Base.size(t::Grassmann) = t.total_size
even(t::Grassmann) = t.even_parity_size
odd(t::Grassmann) = t.total_size .- t.even_parity_size
Base.eltype(t::Grassmann{S, N}) where {S, N} = S 
Base.haskey(t::Grassmann, key::Tuple) = haskey(t.data, key)
index_type(t::Grassmann) = t.index_type
data(t::Grassmann) = t.data
scalar(t::GrassmannScalar) = t[()][1]
nonzero_keys(t::Grassmann) = keys(t.data)
nonzero_vals(t::Grassmann) = values(t.data)
nonzero_pairs(t::Grassmann) = pairs(t.data)
tensor_rank(t::Grassmann{Q, N}) where {Q, N} = N 
Base.getindex(t::Grassmann, key::Tuple) = t.data[key]

function Base.setindex!(t::Grassmann, val::AbstractArray, key::Tuple)
    @assert size(val) == size(t.data[key]) "Value size mismatch"
    t.data[key] = val
    return t
end

function tensor_parity(t::Grassmann)
    first_key = first(nonzero_keys(t))
    return mod(sum(first_key), 2)
end

function Base.convert(
    t::Grassmann{Q1, N}, Q2::Type) where {Q1, N}

    Q = promote_type(Q1, Q2)

    data_pairs = nonzero_pairs(t)
    data_dict = Dict{NTuple{N, Int}, Array{Q, N}}()
    sizehint!(data_dict, length(data_pairs))
    
    @inbounds for (key, block) in data_pairs
        new_block = Array{Q, N}(undef, size(block))
        @simd for i in eachindex(block)
            new_block[i] = Q(block[i])
        end
        data_dict[key] = new_block
    end

    return Grassmann(size(t), even(t), index_type(t), data_dict)
end

"""
Convert the Grassmann tensor to the dense array for testing purposes (type stable function)
"""

function Base.convert(
    ::Type{Array}, 
    t::Grassmann{T, N}) where {T, N}

    N == 0 && return fill(scalar(t))

    total_size = size(t)
    even_size = even(t)

    _, mask_range = _parity_mask(total_size, even_size)

    T_dense = zeros(T, total_size...)
    inds_range = Vector{UnitRange}(undef, N)

    @inbounds for (sector, v) in nonzero_pairs(t)
        for (ind, p) in enumerate(sector)
            inds_range[ind] = mask_range[ind][p+1]
        end
        T_dense[inds_range...] = v
    end

    return T_dense
end

"""
Perform conjugation of a single index type (type stable function)

    :in ==> :out
    :out ==> :in
"""

function conjugate(index_type::Symbol)
    if index_type == :in
        return :out
    elseif index_type == :out
        return :in
    else
        throw(ArgumentError("Only :in and :out are valid inputs"))
    end
end

"""
Conjugate the index type of multiple indices of a given tensor (type stable function).
"""

function index_conjugation(
    t::Grassmann{T, N}, 
    inds::NTuple{NR, Int}) where {T, N, NR}
    
    @assert all(1 ≤ i ≤ N for i in inds) "Indices out of bounds. Valid range: [1, $N]"
    @assert NR <= N "The number of conjugated indices should not be larger than the total number of indices"
    
    total_size = size(t)
    even_size = even(t)
    
    new_index_types = ntuple(N) do i
        i in inds ? conjugate(index_type(t)[i]) : index_type(t)[i]
    end
    
    data_dict = data(t)
    
    return Grassmann(total_size, even_size, new_index_types, data_dict)
end

function index_conjugation(t::Grassmann{T, N}, ind::Int) where {T, N}
    return index_conjugation(t, (ind, ))
end

######################### Save and load the Grassmann tensor #########################

function indextype2num(symin::Symbol)
    (symin === :in) ? 0 : 1
end

function save(
    filename::String, 
    param_str::String, 
    t::Grassmann)

    fid = h5open(filename * ".h5", "cw")
    create_group(fid, param_str)
    fid[param_str]["total_size"] = collect(size(t))
    fid[param_str]["even_size"] = collect(even(t))
    fid[param_str]["index_type"] = map(indextype2num, collect(index_type(t)))
    Tg_array = convert2array(t)
    fid[param_str]["data"] = Tg_array
    fid[param_str]["parity"] = tensor_parity(t)
    close(fid)
end

function num2indextype(numin::Int)
    (numin == 0) ? :in : :out
end

function load(
    filename::String, 
    param_str::String, 
    ::Type{Grassmann})

    fid = h5open(filename * ".h5", "r")
    size_t = read(fid[param_str]["total_size"])
    even_t = read(fid[param_str]["even_size"])
    index_type_t_num = read(fid[param_str]["index_type"])
    index_type_t = map(num2indextype, index_type_t_num)
    t_array = read(fid[param_str]["data"])
    p_num = read(fid[param_str]["parity"])
    p = (p_num == 0) ? :even : :odd
    close(fid)

    t = Grassmann(t_array, Tuple(size_t), Tuple(even_t), Tuple(index_type_t); parity=p)
end

