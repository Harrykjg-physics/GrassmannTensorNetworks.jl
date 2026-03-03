
#################################### auxiliary functions ##########################################

"""
Calculate all the possible Z2 sectors given the total size and even-parity size of the fused dimensions (type stable)
"""

function calculate_sectors(
    total_size::NTuple{N, Int}, 
    even_size::NTuple{N, Int}) where {N}

    odd_size = total_size .- even_size
    iter_vec = Vector{UnitRange{Int}}(undef, N)

    for idx in 1:N
        odd_size[idx] == 0 ? (iter_vec[idx] = 0:0) : (even_size[idx] == 0 ? (iter_vec[idx] = 1:1) : (iter_vec[idx] = 0:1))
    end

    dims = ntuple(i -> length(iter_vec[i]), Val(N))

    sectors = [ntuple(i -> iter_vec[i][s[i]], Val(N)) for s in CartesianIndices(dims)]
end

"""
Calculate the total dimension and the even-parity dimension of the fused index (type stable)
"""

function calculate_fused_size(
    total_size::NTuple{N, Int}, 
    even_size::NTuple{N, Int}) where {N}

    total_dim_fused = prod(total_size)
    odd_size = total_size .- even_size

    idx_sizes = ntuple(Val(N)) do idx
        (even_size[idx], odd_size[idx])
    end

    all_sectors = calculate_sectors(total_size, even_size)
    even_dim_fused = 0

    # Accumulate the total dimension of even-parity sectors
    for sector in all_sectors
        if mod(sum(sector), 2) == 0 
            sector_dim = 1
            for (idx, qn) in enumerate(sector)
                sector_dim *= idx_sizes[idx][qn+1]
            end
            even_dim_fused += sector_dim
        end
    end

    return total_dim_fused, even_dim_fused
end

"""
Prepare the information of the fused Grassmann tensor (type stable)
"""

function prepare_fused_info(
    total_size::NTuple{N1, Int},
    even_size::NTuple{N1, Int},
    index_types::NTuple{N1, Symbol},
    index_type_fused::Symbol,
    inds::NTuple{N2, Int}) where {N1, N2}

    min_ind = minimum(inds)

    total_dim_fused, even_dim_fused = calculate_fused_size(getindices(total_size, inds), getindices(even_size, inds))

    total_size_out = insertafter(deleteat(total_size, inds), min_ind-1, (total_dim_fused, ))
    even_size_out = insertafter(deleteat(even_size, inds), min_ind-1, (even_dim_fused, ))
    index_type_out = insertafter(deleteat(index_types, inds), min_ind-1, (index_type_fused, ))

    return total_size_out, even_size_out, index_type_out
end

#################################### General Z2 fusion ##########################################

"""
Fuse N2 Z2-indices `inds` of a given rank-N1 `tensor` into a single Z2-index (type stable)
    
    The indices to be fused should be nearby and placed in an increasing order. 

    The fused index type is :in by default (`index_type_fused` = :in). 
"""

function fuse(
    tensor::Grassmann{T, N1, AT1},
    inds::NTuple{N2, Int}; 
    index_type_fused::Symbol=:in) where {T, N1, N2, AT1}

    N1 >= N2 || throw(ArgumentError("The rank of the given tensor should not be smaller than the number of indices to be fused ! "))
    inds[N2] == inds[1] + N2 - 1 || throw(ArgumentError("The indices to be fused should be nearby with an increasing order ! "))

    N_out = N1 - N2 + 1
    min_ind = minimum(inds)

    total_size, even_size = size(tensor), even(tensor)
    total_size_fused = getindices(total_size, inds)
    even_size_fused = getindices(even_size, inds)
    # The sectors_target fix the order of fused sectors given the same total_size_fused and even_size_fused
    sectors_target = calculate_sectors(total_size_fused, even_size_fused)
    
    data_dict_out = Dict{NTuple{N_out, Int}, AbstractArray{T, N_out}}()
    iter_dict = Dict{NTuple{N_out, Int}, Vector{AbstractArray{T, N_out}}}()
    grouped_keys = Dict{NTuple{N2, Int}, Vector{NTuple{N1, Int}}}()
    data_keys = nonzero_keys(tensor)
    sizehint!(data_dict_out, length(data_keys))
    sizehint!(grouped_keys, length(data_keys))

    @inbounds for qn_full in data_keys
        qn_target = getindices(qn_full, inds)
        push!(get!(()->Vector{NTuple{N1, Int}}(), grouped_keys, qn_target), qn_full)
    end

    empty_qn_fulls = NTuple{N1, Int}[]

    for qn_target in sectors_target

        # The Z2 quantum number(or total-parity) wouldn't change before/after Z2 fusion 
        qn_fused = mod(sum(qn_target), 2)

        qn_fulls = get(grouped_keys, qn_target, empty_qn_fulls)
        for qn_full in qn_fulls
            qn_del = deleteat(qn_full, inds)
            qn_out = insertafter(qn_del, min_ind-1, (qn_fused, ))
            # Perform tensor reshaping
            block = tensor[qn_full]
            size_full = size(block)
            size_del = deleteat(size_full, inds)
            fused_dim = prod(getindices(size_full, inds))
            size_fused = insertafter(size_del, min_ind-1, (fused_dim, ))
            tensor_out = reshape(block, size_fused)
            push!(get!(()->Vector{AbstractArray{T, N_out}}(), iter_dict, qn_out), tensor_out)
        end
    end

    for (qn_out, block) in pairs(iter_dict)
        data_dict_out[qn_out] = cat(block..., dims=min_ind)
    end

    total_size_out, even_size_out, index_type_out = prepare_fused_info(
        total_size, even_size, index_type(tensor), index_type_fused, inds)

    return Grassmann(total_size_out, even_size_out, index_type_out, data_dict_out)
end

#################################### General Z2 splitting ###########################################

"""
Split a single Z2-index specified by `ind` of a given Grassmann tensor `tensor` into multiple Z2-indices (type stable)

    Z2 split should satisfy :

        total_size, even_size, index_type_in = size(tensor), even(tensor), index_type(tensor)

        tensor_fused = fuse(tensor, inds)
      
        tensor == split(tensor_fused, minimum(inds), total_size, even_size, index_type_in) 
"""

function Base.split(
    tensor::Grassmann{T, N1, AT1},
    ind::Int, 
    total_size_split::NTuple{N2, Int}, 
    even_size_split::NTuple{N2, Int}, 
    index_type_split::NTuple{N2, Symbol}) where {T, N1, N2, AT1}

    N1 < N2 || throw(ArgumentError("The rank of the splitted tensor should be larger than that of the original tensor ! "))
    N = N2 - N1 # The number of increased indices
    inds_split = ntuple(i->i-1+ind, N+1)

    # Check whether the information of the splitted tensor is valid
    total_size_orig, even_size_orig, index_type_orig = size(tensor), even(tensor), index_type(tensor)
    total_size_test, even_size_test, index_type_test = prepare_fused_info(
        total_size_split, even_size_split, index_type_split, index_type_orig[ind], inds_split)
    (total_size_orig == total_size_test && even_size_orig == even_size_test && index_type_orig == index_type_test) || throw(ArgumentError(
        "The tensor size or index type of the splitted tensor does not match the original tensor ! "))

    # The sectors_target fix the order of splitted sectors given the same total_size_split and even_size_split
    sectors_target = calculate_sectors(getindices(total_size_split, inds_split), getindices(even_size_split, inds_split))
    # The Z2 quantum number(or total-parity) wouldn't change before/after Z2 split
    p_flag = (tensor_parity(tensor) == 0 ? :even : :odd)
    block_dict_out = _fixed_parity_blocks(total_size_split, even_size_split; parity=p_flag)
    data_pairs = pairs(block_dict_out)
    grouped_qn_out = Dict{NTuple{N + 1, Int}, Vector{NTuple{N2, Int}}}()
    sizehint!(grouped_qn_out, length(block_dict_out))
    @inbounds for (qn_out, _) in data_pairs
        qn_target = getindices(qn_out, inds_split)
        push!(get!(()->Vector{NTuple{N2, Int}}(), grouped_qn_out, qn_target), qn_out)
    end

    iter_dict = Dict{NTuple{N1, Int}, Int}()
    data_dict_out = Dict{NTuple{N2, Int}, AbstractArray{T, N2}}()
    sizehint!(data_dict_out, length(data_pairs))
    empty_qn_outs = NTuple{N2, Int}[]

    for qn_target in sectors_target

        qn = mod(sum(qn_target), 2)

        qn_outs = get(grouped_qn_out, qn_target, empty_qn_outs)
        for qn_out in qn_outs
            block = block_dict_out[qn_out]

            qn_del = deleteat(qn_out, inds_split)
            qn_orig = insertafter(qn_del, ind-1, (qn, ))
            # The tensor size of the current sector of the splitted tensor
            size_split = ntuple(i -> length(block[i]::UnitRange{Int}), N2)
            # The total dimension of the current sector corresponding to the splitted indices
            inds_split_dim = prod(getindices(size_split, inds_split))

            # Since a single `qn` corresponds to multiple `qn_target`
            # Therefore a single `qn_orig` corresponds to multiple `qn_out`
            # Each tensor_out[qn_out] is splitted from tensor[qn_orig]
            # i.e. tensor_out[qn_out] = reshape(tensor[qn_orig], (:, :, range1:range2, :, :))
            last_end = get(iter_dict, qn_orig, 0)
            range_iter = (last_end + 1):(last_end + inds_split_dim)
            iter_dict[qn_orig] = last_end + inds_split_dim

            range_inds = ntuple(i -> i == ind ? range_iter : Colon(), Val(N1))
            tensor_out = reshape(tensor[qn_orig][range_inds...], size_split)
            data_dict_out[qn_out] = tensor_out
        end
    end

    return Grassmann(total_size_split, even_size_split, index_type_split, data_dict_out)
end
