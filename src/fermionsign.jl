
################################ trivial fermionic sign functions ################################

function trivial_sign(args...)
    return 1
end

################################ automatical fermionic sign computations ################################

"""
Compute fermionic sign factors automatically when sign_functions=auto_sign is enabled

    :in (Symbol type) is associated with an ordinary Grassmann number
    :out (Symbol type) is associated with a dual Grassmann number
    the dual index should be moved to the right-hand side of the ordinary one 
    while performing Grassmann contractions (i.e. :in - :out)
"""

"""
Fermionic sign factor from boundary conditions during tracing operations
"""

function auto_sign(
    QN::NTuple{N, Int}, 
    pbc::NTuple{N, Bool}) where {N}

    sign = 1
    @inbounds for idx = 1:N
        if QN[idx] * Int(!pbc[idx]) == 1
            sign = -sign
        end
    end
    return sign
end

"""
Fermionic sign factor from performing a single Grassmann contraction between a pair of Grassmann numbers

Arguments:
    `QN` : Z2 quantum numbers corresponding to the indices(Grassmann numbers)
    `index` : the location of the first and second index(Grassmann number)
    `index_type` : the index type of the first and second index(Grassmann number)

    e.g.  QN = (a, b, c, d, e, f, g)
          index = (3, 5)
          index_type = (:out, :in)

    sign = p(e) × (p(d) + p(c))
"""

function auto_sign(
    QN::NTuple{N, Int}, 
    index::Tuple{Int, Int}, 
    index_type::Tuple{Symbol, Symbol}) where {N}

    @assert (index_type[1] !== index_type[2]) "A conjugated index should contract with a non-conjugated index"

    idx_start = (index_type[1] === :in ? index[1] + 1 : index[1])
    idx_stop = index[2]

    if idx_start > idx_stop-1
        # If the indices to be contracted are adjacent initially, with the order :in  :out 
        return 1
    else    
        QN_interm = QN[idx_start:idx_stop-1]
        flag = mod(sum(QN_interm) * QN[idx_stop], 2)
        sign = (flag == 1 ? -1 : 1)
        return sign
    end
end


"""
Fermionic sign factor from performing multiple Grassmann contractions between several pairs of Grassmann numbers

Arguments:
    `QN` : Z2 quantum numbers corresponding to the indices(Grassmann numbers)
    `index` : the location of each index(Grassmann number)
    `index_type` : the index type of each index(Grassmann number)

    e.g.  QN = (a, b, c, d, e, f, g)
          index = ((2, 3), (6, 5))
          index_type = ((:out, :in), (:in, :out))

    sign = p(f) × (p(b) + p(c) + p(d) + p(e)) + p(e) × p(d)
"""

function auto_sign(
    QN::NTuple{N1, Int}, 
    index::Tuple{NTuple{N2, Int}, NTuple{N2, Int}}, 
    index_type::Tuple{NTuple{N2, Symbol}, NTuple{N2, Symbol}}) where {N1, N2}

    index1, index2 = index
    index_type1, index_type2 = index_type

    index_record = [N1 + 1, ]  
    sign = 1

    @inbounds for idx = 1:N2

        index_new1 = index1[idx]
        index_new2 = index2[idx]

        n1 = sum(index_new1 .> index_record)
        n2 = sum(index_new2 .> index_record)

        index1_iter = index_new1 - n1
        index2_iter = index_new2 - n2

        sign *= auto_sign(QN, (index1_iter, index2_iter), (index_type1[idx], index_type2[idx]))
        QN = deleteat(QN, (index1_iter, index2_iter))
        
        push!(index_record, index_new1)
        push!(index_record, index_new2)
    end

    return sign
end

"""
Fermionic sign factor from contracting a single index 

Arguments:
    `QN1` : Z2 quantum numbers corresponding to the index of the first tensor
    `QN2` : Z2 quantum numbers corresponding to the index of the second tensor
    `index` : the location of the first and the second index
    `index_type` : the index type of the first and the second index

    e.g.  QN1 = (a, b, c, d); QN2 = (e, f, g)
          index = (3, 2)
          index_type = (:out, :in)

    sign = p(f) × (p(e) + p(d) + p(c))
"""

function auto_sign(
    QN1::NTuple{N1, Int}, 
    QN2::NTuple{N2, Int}, 
    index::Tuple{Int, Int}, 
    index_type::Tuple{Symbol, Symbol}) where {N1, N2}

    auto_sign(
        flatten(QN1, QN2), 
        (index[1], N1+index[2]), 
        index_type)
end

"""
Fermionic sign factor from contracting multiple indices 

Arguments:
    `QN1` : Z2 quantum numbers corresponding to the indices(Grassmann numbers) of the first tensor
    `QN2` : Z2 quantum numbers corresponding to the indices(Grassmann numbers) of the second tensor
    `index` : the location of each index(Grassmann number)
    `index_type` : the index type of each index(Grassmann number)

    e.g.  QN1 = (a, b, c, d); QN2 = (e, f, g)
          index = ((3, 4), (2, 3))
          index_type = ((:out, :in), (:in, :out))

    sign = p(f) × (p(e) + p(d) + p(c)) + p(g) × p(e)
"""

function auto_sign(
    QN1::NTuple{N1, Int}, 
    QN2::NTuple{N2, Int}, 
    index::Tuple{NTuple{N3, Int}, NTuple{N3, Int}}, 
    index_type::Tuple{NTuple{N3, Symbol}, NTuple{N3, Symbol}}) where {N1, N2, N3}

    @inbounds begin
        auto_sign(
            flatten(QN1, QN2), 
            (index[1], N1.+index[2]), 
            index_type)
    end
end

"""
Fermionic sign factor from Grassmann number permutation 

Arguments:
    `QN` : Z2 quantum numbers corresponding to the index(Grassmann numbers)
    `dst` : the new order after permutation

    e.g.  QN = (a, b, c, d, e)
          dst = (4, 2, 5, 1, 3)

    sign = p(d) × (p(a) + p(b) + p(c)) + p(b) × p(a) + p(e) × (p(c) + p(a))
"""

function auto_sign(::Tuple{}, ::Tuple{})
    return 1
end

function auto_sign(
    QN::NTuple{N, Int}, 
    dst::NTuple{N, Int}) where {N}

    sign = 1
    pos_tup = ntuple(i->i, N)

    @inbounds for i = 1:N

        pos = findfirst(x -> x == dst[i], pos_tup)
        
        if pos > 1
            parity_sum = 0
            for j in 1:(pos-1)
                parity_sum += QN[j]
            end
            flag = mod(parity_sum * QN[pos], 2)
            if flag == 1
                sign = -sign
            end
        end

        pos_tup = deleteat(pos_tup, pos)
        QN = deleteat(QN, pos)
    end

    return sign
end

"""
Fermionic sign factor given a single Z2 QN
"""

auto_sign(qn::Int) = (-1)^qn

##########################################################################################################

"""
Incorporate a parity sign factor (-1)^p(i) into the coefficient tensor (type-stable implementation)

    A ←←←←←←←←←← B <===> A →→→→→ P →→→→ B
"""

function add_parity_sign(
    t::Grassmann{T, N, AT}, 
    ind::Int;
    sign_function::Function=auto_sign) where {T, N, AT}
    
    1 ≤ ind ≤ N || throw(ArgumentError("Index $ind out of bounds [1, $N]"))
    
    data_pairs = nonzero_pairs(t)
    data_dict = Dict{NTuple{N, Int}, AT}()
    sizehint!(data_dict, length(data_pairs))
    
    @inbounds for (key, block) in data_pairs
        qn = key[ind]
        sign = sign_function(qn)
        new_block = similar(block)
        @simd for i in eachindex(block)
            new_block[i] = sign * block[i]
        end
        data_dict[key] = new_block
    end
    
    return Grassmann(size(t), even(t), index_type(t), data_dict)
end

"""
In-place version of add_parity_sign.
"""

function add_parity_sign!(
    t::Grassmann{T, N, AT}, 
    ind::Int;
    sign_function::Function=auto_sign) where {T, N, AT}
    
    1 ≤ ind ≤ N || throw(ArgumentError("Index $ind out of bounds [1, $N]"))
    
    @inbounds for (key, block) in nonzero_pairs(t)
        qn = key[ind]
        sign = sign_function(qn)
        @simd for i in eachindex(block)
            block[i] *= sign
        end
    end
    
    return t
end

"""
Incorporate a permutation sign factor into the coefficient tensor(type-stable implementation)
"""

function add_perm_sign(
    t::Grassmann{T, N, AT}, 
    dst::NTuple{N, Int}; 
    sign_function::Function=auto_sign) where {T, N, AT}

    data_dict = Dict{NTuple{N, Int}, AT}()
    data_pairs = nonzero_pairs(t)
    sizehint!(data_dict, length(data_pairs))

    @inbounds for (inds, block) in data_pairs
        sign_num = sign_function(inds, dst)
        new_block = similar(block)
        @simd for i in eachindex(block)
            new_block[i] = block[i] * sign_num
        end
        data_dict[inds] = new_block
    end

    return Grassmann(size(t), even(t), index_type(t), data_dict)
end

"""
In-place version of add_perm_sign. 
"""

function add_perm_sign!(
    t::Grassmann{T, N, AT}, 
    dst::NTuple{N, Int}; 
    sign_function::Function=auto_sign) where {T, N, AT}

    @inbounds for (inds, block) in nonzero_pairs(t)
        sign_num = sign_function(inds, dst)
        @simd for i in eachindex(block)
            block[i] *= sign_num
        end
    end

    return t
end
