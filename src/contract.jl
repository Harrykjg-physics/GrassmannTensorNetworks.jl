
############################### Fermionic tensor trace ###############################

function trace_size(
    tup::NTuple{N1, Union{Symbol, Int}}, 
    inds::NTuple{2, NTuple{N2, Int}}) where {N1, N2}

    N1 >= 2*N2 || throw(ArgumentError("invalid trace indices"))

    inds_tr = linearize(inds)
    tup_tr = deleteat(tup, inds_tr)
end

function trace_size(
    total_size::NTuple{N1, Int}, 
    even_size::NTuple{N1, Int}, 
    index_types::NTuple{N1, Symbol}, 
    inds::NTuple{2, NTuple{N2, Int}}) where {N1, N2}

    total_size_tr = trace_size(total_size, inds)
    even_size_tr = trace_size(even_size, inds)
    index_type_tr = trace_size(index_types, inds)

    return total_size_tr, even_size_tr, index_type_tr
end

function trace_structure(
    T::Grassmann{S, N1, AT}, 
    inds_tr::NTuple{2, NTuple{N2, Int}}; 
    cj::Bool=false, 
    perm::NTuple{N3, Int}=ntuple(i->i, N1-2*N2)) where {N1, N2, N3, S, AT}

    total_size, even_size, index_types = size(T), even(T), index_type(T)
    index_types = cj ? map(conjugate, index_types) : index_types
    
    total_size_out, even_size_out, index_types_out = trace_size(total_size, even_size, index_types, inds_tr)

    total_size_out = permute(total_size_out, perm)
    even_size_out = permute(even_size_out, perm)
    index_types_out = permute(index_types_out, perm)

    return total_size_out, even_size_out, index_types_out
end

"""
Fermionic tensor trace operation with support to anti-periodic condition (APBC) (type stable)

Arguments:
    `T` : the rank-N1 Grassmann tensor to be traced
    `inds_tr` : the 2 × N2 indices to be traced out 
    `sign_function` : specify the Fermionic sign function (default=trivial_sign) 
    `cj` : whether `T` is conjugated (true) or not (false) (default=false)
    `perm` : whether to permute of the traced tensor (default=(1, 2, ..., N3), means no permutation)
    `pbc` : whether to perform ordinary (PBC) trace or APBC trace (by default, pbc=true for each traced index)
"""

# trace a single index
function trace(
    T::Grassmann{S, N1, AT}, 
    inds_tr::NTuple{2, Int}; 
    sign_function::F=trivial_sign,
    cj::Bool=false, 
    perm::NTuple{N2, Int}=ntuple(i->i, N1-2), 
    pbc::Bool=true) where {N1, N2, S, AT, F}

    trace(
        T, 
        ((inds_tr[1], ), (inds_tr[2], ));
        sign_function=sign_function, 
        cj=cj, 
        perm=perm, 
        pbc=(pbc, )
        )
end

# trace multiple indices
function trace(
    T::Grassmann{S, N1, AT}, 
    inds_tr::NTuple{2, NTuple{N2, Int}}; 
    sign_function::F=trivial_sign,
    cj::Bool=false, 
    perm::NTuple{N3, Int}=ntuple(i->i, N1-2*N2), 
    pbc::NTuple{N2, Bool}=ntuple(i->true, N2)) where {N1, N2, N3, S, AT, F}

    total_size_out, even_size_out, index_types_out = trace_structure(T, inds_tr; cj=cj, perm=perm)
    tensor_parity(T) == 0 ? p_flag = :even : p_flag = :odd
    To = Grassmann(total_size_out, even_size_out, index_types_out, S; init=:undef, parity=p_flag)
    trace!(To, T, inds_tr; perm=perm, cj=cj, pbc=pbc, sign_function=sign_function)
    return To
end

# the mutating implementation
function trace!(
    To::Grassmann{S, N, ATO}, 
    T::Grassmann{S, N1, AT}, 
    inds_tr::NTuple{2, NTuple{N2, Int}}; 
    sign_function::F=trivial_sign,
    cj::Bool=false, 
    perm::NTuple{N, Int}=ntuple(i->i, N1-2*N2), 
    pbc::NTuple{N2, Bool}=ntuple(i->true, N2)) where {N, N1, N2, S, ATO, AT, F}

    inds_tr1 = inds_tr[1]
    inds_tr2 = inds_tr[2]
    inds_tr_flat = linearize(inds_tr)
    sort_tup = TupleTools.sortperm(inds_tr_flat)
    inds_tr_sort = permute(inds_tr_flat, sort_tup)

    perm_conj = (cj ? ntuple(i -> N1 - i + 1, N1) : ntuple(i -> i, N1))

    index_type1 = getindices(index_type(T), inds_tr1)
    index_type2 = getindices(index_type(T), inds_tr2)
    index_type1a = (cj ? map(conjugate, index_type1) : index_type1)
    index_type2a = (cj ? map(conjugate, index_type2) : index_type2)
    index_type_tr = (index_type1a, index_type2a)

    permutation = deleteat(ntuple(i->i, N1), inds_tr_flat)
    permutation = permute(permutation, perm)
    inv_perm = invperm(perm)
    inner_space = ntuple(_ -> 0:1, N2)

    for inds in nonzero_keys(To)

        inds_invp = permute(inds, inv_perm)
        sign_perm_num = sign_function(inds_invp, perm)

        count = 0

        for inner_inds in Iterators.product(inner_space...)

            inds_iter = inds_invp

            inner_inds_flat = linearize((inner_inds, inner_inds))
            inner_inds_sort = permute(inner_inds_flat, sort_tup) 

            @inbounds for idx in 1:2*N2
                inds_iter = insertafter(inds_iter, inds_tr_sort[idx]-1, (inner_inds_sort[idx], ))
            end

            sign_conj_num = sign_function(inds_iter, perm_conj)
            sign_tr = sign_function(inds_iter, inds_tr, index_type_tr)
            sign_bc = sign_function(inner_inds, pbc)
            sign_num = sign_perm_num * sign_conj_num * sign_tr * sign_bc

            if haskey(T, inds_iter) 
                β = (count > 0 ? one(S) : zero(S))
                tensortrace!(To[inds], T[inds_iter], (permutation, ()), inds_tr, cj, sign_num, β)
                count += 1
            end
        end
    end
end

################################ Fermionic tensor contraction ###############################

function contract_size(
    tup1::NTuple{N1, T1}, 
    tup2::NTuple{N2, T2}, 
    inds_contr::NTuple{2, NTuple{N, Int}}) where {N1, N2, N, T1, T2}

    tup1_del = deleteat(tup1, inds_contr[1])
    tup2_del = deleteat(tup2, inds_contr[2])
    tup_contr = flatten(tup1_del, tup2_del)
end

function contract_size(
    total_size1::NTuple{N1, Int}, even_size1::NTuple{N1, Int}, index_type1::NTuple{N1, Symbol}, 
    total_size2::NTuple{N2, Int}, even_size2::NTuple{N2, Int}, index_type2::NTuple{N2, Symbol}, 
    inds_contr::NTuple{2, Int}) where {N1, N2}

    return contract_size(
        total_size1, even_size1, index_type1, 
        total_size2, even_size2, index_type2, 
        ((inds_contr[1], ), (inds_contr[2], )))
end

function contract_size(
    total_size1::NTuple{N1, Int}, even_size1::NTuple{N1, Int}, index_type1::NTuple{N1, Symbol}, 
    total_size2::NTuple{N2, Int}, even_size2::NTuple{N2, Int}, index_type2::NTuple{N2, Symbol}, 
    inds_contr::NTuple{2, NTuple{N3, Int}}) where {N1, N2, N3}

    total_size_contr = contract_size(total_size1, total_size2, inds_contr)
    even_size_contr = contract_size(even_size1, even_size2, inds_contr)
    index_type_contr = contract_size(index_type1, index_type2, inds_contr)

    return total_size_contr, even_size_contr, index_type_contr
end

function contract_sectors(
    T1::Grassmann{S1, N1, AT1}, 
    T2::Grassmann{S2, N2, AT2}, 
    contr_inds::NTuple{2, NTuple{N3, Int}}) where {S1, S2, N1, N2, N3, AT1, AT2}

    # Construct all the sectors of the contracted indices
    total_size_contr::NTuple{N3, Int}  = getindices(size(T1), contr_inds[1])
    even_size_contr::NTuple{N3, Int} = getindices(even(T1), contr_inds[1])
    contr_sector_dict_even = _fixed_parity_blocks(total_size_contr, even_size_contr; parity=:even)
    contr_sector_dict_odd = _fixed_parity_blocks(total_size_contr, even_size_contr; parity=:odd)
    contr_sector_dict_all = merge(contr_sector_dict_even, contr_sector_dict_odd)
end

function contract_structure(
    T1::Grassmann{S1, N1, AT1}, 
    T2::Grassmann{S2, N2, AT2}, 
    contr_inds::NTuple{2, NTuple{N3, Int}}; 
    perm::NTuple{N4, Int}=ntuple(i->i, N1+N2-2*N3), 
    cj::NTuple{2, Bool}=(false, false)) where {S1, S2, N1, N2, N3, N4, AT1, AT2}

    total_size1, even_size1, index_type1 = size(T1), even(T1), index_type(T1)
    total_size2, even_size2, index_type2 = size(T2), even(T2), index_type(T2)

    index_type1 = cj[1] ? map(conjugate, index_type1) : index_type1
    index_type2 = cj[2] ? map(conjugate, index_type2) : index_type2

    total_size_out, even_size_out, index_type_out = contract_size(
        total_size1, even_size1, index_type1, 
        total_size2, even_size2, index_type2, 
        contr_inds)

    total_size_out = permute(total_size_out, perm)
    even_size_out = permute(even_size_out, perm)
    index_type_out = permute(index_type_out, perm)

    return total_size_out, even_size_out, index_type_out
end

"""
Fermionic tensor contraction of two Grassmann tensors

Arguments:
    `T1` and `T2` : the rank-N1 Grassmann tensor and the rank-N2 Grassmann tensor to be contracted
    `contr_inds` : the 2 × N2 indices to be contracted corresponding to `T1` and `T2` respectively
    `sign_function` : the Fermionic sign functions (defalut=trivial_sign, means no sign factors)
    `perm` : the permutation of the contracted tensor (default=(1, 2, ..., N4), means no permutation)
    `cj` : whether `T1`/`T2` is conjugated (true) or not (false) (default=(false, false) means no conjugation)
"""

# contract 0 index(a.k.a. direct product)
function contract(
    T1::Grassmann{S1, N1, AT1}, 
    T2::Grassmann{S2, N2, AT2};
    sign_function::F=trivial_sign, 
    perm::NTuple{N3, Int}=ntuple(i->i, N1+N2), 
    cj::NTuple{2, Bool}=(false, false)) where {S1, S2, N1, N2, N3, AT1, AT2, F}

    return contract(
        T1, T2, 
        ((), ()); 
        sign_function=sign_function, 
        perm=perm, 
        cj=cj)
end

# contract a single index
function contract(
    T1::Grassmann{S1, N1, AT1}, 
    T2::Grassmann{S2, N2, AT2}, 
    contr_inds::NTuple{2, Int};
    sign_function::F=trivial_sign, 
    perm::NTuple{N3, Int}=ntuple(i->i, N1+N2-2), 
    cj::NTuple{2, Bool}=(false, false)) where {S1, S2, N1, N2, N3, AT1, AT2, F}

    return contract(
        T1, T2, 
        ((contr_inds[1], ), (contr_inds[2], )); 
        sign_function=sign_function, 
        perm=perm, 
        cj=cj)
end

# contract multiple indices
function contract(
    T1::Grassmann{S1, N1, AT1}, 
    T2::Grassmann{S2, N2, AT2}, 
    contr_inds::NTuple{2, NTuple{N3, Int}};
    sign_function::F=trivial_sign, 
    perm::NTuple{N4, Int}=ntuple(i->i, N1+N2-2*N3), 
    cj::NTuple{2, Bool}=(false, false)) where {S1, S2, N1, N2, N3, N4, AT1, AT2, F}

    S3 = promote_type(S1, S2)
    total_size_out, even_size_out, index_type_out = contract_structure(T1, T2, contr_inds; perm=perm, cj=cj)
    (tensor_parity(T1) == tensor_parity(T2)) ? p_out = :even : p_out = :odd
    To = Grassmann(total_size_out, even_size_out, index_type_out, S3; init=:undef, parity=p_out)
    contract!(To, T1, T2, contr_inds; perm=perm, cj=cj, sign_function=sign_function)
    return To
end

# the mutating implementation 
function contract!(
    To::Grassmann{S, N, ATO}, 
    T1::Grassmann{S1, N1, AT1}, 
    T2::Grassmann{S2, N2, AT2}, 
    contr_inds::NTuple{2, NTuple{N3, Int}};
    sign_function::F=trivial_sign, 
    perm::NTuple{N, Int}=ntuple(i->i, N1+N2-2*N3), 
    cj::NTuple{2, Bool}=(false, false)) where {S, S1, S2, N, N1, N2, N3, ATO, AT1, AT2, F}

    ind_contr1 = contr_inds[1]
    ind_contr2 = contr_inds[2]
    sort_tup1 = TupleTools.sortperm(ind_contr1)
    sort_tup2 = TupleTools.sortperm(ind_contr2)

    N1_open = N1 - N3
    N2_open = N2 - N3
    inds_open1::NTuple{N1_open, Int64} = deleteat(ntuple(i->i, N1), ind_contr1)
    inds_open2::NTuple{N2_open, Int64} = deleteat(ntuple(i->i, N2), ind_contr2)
    inds1 = (inds_open1, ind_contr1)
    inds2 = (inds_open2, ind_contr2)

    perm_conj1 = (cj[1] ? ntuple(i -> N1 - i + 1, N1) : ntuple(i -> i, N1))
    perm_conj2 = (cj[2] ? ntuple(i -> N2 - i + 1, N2) : ntuple(i -> i, N2))

    index_type_contr1 = getindices(index_type(T1), contr_inds[1])
    index_type_contr2 = getindices(index_type(T2), contr_inds[2])
    index_type_contr1a = (cj[1] ? map(conjugate, index_type_contr1) : index_type_contr1) 
    index_type_contr2a = (cj[2] ? map(conjugate, index_type_contr2) : index_type_contr2) 
    index_type_contr = (index_type_contr1a, index_type_contr2a)

    sector_keys = keys(contract_sectors(T1, T2, contr_inds))
    inv_perm = invperm(perm)

    for inds in nonzero_keys(To)

        inds_invp = permute(inds, inv_perm)
        sign_perm_num = sign_function(inds_invp, perm)

        inds_open1_iter = inds_invp[1:N1_open]
        inds_open2_iter = inds_invp[N1_open+1:end]

        count = 0

        for sec in sector_keys

            sector1 = inds_open1_iter
            sector2 = inds_open2_iter

            @inbounds for i in 1:N3
                idx1 = sort_tup1[i]
                idx2 = sort_tup2[i]
                sector1 = insertafter(sector1, ind_contr1[idx1]-1, (sec[idx1], ))
                sector2 = insertafter(sector2, ind_contr2[idx2]-1, (sec[idx2], ))
            end

            sign_contr_num = sign_function(sector1, sector2, contr_inds, index_type_contr)
            sign_conj_num1 = sign_function(sector1, perm_conj1)
            sign_conj_num2 = sign_function(sector2, perm_conj2)
            sign_num = sign_perm_num * sign_contr_num * sign_conj_num1 * sign_conj_num2

            if haskey(T1, sector1) && haskey(T2, sector2)
                β = (count > 0 ? one(S) : zero(S))
                tensorcontract!(To[inds], T1[sector1], (inds1[1], inds1[2]), cj[1], T2[sector2], (inds2[2], inds2[1]), cj[2], (perm, ()), sign_num, β)
                count += 1
            end
        end
        
        if count == 0
            # The sector `inds` is redundant and set to zeros
            fill!(To[inds], zero(S))
        end
    end
end
