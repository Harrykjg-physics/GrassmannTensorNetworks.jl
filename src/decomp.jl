
############################### auxiliary functions ###############################

function check_parity(
    tot_dim_row::Int, even_dim_row::Int, 
    tot_dim_col::Int, even_dim_col::Int)

    odd_dim_row = tot_dim_row - even_dim_row
    odd_dim_col = tot_dim_col - even_dim_col

    # flag = 2 : the index has both even and odd-parity states
    # flag = 0 : the index has only even-parity states
    # flag = 1 : the index has only odd-parity states

    even_dim_row == 0 ? flag_row = 1 : odd_dim_row == 0 ? flag_row = 0 : flag_row = 2
    even_dim_col == 0 ? flag_col = 1 : odd_dim_col == 0 ? flag_col = 0 : flag_col = 2

    return flag_row, flag_col
end

############################### GSVD for Grassmann Matrix ###############################

"""
Grassmann Singular Value Decomposition (GSVD) (type stable)

     T_ee    0     -->    U_ee_trunc   0      S_ee_trunc  0      Vdag_ee_trunc   0
      0    T_oo    -->      0    U_oo_trunc    0   S_odd_trunc     0    Vdag_oo_trunc

We only assume the following cases, otherwise the GSVD is not well-defined:

    (1) The Grassmann matrix has both (even, even) and (odd, odd) sectors
    (2) The Grassmann matrix only has a (even, even) sector
    (3) The Grassmann matrix only has a (odd, odd) sector


Arguments: 

    `tensor` : a Grassmann matrix with a block structure, i.e. tensor = tensor_ee ⊕ tensor_oo, 
    where tensor_ee/tensor_oo is a square matrix which represents the (0, 0)/(1, 1) sector.
    `Dcut` : the kept total dimension of the new index, Dcut = Dcut_even + Dcut_odd, 
    where Dcut_even(odd) is determined automatically according to the magnitude of the singular 
    values in the (even, even) ((odd, odd)) sector, as long as average_trunc=false(see below)
    `trunc` : whether to perform tensor truncation (default=true)
    `average_trunc`: if trunc=true, fix Dcut_even = Dcut_odd = Dcut/2 (default=false)
"""

function gsvd(
    tensor::GrassmannMatrix{T, AT}, 
    Dcut::Int; 
    trunc::Bool=true, 
    average_trunc::Bool=false) where {T, AT}

    (tot_dim_row, tot_dim_col), (even_dim_row, even_dim_col), index_types = size(tensor), even(tensor), index_type(tensor)
    # total_min is equal to the number of total singular values from SVD
    total_dim_min = minimum([tot_dim_row, tot_dim_col])

    flag1, flag2 = check_parity(tot_dim_row, even_dim_row, tot_dim_col, even_dim_col)
    (flag1 == flag2) || throw(ArgumentError("The Grassmann matrix should have the same parity-structure 
    for the row and column index"))

    data_U_dict = Dict{NTuple{2, Int}, Matrix{T}}()
    data_S_dict = Dict{NTuple{2, Int}, Matrix{Float64}}()
    data_V_dict = Dict{NTuple{2, Int}, Matrix{T}}()

    if flag1 == 2
        # The Grassmann matrix has both (even, even) sector and (odd, odd) sector
        # Perform ordinary SVD to the (even, even) and the (odd, odd) sector respectively
        block_ee = tensor[(0, 0)]
        U_ee, S_ee, V_ee = svd(block_ee)
        block_oo = tensor[(1, 1)]
        U_oo, S_oo, V_oo = svd(block_oo)

        if trunc 
            if average_trunc && (Dcut ÷ 2 < length(S_ee)) && (Dcut ÷ 2 < length(S_oo)) 
                # Set Dcut_even = Dcut_odd = Dcut/2
                S = cat(S_ee, S_oo; dims=1)
                even_dim_new = Dcut ÷ 2
                odd_dim_new = Dcut ÷ 2
                S_trunc = cat(S_ee[1:even_dim_new], S_oo[1:odd_dim_new]; dims=1)
                trunc_err = 1 - sum(S_trunc) / sum(S)
            else
                # Concatenate and truncate the even-part and odd-part of the new index respectively, 
                # To the desired total dimension Dcut according to the magnitude of the singular values
                S = cat(S_ee, S_oo; dims=1)
                S_trunc, even_dim_new, odd_dim_new = truncation(S, length(S_ee), Dcut)
                trunc_err = 1 - sum(S_trunc) / sum(S)
            end
        else
            even_dim_new = length(S_ee)
            odd_dim_new = length(S_oo)
            trunc_err = 1.0
        end

        U_ee_new = U_ee[:, 1:even_dim_new]
        U_oo_new = U_oo[:, 1:odd_dim_new]
        V_ee_new = V_ee[:, 1:even_dim_new]
        V_oo_new = V_oo[:, 1:odd_dim_new]
        S_ee_new = Diagonal(S_ee[1:even_dim_new])
        S_oo_new = Diagonal(S_oo[1:odd_dim_new])

        total_dim_new = even_dim_new + odd_dim_new

        push!(data_U_dict, (0, 0) => U_ee_new)
        push!(data_U_dict, (1, 1) => U_oo_new)
        push!(data_S_dict, (0, 0) => S_ee_new)
        push!(data_S_dict, (1, 1) => S_oo_new)
        push!(data_V_dict, (0, 0) => V_ee_new)
        push!(data_V_dict, (1, 1) => V_oo_new)

    elseif flag1 == 0
        # The Grassmann matrix only has (even, even) sector
        # Perform ordinary SVD to the (even, even) sector
        block_ee = tensor[(0, 0)]
        U_ee, S_ee, V_ee = svd(block_ee)

        total_dim_new = (trunc ? minimum([Dcut, total_dim_min]) : total_dim_min)
        even_dim_new = total_dim_new
        odd_dim_new = 0
        
        U_ee_new = U_ee[:, 1:total_dim_new]
        V_ee_new=  V_ee[:, 1:total_dim_new]
        S_new = Diagonal(S_ee[1:total_dim_new])

        trunc_err = (trunc ? 1 - sum(S_new)/sum(S_ee) : 1.0)

        push!(data_U_dict, (0, 0) => U_ee_new)
        push!(data_S_dict, (0, 0) => S_new)
        push!(data_V_dict, (0, 0) => V_ee_new)

    else 
        # The Grassmann matrix only has (odd, odd) sector
        # Perform ordinary SVD to the (odd, odd) sector
        block_oo = tensor[(1, 1)]
        U_oo, S_oo, V_oo = svd(block_oo)

        total_dim_new = (trunc ? minimum([Dcut, total_dim_min]) : total_dim_min)
        even_dim_new = 0
        odd_dim_new = total_dim_new
        
        U_oo_new = U_oo[:, 1:total_dim_new]
        V_oo_new = V_oo[:, 1:total_dim_new]
        S_new = Diagonal(S_oo[1:total_dim_new])

        trunc_err = (trunc ? 1 - sum(S_new)/sum(S_oo) : 1.0)

        push!(data_U_dict, (1, 1) => U_oo_new)
        push!(data_S_dict, (1, 1) => S_new)
        push!(data_V_dict, (1, 1) => V_oo_new)
    end

    total_size_U_out = (tot_dim_row, total_dim_new)
    even_size_U_out = (even_dim_row, even_dim_new)
    index_type_U_out = (index_types[1], :in)

    total_size_S_out = (total_dim_new, total_dim_new)
    even_size_S_out = (even_dim_new, even_dim_new)
    index_type_S_out = (:out, :in)

    total_size_V_out = (tot_dim_col, total_dim_new)
    even_size_V_out = (even_dim_col, even_dim_new)
    index_type_V_out = (conjugate(index_types[2]), :in)

    return Grassmann(total_size_U_out, even_size_U_out, index_type_U_out, data_U_dict), Grassmann(total_size_S_out, even_size_S_out, index_type_S_out, data_S_dict), 
    Grassmann(total_size_V_out, even_size_V_out, index_type_V_out, data_V_dict), trunc_err
end

################################## General GSVD ##################################

"""
General GSVD applying to a rank-N Grassmann tensor (type stable):

    T[(i1, i2, ..., iN1), (j1, j2, ..., jN2)] ==> U[(i1, i2, ..., iN1), x], Λ[x, y], V[(j1, j2, ..., jN2), y] 

    Note that the conjugation of the Grassmann tensor Vdag[y, (j1, j2, ..., jN2)] should be V[(jN2,  ..., j2, j1), y]
    instead of V[(j1, j2, ..., jN2), y] 

    Therefore, we have to consider a permutation sign factor from (jN2,  ..., j2, j1, y) to (j1, j2, ..., jN2, y)
    if sign_function=auto_sign is enabled (Remind that the order of the indices should correspond to the order of the 
    Grassmann numbers if sign_function=auto_sign)
"""

function gsvd(
    tensor::Grassmann{T, N, AT}, 
    rowinds::NTuple{N1, Int}, 
    colinds::NTuple{N2, Int}, 
    Dcut::Int; 
    sign_function::F=trivial_sign, 
    trunc::Bool=true, 
    average_trunc::Bool=false) where {T, N, AT, N1, N2, F}

    N1 + N2 == N || throw(DimensionMismatch("The total length of rowinds and colinds should be equal to the total number of indices of the tensor"))

    total_size, even_size, index_types = size(tensor), even(tensor), index_type(tensor)
    total_size_row, even_size_row, index_type_row = getindices(total_size, rowinds), getindices(even_size, rowinds), getindices(index_types, rowinds)
    total_size_col, even_size_col, index_type_col = getindices(total_size, colinds), getindices(even_size, colinds), getindices(index_types, colinds)

    N1 > 1 ? tensor_fused1 = fuse(tensor, rowinds) : tensor_fused1 = tensor
    N2 > 1 ? tensor_fused2 = fuse(tensor_fused1, colinds .- (N1-1)) : tensor_fused2 = tensor_fused1

    U, S_out, V, trunc_err  = gsvd(tensor_fused2, Dcut; trunc=trunc, average_trunc=average_trunc)

    _, dim_U_col = size(U)
    _, edim_U_col = even(U)
    _, dim_V_col = size(V)
    _, edim_V_col = even(V)

    # Prepare the output information for the U tensor
    total_size_U = insertafter(total_size_row, N1, (dim_U_col, ))
    even_size_U = insertafter(even_size_row, N1, (edim_U_col, ))
    index_type_U = insertafter(index_type_row, N1, (:in, ))
    # Prepare the output information for the V tensor
    total_size_V = insertafter(total_size_col, N2, (dim_V_col, ))
    even_size_V = insertafter(even_size_col, N2, (edim_V_col, ))
    index_type_col_conj = conjugate.(index_type_col)
    index_type_V = insertafter(index_type_col_conj, N2, (:in, ))

    N1 > 1 ? U_out = split(U, 1, total_size_U, even_size_U, index_type_U) : U_out = U
    N2 > 1 ? V_out = split(V, 1, total_size_V, even_size_V, index_type_V) : V_out = V

    perm_tup = ntuple(i -> N2 - i + 1, N2)
    perm_tup = insertafter(perm_tup, N2, (N2+1, ))
    N2 > 1 ? V_out = add_perm_sign(V_out, perm_tup; sign_function=sign_function) : nothing

    return U_out, S_out, V_out, trunc_err
end

############################### GEVD for Grassmann Matrix ###############################

"""
Grassmann Eigenvalue Decomposition (GEVD)

      T_ee    0     -->    U_ee_trunc   0      Λ_ee_trunc  0      Udag_ee_trunc   0
      0    T_oo    -->      0    U_oo_trunc    0   Λ_odd_trunc     0    Udag_oo_trunc

We only assume the following cases, otherwise the GEVD is not well-defined:

    (1) The Grassmann matrix has both (even, even) and (odd, odd) sectors
    (2) The Grassmann matrix only has a (even, even) sector
    (3) The Grassmann matrix only has a (odd, odd) sector
    (4) The (even, even) or (odd, odd) sector should be square matrices

Arguments:
    `symflag` : whether to symmetrize the (even, even) or (odd, odd) sector before the EVD (default=false)
    `trunc` : whether to perform truncation according to the absolute values of the eigenvalues (default=true)
"""

function gevd(
    tensor::GrassmannMatrix{T, AT}, 
    Dcut::Int; 
    symflag::Bool=false, 
    trunc::Bool=true, 
    average_trunc::Bool=false) where {T, AT}

    (tot_dim_row, tot_dim_col), (even_dim_row, even_dim_col), index_types = size(tensor), even(tensor), index_type(tensor)
    odd_dim_row = tot_dim_row - even_dim_row
    ((tot_dim_row == tot_dim_col) && (even_dim_row == even_dim_col)) || throw(ArgumentError("The 
    Grassmann matrix should satisfy : (tot_dim_row == tot_dim_col) && (even_dim_row == even_dim_col)"))

    data_U_dict = Dict{NTuple{2, Int}, Matrix{T}}()
    data_Λ_dict = Dict{NTuple{2, Int}, Matrix{T}}()

    if (even_dim_row != 0) && (odd_dim_row != 0)
        # The Grassmann matrix has both (even, even) sector and (odd, odd) sector
        # Perform EVD to the (even, even) and (odd, odd) sector respectively
        block_ee = tensor[(0, 0)]
        block_oo = tensor[(1, 1)]

        if symflag == true
            block_ee = (block_ee + block_ee') / 2
            block_oo = (block_oo + block_oo') / 2
        end

        Λ_ee, U_ee = eigen(block_ee)
        Λ_oo, U_oo = eigen(block_oo)
        # Sort the eigenvectors according to the the absolute eigenvalues
        idx_ee = sortperm(Λ_ee; by=abs, rev=true)
        Λ_ee_sorted = Λ_ee[idx_ee]
        Λ_ee_sorted_abs = abs.(Λ_ee_sorted)
        U_ee_sorted = U_ee[:, idx_ee]

        idx_oo = sortperm(Λ_oo; by=abs, rev=true)
        Λ_oo_sorted = Λ_oo[idx_oo]
        Λ_oo_sorted_abs = abs.(Λ_oo_sorted)
        U_oo_sorted = U_oo[:, idx_oo]

        Λ_abs = cat(Λ_ee_sorted_abs, Λ_oo_sorted_abs; dims=1)

        # Truncate the even-part and odd-part of the new index to the desired dimension
        if trunc 
            if average_trunc && (Dcut ÷ 2 < even_dim_row) && (Dcut ÷ 2 < odd_dim_row)
                even_dim_new = Dcut ÷ 2
                odd_dim_new = Dcut ÷ 2
                Λ_trunc_abs = cat(Λ_ee_sorted_abs[1:even_dim_new], Λ_oo_sorted_abs[1:odd_dim_new]; dims=1)
            else
                # truncation according to the magnitude of the eigenvalues
                Λ_trunc_abs, even_dim_new, odd_dim_new = truncation(Λ_abs, even_dim_row, Dcut)
            end
            trunc_err = 1 - sum(Λ_trunc_abs) / sum(Λ_abs)
        else
            even_dim_new = length(Λ_ee)
            odd_dim_new = length(Λ_oo)
            trunc_err = 1.0
        end

        # The truncation is performed
        U_ee_new = U_ee_sorted[:, 1:even_dim_new]
        U_oo_new = U_oo_sorted[:, 1:odd_dim_new]
        Λ_ee_new = Diagonal(Λ_ee_sorted[1:even_dim_new])
        Λ_oo_new = Diagonal(Λ_oo_sorted[1:odd_dim_new])

        total_dim_new = even_dim_new + odd_dim_new

        push!(data_U_dict, (0, 0) => U_ee_new)
        push!(data_U_dict, (1, 1) => U_oo_new)
        push!(data_Λ_dict, (0, 0) => Λ_ee_new)
        push!(data_Λ_dict, (1, 1) => Λ_oo_new)

    elseif (odd_dim_row == 0)
        # The Grassmann matrix only has (even, even) sector
        # Perform EVD to the (even, even) sector
        block_ee = tensor[(0, 0)]

        if symflag == true
            block_ee = (block_ee + block_ee') / 2
        end

        Λ_ee, U_ee = eigen(block_ee)

        total_dim_new = (trunc ? minimum([tot_dim_row, Dcut]) : tot_dim_row)
        even_dim_new = total_dim_new
        odd_dim_new = 0

        idx_ee = sortperm(Λ_ee; by=abs, rev=true)
        Λ_ee_sorted = Λ_ee[idx_ee]
        Λ_ee_sorted_abs = abs.(Λ_ee_sorted)
        U_ee_sorted = U_ee[:, idx_ee]

        Λ_ee_trunc_abs = Λ_ee_sorted_abs[1:total_dim_new]
        trunc_err = (trunc ? 1 - sum(Λ_ee_trunc_abs)/sum(Λ_ee_sorted_abs) : 1.0)

        U_ee_new = U_ee_sorted[:, 1:total_dim_new]
        Λ_ee_new = Diagonal(Λ_ee_sorted[1:total_dim_new])

        push!(data_U_dict, (0, 0) => U_ee_new)
        push!(data_Λ_dict, (0, 0) => Λ_ee_new)
    else
        # The Grassmann matrix only has (odd, odd) sector
        # Perform EVD to the (odd, odd) sector
        block_oo = tensor[(1, 1)]

        if symflag == true
            block_oo = (block_oo + block_oo') / 2
        end

        Λ_oo, U_oo = eigen(block_oo)

        total_dim_new = (trunc ? minimum([tot_dim_row, Dcut]) : tot_dim_row)
        even_dim_new = 0
        odd_dim_new = total_dim_new

        idx_oo = sortperm(Λ_oo; by=abs, rev=true)
        Λ_oo_sorted = Λ_oo[idx_oo]
        Λ_oo_sorted_abs = abs.(Λ_oo_sorted)
        U_oo_sorted = U_oo[:, idx_oo]

        Λ_oo_trunc_abs = Λ_oo_sorted_abs[1:total_dim_new]
        trunc_err = (trunc ? 1 - sum(Λ_oo_trunc_abs)/sum(Λ_oo_sorted_abs) : 1.0)

        U_oo_trunc = U_oo_sorted[:, 1:total_dim_new]
        Λ_oo_trunc = Diagonal(Λ_oo_sorted[1:total_dim_new])

        push!(data_U_dict, (1, 1) => U_oo_trunc)
        push!(data_Λ_dict, (1, 1) => Λ_oo_trunc)
    end

    total_size_U_out = (tot_dim_row, total_dim_new)
    even_size_U_out = (even_dim_row, even_dim_new)
    index_type_U_out = (index_types[1], :in)
    total_size_Λ_out = (total_dim_new, total_dim_new)
    even_size_Λ_out = (even_dim_new, even_dim_new)
    index_type_Λ_out = (:out, :in)

    return Grassmann(total_size_U_out, even_size_U_out, index_type_U_out, data_U_dict), Grassmann(total_size_Λ_out, even_size_Λ_out, index_type_Λ_out, data_Λ_dict), 
    trunc_err
end

####################################### General GEVD #######################################

"""
General GEVD applying to a rank-N Grassmann tensor (type stable):
"""

function gevd(
    tensor::Grassmann{T, N, AT}, 
    rowinds::NTuple{N1, Int}, 
    colinds::NTuple{N2, Int}, 
    Dcut::Int; 
    symflag::Bool=false, 
    trunc::Bool=true, 
    average_trunc::Bool=false) where {T, N, AT, N1, N2}

    N1 + N2 == N || throw(DimensionMismatch("The total length of rowinds and colinds should be equal to the total number of indices of the tensor "))

    total_size, even_size, index_types = size(tensor), even(tensor), index_type(tensor)
    total_size_row, even_size_row, index_type_row = getindices(total_size, rowinds), getindices(even_size, rowinds), getindices(index_types, rowinds)
    total_size_col, even_size_col, index_type_col = getindices(total_size, colinds), getindices(even_size, colinds), getindices(index_types, colinds)

    N1 > 1 ? tensor_fused1 = fuse(tensor, rowinds) : tensor_fused1 = tensor
    N2 > 1 ? tensor_fused2 = fuse(tensor_fused1, colinds .- (N1-1)) : tensor_fused2 = tensor_fused1

    U, Λ_out, trunc_err = gevd(tensor_fused2, Dcut; symflag=symflag, trunc=trunc, average_trunc=average_trunc)

    _, dim_U_col = size(U)
    _, edim_U_col = even(U)

    # Prepare the output information for the U tensor
    total_size_U = insertafter(total_size_row, N1, (dim_U_col, ))
    even_size_U = insertafter(even_size_row, N1, (edim_U_col, ))
    index_type_U = insertafter(index_type_row, N1, (:in, ))

    N1 > 1 ? U_out = split(U, 1, total_size_U, even_size_U, index_type_U) : U_out = U

    return U_out, Λ_out, trunc_err
end

############################### GQR/GLQ for Grassmann Matrix ###############################

"""
Grassmann Orthogonal decomposition (Gortho) : GQR or GLQ decomposition

    (1) GQR decomposition

        T = QR, where Q is an unitary matrix and R is an upper triangular matrix

        T_ee    0     -->    Q_ee_trunc   0      R_ee_trunc   0
         0    T_oo    -->      0      Q_oo_trunc    0      R_oo_trunc

        Set alg = LinearAlgebra.qr

    (2) GLQ decomposition

        T = LQ, where Q is an unitary matrix and L is an lower triangular matrix

        T_ee    0     -->    L_ee_trunc   0      Q_ee_trunc   0
          0    T_oo    -->      0      L_oo_trunc    0      Q_oo_trunc

        Set alg = LinearAlgebra.lq
"""

function gortho(
    tensor::GrassmannMatrix{T, AT}; 
    alg::F=LinearAlgebra.qr) where {T, AT, F}

    (tot_dim_row, tot_dim_col), (even_dim_row, even_dim_col), index_types = size(tensor), even(tensor), index_type(tensor)

    flag1, flag2 = check_parity(tot_dim_row, even_dim_row, tot_dim_col, even_dim_col)
    (flag1 == flag2) || throw(ArgumentError("The Grassmann matrix should have the same parity structure 
    for the row and column indices"))

    idx_min = argmin([tot_dim_row, tot_dim_col])
    tot_dim_min = size(tensor)[idx_min]
    even_dim_min = even(tensor)[idx_min]

    data_M1_dict = Dict{NTuple{2, Int}, Matrix{T}}()
    data_M2_dict = Dict{NTuple{2, Int}, Matrix{T}}()

    if flag1 == 2
        # The Grassmann matrix has both (even, even) sector and (odd, odd) sector
        # Do the QR/LQ to the (even, even) and the (odd, odd) sector respectively
        block_ee = tensor[(0, 0)]
        M1_ee, M2_ee = alg(block_ee)
        block_oo = tensor[(1, 1)]
        M1_oo, M2_oo = alg(block_oo)
        push!(data_M1_dict, (0, 0) => M1_ee)
        push!(data_M1_dict, (1, 1) => M1_oo)
        push!(data_M2_dict, (0, 0) => M2_ee)
        push!(data_M2_dict, (1, 1) => M2_oo)

    elseif flag1 == 0
        # The Grassmann matrix only has (even, even) sector
        # Perform ordinary QR/LQ to the (even, even) sector
        block_ee = tensor[(0, 0)]
        M1_ee, M2_ee = alg(block_ee)
        push!(data_M1_dict, (0, 0) => M1_ee)
        push!(data_M2_dict, (0, 0) => M2_ee)

    else 
        # The Grassmann matrix only has (odd, odd) sector
        # Perform ordinary QR/LQ to the (odd, odd) sector
        block_oo = tensor[(1, 1)]
        M1_oo, M2_oo = alg(block_oo)
        push!(data_M1_dict, (1, 1) => M1_oo)
        push!(data_M2_dict, (1, 1) => M2_oo)
    end

    total_size_M1_out = (tot_dim_row, tot_dim_min)
    even_size_M1_out = (even_dim_row, even_dim_min)
    index_type_M1_out = (index_types[1], :in)
    total_size_M2_out = (tot_dim_min, tot_dim_col)
    even_size_M2_out = (even_dim_min, even_dim_col)
    index_type_M2_out = (:out, index_types[2])

    return Grassmann(total_size_M1_out, even_size_M1_out, index_type_M1_out, data_M1_dict), 
    Grassmann(total_size_M2_out, even_size_M2_out, index_type_M2_out, data_M2_dict)
end

########################################## General GQR/GLQ ##########################################

function gortho(
    tensor::Grassmann{T, N, AT}, 
    rowinds::NTuple{N1, Int}, 
    colinds::NTuple{N2, Int}; 
    alg::F=LinearAlgebra.qr) where {T, AT, N, N1, N2, F}

    N1 + N2 == N || throw(DimensionMismatch("The total length of rowinds and colinds should be equal to the total number of indices of the tensor "))

    total_size, even_size, index_types = size(tensor), even(tensor), index_type(tensor)
    total_size_row, even_size_row, index_type_row = getindices(total_size, rowinds), getindices(even_size, rowinds), getindices(index_types, rowinds)
    total_size_col, even_size_col, index_type_col = getindices(total_size, colinds), getindices(even_size, colinds), getindices(index_types, colinds)

    N1 > 1 ? tensor_fused1 = fuse(tensor, rowinds) : tensor_fused1 = tensor
    N2 > 1 ? tensor_fused2 = fuse(tensor_fused1, colinds .- (N1-1)) : tensor_fused2 = tensor_fused1

    M1, M2 = gortho(tensor_fused2; alg=alg)

    _, dim_M1_col = size(M1)
    _, edim_M1_col = even(M1)
    dim_M2_row, _ = size(M2)
    edim_M2_row, _ = even(M2)

    # Prepare the output information for the M1 tensor
    total_size_M1 = insertafter(total_size_row, N1, (dim_M1_col, ))
    even_size_M1 = insertafter(even_size_row, N1, (edim_M1_col, ))
    index_type_M1 = insertafter(index_type_row, N1, (:in, ))
    # Prepare the output information for the M2 tensor
    total_size_M2 = insertafter(total_size_col, 0, (dim_M2_row, ))
    even_size_M2 = insertafter(even_size_col, 0, (edim_M2_row, ))
    index_type_M2 = insertafter(index_type_col, 0, (:out, ))

    N1 > 1 ? M1_out = split(M1, 1, total_size_M1, even_size_M1, index_type_M1) : M1_out = M1
    N2 > 1 ? M2_out = split(M2, 2, total_size_M2, even_size_M2, index_type_M2) : M2_out = M2

    return M1_out, M2_out
end

############################### truncation ###############################

function truncation(S::GrassmannMatrix{Float64}, Dcut::Int)

    if haskey(S, (0, 0)) && haskey(S, (1, 1))

        S_00 = diag(S[(0, 0)])
        S_11 = diag(S[(1, 1)])
        S_cat = cat(S_00, S_11; dims=1)

        _, dim_even_trunc, dim_odd_trunc = truncation(S_cat, length(S_00), Dcut)
        dim_total_trunc = dim_even_trunc + dim_odd_trunc
    
        S_00_trunc = S_00[1:dim_even_trunc]
        S_11_trunc = S_11[1:dim_odd_trunc]
        S_dict = Dict{NTuple{2, Int}, Matrix{Float64}}((0, 0) => diagm(S_00_trunc), (1, 1) => diagm(S_11_trunc))

    elseif haskey(S, (0, 0)) && !haskey(S, (1, 1))

        S_00 = diag(S[(0, 0)])
        dim_even_trunc = minimum([Dcut, length(S_00)])
        dim_odd_trunc = 0
        dim_total_trunc = dim_even_trunc
        S_00_trunc = S_00[1:dim_even_trunc]
        S_dict = Dict{NTuple{2, Int}, Matrix{Float64}}((0, 0) => diagm(S_00_trunc))

    elseif !haskey(S, (0, 0)) && haskey(S, (1, 1))

        S_11 = diag(S[(1, 1)])
        dim_odd_trunc = minimum([Dcut, length(S_11)])
        dim_even_trunc = 0
        dim_total_trunc = dim_odd_trunc
        S_11_trunc = S_11[1:dim_odd_trunc]
        S_dict = Dict{NTuple{2, Int}, Matrix{Float64}}((1, 1) => diagm(S_11_trunc))

    else
        throw(ArgumentError("Empty sector is not allowed !"))
    end

    return Grassmann((dim_total_trunc, dim_total_trunc), (dim_even_trunc, dim_even_trunc), (:out, :in), S_dict), dim_even_trunc, dim_odd_trunc
end

"""
A naive (but efficient) implementation of truncating the Z2 index to desired dimension (type stable)

Arguments:
    `S` : the singular value vector to be truncated, where S = S_even ⊕ S_odd
     S_even and S_odd are already sorted in the descending order
    `size_even_k`: the even-parity dimension of index ( or length(S_even) )
    `Dcut` : the desired dimension of the new index
"""

function truncation(
    S::Vector{Float64}, 
    size_even_k::Int, 
    Dcut::Int)

    k = length(S)

    size_odd_k = k - size_even_k

    # Record the number of selected even-parity singular values 
    size_even_trunc = 0
    # Record the number of selected odd-parity singular values 
    size_odd_trunc = 0

    if Dcut <= k
        # The index need to be truncated
        S_trunc = zeros(eltype(S), Dcut)
        if (0 < size_odd_k < k)
            # The index to be truncated contain both even and odd-parity states
            count = 1
            # The biggest singular value in the even-parity sector
            s_even_compare = S[1]
            # The biggest singular value in the odd-parity sector
            s_odd_compare = S[size_even_k+1]
            new_s_even_compare = 0.0
            new_s_odd_compare = 0.0
            while count <= Dcut
                # The desired dimension is not reached
                if (size_even_trunc < size_even_k) && (size_odd_trunc < size_odd_k)
                    # The next singular values should be chosen by comparing the largest one in each parity sector
                    if s_even_compare >= s_odd_compare
                        # The largest one in the even-parity sector is selected
                        size_even_trunc += 1
                        if size_even_trunc < size_even_k
                            # There are still singlular values left in the even-parity sector
                            # The next singular value becomes the largest one in the even-parity sector for the next iteration
                            new_s_even_compare = S[size_even_trunc+1]
                        else
                            # There are no singlular values left in the even-parity sector
                            # Then we always choose odd-parity singular values for the following iterations
                            new_s_even_compare = 0.0
                        end
                        # The largest singular value in the odd-parity sector remain the same for the next iteration
                        new_s_odd_compare = s_odd_compare
                    else
                        # The largest one in the odd-parity sector is selected
                        size_odd_trunc += 1
                        if size_odd_trunc < size_odd_k
                            # There are still singlular values left in the odd-parity sector
                            # The next singular value becomes the largest one in the odd-parity sector for the next iteration
                            new_s_odd_compare = S[size_even_k+size_odd_trunc+1]
                        else
                            # There are no singlular values left in the odd-parity sector
                            # Then we always choose even-parity singular values for the following iterations
                            new_s_odd_compare = 0.0
                        end
                        # The largest singular value in the even-parity sector remain the same for the next iteration
                        new_s_even_compare = s_even_compare
                    end
                elseif size_even_trunc == size_even_k
                    # The next singular values should be chosen only from the odd-parity sector
                    size_odd_trunc += 1
                elseif size_odd_trunc == size_odd_k
                    # The next singular values should be chosen only from the even-parity sector
                    size_even_trunc += 1
                end
                count += 1
                s_even_compare = new_s_even_compare
                s_odd_compare = new_s_odd_compare
            end
            # trucation ! 
            for i = 1:size_even_trunc
                S_trunc[i] = S[i]
            end
            for i = 1:size_odd_trunc
                S_trunc[size_even_trunc+i] = S[size_even_k+i]
            end
        elseif (size_odd_k == 0)
            # The index to be truncated contain only even-parity states
            size_even_trunc = Dcut
            for i = 1:size_even_trunc
                S_trunc[i] = S[i]
            end

        else
            # The index to be truncated contain only odd-parity states
            size_odd_trunc = Dcut
            for i = 1:size_odd_trunc
                S_trunc[i] = S[i]
            end
        end
    else
        # The index doesn't need to be truncated
        size_even_trunc = size_even_k
        size_odd_trunc = size_odd_k
        S_trunc = S
    end

    return S_trunc, size_even_trunc, size_odd_trunc
end
