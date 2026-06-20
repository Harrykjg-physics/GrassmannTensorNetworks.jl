################################ AD rules for decomp.jl operations ################################
function _tuple_cotangent_get(Δout, i::Int)
    Δ = ChainRulesCore.unthunk(Δout)
    Δ isa AbstractZero && return ZeroTangent()

    if Δ isa Tuple
        return i <= length(Δ) ? Δ[i] : ZeroTangent()
    elseif Δ isa NamedTuple
        key = Symbol(string(i))
        key in keys(Δ) && return Δ[key]
        :tuple in keys(Δ) && return _tuple_cotangent_get(Δ[:tuple], i)
        :args in keys(Δ) && return _tuple_cotangent_get(Δ[:args], i)
        return ZeroTangent()
    end

    try
        return Δ[i]
    catch
    end

    try
        return _tuple_cotangent_get(getfield(Δ, :backing), i)
    catch
    end

    try
        return getproperty(Δ, Symbol(string(i)))
    catch
    end

    return ZeroTangent()
end

function _deep_unthunk(x)
    y = x
    for _ in 1:8
        z = ChainRulesCore.unthunk(y)
        z === y && return z
        y = z
    end
    return y
end

function _zero_block(::Type{S}, dims::Vararg{Int}) where {S}
    return zeros(S, dims...)
end

function _normalize_block_cotangent(block, ::Type{S}, dims::Vararg{Int}) where {S}
    block_unthunked = _deep_unthunk(block)
    (block_unthunked isa AbstractZero || block_unthunked === nothing) && return _zero_block(S, dims...)

    for field in (:backing, :value, :val)
        try
            block_unthunked = _deep_unthunk(getfield(block_unthunked, field))
            break
        catch
        end
    end

    block_unthunked isa AbstractArray || return _zero_block(S, dims...)
    size(block_unthunked) == dims || return _zero_block(S, dims...)
    return Array(block_unthunked)
end

function _block_cotangent_or_zeros(Δ, sec, ::Type{S}, dims::Vararg{Int}) where {S}
    Δ_unthunked = _deep_unthunk(Δ)
    (Δ_unthunked isa AbstractZero || Δ_unthunked === nothing) && return _zero_block(S, dims...)

    try
        if haskey(Δ_unthunked, sec)
            return _normalize_block_cotangent(Δ_unthunked[sec], S, dims...)
        end
        return _zero_block(S, dims...)
    catch
    end

    try
        data_blocks = _deep_unthunk(getproperty(Δ_unthunked, :data))
        if haskey(data_blocks, sec)
            return _normalize_block_cotangent(data_blocks[sec], S, dims...)
        end
        return _zero_block(S, dims...)
    catch
    end

    try
        backing = _deep_unthunk(getfield(Δ_unthunked, :backing))
        data_blocks = _deep_unthunk(backing[:data])
        if haskey(data_blocks, sec)
            return _normalize_block_cotangent(data_blocks[sec], S, dims...)
        end
    catch
    end

    return _zero_block(S, dims...)
end
# ─── Block-level SVD reverse-mode pullback ────────────────────────────────────
#
# For A = U * diagm(S) * V'  (real, economy SVD; Julia returns V, not V').
#
# Given cotangents dU, dS_vec, dV, computes dA using the closed-form SVD
# pullback (Giles 2008 / Townsend 2016).
#
# The "economy" size convention:
#   A : m x n,   U : m x k,   S : length k,   V : n x k
#
# Fᵢⱼ = 0  if i=j or |σⱼ² - σᵢ²| < tol,  else  1 / (σⱼ² − σᵢ²)

function _svd_block_rev(
    U::Matrix{S},
    S_vec::Vector{Float64},
    V::Matrix{S},
    dU::Matrix{S},
    dS_vec::Vector{Float64},
    dV::Matrix{S}) where {S<:Number}

    m, k = size(U)          # U : m × k
    n, kV = size(V)         # V : n x k
    @assert k == kV == length(S_vec)

    S2 = S_vec .^ 2
    eps_sq = eps(Float64)

    # ---- F matrix -----------------------------------------------------------
    F = zeros(S, k, k)
    @inbounds for j in 1:k, i in 1:k
        i == j && continue
        denom = S2[j] - S2[i]
        abs(denom) > 1e3 * eps_sq * (S2[i] + S2[j] + 1) || continue
        F[i, j] = 1 / denom
    end

    # ---- core term ----------------------------------------------------------
    S_mat = diagm(S_vec)
    dS_mat = diagm(dS_vec)

    Ut_dU = U' * dU                     # k × k
    Vt_dV = V' * dV                    # k x k

    term = dS_mat
    term += (F .* (Ut_dU - Ut_dU')) * S_mat
    term += S_mat * (F .* (Vt_dV - Vt_dV'))

    dA_core = U * term * V'

    # ---- orthogonal complement (zero for square full-rank) -------------------
    Im = Matrix{S}(I, m, m)
    In = Matrix{S}(I, n, n)

    # regularised inverse singular values
    S_inv = zeros(S, k)
    @inbounds for i in 1:k
        S_inv[i] = abs(S_vec[i]) > 1e3 * eps_sq ? inv(S_vec[i]) : zero(S)
    end
    S_inv_mat = diagm(S_inv)

    dA_orth = (Im - U * U') * dU * S_inv_mat * V' +
              U * S_inv_mat * dV' * (In - V * V')

    return dA_core + dA_orth
end

# ─── Block-level symmetric EVD reverse-mode pullback ─────────────────────────
#
# For A = U * diagm(Λ) * U'  (real symmetric, U orthogonal).
#
# Given cotangents dΛ_vec, dU, computes dA.
#
# Fᵢⱼ = 0  if i=j or |λⱼ − λᵢ| < tol,  else  1 / (λⱼ − λᵢ)

function _eigen_block_rev(
    Λ_vec::Vector{S},
    U::Matrix{S},
    dΛ_vec::Vector{S},
    dU::Matrix{S}) where {S<:Number}

    n = length(Λ_vec)
    @assert size(U) == (n, n)

    eps_val = eps(Float64)

    # ---- F matrix -----------------------------------------------------------
    F = zeros(S, n, n)
    @inbounds for j in 1:n, i in 1:n
        i == j && continue
        denom = Λ_vec[j] - Λ_vec[i]
        abs(denom) > 1e3 * eps_val * (abs(Λ_vec[i]) + abs(Λ_vec[j]) + 1) || continue
        F[i, j] = 1 / denom
    end

    Ut_dU = U' * dU                     # n × n

    dA = U * (diagm(dΛ_vec) + F .* (Ut_dU - Ut_dU')) * U'

    return dA
end


# ─── Block-level QR reverse-mode pullback ────────────────────────────────────
#
# For A = Q * R  (real, Q orthogonal, R upper triangular).
#
# The tangent space:  dQ = Q * K  with K skew-symmetric,  dR upper triangular.
# We project arbitrary cotangents onto the valid tangent space before applying
# the product rule  dA = dQ_proj * R + Q * dR_proj.

function _qr_block_rev(
    Q::Matrix{S},
    R::Matrix{S},
    dQ::Matrix{S},
    dR::Matrix{S}) where {S<:Number}

    # project dQ onto tangent space of orthogonal group at Q
    K = Q' * dQ
    K_skew = (K - K') / 2
    dQ_proj = Q * K_skew

    # project dR onto upper triangular tangent space
    dR_proj = UpperTriangular(dR)

    return dQ_proj * R + Q * Matrix(dR_proj)
end


# ─── Block-level LQ reverse-mode pullback ────────────────────────────────────
#
# For A = L * Q  (real, Q orthogonal, L lower triangular).
#
# The tangent space:  dQ = K * Q  with K skew-symmetric,  dL lower triangular.

function _lq_block_rev(
    L::Matrix{S},
    Q::Matrix{S},
    dL::Matrix{S},
    dQ::Matrix{S}) where {S<:Number}

    # project dQ onto tangent space of orthogonal group at Q
    K = dQ * Q'
    K_skew = (K - K') / 2
    dQ_proj = K_skew * Q

    # project dL onto lower triangular tangent space
    dL_proj = LowerTriangular(dL)

    return Matrix(dL_proj) * Q + L * dQ_proj
end


# ══════════════════════════════════════════════════════════════════════════════
# rrule for gsvd (GrassmannMatrix version)
# ══════════════════════════════════════════════════════════════════════════════

function ChainRulesCore.rrule(
    ::typeof(gsvd),
    tensor::Grassmann{S, 2, AT},
    Dcut::Int;
    trunc::Bool=true,
    average_trunc::Bool=false) where {S, AT}

    (tot_dim_row, tot_dim_col) = size(tensor)
    (even_dim_row, even_dim_col) = even(tensor)
    index_types = index_type(tensor)

    odd_dim_row = tot_dim_row - even_dim_row
    odd_dim_col = tot_dim_col - even_dim_col

    total_dim_min = minimum([tot_dim_row, tot_dim_col])

    flag1 = (even_dim_row == 0 ? 1 : odd_dim_row == 0 ? 0 : 2)
    flag2 = (even_dim_col == 0 ? 1 : odd_dim_col == 0 ? 0 : 2)
    flag1 == flag2 || throw(ArgumentError(
        "The Grassmann matrix should have the same parity-structure for the row and column index"))

    # ---------- storage for the backward pass ----------
    # Store per-sector SVD results: Dict sector => (U_full, S_full_vec, Vt_full, k_trunc)
    block_U_full = Dict{NTuple{2,Int}, Matrix{S}}()
    block_S_full = Dict{NTuple{2,Int}, Vector{Float64}}()
    block_Vt_full = Dict{NTuple{2,Int}, Matrix{S}}()
    block_k_trunc = Dict{NTuple{2,Int}, Int}()

    # ---------- forward pass (duplicated to capture full SVD results) ----------
    data_U_dict = Dict{NTuple{2, Int}, Matrix{S}}()
    data_S_dict = Dict{NTuple{2, Int}, Matrix{Float64}}()
    data_V_dict = Dict{NTuple{2, Int}, Matrix{S}}()

    if flag1 == 2
        # ---- both (0,0) and (1,1) sectors ----
        for sector in [(0, 0), (1, 1)]
            block = tensor[sector]
            U_full, S_full_vec, Vt_full = svd(block)
            k_full = length(S_full_vec)

            if trunc
                if average_trunc && (Dcut ÷ 2 < length(S_full_vec))
                    k_trunc = Dcut ÷ 2
                else
                    # placeholder, resolved below
                    k_trunc = -1
                end
            else
                k_trunc = k_full
            end

            block_U_full[sector] = U_full
            block_S_full[sector] = S_full_vec
            block_Vt_full[sector] = Vt_full
            block_k_trunc[sector] = k_trunc
        end

        # resolve per-sector truncation for the non-average_trunc case
        if trunc && !average_trunc
            S_cat = vcat(block_S_full[(0,0)], block_S_full[(1,1)])
            _, even_trunc, odd_trunc = truncation(S_cat, length(block_S_full[(0,0)]), Dcut)
            block_k_trunc[(0,0)] = even_trunc
            block_k_trunc[(1,1)] = odd_trunc
        end

        # build output blocks
        for sector in [(0,0), (1,1)]
            kt = block_k_trunc[sector]
            data_U_dict[sector] = block_U_full[sector][:, 1:kt]
            data_S_dict[sector] = Diagonal(block_S_full[sector][1:kt])
            data_V_dict[sector] = block_Vt_full[sector][:, 1:kt]
        end

        even_dim_new = block_k_trunc[(0,0)]
        odd_dim_new = block_k_trunc[(1,1)]

        # truncation error
        if trunc
            S_all = vcat(block_S_full[(0,0)], block_S_full[(1,1)])
            S_trunc = vcat(block_S_full[(0,0)][1:block_k_trunc[(0,0)]],
                            block_S_full[(1,1)][1:block_k_trunc[(1,1)]])
            trunc_err = 1 - sum(S_trunc) / sum(S_all)
        else
            trunc_err = 1.0
        end

    elseif flag1 == 0
        # ---- only (0,0) sector ----
        sector = (0, 0)
        block = tensor[sector]
        U_full, S_full_vec, Vt_full = svd(block)

        total_dim_new = trunc ? min(Dcut, total_dim_min) : total_dim_min
        even_dim_new = total_dim_new
        odd_dim_new = 0

        block_U_full[sector] = U_full
        block_S_full[sector] = S_full_vec
        block_Vt_full[sector] = Vt_full
        block_k_trunc[sector] = even_dim_new

        data_U_dict[sector] = U_full[:, 1:even_dim_new]
        data_S_dict[sector] = Diagonal(S_full_vec[1:even_dim_new])
        data_V_dict[sector] = Vt_full[:, 1:even_dim_new]

        trunc_err = trunc ? 1 - sum(S_full_vec[1:even_dim_new]) / sum(S_full_vec) : 1.0

    else
        # ---- only (1,1) sector ----
        sector = (1, 1)
        block = tensor[sector]
        U_full, S_full_vec, Vt_full = svd(block)

        total_dim_new = trunc ? min(Dcut, total_dim_min) : total_dim_min
        even_dim_new = 0
        odd_dim_new = total_dim_new

        block_U_full[sector] = U_full
        block_S_full[sector] = S_full_vec
        block_Vt_full[sector] = Vt_full
        block_k_trunc[sector] = odd_dim_new

        data_U_dict[sector] = U_full[:, 1:odd_dim_new]
        data_S_dict[sector] = Diagonal(S_full_vec[1:odd_dim_new])
        data_V_dict[sector] = Vt_full[:, 1:odd_dim_new]

        trunc_err = trunc ? 1 - sum(S_full_vec[1:odd_dim_new]) / sum(S_full_vec) : 1.0
    end

    total_dim_new = even_dim_new + odd_dim_new

    # assemble output Grassmann tensors
    total_size_U_out = (tot_dim_row, total_dim_new)
    even_size_U_out = (even_dim_row, even_dim_new)
    index_type_U_out = (index_types[1], :in)

    total_size_S_out = (total_dim_new, total_dim_new)
    even_size_S_out = (even_dim_new, even_dim_new)
    index_type_S_out = (:out, :in)

    total_size_V_out = (tot_dim_col, total_dim_new)
    even_size_V_out = (even_dim_col, even_dim_new)
    index_type_V_out = (conjugate(index_types[2]), :in)

    U_out = Grassmann(total_size_U_out, even_size_U_out, index_type_U_out, data_U_dict)
    S_out = Grassmann(total_size_S_out, even_size_S_out, index_type_S_out, data_S_dict)
    V_out = Grassmann(total_size_V_out, even_size_V_out, index_type_V_out, data_V_dict)

    # ────── pullback ────────────────────────────────────────────────────────
    function gsvd_pullback(Δout)
        ΔU = ChainRulesCore.unthunk(_tuple_cotangent_get(Δout, 1))
        ΔS = ChainRulesCore.unthunk(_tuple_cotangent_get(Δout, 2))
        ΔV = ChainRulesCore.unthunk(_tuple_cotangent_get(Δout, 3))

        if ΔU isa AbstractZero && ΔS isa AbstractZero && ΔV isa AbstractZero
            return (NoTangent(), ZeroTangent(), NoTangent())
        end

        data_dict = Dict{NTuple{2, Int}, AT}()

        for sec in keys(block_U_full)
            U_f = block_U_full[sec]
            S_f = block_S_full[sec]
            Vt_f = block_Vt_full[sec]
            kt = block_k_trunc[sec]
            kf = length(S_f)

            # extract block cotangents (or zeros for missing blocks)
            dU_blk = _block_cotangent_or_zeros(ΔU, sec, S, size(U_f, 1), kt)
            dS_blk = _block_cotangent_or_zeros(ΔS, sec, S, kt, kt)
            dS_blk_vec = diag(dS_blk)
            dV_blk = _block_cotangent_or_zeros(ΔV, sec, S, size(Vt_f, 1), kt)

            # pad with zeros for the truncated dimensions
            dU_pad = hcat(dU_blk, zeros(S, size(U_f, 1), kf - kt))
            dS_pad = vcat(dS_blk_vec, zeros(S, kf - kt))
            dV_pad = hcat(dV_blk, zeros(S, size(Vt_f, 1), kf - kt))

            dA_block = _svd_block_rev(U_f, S_f, Vt_f, dU_pad, dS_pad, dV_pad)
            data_dict[sec] = dA_block
        end

        Δtensor = Grassmann(size(tensor), even(tensor), index_type(tensor), data_dict)
        return (NoTangent(), Δtensor, NoTangent())
    end

    return (U_out, S_out, V_out, trunc_err), gsvd_pullback
end


# ══════════════════════════════════════════════════════════════════════════════
# rrule for gevd (GrassmannMatrix version)
# ══════════════════════════════════════════════════════════════════════════════

function ChainRulesCore.rrule(
    ::typeof(gevd),
    tensor::Grassmann{S, 2, AT},
    Dcut::Int;
    symflag::Bool=false,
    trunc::Bool=true,
    average_trunc::Bool=false) where {S, AT}

    (tot_dim_row, tot_dim_col) = size(tensor)
    (even_dim_row, even_dim_col) = even(tensor)
    index_types = index_type(tensor)
    odd_dim_row = tot_dim_row - even_dim_row

    (tot_dim_row == tot_dim_col && even_dim_row == even_dim_col) || throw(ArgumentError(
        "The Grassmann matrix should satisfy: (tot_dim_row == tot_dim_col) && (even_dim_row == even_dim_col)"))

    # ---------- storage for backward pass ----------
    block_Λ_full = Dict{NTuple{2,Int}, Vector{S}}()
    block_U_full = Dict{NTuple{2,Int}, Matrix{S}}()
    block_k_trunc = Dict{NTuple{2,Int}, Int}()

    # ---------- forward pass ----------
    data_U_dict = Dict{NTuple{2, Int}, Matrix{S}}()
    data_Λ_dict = Dict{NTuple{2, Int}, Matrix{S}}()

    if (even_dim_row != 0) && (odd_dim_row != 0)
        # ---- both sectors ----
        for sector in [(0, 0), (1, 1)]
            block = tensor[sector]
            if symflag
                block = (block + block') / 2
            end
            Λ_vec, U_full = eigen(block)

            # sort by absolute eigenvalue descending
            idx = sortperm(Λ_vec; by=abs, rev=true)
            Λ_full_sorted = Λ_vec[idx]
            U_full_sorted = U_full[:, idx]
            k_full = length(Λ_full_sorted)

            if trunc && average_trunc && (Dcut ÷ 2 < even_dim_row)
                k_trunc = Dcut ÷ 2
            elseif trunc && !average_trunc
                k_trunc = -1   # resolved after seeing both sectors
            else
                k_trunc = k_full
            end

            block_Λ_full[sector] = Λ_full_sorted
            block_U_full[sector] = U_full_sorted
            block_k_trunc[sector] = k_trunc
        end

        # resolve per-sector truncation for non-average_trunc
        if trunc && !average_trunc
            Λ_abs_cat = vcat(abs.(block_Λ_full[(0,0)]), abs.(block_Λ_full[(1,1)]))
            _, even_trunc, odd_trunc = truncation(Λ_abs_cat, length(block_Λ_full[(0,0)]), Dcut)
            block_k_trunc[(0,0)] = even_trunc
            block_k_trunc[(1,1)] = odd_trunc
        end

        for sector in [(0,0), (1,1)]
            kt = block_k_trunc[sector]
            data_U_dict[sector] = block_U_full[sector][:, 1:kt]
            data_Λ_dict[sector] = Diagonal(block_Λ_full[sector][1:kt])
        end

        even_dim_new = block_k_trunc[(0,0)]
        odd_dim_new = block_k_trunc[(1,1)]

        if trunc
            Λ_abs_all = vcat(abs.(block_Λ_full[(0,0)]), abs.(block_Λ_full[(1,1)]))
            Λ_trunc_abs = vcat(
                abs.(block_Λ_full[(0,0)][1:block_k_trunc[(0,0)]]),
                abs.(block_Λ_full[(1,1)][1:block_k_trunc[(1,1)]]))
            trunc_err = 1 - sum(Λ_trunc_abs) / sum(Λ_abs_all)
        else
            trunc_err = 1.0
        end

    elseif odd_dim_row == 0
        # ---- only (0,0) sector ----
        sector = (0, 0)
        block = tensor[sector]
        if symflag
            block = (block + block') / 2
        end
        Λ_vec, U_full = eigen(block)
        idx = sortperm(Λ_vec; by=abs, rev=true)
        Λ_full_sorted = Λ_vec[idx]
        U_full_sorted = U_full[:, idx]

        total_dim_new = trunc ? min(tot_dim_row, Dcut) : tot_dim_row
        even_dim_new = total_dim_new
        odd_dim_new = 0

        block_Λ_full[sector] = Λ_full_sorted
        block_U_full[sector] = U_full_sorted
        block_k_trunc[sector] = even_dim_new

        data_U_dict[sector] = U_full_sorted[:, 1:even_dim_new]
        data_Λ_dict[sector] = Diagonal(Λ_full_sorted[1:even_dim_new])

        trunc_err = trunc ? 1 - sum(abs.(Λ_full_sorted[1:even_dim_new])) / sum(abs.(Λ_full_sorted)) : 1.0

    else
        # ---- only (1,1) sector ----
        sector = (1, 1)
        block = tensor[sector]
        if symflag
            block = (block + block') / 2
        end
        Λ_vec, U_full = eigen(block)
        idx = sortperm(Λ_vec; by=abs, rev=true)
        Λ_full_sorted = Λ_vec[idx]
        U_full_sorted = U_full[:, idx]

        total_dim_new = trunc ? min(tot_dim_row, Dcut) : tot_dim_row
        even_dim_new = 0
        odd_dim_new = total_dim_new

        block_Λ_full[sector] = Λ_full_sorted
        block_U_full[sector] = U_full_sorted
        block_k_trunc[sector] = odd_dim_new

        data_U_dict[sector] = U_full_sorted[:, 1:odd_dim_new]
        data_Λ_dict[sector] = Diagonal(Λ_full_sorted[1:odd_dim_new])

        trunc_err = trunc ? 1 - sum(abs.(Λ_full_sorted[1:odd_dim_new])) / sum(abs.(Λ_full_sorted)) : 1.0
    end

    total_dim_new = even_dim_new + odd_dim_new

    total_size_U_out = (tot_dim_row, total_dim_new)
    even_size_U_out = (even_dim_row, even_dim_new)
    index_type_U_out = (index_types[1], :in)
    total_size_Λ_out = (total_dim_new, total_dim_new)
    even_size_Λ_out = (even_dim_new, even_dim_new)
    index_type_Λ_out = (:out, :in)

    U_out = Grassmann(total_size_U_out, even_size_U_out, index_type_U_out, data_U_dict)
    Λ_out = Grassmann(total_size_Λ_out, even_size_Λ_out, index_type_Λ_out, data_Λ_dict)

    # ────── pullback ────────────────────────────────────────────────────────
    function gevd_pullback(Δout)
        ΔU = ChainRulesCore.unthunk(_tuple_cotangent_get(Δout, 1))
        ΔΛ = ChainRulesCore.unthunk(_tuple_cotangent_get(Δout, 2))

        if ΔU isa AbstractZero && ΔΛ isa AbstractZero
            return (NoTangent(), ZeroTangent(), NoTangent())
        end

        data_dict = Dict{NTuple{2, Int}, AT}()

        for sec in keys(block_U_full)
            Λ_f = block_Λ_full[sec]
            U_f = block_U_full[sec]
            kt = block_k_trunc[sec]
            kf = length(Λ_f)

            dU_blk = _block_cotangent_or_zeros(ΔU, sec, S, size(U_f, 1), kt)
            dΛ_blk = _block_cotangent_or_zeros(ΔΛ, sec, S, kt, kt)
            dΛ_blk_vec = diag(dΛ_blk)

            # pad with zeros for truncated dimensions
            dU_pad = hcat(dU_blk, zeros(S, size(U_f, 1), kf - kt))
            dΛ_pad = vcat(dΛ_blk_vec, zeros(S, kf - kt))

            dA_block = _eigen_block_rev(Λ_f, U_f, dΛ_pad, dU_pad)
            data_dict[sec] = dA_block
        end

        Δtensor = Grassmann(size(tensor), even(tensor), index_type(tensor), data_dict)
        return (NoTangent(), Δtensor, NoTangent())
    end

    return (U_out, Λ_out, trunc_err), gevd_pullback
end


# ══════════════════════════════════════════════════════════════════════════════
# rrule for gortho (GrassmannMatrix version, QR / LQ)
# ══════════════════════════════════════════════════════════════════════════════

function ChainRulesCore.rrule(
    ::typeof(gortho),
    tensor::Grassmann{S, 2, AT};
    alg::F=LinearAlgebra.qr) where {S, AT, F}

    (tot_dim_row, tot_dim_col) = size(tensor)
    (even_dim_row, even_dim_col) = even(tensor)
    index_types = index_type(tensor)

    flag1 = (even_dim_row == 0 ? 1 : (tot_dim_row - even_dim_row) == 0 ? 0 : 2)
    flag2 = (even_dim_col == 0 ? 1 : (tot_dim_col - even_dim_col) == 0 ? 0 : 2)
    flag1 == flag2 || throw(ArgumentError(
        "The Grassmann matrix should have the same parity structure for the row and column indices"))

    idx_min = argmin([tot_dim_row, tot_dim_col])
    tot_dim_min = [tot_dim_row, tot_dim_col][idx_min]
    even_dim_min = [even_dim_row, even_dim_col][idx_min]

    # ---------- storage for backward pass ----------
    block_M1 = Dict{NTuple{2,Int}, Matrix{S}}()
    block_M2 = Dict{NTuple{2,Int}, Matrix{S}}()

    data_M1_dict = Dict{NTuple{2, Int}, Matrix{S}}()
    data_M2_dict = Dict{NTuple{2, Int}, Matrix{S}}()

    if flag1 == 2
        for sector in [(0, 0), (1, 1)]
            block = tensor[sector]
            M1, M2 = alg(block)
            block_M1[sector] = M1
            block_M2[sector] = M2
            data_M1_dict[sector] = M1
            data_M2_dict[sector] = M2
        end
    elseif flag1 == 0
        sector = (0, 0)
        block = tensor[sector]
        M1, M2 = alg(block)
        block_M1[sector] = M1
        block_M2[sector] = M2
        data_M1_dict[sector] = M1
        data_M2_dict[sector] = M2
    else
        sector = (1, 1)
        block = tensor[sector]
        M1, M2 = alg(block)
        block_M1[sector] = M1
        block_M2[sector] = M2
        data_M1_dict[sector] = M1
        data_M2_dict[sector] = M2
    end

    is_qr = (alg === LinearAlgebra.qr)

    total_size_M1_out = (tot_dim_row, tot_dim_min)
    even_size_M1_out = (even_dim_row, even_dim_min)
    index_type_M1_out = (index_types[1], :in)
    total_size_M2_out = (tot_dim_min, tot_dim_col)
    even_size_M2_out = (even_dim_min, even_dim_col)
    index_type_M2_out = (:out, index_types[2])

    M1_out = Grassmann(total_size_M1_out, even_size_M1_out, index_type_M1_out, data_M1_dict)
    M2_out = Grassmann(total_size_M2_out, even_size_M2_out, index_type_M2_out, data_M2_dict)

    # ────── pullback ────────────────────────────────────────────────────────
    function gortho_pullback(Δout)
        ΔM1 = ChainRulesCore.unthunk(_tuple_cotangent_get(Δout, 1))
        ΔM2 = ChainRulesCore.unthunk(_tuple_cotangent_get(Δout, 2))

        if ΔM1 isa AbstractZero && ΔM2 isa AbstractZero
            return (NoTangent(), ZeroTangent())
        end

        data_dict = Dict{NTuple{2, Int}, AT}()

        for sec in keys(block_M1)
            M1 = block_M1[sec]
            M2 = block_M2[sec]

            dM1_blk = _block_cotangent_or_zeros(ΔM1, sec, S, size(M1)...)
            dM2_blk = _block_cotangent_or_zeros(ΔM2, sec, S, size(M2)...)

            dA_block = if is_qr
                _qr_block_rev(M1, M2, dM1_blk, dM2_blk)
            else
                _lq_block_rev(M1, M2, dM1_blk, dM2_blk)
            end
            data_dict[sec] = dA_block
        end

        Δtensor = Grassmann(size(tensor), even(tensor), index_type(tensor), data_dict)
        return (NoTangent(), Δtensor)
    end

    return (M1_out, M2_out), gortho_pullback
end


# ─── Non-differentiable declarations ─────────────────────────────────────────

@non_differentiable check_parity(::Any...)

@non_differentiable truncation(::Grassmann{Float64, 2}, ::Any)
@non_differentiable truncation(::Vector{Float64}, ::Any, ::Any)
