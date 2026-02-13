################################ AD rules for contract.jl operations ################################

"""
Compute Grassmann inner product <a, b> = sum(conj(a) .* b) over shared sectors.
"""

function _grassmann_inner(
    a::Grassmann{T1, N1, AT1}, 
    b::Grassmann{T2, N2, AT2}) where {T1, T2, N1, N2, AT1, AT2}

    N1 == N2 || throw(ArgumentError("Rank mismatch in _grassmann_inner"))
    acc = zero(promote_type(T1, T2))
    @inbounds for (k, av) in nonzero_pairs(a)
        haskey(b, k) || continue
        acc += sum(conj.(av) .* b[k])
    end
    return acc
end

"""
Conjugate data blocks of a Grassmann tensor in-place (without changing index types).
Used to correct the pullback when the forward operation involves conjugation of the input.
"""
function _conj_data!(t::Grassmann)
    @inbounds for (key, block) in nonzero_pairs(t)
        t[key] = conj.(block)
    end
    return t
end

"""
Given a linear map `op`, build the reverse-mode pullback by explicit basis projection.
"""

function _adjoint_linear_map(
    input::Grassmann{T, N, AT}, 
    Δy::Grassmann, 
    op::F) where {T, N, AT, F}

    data_dict = Dict{NTuple{N, Int}, AT}()
    sizehint!(data_dict, length(nonzero_keys(input)))

    @inbounds for (key, block) in nonzero_pairs(input)
        grad_block = similar(block)
        basis_block = similar(block)
        fill!(grad_block, zero(eltype(block)))
        fill!(basis_block, zero(eltype(block)))

        basis_dict = Dict{NTuple{N, Int}, AT}(key => basis_block)
        basis_tensor = Grassmann(size(input), even(input), index_type(input), basis_dict)

        for idx in eachindex(block)
            basis_block[idx] = one(eltype(block))
            y_basis = op(basis_tensor)
            grad_block[idx] = _grassmann_inner(y_basis, Δy)
            basis_block[idx] = zero(eltype(block))
        end

        data_dict[key] = grad_block
    end

    return Grassmann(size(input), even(input), index_type(input), data_dict)
end

function _trace_pullback(
    T::Grassmann{S, N1, AT},
    inds_tr::NTuple{2, NTuple{N2, Int}},
    Δy;
    sign_function::F,
    cj::Bool,
    perm::NTuple{N3, Int},
    pbc::NTuple{N2, Bool}) where {S, N1, N2, N3, AT, F}

    Δy_unthunk = unthunk(Δy)
    if Δy_unthunk isa AbstractZero
        return ZeroTangent()
    end
    result = _adjoint_linear_map(
        T,
        Δy_unthunk,
        t -> trace(t, inds_tr; sign_function=sign_function, cj=cj, perm=perm, pbc=pbc),
    )
    # When cj=true, trace is conjugate-linear: y = L(conj(x)).
    # _adjoint_linear_map computes L^H Δy, but the correct pullback is
    # L^T conj(Δy) = conj(L^H Δy), so we conjugate the data blocks.
    cj && _conj_data!(result)
    return result
end

# ─── trace: single traced index ────────────────────────────────────────────────
function ChainRulesCore.rrule(
    ::typeof(trace),
    T::Grassmann{S, N1, AT},
    inds_tr::NTuple{2, Int};
    sign_function::F=trivial_sign,
    cj::Bool=false,
    perm::NTuple{N2, Int}=ntuple(i -> i, N1 - 2),
    pbc::Bool=true) where {S, N1, N2, AT, F}

    y = trace(T, inds_tr; sign_function=sign_function, cj=cj, perm=perm, pbc=pbc)
    inds_tr_tuple = ((inds_tr[1],), (inds_tr[2],))
    function trace1_pullback(Δy)
        ΔT = _trace_pullback(
            T,
            inds_tr_tuple,
            Δy;
            sign_function=sign_function,
            cj=cj,
            perm=perm,
            pbc=(pbc,),
        )
        return (NoTangent(), ΔT, NoTangent())
    end
    return y, trace1_pullback
end

# ─── trace: multiple traced indices ────────────────────────────────────────────
function ChainRulesCore.rrule(
    ::typeof(trace),
    T::Grassmann{S, N1, AT},
    inds_tr::NTuple{2, NTuple{N2, Int}};
    sign_function::F=trivial_sign,
    cj::Bool=false,
    perm::NTuple{N3, Int}=ntuple(i -> i, N1 - 2 * N2),
    pbc::NTuple{N2, Bool}=ntuple(i -> true, N2)) where {S, N1, N2, N3, AT, F}
    
    y = trace(T, inds_tr; sign_function=sign_function, cj=cj, perm=perm, pbc=pbc)
    function tracen_pullback(Δy)
        ΔT = _trace_pullback(
            T,
            inds_tr,
            Δy;
            sign_function=sign_function,
            cj=cj,
            perm=perm,
            pbc=pbc,
        )
        return (NoTangent(), ΔT, NoTangent())
    end
    return y, tracen_pullback
end

function _contract_pullback(
    T1::Grassmann{S1, N1, AT1},
    T2::Grassmann{S2, N2, AT2},
    contr_inds::NTuple{2, NTuple{N3, Int}},
    Δy;
    sign_function::F,
    perm::NTuple{N4, Int},
    cj::NTuple{2, Bool},) where {S1, S2, N1, N2, N3, N4, AT1, AT2, F}

    Δy_unthunk = unthunk(Δy)
    if Δy_unthunk isa AbstractZero
        return ZeroTangent(), ZeroTangent()
    end
    ΔT1 = _adjoint_linear_map(
        T1,
        Δy_unthunk,
        t -> contract(t, T2, contr_inds; sign_function=sign_function, perm=perm, cj=cj),
    )
    ΔT2 = _adjoint_linear_map(
        T2,
        Δy_unthunk,
        t -> contract(T1, t, contr_inds; sign_function=sign_function, perm=perm, cj=cj),
    )
    # When cj[1]=true, contraction is conjugate-linear in T1: y = M(conj(T1), T2).
    # _adjoint_linear_map computes M^H Δy, but correct pullback is conj(M^H Δy).
    cj[1] && _conj_data!(ΔT1)
    # When cj[2]=true, contraction is conjugate-linear in T2: y = M(T1, conj(T2)).
    cj[2] && _conj_data!(ΔT2)
    return ΔT1, ΔT2
end

# ─── contract: direct product (0 contracted indices) ──────────────────────────
function ChainRulesCore.rrule(
    ::typeof(contract),
    T1::Grassmann{S1, N1, AT1},
    T2::Grassmann{S2, N2, AT2};
    sign_function::F=trivial_sign,
    perm::NTuple{N3, Int}=ntuple(i -> i, N1 + N2),
    cj::NTuple{2, Bool}=(false, false)) where {S1, S2, N1, N2, N3, AT1, AT2, F}

    y = contract(T1, T2; sign_function=sign_function, perm=perm, cj=cj)
    contr_inds = ((), ())
    function contract0_pullback(Δy)
        ΔT1, ΔT2 = _contract_pullback(
            T1,
            T2,
            contr_inds,
            Δy;
            sign_function=sign_function,
            perm=perm,
            cj=cj,
        )
        return (NoTangent(), ΔT1, ΔT2)
    end
    return y, contract0_pullback
end

# ─── contract: single contracted index ─────────────────────────────────────────
function ChainRulesCore.rrule(
    ::typeof(contract),
    T1::Grassmann{S1, N1, AT1},
    T2::Grassmann{S2, N2, AT2},
    contr_inds::NTuple{2, Int};
    sign_function::F=trivial_sign,
    perm::NTuple{N3, Int}=ntuple(i -> i, N1 + N2 - 2),
    cj::NTuple{2, Bool}=(false, false)) where {S1, S2, N1, N2, N3, AT1, AT2, F}

    y = contract(T1, T2, contr_inds; sign_function=sign_function, perm=perm, cj=cj)
    contr_inds_tuple = ((contr_inds[1],), (contr_inds[2],))
    function contract1_pullback(Δy)
        ΔT1, ΔT2 = _contract_pullback(
            T1,
            T2,
            contr_inds_tuple,
            Δy;
            sign_function=sign_function,
            perm=perm,
            cj=cj,
        )
        return (NoTangent(), ΔT1, ΔT2, NoTangent())
    end
    return y, contract1_pullback
end

# ─── contract: multiple contracted indices ─────────────────────────────────────
function ChainRulesCore.rrule(
    ::typeof(contract),
    T1::Grassmann{S1, N1, AT1},
    T2::Grassmann{S2, N2, AT2},
    contr_inds::NTuple{2, NTuple{N3, Int}};
    sign_function::F=trivial_sign,
    perm::NTuple{N4, Int}=ntuple(i -> i, N1 + N2 - 2 * N3),
    cj::NTuple{2, Bool}=(false, false)) where {S1, S2, N1, N2, N3, N4, AT1, AT2, F}

    y = contract(T1, T2, contr_inds; sign_function=sign_function, perm=perm, cj=cj)
    function contractn_pullback(Δy)
        ΔT1, ΔT2 = _contract_pullback(
            T1,
            T2,
            contr_inds,
            Δy;
            sign_function=sign_function,
            perm=perm,
            cj=cj,
        )
        return (NoTangent(), ΔT1, ΔT2, NoTangent())
    end
    return y, contractn_pullback
end

