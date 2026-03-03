
################################## helper functions ##################################

function compute_QN(
    tup::NTuple{N, Int}, 
    even_tup::NTuple{N, Int}) where {N}

    tup_out = ()

    for i in 1:N
        tup[i] > even_tup[i] ? QN = 1 : QN = 0
        tup_out = flatten(tup_out, (QN,))
    end

    return tup_out
end

function sign_trace_conj1(qn::NTuple{6, Int})
    sign_flag = 
    qn[1] * (qn[2] + qn[3] + qn[4] + qn[5] + qn[6]) + qn[2] * (qn[3] + qn[4] + qn[5] + qn[6]) +
    qn[3] * (qn[4] + qn[5] + qn[6]) + qn[4] * (qn[5] + qn[6]) + qn[5] * qn[6]
    (mod(sign_flag, 2) == 0 ? 1 : -1)
end

function sign_trace_perm1(qn::NTuple{6, Int})
    sign_flag = 
    qn[6] * (qn[4] + qn[3] + qn[1]) + qn[3] * qn[1] + qn[4] * qn[1]
    (mod(sign_flag, 2) == 0 ? 1 : -1)
end

function sign_tr1(qn::NTuple{6, Int}; cj::Bool=false)
    sign_flag = (cj ? qn[2] * (qn[4] + qn[3] + qn[2]) : qn[2] * (qn[4] + qn[3]) )
    (mod(sign_flag, 2) == 0 ? 1 : -1)
end

function sign_trace_bc1(qn::NTuple{6, Int})
    sign_flag = qn[2] * qn[5]
    (mod(sign_flag, 2) == 0 ? 1 : -1)
end

sign_trace_conj2 = sign_trace_conj1

function sign_trace_perm2(qn::NTuple{6, Int})
    sign_flag = qn[6] * qn[1]
    (mod(sign_flag, 2) == 0 ? 1 : -1)
end

function sign_tr2(qn::NTuple{6, Int}; cj::Bool=false)
    sign_flag = (cj ? qn[2]  : qn[3])
    (mod(sign_flag, 2) == 0 ? 1 : -1)
end

function sign_trace_bc2(qn::NTuple{6, Int}, pbc::NTuple{2, Bool})

    if !pbc[1]
        sign_flag1 = qn[2] * qn[5]
        sign_num1 = (mod(sign_flag1, 2) == 0 ? 1 : -1)
    else
        sign_num1 = 1
    end

    if !pbc[2]
        sign_flag2 = qn[3] * qn[4]
        sign_num2 = (mod(sign_flag2, 2) == 0 ? 1 : -1)
    else
        sign_num2 = 1
    end

    return sign_num1 * sign_num2
end

sign_trace_conj3 = sign_trace_conj1

function sign_tr3(qn::NTuple{6, Int}; cj::Bool=false)
    sign_flag = (cj ? qn[2] : qn[1] + qn[3])
    (mod(sign_flag, 2) == 0 ? 1 : -1)
end

function sign_trace_bc3(qn::NTuple{6, Int}, pbc::NTuple{3, Bool})

    if !pbc[1]
        sign_flag1 = qn[2] * qn[5]
        sign_num1 = (mod(sign_flag1, 2) == 0 ? 1 : -1)
    else
        sign_num1 = 1
    end

    if !pbc[2]
        sign_flag2 = qn[3] * qn[4]
        sign_num2 = (mod(sign_flag2, 2) == 0 ? 1 : -1)
    else
        sign_num2 = 1
    end

    if !pbc[3]
        sign_flag3 = qn[1] * qn[6]
        sign_num3 = (mod(sign_flag3, 2) == 0 ? 1 : -1)
    else
        sign_num3 = 1
    end

    return sign_num1 * sign_num2 * sign_num3
end

function sign_contr_conj1_0(qn::NTuple{4, Int})
    sign_flag = qn[1] * (qn[2] + qn[3] + qn[4]) + qn[2] * (qn[3] + qn[4]) + qn[3] * qn[4]
    (mod(sign_flag, 2) == 0 ? 1 : -1)
end

function sign_contr_conj2_0(qn::NTuple{4, Int})
    sign_flag = qn[1] * (qn[2] + qn[3] + qn[4]) + qn[2] * (qn[3] + qn[4]) + qn[3] * qn[4]
    (mod(sign_flag, 2) == 0 ? 1 : -1)
end

function sign_contr_perm_0(qn1::NTuple{4, Int}, qn2::NTuple{4, Int})
    sign_flag = qn1[3] * (qn1[1] + qn1[2]) + qn2[3] * (qn2[1] + qn2[2])
    (mod(sign_flag, 2) == 0 ? 1 : -1)
end

sign_contr_conj1 = sign_contr_conj1_0
sign_contr_conj2 = sign_contr_conj2_0

function sign_contract(qn1::NTuple{4, Int}, qn2::NTuple{4, Int}; cj::NTuple{2, Bool}=(false, false))

    if !cj[1] && !cj[2]
        sign_flag = qn1[2] * (qn2[3] + qn2[2] + qn2[1] + qn1[4] + qn1[3])
    elseif cj[1] && cj[2]
        sign_flag = qn1[2] * (qn2[3] + qn2[2] + qn2[1] + qn1[4] + qn1[3] + qn1[2])
    else
        throw(ArgumentError("Not implemented !"))
    end
    (mod(sign_flag, 2) == 0 ? 1 : -1)
end

function sign_contr_perm(qn1::NTuple{4, Int}, qn2::NTuple{4, Int})
    sign_flag = qn2[2] * (qn2[1] + qn1[4] + qn1[3] + qn1[1]) + qn1[4] * qn1[3]
    (mod(sign_flag, 2) == 0 ? 1 : -1)
end

sign_contr_conj1_a = sign_contr_conj1
sign_contr_conj2_a = sign_contr_conj2

function sign_contract_a(qn1::NTuple{4, Int}, qn2::NTuple{4, Int}; cj::NTuple{2, Bool}=(false, false))

    if !cj[1] && !cj[2]
        sign_flag = qn1[2] * (qn2[3] + qn2[1] + qn1[4]) + qn1[3] * (qn2[1] + qn1[4] + qn1[3])
    elseif cj[1] && cj[2]
        sign_flag = qn1[2] * (qn2[3] + qn2[1] + qn1[4] + qn1[2]) + qn1[3] * (qn2[1] + qn1[4])
    else
        throw(ArgumentError("Not implemented !"))
    end
    (mod(sign_flag, 2) == 0 ? 1 : -1)
end

function sign_contr_perm_a(qn1::NTuple{4, Int}, qn2::NTuple{4, Int})
    sign_flag = qn2[1] * (qn1[4] + qn1[1]) + qn2[3] * qn1[4]
    (mod(sign_flag, 2) == 0 ? 1 : -1)
end

sign_contr_conj1_b = sign_contr_conj1
sign_contr_conj2_b = sign_contr_conj2

function sign_contract_b(qn1::NTuple{4, Int}, qn2::NTuple{4, Int}; cj::NTuple{2, Bool}=(false, false))

    if !cj[1] && !cj[2]
        sign_flag = qn1[2] * qn1[1] + qn1[3] * (qn1[1] + qn1[4] + qn1[3]) + qn1[4] * qn1[1] + qn1[1]
    elseif cj[1] && cj[2]
        sign_flag = qn1[2] * (qn1[1] + qn1[2]) + qn1[3] * (qn1[1] + qn1[4]) + qn1[4] * (qn1[1] + qn1[4])
    else
        throw(ArgumentError("Not implemented !"))
    end
    (mod(sign_flag, 2) == 0 ? 1 : -1)
end

####################### test functions #######################

"""
Trace a single index :

@tensor T_tr[f, c, d, a] := T[a, b, c, d, b, f]

        sign_conj = p(a) × (p(b) + p(c) + p(d) + p(e) + p(f))    (a, b, c, d, e, f) <-- (f, e, d, c, b, a)
                  + p(b) × (p(c) + p(d) + p(e) + p(f))
                  + p(c) × (p(d) + p(e) + p(f))
                  + p(d) × (p(e) + p(f))
                  + p(d) × p(f)

        sign_tr = p(b) × (p(d) + p(c))    (a, b, c, d, b, f) (:out, :in, :out, :in, :out, :in)
           or 
        sign_tr = p(b) × (p(d) + p(c) + p(b))    (a, b, c, d, b, f) (:in, :out, :in, :out, :in, :out)

        sign_perm = p(f) × (p(d) + p(c) + p(a))     (f, c, d, a) <-- (a, c, d, f)
                  + p(c) × p(a)
                  + p(d) × p(a)
"""

function manual_tr1(
    T_array::Array{T, 6}, 
    total_size::NTuple{6, Int}, 
    even_size::NTuple{6, Int};
    perm::NTuple{4, Int}=(1, 2, 3, 4), 
    cj::Bool=false,
    sign_conj::Bool=false, 
    sign_tr::Bool=false, 
    sign_perm::Bool=false, 
    sign_bc::Bool=false) where {T}

    T_array = (cj ? conj(T_array) : T_array)

    dim1, dim2, dim3, dim4, dim5, dim6 = total_size
    edim1, edim2, edim3, edim4, edim5, edim6 = even_size

    total_size_out = deleteat(total_size, (2, 5))
    total_size_out = permute(total_size_out, perm)
    T_array_tr = bufferfrom(zeros(T, total_size_out))

    for a in 1:dim1, c in 1:dim3, d in 1:dim4, f in 1:dim6
        for b in 1:dim2

            qn = compute_QN((a, b, c, d, b, f), even_size)

            sign_num_conj = (sign_conj ? sign_trace_conj1(qn) : 1)
            sign_num_perm = (sign_perm ? sign_trace_perm1(qn) : 1)
            sign_num_tr = (sign_tr ? sign_tr1(qn; cj=cj) : 1)
            sign_num_bc = (sign_bc ? sign_trace_bc1(qn) : 1)

            if perm != (1, 2, 3, 4)
                T_array_tr[f, c, d, a] += T_array[a, b, c, d, b, f] * sign_num_conj * sign_num_perm * sign_num_tr * sign_num_bc
            else
                T_array_tr[a, c, d, f] += T_array[a, b, c, d, b, f] * sign_num_conj * sign_num_perm * sign_num_tr * sign_num_bc
            end    
        end
    end
    return copy(T_array_tr)
end

function test_tr1(
    total_size::NTuple{6, Int}, 
    even_size::NTuple{6, Int},
    p_flag::Symbol, 
    elemtype::Type; 
    sign_function::F=trivial_sign, 
    perm::NTuple{4, Int}=(1, 2, 3, 4),
    cj::Bool=false,
    pbc::Bool=true, 
    sign_conj::Bool=false, 
    sign_tr::Bool=false, 
    sign_perm::Bool=false, 
    sign_bc::Bool=false) where {F}

    T = Grassmann(total_size, even_size, (:out, :in, :out, :in, :out, :in), elemtype; 
    init=:random, parity=p_flag)

    T_tr = trace(T, (2, 5); perm=perm, cj=cj, pbc=pbc, sign_function=sign_function)
    T_tr_array = convert(Array, T_tr)
    T_array = convert(Array, T)

    T_array_tr = manual_tr1(T_array, total_size, even_size; perm=perm, cj=cj, sign_conj=sign_conj, sign_tr=sign_tr, 
    sign_perm=sign_perm, sign_bc=sign_bc)

    (T_tr_array ≈ T_array_tr)
end

"""
trace two indices :

@tensor T_tr[a, f] := T[a, b, c, c, b, f]

        sign_conj = p(a) × (p(b) + p(c) + p(d) + p(e) + p(f))    (a, b, c, d, e, f) <-- (f, e, d, c, b, a)
                  + p(b) × (p(c) + p(d) + p(e) + p(f))
                  + p(c) × (p(d) + p(e) + p(f))
                  + p(d) × (p(e) + p(f))
                  + p(d) × p(f)

        sign_tr = p(c)   (a, b, c, c, b, f) (:out, :in, :out, :in, :out, :in)
           or
        sign_tr = p(b)      (a, b, c, c, b, f) (:in, :out, :in, :out, :in, :out)

        sign_perm = p(f) × p(a)    (f, a) <-- (a, f)
"""

function manual_tr2(
    T_array::Array{T, 6}, 
    total_size::NTuple{6, Int}, 
    even_size::NTuple{6, Int};
    perm::NTuple{2, Int}=(1, 2), 
    cj::Bool=false,
    sign_conj::Bool=false, 
    sign_tr::Bool=false, 
    sign_perm::Bool=false, 
    sign_bc::Bool=false,
    pbc::NTuple{2, Bool}=(true, true)) where {T}

    T_array = (cj ? conj(T_array) : T_array)

    dim1, dim2, dim3, dim4, dim5, dim6 = total_size
    edim1, edim2, edim3, edim4, edim5, edim6 = even_size

    total_size_out = deleteat(total_size, (2, 3, 5, 4))
    total_size_out = permute(total_size_out, perm)
    T_array_tr = bufferfrom(zeros(T, total_size_out))

    for a in 1:dim1, f in 1:dim6
        for b in 1:dim2, c in 1:dim3

            qn = compute_QN((a, b, c, c, b, f), even_size)

            sign_num_conj = (sign_conj ? sign_trace_conj2(qn) : 1)
            sign_num_perm = (sign_perm ? sign_trace_perm2(qn) : 1)
            sign_num_tr = (sign_tr ? sign_tr2(qn; cj=cj) : 1)
            sign_num_bc = (sign_bc ? sign_trace_bc2(qn, pbc) : 1)

            if perm != (1, 2)
                T_array_tr[f, a] += T_array[a, b, c, c, b, f] * sign_num_conj * sign_num_perm * sign_num_tr * sign_num_bc
            else
                T_array_tr[a, f] += T_array[a, b, c, c, b, f] * sign_num_conj * sign_num_perm * sign_num_tr * sign_num_bc
            end    
        end
    end

    return copy(T_array_tr)
end

function test_tr2(
    total_size::NTuple{6, Int}, 
    even_size::NTuple{6, Int},
    p_flag::Symbol, 
    elemtype::Type; 
    sign_function::Function=trivial_sign, 
    perm::NTuple{2, Int}=(1, 2),
    cj::Bool=false,
    pbc::NTuple{2, Bool}=(true, true), 
    sign_conj::Bool=false, 
    sign_tr::Bool=false, 
    sign_perm::Bool=false, 
    sign_bc::Bool=false)

    T = Grassmann(total_size, even_size, (:out, :in, :out, :in, :out, :in), elemtype; 
    init=:random, parity=p_flag)

    T_tr = trace(T, ((2, 3), (5, 4)); perm=perm, cj=cj, pbc=pbc, sign_function=sign_function)
    T_tr_array = convert(Array, T_tr)
    T_array = convert(Array, T)

    T_array_tr = manual_tr2(T_array, total_size, even_size; perm=perm, cj=cj, sign_conj=sign_conj, sign_tr=sign_tr, 
    sign_perm=sign_perm, sign_bc=sign_bc, pbc=pbc)

    (T_tr_array ≈ T_array_tr)
end

"""
trace 3 indices : 

@tensor T_tr[] := T[a, b, c, c, b, a]

        sign_conj = p(a) × (p(b) + p(c) + p(d) + p(e) + p(f))    (a, b, c, d, e, f) <-- (f, e, d, c, b, a)
                  + p(b) × (p(c) + p(d) + p(e) + p(f))
                  + p(c) × (p(d) + p(e) + p(f))
                  + p(d) × (p(e) + p(f))
                  + p(d) × p(f)

        sign_tr = p(c) + p(a)  (a, b, c, c, b, a) (:out, :in, :out, :in, :out, :in)
           or
        sign_tr = p(b)      (a, b, c, c, b, a) (:in, :out, :in, :out, :in, :out)
"""

function manual_tr3(
    T_array::Array{T, 6}, 
    total_size::NTuple{6, Int}, 
    even_size::NTuple{6, Int};
    cj::Bool=false,
    pbc::NTuple{3, Bool}=(true, true, true), 
    sign_conj::Bool=false, 
    sign_tr::Bool=false, 
    sign_bc::Bool=false) where {T}

    T_array = (cj ? conj(T_array) : T_array)

    out = zero(T)

    dim1, dim2, dim3, dim4, dim5, dim6 = total_size
    edim1, edim2, edim3, edim4, edim5, edim6 = even_size

    for a in 1:dim1, b in 1:dim2, c in 1:dim3

        qn = compute_QN((a, b, c, c, b, a), even_size)

        sign_num_conj = (sign_conj ? sign_trace_conj3(qn) : 1)
        sign_num_tr = (sign_tr ? sign_tr3(qn; cj=cj) : 1)
        sign_num_bc = (sign_bc ? sign_trace_bc3(qn, pbc) : 1)

        out += T_array[a, b, c, c, b, a] * sign_num_conj * sign_num_tr * sign_num_bc   
    end

    return out
end

function test_tr3(
    total_size::NTuple{6, Int}, 
    even_size::NTuple{6, Int}, 
    p_flag::Symbol, 
    elemtype::Type; 
    sign_function::Function=trivial_sign, 
    cj::Bool=false,
    pbc::NTuple{3, Bool}=(true, true, true), 
    sign_conj::Bool=false, 
    sign_tr::Bool=false, 
    sign_bc::Bool=false)

    T = Grassmann(total_size, even_size, (:out, :in, :out, :in, :out, :in), elemtype; 
    init=:random, parity=p_flag)

    T_tr = trace(T, ((2, 3, 1), (5, 4, 6)); cj=cj, pbc=pbc, sign_function=sign_function)
    T_array = convert(Array, T)

    out = manual_tr3(T_array, total_size, even_size; cj=cj, sign_conj=sign_conj, sign_tr=sign_tr, 
    sign_bc=sign_bc, pbc=pbc)

    (scalar(T_tr) ≈ out)
end

"""
contract 0 index(a.k.a direct tensor product) : 

T_contr[c, a, b, d, g, e, f, h] <-- T_contr[a, b, c, d, e, f, g, h] := T1[a, b, c, d] * T2[e, f, g, h]

        sign_conj1 = p(a) × (p(b) + p(c) + p(d))  (a, b, c, d) <-- (d, c, b, a)
                   + p(b) × (p(c) + p(d))
                   + p(c) × p(d)

        sign_conj2 = p(e) × (p(f) + p(g) + p(h))  (e, f, g, h) <-- (h, g, f, e)
                   + p(f) × (p(g) + p(h))
                   + p(g) × p(h)

        sign_perm = p(c) × (p(a) + p(b))  (c, a, b, d, g, e, f, h) <-- (a, b, c, d, e, f, g, h) 
                  + p(g) × (p(e) + p(f))  qn1[3] * (qn1[1] + qn1[2]) + qn2[3] * (qn2[1] + qn2[2])
"""

function manual_contr0(
    T1_array::Array{T1, 4}, 
    T2_array::Array{T2, 4}, 
    total_size1::NTuple{4, Int}, 
    even_size1::NTuple{4, Int}, 
    total_size2::NTuple{4, Int}, 
    even_size2::NTuple{4, Int};
    perm::NTuple{8, Int}=(1, 2, 3, 4, 5, 6, 7, 8),
    cj::NTuple{2, Bool}=(false, false), 
    sign_conj1::Bool=false, 
    sign_conj2::Bool=false, 
    sign_perm::Bool=false) where {T1, T2}

    T = promote_type(T1, T2)
    T1_array = (cj[1] ? conj(T1_array) : T1_array)
    T2_array = (cj[2] ? conj(T2_array) : T2_array)

    total_size_out1 = flatten(total_size1, total_size2)
    # avoid AD error
    # total_size_out2 = permute(total_size_out1, perm) 
    total_size_out2 = ntuple(i -> total_size_out1[perm[i]], Val(8))
    T_array_contr = bufferfrom(zeros(T, total_size_out2))

    dim_a, dim_b, dim_c, dim_d = total_size1
    dim_e, dim_f, dim_g, dim_h = total_size2

    for a in 1:dim_a, b in 1:dim_b, c in 1:dim_c, d in 1:dim_d 
        for e in 1:dim_e, f in 1:dim_f, g in 1:dim_g, h in 1:dim_h

            qn1 = compute_QN((a, b, c, d), even_size1)
            qn2 = compute_QN((e, f, g, h), even_size2)

            sign_num_conj1 = (sign_conj1 ? sign_contr_conj1_0(qn1) : 1)
            sign_num_conj2 = (sign_conj2 ? sign_contr_conj2_0(qn2) : 1)
            sign_num_perm = (sign_perm ? sign_contr_perm_0(qn1, qn2) : 1)

            if perm != (1, 2, 3, 4, 5, 6, 7, 8)
                T_array_contr[c, a, b, d, g, e, f, h] += T1_array[a, b, c, d] * T2_array[e, f, g, h] * sign_num_conj1 * sign_num_conj2 * 
                sign_num_perm 
            else
                T_array_contr[a, b, c, d, e, f, g, h] += T1_array[a, b, c, d] * T2_array[e, f, g, h] * sign_num_conj1 * sign_num_conj2 * 
                sign_num_perm
            end    
        end
    end

    return copy(T_array_contr)
end

function test_contr0(
    total_size1::NTuple{4, Int}, 
    even_size1::NTuple{4, Int}, 
    p_flag1::Symbol, 
    total_size2::NTuple{4, Int}, 
    even_size2::NTuple{4, Int}, 
    p_flag2::Symbol,
    elemtype::Type; 
    sign_function::F=trivial_sign, 
    perm::NTuple{8, Int}=(1, 2, 3, 4, 5, 6, 7, 8),
    cj::NTuple{2, Bool}=(false, false), 
    sign_conj1::Bool=false, 
    sign_conj2::Bool=false, 
    sign_perm::Bool=false) where {F}

    T1 = Grassmann(total_size1, even_size1, (:out, :in, :out, :in), elemtype; 
    init=:random, parity=p_flag1)
    T2 = Grassmann(total_size2, even_size2, (:in, :in, :out, :out), elemtype; 
    init=:random, parity=p_flag2)

    T_contr = contract(T1, T2; perm=perm, cj=cj, sign_function=sign_function)
    T_contr_array = convert(Array, T_contr)

    T1_array = convert(Array, T1)
    T2_array = convert(Array, T2)

    T_array_contr = manual_contr0(T1_array, T2_array, total_size1, even_size1, total_size2, even_size2; 
    perm=perm, cj=cj, sign_conj1=sign_conj1, sign_conj2=sign_conj2, sign_perm=sign_perm)

    (T_contr_array ≈ T_array_contr)
end

"""
contract 1 index : 

T_contr[f, a, d, c, e, g] <-- T_contr[a, c, d, e, f, g] := T1[a, dum, c, d] * T2[e, f, g, dum]

        sign_conj1 = p(a) × (p(b) + p(c) + p(d))  (a, b, c, d) <-- (d, c, b, a)
                   + p(b) × (p(c) + p(d))
                   + p(c) × p(d)

        sign_conj2 = p(e) × (p(f) + p(g) + p(h))  (e, f, g, h) <-- (h, g, f, e)
                   + p(f) × (p(g) + p(h))
                   + p(g) × p(h)

        sign_contr = p(dum) × (p(g) + p(f) + p(e) + p(d) + p(c))  (a, dum, c, d) and (e, f, g, dum),  
        (:out, :in, :out, :in) and (:in, :in, :out, :out)

        (f, a, d, c, e, g) <-- (a, c, d, e, f, g) :

        sign_perm = p(f) × (p(e) + p(d) + p(c) + p(a))  (f, a, c, d, e, g) <-- (a, c, d, e, f, g) 
                  + p(d) × p(c)       (f, a, d, c, e, g) <-- (f, a, c, d, e, g)
"""

function manual_contr1(
    T1_array::Array{T1, 4}, 
    T2_array::Array{T2, 4}, 
    total_size1::NTuple{4, Int}, 
    even_size1::NTuple{4, Int}, 
    total_size2::NTuple{4, Int}, 
    even_size2::NTuple{4, Int};
    perm::NTuple{6, Int}=(1, 2, 3, 4, 5, 6),
    cj::NTuple{2, Bool}=(false, false), 
    sign_conj1::Bool=false, 
    sign_conj2::Bool=false, 
    sign_contr::Bool=false, 
    sign_perm::Bool=false) where {T1, T2}

    T = promote_type(T1, T2)
    T1_array = (cj[1] ? conj(T1_array) : T1_array)
    T2_array = (cj[2] ? conj(T2_array) : T2_array)

    total_size_out1 = deleteat(total_size1, (2, ))
    total_size_out2 = deleteat(total_size2, (4, ))
    total_size_out3 = flatten(total_size_out1, total_size_out2)
    # total_size_out = permute(total_size_out, perm)
    total_size_out4 = ntuple(i -> total_size_out3[perm[i]], Val(6))
    T_array_contr = bufferfrom(zeros(T, total_size_out4))

    dim_a, dim_b, dim_c, dim_d = total_size1
    dim_e, dim_f, dim_g, dim_h = total_size2

    for a in 1:dim_a, c in 1:dim_c, d in 1:dim_d, e in 1:dim_e, f in 1:dim_f, g in 1:dim_g
        for dum in 1:dim_b

            qn1 = compute_QN((a, dum, c, d), even_size1)
            qn2 = compute_QN((e, f, g, dum), even_size2)

            sign_num_conj1 = (sign_conj1 ? sign_contr_conj1(qn1) : 1)
            sign_num_conj2 = (sign_conj2 ? sign_contr_conj2(qn2) : 1)
            sign_num_perm = (sign_perm ? sign_contr_perm(qn1, qn2) : 1)
            sign_num_contr = (sign_contr ? sign_contract(qn1, qn2; cj=cj) : 1)

            if perm != (1, 2, 3, 4, 5, 6)
                T_array_contr[f, a, d, c, e, g] += T1_array[a, dum, c, d] * T2_array[e, f, g, dum] * sign_num_conj1 * sign_num_conj2 * 
                sign_num_perm * sign_num_contr 
            else
                T_array_contr[a, c, d, e, f, g] += T1_array[a, dum, c, d] * T2_array[e, f, g, dum] * sign_num_conj1 * sign_num_conj2 * 
                sign_num_perm * sign_num_contr
            end    
        end
    end

    return copy(T_array_contr)
end

function test_contr1(
    total_size1::NTuple{4, Int}, 
    even_size1::NTuple{4, Int}, 
    p_flag1::Symbol, 
    total_size2::NTuple{4, Int}, 
    even_size2::NTuple{4, Int}, 
    p_flag2::Symbol,
    elemtype::Type;
    sign_function::F=trivial_sign, 
    perm::NTuple{6, Int}=(1, 2, 3, 4, 5, 6),
    cj::NTuple{2, Bool}=(false, false), 
    sign_conj1::Bool=false, 
    sign_conj2::Bool=false, 
    sign_contr::Bool=false, 
    sign_perm::Bool=false) where {F}

    T1 = Grassmann(total_size1, even_size1, (:out, :in, :out, :in), elemtype; 
    init=:random, parity=p_flag1)
    T2 = Grassmann(total_size2, even_size2, (:in, :in, :out, :out), elemtype; 
    init=:random, parity=p_flag2)

    T_contr = contract(T1, T2, (2, 4); perm=perm, cj=cj, sign_function=sign_function)
    T_contr_array = convert(Array, T_contr)

    T1_array = convert(Array, T1)
    T2_array = convert(Array, T2)

    T_array_contr = manual_contr1(T1_array, T2_array, total_size1, even_size1, total_size2, even_size2; 
    perm=perm, cj=cj, sign_conj1=sign_conj1, sign_conj2=sign_conj2, sign_contr=sign_contr, sign_perm=sign_perm)

    (T_contr_array ≈ T_array_contr)
end

"""
contract 2 indices : 

T_contr[e, a, g, d] <-- T_contr[a, d, e, g] := T1[a, dum1, dum2, d] * T2[e, dum2, g, dum1]

        sign_conj1 = p(a) × (p(b) + p(c) + p(d))  (a, b, c, d) <-- (d, c, b, a)
                   + p(b) × (p(c) + p(d))
                   + p(c) × p(d)

        sign_conj2 = p(e) × (p(f) + p(g) + p(h))  (e, f, g, h) <-- (h, g, f, e)
                   + p(f) × (p(g) + p(h))
                   + p(g) × p(h)

        sign_contr = p(dum1) × (p(g) + p(e) + p(d)) + p(dum2) × (p(e) + p(d) + p(dum2))  (a, dum1, dum2, d) and (e, dum2, g, dum1),  
        (:out, :in, :out, :in) and (:in, :in, :out, :out)  

        (e, a, g, d) <-- (a, d, e, g)  :

        sign_perm = p(e) × (p(d) + p(a))  (e, a, d, g)  <-- (a, d, e, g)  
                  + p(g) × p(d)       (e, a, g, d) <-- (e, a, d, g)
"""

function manual_contr2(
    T1_array::Array{T1, 4}, 
    T2_array::Array{T2, 4}, 
    total_size1::NTuple{4, Int}, 
    even_size1::NTuple{4, Int}, 
    total_size2::NTuple{4, Int}, 
    even_size2::NTuple{4, Int};
    perm::NTuple{4, Int}=(1, 2, 3, 4),
    cj::NTuple{2, Bool}=(false, false), 
    sign_conj1::Bool=false, 
    sign_conj2::Bool=false, 
    sign_contr::Bool=false, 
    sign_perm::Bool=false) where {T1, T2}

    T = promote_type(T1, T2)
    T1_array = (cj[1] ? conj(T1_array) : T1_array)
    T2_array = (cj[2] ? conj(T2_array) : T2_array)

    total_size_out1 = deleteat(total_size1, (2, 3))
    total_size_out2 = deleteat(total_size2, (4, 2))
    total_size_out = flatten(total_size_out1, total_size_out2)
    total_size_out = permute(total_size_out, perm)
    T_array_contr = bufferfrom(zeros(T, total_size_out))

    dim_a, dim_b, dim_c, dim_d = total_size1
    dim_e, dim_f, dim_g, dim_h = total_size2

    for a in 1:dim_a, d in 1:dim_d, e in 1:dim_e, g in 1:dim_g
        for dum1 in 1:dim_b, dum2 in 1:dim_c

            qn1 = compute_QN((a, dum1, dum2, d), even_size1)
            qn2 = compute_QN((e, dum2, g, dum1), even_size2)

            sign_num_conj1 = (sign_conj1 ? sign_contr_conj1_a(qn1) : 1)
            sign_num_conj2 = (sign_conj2 ? sign_contr_conj2_a(qn2) : 1)
            sign_num_perm = (sign_perm ? sign_contr_perm_a(qn1, qn2) : 1)
            sign_num_contr = (sign_contr ? sign_contract_a(qn1, qn2; cj=cj) : 1)

            if perm != (1, 2, 3, 4)
                T_array_contr[e, a, g, d] += T1_array[a, dum1, dum2, d] * T2_array[e, dum2, g, dum1] * sign_num_conj1 * sign_num_conj2 * 
                sign_num_perm * sign_num_contr 
            else
                T_array_contr[a, d, e, g] += T1_array[a, dum1, dum2, d] * T2_array[e, dum2, g, dum1] * sign_num_conj1 * sign_num_conj2 * 
                sign_num_perm * sign_num_contr
            end    
        end
    end

    return copy(T_array_contr)
end

function test_contr2(
    total_size1::NTuple{4, Int}, 
    even_size1::NTuple{4, Int}, 
    p_flag1::Symbol, 
    total_size2::NTuple{4, Int}, 
    even_size2::NTuple{4, Int}, 
    p_flag2::Symbol,
    elemtype::Type; 
    sign_function::F=trivial_sign, 
    perm::NTuple{4, Int}=(1, 2, 3, 4),
    cj::NTuple{2, Bool}=(false, false), 
    sign_conj1::Bool=false, 
    sign_conj2::Bool=false, 
    sign_contr::Bool=false, 
    sign_perm::Bool=false) where {F}

    T1 = Grassmann(total_size1, even_size1, (:out, :in, :out, :in), elemtype; 
    init=:random, parity=p_flag1)
    T2 = Grassmann(total_size2, even_size2, (:in, :in, :out, :out), elemtype; 
    init=:random, parity=p_flag2)

    T_contr = contract(T1, T2, ((2, 3), (4, 2)); perm=perm, cj=cj, sign_function=sign_function)
    T_contr_array = convert(Array, T_contr)

    T1_array = convert(Array, T1)
    T2_array = convert(Array, T2)

    T_array_contr = manual_contr2(T1_array, T2_array, total_size1, even_size1, total_size2, even_size2; 
    perm=perm, cj=cj, sign_conj1=sign_conj1, sign_conj2=sign_conj2, sign_contr=sign_contr, sign_perm=sign_perm)

    (T_contr_array ≈ T_array_contr)
end

"""
contract all indices : 

        T_contr[] := T1[dum4, dum1, dum2, dum3] * T2[dum4, dum2, dum3, dum1] 

        sign_conj1 = p(a) × (p(b) + p(c) + p(d))  (a, b, c, d) <-- (d, c, b, a)
                   + p(b) × (p(c) + p(d))
                   + p(c) × p(d)

        sign_conj2 = p(e) × (p(f) + p(g) + p(h))  (e, f, g, h) <-- (h, g, f, e)
                   + p(f) × (p(g) + p(h))
                   + p(g) × p(h)

        sign_contr = p(dum1) × p(dum4) + p(dum2) × (p(dum4) + p(dum3) + p(dum2))  (dum4, dum1, dum2, dum3) and (dum4, dum2, dum3, dum1), (:out, :in, :out, :in) and (:in, :in, :out, :out)  
                   + p(dum3) × p(dum4)
                   + p(dum4)
"""

function manual_contr3(
    T1_array::Array{T1, 4}, 
    T2_array::Array{T2, 4}, 
    total_size1::NTuple{4, Int}, 
    even_size1::NTuple{4, Int}, 
    total_size2::NTuple{4, Int}, 
    even_size2::NTuple{4, Int};
    cj::NTuple{2, Bool}=(false, false), 
    sign_conj1::Bool=false, 
    sign_conj2::Bool=false, 
    sign_contr::Bool=false) where {T1, T2}

    T = promote_type(T1, T2)
    T1_array = (cj[1] ? conj(T1_array) : T1_array)
    T2_array = (cj[2] ? conj(T2_array) : T2_array)

    out_array = zero(T)

    dim_a, dim_b, dim_c, dim_d = total_size1
    dim_e, dim_f, dim_g, dim_h = total_size2

    for dum4 in 1:dim_a, dum1 in 1:dim_b, dum2 in 1:dim_c, dum3 in 1:dim_d

        qn1 = compute_QN((dum4, dum1, dum2, dum3), even_size1)
        qn2 = compute_QN((dum4, dum2, dum3, dum1), even_size2)

        sign_num_conj1 = (sign_conj1 ? sign_contr_conj1_b(qn1) : 1)
        sign_num_conj2 = (sign_conj2 ? sign_contr_conj2_b(qn2) : 1)
        sign_num_contr = (sign_contr ? sign_contract_b(qn1, qn2; cj=cj) : 1)

        out_array += T1_array[dum4, dum1, dum2, dum3] * T2_array[dum4, dum2, dum3, dum1] * sign_num_conj1 * sign_num_conj2 * sign_num_contr  
    end

    return out_array
end

function test_contr3(
    total_size1::NTuple{4, Int}, 
    even_size1::NTuple{4, Int}, 
    p_flag1::Symbol, 
    total_size2::NTuple{4, Int}, 
    even_size2::NTuple{4, Int}, 
    p_flag2::Symbol,
    elemtype::Type; 
    sign_function::F=trivial_sign, 
    cj::NTuple{2, Bool}=(false, false), 
    sign_conj1::Bool=false, 
    sign_conj2::Bool=false, 
    sign_contr::Bool=false) where {F}

    T1 = Grassmann(total_size1, even_size1, (:out, :in, :out, :in), elemtype; 
    init=:random, parity=p_flag1)
    T2 = Grassmann(total_size2, even_size2, (:in, :in, :out, :out), elemtype; 
    init=:random, parity=p_flag2)

    T_contr = contract(T1, T2, ((2, 3, 4, 1), (4, 2, 3, 1));  cj=cj, sign_function=sign_function)

    T1_array = convert(Array, T1)
    T2_array = convert(Array, T2)

    out_array = manual_contr3(T1_array, T2_array, total_size1, even_size1, total_size2, even_size2; 
    cj=cj, sign_conj1=sign_conj1, sign_conj2=sign_conj2, sign_contr=sign_contr)

    (scalar(T_contr) ≈ out_array)
end

################################## testing ##################################

# ------------------------- trace a single index -------------------------

@timedtestset "test trace operations on one index for even-parity tensor" verbose=true begin
    @timedtestset "test ordinary trace operations" verbose=true begin
        # Float64
        @test test_tr1((4, 5, 6, 3, 5, 4), (2, 2, 3, 2, 2, 2), :even, Float64)
        @test test_tr1((4, 5, 6, 3, 5, 4), (2, 2, 3, 2, 2, 2), :even, Float64; perm=(4, 2, 3, 1))
        @test test_tr1((4, 5, 6, 3, 5, 4), (2, 2, 3, 2, 2, 2), :even, Float64; cj=true)
        @test test_tr1((4, 5, 6, 3, 5, 4), (2, 2, 3, 2, 2, 2), :even, Float64; perm=(4, 2, 3, 1), cj=true)
        # ComplexF64
        @test test_tr1((4, 5, 6, 3, 5, 4), (2, 2, 3, 2, 2, 2), :even, ComplexF64)
        @test test_tr1((4, 5, 6, 3, 5, 4), (2, 2, 3, 2, 2, 2), :even, ComplexF64; perm=(4, 2, 3, 1))
        @test test_tr1((4, 5, 6, 3, 5, 4), (2, 2, 3, 2, 2, 2), :even, ComplexF64; cj=true)
        @test test_tr1((4, 5, 6, 3, 5, 4), (2, 2, 3, 2, 2, 2), :even, ComplexF64; perm=(4, 2, 3, 1), cj=true)
    end
    @timedtestset "test Fermionic trace operations" verbose=true begin
        # at least sign_tr=true should be enabled if sign_function=auto_sign is enabled
        # Float64
        @test test_tr1((4, 5, 6, 3, 5, 4), (2, 2, 3, 2, 2, 2), :even, Float64; sign_tr=true, sign_function=auto_sign)
        @test test_tr1((4, 5, 6, 3, 5, 4), (2, 2, 3, 2, 2, 2), :even, Float64; sign_tr=true, sign_bc=true, pbc=false, sign_function=auto_sign)
        @test test_tr1((4, 5, 6, 3, 5, 4), (2, 2, 3, 2, 2, 2), :even, Float64; sign_tr=true, sign_perm=true, perm=(4, 2, 3, 1), sign_function=auto_sign)
        @test test_tr1((4, 5, 6, 3, 5, 4), (2, 2, 3, 2, 2, 2), :even, Float64; sign_tr=true, sign_conj=true, cj=true, sign_function=auto_sign)
        @test test_tr1((4, 5, 6, 3, 5, 4), (2, 2, 3, 2, 2, 2), :even, Float64; sign_tr=true, sign_conj=true, sign_perm=true,
        cj=true, perm=(4, 2, 3, 1), sign_function=auto_sign)
        @test test_tr1((4, 5, 6, 3, 5, 4), (2, 2, 3, 2, 2, 2), :even, Float64; sign_tr=true, sign_conj=true,
        sign_perm=true,
        sign_bc=true,
        pbc=false, cj=true, perm=(4, 2, 3, 1), sign_function=auto_sign)
        # ComplexF64
        @test test_tr1((4, 5, 6, 3, 5, 4), (2, 2, 3, 2, 2, 2), :even, ComplexF64; sign_tr=true, sign_function=auto_sign)
        @test test_tr1((4, 5, 6, 3, 5, 4), (2, 2, 3, 2, 2, 2), :even, ComplexF64; sign_tr=true, sign_bc=true, pbc=false, sign_function=auto_sign)
        @test test_tr1((4, 5, 6, 3, 5, 4), (2, 2, 3, 2, 2, 2), :even, ComplexF64; sign_tr=true, sign_perm=true, perm=(4, 2, 3, 1), sign_function=auto_sign)
        @test test_tr1((4, 5, 6, 3, 5, 4), (2, 2, 3, 2, 2, 2), :even, ComplexF64; sign_tr=true, sign_conj=true, cj=true, sign_function=auto_sign)
        @test test_tr1((4, 5, 6, 3, 5, 4), (2, 2, 3, 2, 2, 2), :even, ComplexF64; sign_tr=true, sign_conj=true, sign_perm=true,
        cj=true, perm=(4, 2, 3, 1), sign_function=auto_sign)
        @test test_tr1((4, 5, 6, 3, 5, 4), (2, 2, 3, 2, 2, 2), :even, ComplexF64; sign_tr=true, sign_conj=true,
        sign_perm=true,
        sign_bc=true,
        pbc=false, cj=true, perm=(4, 2, 3, 1), sign_function=auto_sign)
    end
end

@timedtestset "test trace operations on one index for odd-parity tensor" verbose=true begin
    @timedtestset "test ordinary trace operations" verbose=true begin
        # Float64
        @test test_tr1((4, 5, 6, 3, 5, 4), (2, 2, 3, 2, 2, 2), :odd, Float64)
        @test test_tr1((4, 5, 6, 3, 5, 4), (2, 2, 3, 2, 2, 2), :odd, Float64; perm=(4, 2, 3, 1))
        @test test_tr1((4, 5, 6, 3, 5, 4), (2, 2, 3, 2, 2, 2), :odd, Float64; cj=true)
        @test test_tr1((4, 5, 6, 3, 5, 4), (2, 2, 3, 2, 2, 2), :odd, Float64; perm=(4, 2, 3, 1), cj=true)
        # ComplexF64
        @test test_tr1((4, 5, 6, 3, 5, 4), (2, 2, 3, 2, 2, 2), :odd, ComplexF64)
        @test test_tr1((4, 5, 6, 3, 5, 4), (2, 2, 3, 2, 2, 2), :odd, ComplexF64; perm=(4, 2, 3, 1))
        @test test_tr1((4, 5, 6, 3, 5, 4), (2, 2, 3, 2, 2, 2), :odd, ComplexF64; cj=true)
        @test test_tr1((4, 5, 6, 3, 5, 4), (2, 2, 3, 2, 2, 2), :odd, ComplexF64; perm=(4, 2, 3, 1), cj=true)
    end
    @timedtestset "test Fermionic trace operations" verbose=true begin
        # at least sign_tr=true should be enabled if sign_function=auto_sign is enabled
        # Float64
        @test test_tr1((4, 5, 6, 3, 5, 4), (2, 2, 3, 2, 2, 2), :odd, Float64; sign_tr=true, sign_function=auto_sign)
        @test test_tr1((4, 5, 6, 3, 5, 4), (2, 2, 3, 2, 2, 2), :odd, Float64; sign_tr=true, sign_bc=true, pbc=false, sign_function=auto_sign)
        @test test_tr1((4, 5, 6, 3, 5, 4), (2, 2, 3, 2, 2, 2), :odd, Float64; sign_tr=true, sign_perm=true, perm=(4, 2, 3, 1), sign_function=auto_sign)
        @test test_tr1((4, 5, 6, 3, 5, 4), (2, 2, 3, 2, 2, 2), :odd, Float64; sign_tr=true, sign_conj=true, cj=true, sign_function=auto_sign)
        @test test_tr1((4, 5, 6, 3, 5, 4), (2, 2, 3, 2, 2, 2), :odd, Float64; sign_tr=true, sign_conj=true, sign_perm=true,
        cj=true, perm=(4, 2, 3, 1), sign_function=auto_sign)
        @test test_tr1((4, 5, 6, 3, 5, 4), (2, 2, 3, 2, 2, 2), :odd, Float64; sign_tr=true,
        sign_conj=true,
        sign_perm=true,
        sign_bc=true,
        pbc=false, cj=true, perm=(4, 2, 3, 1), sign_function=auto_sign)
        # ComplexF64
        @test test_tr1((4, 5, 6, 3, 5, 4), (2, 2, 3, 2, 2, 2), :odd, ComplexF64; sign_tr=true, sign_function=auto_sign)
        @test test_tr1((4, 5, 6, 3, 5, 4), (2, 2, 3, 2, 2, 2), :odd, ComplexF64; sign_tr=true, sign_bc=true, pbc=false, sign_function=auto_sign)
        @test test_tr1((4, 5, 6, 3, 5, 4), (2, 2, 3, 2, 2, 2), :odd, ComplexF64; sign_tr=true, sign_perm=true, perm=(4, 2, 3, 1), sign_function=auto_sign)
        @test test_tr1((4, 5, 6, 3, 5, 4), (2, 2, 3, 2, 2, 2), :odd, ComplexF64; sign_tr=true, sign_conj=true, cj=true, sign_function=auto_sign)
        @test test_tr1((4, 5, 6, 3, 5, 4), (2, 2, 3, 2, 2, 2), :odd, ComplexF64; sign_tr=true, sign_conj=true, sign_perm=true,
        cj=true, perm=(4, 2, 3, 1), sign_function=auto_sign)
        @test test_tr1((4, 5, 6, 3, 5, 4), (2, 2, 3, 2, 2, 2), :odd, ComplexF64; sign_tr=true,
        sign_conj=true,
        sign_perm=true,
        sign_bc=true,
        pbc=false, cj=true, perm=(4, 2, 3, 1), sign_function=auto_sign)
    end
end

# ------------------------- trace two indices -------------------------

@timedtestset "test trace operations on two indices for even-parity tensor" verbose=true begin
    @timedtestset "test ordinary trace operations" verbose=true begin
        # Float64
        @test test_tr2((4, 5, 3, 3, 5, 4), (2, 2, 2, 2, 2, 2), :even, Float64)
        @test test_tr2((4, 5, 3, 3, 5, 4), (2, 2, 2, 2, 2, 2), :even, Float64; perm=(2, 1))
        @test test_tr2((4, 5, 3, 3, 5, 4), (2, 2, 2, 2, 2, 2), :even, Float64; cj=true)
        @test test_tr2((4, 5, 3, 3, 5, 4), (2, 2, 2, 2, 2, 2), :even, Float64; perm=(2, 1), cj=true)
        # ComplexF64
        @test test_tr2((4, 5, 3, 3, 5, 4), (2, 2, 2, 2, 2, 2), :even, ComplexF64)
        @test test_tr2((4, 5, 3, 3, 5, 4), (2, 2, 2, 2, 2, 2), :even, ComplexF64; perm=(2, 1))
        @test test_tr2((4, 5, 3, 3, 5, 4), (2, 2, 2, 2, 2, 2), :even, ComplexF64; cj=true)
        @test test_tr2((4, 5, 3, 3, 5, 4), (2, 2, 2, 2, 2, 2), :even, ComplexF64; perm=(2, 1), cj=true)
    end
    @timedtestset "test Fermionic trace operations" verbose=true begin
        # at least sign_tr=true should be enabled if sign_function=auto_sign is enabled
        # Float64
        @test test_tr2((4, 5, 3, 3, 5, 4), (2, 2, 2, 2, 2, 2), :even, Float64; sign_tr=true, sign_function=auto_sign)
        @test test_tr2((4, 5, 3, 3, 5, 4), (2, 2, 2, 2, 2, 2), :even, Float64; sign_tr=true, sign_bc=true, pbc=(false, false), sign_function=auto_sign)
        @test test_tr2((4, 5, 3, 3, 5, 4), (2, 2, 2, 2, 2, 2), :even, Float64; sign_tr=true, sign_perm=true, perm=(2, 1), sign_function=auto_sign)
        @test test_tr2((4, 5, 3, 3, 5, 4), (2, 2, 2, 2, 2, 2), :even, Float64; sign_tr=true, sign_conj=true, cj=true, sign_function=auto_sign)
        @test test_tr2((4, 5, 3, 3, 5, 4), (2, 2, 2, 2, 2, 2), :even, Float64; sign_tr=true, sign_conj=true, sign_perm=true,
        cj=true, perm=(2, 1), sign_function=auto_sign)
        @test test_tr2((4, 5, 3, 3, 5, 4), (2, 2, 2, 2, 2, 2), :even, Float64; sign_tr=true,
        sign_conj=true,
        sign_perm=true,
        sign_bc=true,
        pbc=(false, false), cj=true, perm=(2, 1), sign_function=auto_sign)
        # ComplexF64
        @test test_tr2((4, 5, 3, 3, 5, 4), (2, 2, 2, 2, 2, 2), :even, ComplexF64; sign_tr=true, sign_function=auto_sign)
        @test test_tr2((4, 5, 3, 3, 5, 4), (2, 2, 2, 2, 2, 2), :even, ComplexF64; sign_tr=true, sign_bc=true, pbc=(false, false), sign_function=auto_sign)
        @test test_tr2((4, 5, 3, 3, 5, 4), (2, 2, 2, 2, 2, 2), :even, ComplexF64; sign_tr=true, sign_perm=true, perm=(2, 1), sign_function=auto_sign)
        @test test_tr2((4, 5, 3, 3, 5, 4), (2, 2, 2, 2, 2, 2), :even, ComplexF64; sign_tr=true, sign_conj=true, cj=true, sign_function=auto_sign)
        @test test_tr2((4, 5, 3, 3, 5, 4), (2, 2, 2, 2, 2, 2), :even, ComplexF64; sign_tr=true, sign_conj=true, sign_perm=true,
        cj=true, perm=(2, 1), sign_function=auto_sign)
        @test test_tr2((4, 5, 3, 3, 5, 4), (2, 2, 2, 2, 2, 2), :even, ComplexF64; sign_tr=true,
        sign_conj=true,
        sign_perm=true,
        sign_bc=true,
        pbc=(false, false), cj=true, perm=(2, 1), sign_function=auto_sign)
    end
end

@timedtestset "test trace operations on two indices for odd-parity tensor" verbose=true begin
    @timedtestset "test ordinary trace operations" verbose=true begin
        # Float64
        @test test_tr2((4, 5, 3, 3, 5, 4), (2, 2, 2, 2, 2, 2), :odd, Float64)
        @test test_tr2((4, 5, 3, 3, 5, 4), (2, 2, 2, 2, 2, 2), :odd, Float64; perm=(2, 1))
        @test test_tr2((4, 5, 3, 3, 5, 4), (2, 2, 2, 2, 2, 2), :odd, Float64; cj=true)
        @test test_tr2((4, 5, 3, 3, 5, 4), (2, 2, 2, 2, 2, 2), :odd, Float64; perm=(2, 1), cj=true)
        # ComplexF64
        @test test_tr2((4, 5, 3, 3, 5, 4), (2, 2, 2, 2, 2, 2), :odd, ComplexF64)
        @test test_tr2((4, 5, 3, 3, 5, 4), (2, 2, 2, 2, 2, 2), :odd, ComplexF64; perm=(2, 1))
        @test test_tr2((4, 5, 3, 3, 5, 4), (2, 2, 2, 2, 2, 2), :odd, ComplexF64; cj=true)
        @test test_tr2((4, 5, 3, 3, 5, 4), (2, 2, 2, 2, 2, 2), :odd, ComplexF64; perm=(2, 1), cj=true)
    end
    @timedtestset "test Fermionic trace operations" verbose=true begin
        # at least sign_tr=true should be enabled if sign_function=auto_sign is enabled
        # Float64
        @test test_tr2((4, 5, 3, 3, 5, 4), (2, 2, 2, 2, 2, 2), :odd, Float64; sign_tr=true, sign_function=auto_sign)
        @test test_tr2((4, 5, 3, 3, 5, 4), (2, 2, 2, 2, 2, 2), :odd, Float64; sign_tr=true, sign_bc=true, pbc=(false, false), sign_function=auto_sign)
        @test test_tr2((4, 5, 3, 3, 5, 4), (2, 2, 2, 2, 2, 2), :odd, Float64; sign_tr=true, sign_perm=true, perm=(2, 1), sign_function=auto_sign)
        @test test_tr2((4, 5, 3, 3, 5, 4), (2, 2, 2, 2, 2, 2), :odd, Float64; sign_tr=true, sign_conj=true, cj=true, sign_function=auto_sign)
        @test test_tr2((4, 5, 3, 3, 5, 4), (2, 2, 2, 2, 2, 2), :odd, Float64; sign_tr=true, sign_conj=true, sign_perm=true,
        cj=true, perm=(2, 1), sign_function=auto_sign)
        @test test_tr2((4, 5, 3, 3, 5, 4), (2, 2, 2, 2, 2, 2), :odd, Float64; sign_tr=true, sign_conj=true, sign_perm=true, sign_bc=true, 
        pbc=(false, false), cj=true, perm=(2, 1), sign_function=auto_sign)
        # ComplexF64
        @test test_tr2((4, 5, 3, 3, 5, 4), (2, 2, 2, 2, 2, 2), :odd, ComplexF64; sign_tr=true, sign_function=auto_sign)
        @test test_tr2((4, 5, 3, 3, 5, 4), (2, 2, 2, 2, 2, 2), :odd, ComplexF64; sign_tr=true, sign_bc=true, pbc=(false, false), sign_function=auto_sign)
        @test test_tr2((4, 5, 3, 3, 5, 4), (2, 2, 2, 2, 2, 2), :odd, ComplexF64; sign_tr=true, sign_perm=true, perm=(2, 1), sign_function=auto_sign)
        @test test_tr2((4, 5, 3, 3, 5, 4), (2, 2, 2, 2, 2, 2), :odd, ComplexF64; sign_tr=true, sign_conj=true, cj=true, sign_function=auto_sign)
        @test test_tr2((4, 5, 3, 3, 5, 4), (2, 2, 2, 2, 2, 2), :odd, ComplexF64; sign_tr=true, sign_conj=true, sign_perm=true,
        cj=true, perm=(2, 1), sign_function=auto_sign)
        @test test_tr2((4, 5, 3, 3, 5, 4), (2, 2, 2, 2, 2, 2), :odd, ComplexF64; sign_tr=true, sign_conj=true, sign_perm=true, sign_bc=true, 
        pbc=(false, false), cj=true, perm=(2, 1), sign_function=auto_sign)
    end
end

# ------------------------- trace all the indices -------------------------

@timedtestset "test trace operations on all indices for even-parity tensor" verbose=true begin
    @timedtestset "test ordinary trace operations" verbose=true begin
        # Float64
        @test test_tr3((4, 5, 3, 3, 5, 4), (2, 2, 2, 2, 2, 2), :even, Float64)
        @test test_tr3((4, 5, 3, 3, 5, 4), (2, 2, 2, 2, 2, 2), :even, Float64; cj=true)
        # ComplexF64
        @test test_tr3((4, 5, 3, 3, 5, 4), (2, 2, 2, 2, 2, 2), :even, ComplexF64)
        @test test_tr3((4, 5, 3, 3, 5, 4), (2, 2, 2, 2, 2, 2), :even, ComplexF64; cj=true)
    end
    @timedtestset "test Fermionic trace operations" verbose=true begin
        # at least sign_tr=true should be enabled if sign_function=auto_sign is enabled
        # Float64
        @test test_tr3((4, 5, 3, 3, 5, 4), (2, 2, 2, 2, 2, 2), :even, Float64; sign_tr=true, sign_function=auto_sign)
        @test test_tr3((4, 5, 3, 3, 5, 4), (2, 2, 2, 2, 2, 2), :even, Float64; sign_tr=true, sign_bc=true, pbc=(true, false, false), sign_function=auto_sign)
        @test test_tr3((4, 5, 3, 3, 5, 4), (2, 2, 2, 2, 2, 2), :even, Float64; sign_tr=true, sign_bc=true, pbc=(false, true, false), sign_function=auto_sign)
        @test test_tr3((4, 5, 3, 3, 5, 4), (2, 2, 2, 2, 2, 2), :even, Float64; sign_tr=true, sign_bc=true, pbc=(false, false, true), sign_function=auto_sign)
        @test test_tr3((4, 5, 3, 3, 5, 4), (2, 2, 2, 2, 2, 2), :even, Float64; sign_tr=true, sign_bc=true, pbc=(false, false, false), sign_function=auto_sign)
        @test test_tr3((4, 5, 3, 3, 5, 4), (2, 2, 2, 2, 2, 2), :even, Float64; sign_tr=true, sign_conj=true, cj=true, sign_function=auto_sign)
        @test test_tr3((4, 5, 3, 3, 5, 4), (2, 2, 2, 2, 2, 2), :even, Float64; sign_tr=true, sign_conj=true, sign_bc=true, 
        pbc=(false, false, false), cj=true, sign_function=auto_sign)
        # ComplexF64
        @test test_tr3((4, 5, 3, 3, 5, 4), (2, 2, 2, 2, 2, 2), :even, ComplexF64; sign_tr=true, sign_function=auto_sign)
        @test test_tr3((4, 5, 3, 3, 5, 4), (2, 2, 2, 2, 2, 2), :even, ComplexF64; sign_tr=true, sign_bc=true, pbc=(true, false, false), sign_function=auto_sign)
        @test test_tr3((4, 5, 3, 3, 5, 4), (2, 2, 2, 2, 2, 2), :even, ComplexF64; sign_tr=true, sign_bc=true, pbc=(false, true, false), sign_function=auto_sign)
        @test test_tr3((4, 5, 3, 3, 5, 4), (2, 2, 2, 2, 2, 2), :even, ComplexF64; sign_tr=true, sign_bc=true, pbc=(false, false, true), sign_function=auto_sign)
        @test test_tr3((4, 5, 3, 3, 5, 4), (2, 2, 2, 2, 2, 2), :even, ComplexF64; sign_tr=true, sign_bc=true, pbc=(false, false, false), sign_function=auto_sign)
        @test test_tr3((4, 5, 3, 3, 5, 4), (2, 2, 2, 2, 2, 2), :even, ComplexF64; sign_tr=true, sign_conj=true, cj=true, sign_function=auto_sign)
        @test test_tr3((4, 5, 3, 3, 5, 4), (2, 2, 2, 2, 2, 2), :even, ComplexF64; sign_tr=true, sign_conj=true, sign_bc=true, 
        pbc=(false, false, false), cj=true, sign_function=auto_sign)
    end
end

# ------------------------- contract 0 index(a.k.a direct product) -------------------------

@timedtestset "test contract operations on 0 index (even-even)" verbose=true begin
    @timedtestset "test ordinary contract operations" verbose=true begin
        # Float64
        @test test_contr0((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, Float64)
        @test test_contr0((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, Float64; cj=(true, false))
        @test test_contr0((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, Float64; cj=(false, true))
        @test test_contr0((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, Float64; cj=(true, true))
        @test test_contr0((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, Float64; perm=(3, 1, 2, 4, 7, 5, 6, 8))
        @test test_contr0((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, Float64; cj=(true, true), perm=(3, 1, 2, 4, 7, 5, 6, 8))
        # ComplexF64
        @test test_contr0((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, ComplexF64)
        @test test_contr0((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, ComplexF64; cj=(true, false))
        @test test_contr0((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, ComplexF64; cj=(false, true))
        @test test_contr0((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, ComplexF64; cj=(true, true))
        @test test_contr0((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, ComplexF64; perm=(3, 1, 2, 4, 7, 5, 6, 8))
        @test test_contr0((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, ComplexF64; cj=(true, true), perm=(3, 1, 2, 4, 7, 5, 6, 8))
    end
    @timedtestset "test Fermionic contract operations" verbose=true begin
        # Float64
        @test test_contr0((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, Float64; sign_function=auto_sign)
        @test test_contr0((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, Float64; sign_conj1=true, sign_conj2=true, 
        cj=(true, true), sign_function=auto_sign)
        @test test_contr0((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, Float64; sign_perm=true, 
        perm=(3, 1, 2, 4, 7, 5, 6, 8), sign_function=auto_sign)
        @test test_contr0((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, Float64; sign_perm=true, sign_conj1=true, sign_conj2=true, 
        cj=(true, true), perm=(3, 1, 2, 4, 7, 5, 6, 8), sign_function=auto_sign)
        # ComplexF64
        @test test_contr0((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, ComplexF64; sign_function=auto_sign)
        @test test_contr0((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, ComplexF64; sign_conj1=true, sign_conj2=true, 
        cj=(true, true), sign_function=auto_sign)
        @test test_contr0((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, ComplexF64; sign_perm=true, 
        perm=(3, 1, 2, 4, 7, 5, 6, 8), sign_function=auto_sign)
        @test test_contr0((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, ComplexF64; sign_perm=true, sign_conj1=true, sign_conj2=true, 
        cj=(true, true), perm=(3, 1, 2, 4, 7, 5, 6, 8), sign_function=auto_sign)
    end
end

@timedtestset "test contract operations on 0 index (even-odd)" verbose=true begin
    @timedtestset "test ordinary contract operations" verbose=true begin
        # Float64
        @test test_contr0((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64)
        @test test_contr0((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; cj=(true, false))
        @test test_contr0((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; cj=(false, true))
        @test test_contr0((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; cj=(true, true))
        @test test_contr0((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; perm=(3, 1, 2, 4, 7, 5, 6, 8))
        @test test_contr0((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; cj=(true, true), perm=(3, 1, 2, 4, 7, 5, 6, 8))
        # ComplexF64
        @test test_contr0((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64)
        @test test_contr0((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; cj=(true, false))
        @test test_contr0((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; cj=(false, true))
        @test test_contr0((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; cj=(true, true))
        @test test_contr0((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; perm=(3, 1, 2, 4, 7, 5, 6, 8))
        @test test_contr0((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; cj=(true, true), perm=(3, 1, 2, 4, 7, 5, 6, 8))
    end
    @timedtestset "test Fermionic contract operations" verbose=true begin
        # Float64
        @test test_contr0((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; sign_function=auto_sign)
        @test test_contr0((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; sign_conj1=true, sign_conj2=true, 
        cj=(true, true), sign_function=auto_sign)
        @test test_contr0((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; sign_perm=true, 
        perm=(3, 1, 2, 4, 7, 5, 6, 8), sign_function=auto_sign)
        @test test_contr0((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; sign_perm=true, sign_conj1=true, sign_conj2=true, 
        cj=(true, true), perm=(3, 1, 2, 4, 7, 5, 6, 8), sign_function=auto_sign)
        # ComplexF64
        @test test_contr0((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; sign_function=auto_sign)
        @test test_contr0((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; sign_conj1=true, sign_conj2=true, 
        cj=(true, true), sign_function=auto_sign)
        @test test_contr0((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; sign_perm=true, 
        perm=(3, 1, 2, 4, 7, 5, 6, 8), sign_function=auto_sign)
        @test test_contr0((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; sign_perm=true, sign_conj1=true, sign_conj2=true, 
        cj=(true, true), perm=(3, 1, 2, 4, 7, 5, 6, 8), sign_function=auto_sign)
    end
end

@timedtestset "test contract operations on 0 index (odd-odd)" verbose=true begin
    @timedtestset "test ordinary contract operations" verbose=true begin
        # Float64
        @test test_contr0((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64)
        @test test_contr0((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; cj=(true, false))
        @test test_contr0((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; cj=(false, true))
        @test test_contr0((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; cj=(true, true))
        @test test_contr0((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; perm=(3, 1, 2, 4, 7, 5, 6, 8))
        @test test_contr0((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; cj=(true, true), perm=(3, 1, 2, 4, 7, 5, 6, 8))
        # ComplexF64
        @test test_contr0((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64)
        @test test_contr0((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; cj=(true, false))
        @test test_contr0((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; cj=(false, true))
        @test test_contr0((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; cj=(true, true))
        @test test_contr0((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; perm=(3, 1, 2, 4, 7, 5, 6, 8))
        @test test_contr0((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; cj=(true, true), perm=(3, 1, 2, 4, 7, 5, 6, 8))
    end
    @timedtestset "test Fermionic contract operations" verbose=true begin
        # Float64
        @test test_contr0((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; sign_function=auto_sign)
        @test test_contr0((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; sign_conj1=true, sign_conj2=true, 
        cj=(true, true), sign_function=auto_sign)
        @test test_contr0((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; sign_perm=true, 
        perm=(3, 1, 2, 4, 7, 5, 6, 8), sign_function=auto_sign)
        @test test_contr0((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; sign_perm=true, sign_conj1=true, sign_conj2=true, 
        cj=(true, true), perm=(3, 1, 2, 4, 7, 5, 6, 8), sign_function=auto_sign)
        # ComplexF64
        @test test_contr0((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; sign_function=auto_sign)
        @test test_contr0((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; sign_conj1=true, sign_conj2=true, 
        cj=(true, true), sign_function=auto_sign)
        @test test_contr0((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; sign_perm=true, 
        perm=(3, 1, 2, 4, 7, 5, 6, 8), sign_function=auto_sign)
        @test test_contr0((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; sign_perm=true, sign_conj1=true, sign_conj2=true, 
        cj=(true, true), perm=(3, 1, 2, 4, 7, 5, 6, 8), sign_function=auto_sign)
    end
end

# ------------------------- contract a single index -------------------------

@timedtestset "test contract operations on a single index (even-even)" verbose=true begin
    @timedtestset "test ordinary contract operations" verbose=true begin
        # Float64
        @test test_contr1((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, Float64)
        @test test_contr1((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, Float64; cj=(true, false))
        @test test_contr1((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, Float64; cj=(false, true))
        @test test_contr1((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, Float64; cj=(true, true))
        @test test_contr1((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, Float64; perm = (5, 1, 3, 2, 4, 6))
        @test test_contr1((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, Float64; cj=(true, true), perm = (5, 1, 3, 2, 4, 6))
        # ComplexF64
        @test test_contr1((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, ComplexF64)
        @test test_contr1((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, ComplexF64; cj=(true, false))
        @test test_contr1((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, ComplexF64; cj=(false, true))
        @test test_contr1((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, ComplexF64; cj=(true, true))
        @test test_contr1((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, ComplexF64; perm = (5, 1, 3, 2, 4, 6))
        @test test_contr1((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, ComplexF64; cj=(true, true), perm = (5, 1, 3, 2, 4, 6))
    end
    @timedtestset "test Fermionic contract operations" verbose=true begin
        # at least sign_tr=true should be enabled if sign_function=auto_sign is enabled
        # Float64
        @test test_contr1((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, Float64; sign_contr=true, sign_function=auto_sign)
        @test test_contr1((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, Float64; sign_contr=true, sign_conj1=true, sign_conj2=true, 
        cj=(true, true), sign_function=auto_sign)
        @test test_contr1((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, Float64; sign_contr=true, sign_perm=true, 
        perm = (5, 1, 3, 2, 4, 6), sign_function=auto_sign)
        @test test_contr1((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, Float64; sign_contr=true, sign_perm=true, sign_conj1=true, sign_conj2=true, 
        cj=(true, true), perm = (5, 1, 3, 2, 4, 6), sign_function=auto_sign)
        # ComplexF64
        @test test_contr1((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, ComplexF64; sign_contr=true, sign_function=auto_sign)
        @test test_contr1((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, ComplexF64; sign_contr=true, sign_conj1=true, sign_conj2=true, 
        cj=(true, true), sign_function=auto_sign)
        @test test_contr1((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, ComplexF64; sign_contr=true, sign_perm=true, 
        perm = (5, 1, 3, 2, 4, 6), sign_function=auto_sign)
        @test test_contr1((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, ComplexF64; sign_contr=true, sign_perm=true, sign_conj1=true, sign_conj2=true, 
        cj=(true, true), perm = (5, 1, 3, 2, 4, 6), sign_function=auto_sign)
    end
end

@timedtestset "test contract operations on a single index (even-odd)" verbose=true begin
    @timedtestset "test ordinary contract operations" verbose=true begin
        # Float64
        @test test_contr1((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64)
        @test test_contr1((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; cj=(true, false))
        @test test_contr1((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; cj=(false, true))
        @test test_contr1((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; cj=(true, true))
        @test test_contr1((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; perm = (5, 1, 3, 2, 4, 6))
        @test test_contr1((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; cj=(true, true), perm = (5, 1, 3, 2, 4, 6))
        # ComplexF64
        @test test_contr1((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64)
        @test test_contr1((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; cj=(true, false))
        @test test_contr1((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; cj=(false, true))
        @test test_contr1((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; cj=(true, true))
        @test test_contr1((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; perm = (5, 1, 3, 2, 4, 6))
        @test test_contr1((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; cj=(true, true), perm = (5, 1, 3, 2, 4, 6))
    end
    @timedtestset "test Fermionic contract operations" verbose=true begin
        # at least sign_tr=true should be enabled if sign_function=auto_sign is enabled
        # Float64
        @test test_contr1((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; sign_contr=true, sign_function=auto_sign)
        @test test_contr1((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; sign_contr=true, sign_conj1=true, sign_conj2=true, 
        cj=(true, true), sign_function=auto_sign)
        @test test_contr1((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; sign_contr=true, sign_perm=true, 
        perm = (5, 1, 3, 2, 4, 6), sign_function=auto_sign)
        @test test_contr1((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; sign_contr=true, sign_perm=true, sign_conj1=true, sign_conj2=true, 
        cj=(true, true), perm = (5, 1, 3, 2, 4, 6), sign_function=auto_sign)
        # ComplexF64
        @test test_contr1((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; sign_contr=true, sign_function=auto_sign)
        @test test_contr1((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; sign_contr=true, sign_conj1=true, sign_conj2=true, 
        cj=(true, true), sign_function=auto_sign)
        @test test_contr1((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; sign_contr=true, sign_perm=true, 
        perm = (5, 1, 3, 2, 4, 6), sign_function=auto_sign)
        @test test_contr1((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; sign_contr=true, sign_perm=true, sign_conj1=true, sign_conj2=true, 
        cj=(true, true), perm = (5, 1, 3, 2, 4, 6), sign_function=auto_sign)
    end
end

@timedtestset "test contract operations on a single index (odd-odd)" verbose=true begin
    @timedtestset "test ordinary contract operations" verbose=true begin
        # Float64
        @test test_contr1((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64)
        @test test_contr1((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; cj=(true, false))
        @test test_contr1((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; cj=(false, true))
        @test test_contr1((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; cj=(true, true))
        @test test_contr1((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; perm = (5, 1, 3, 2, 4, 6))
        @test test_contr1((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; cj=(true, true), perm = (5, 1, 3, 2, 4, 6))
        # ComplexF64
        @test test_contr1((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64)
        @test test_contr1((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; cj=(true, false))
        @test test_contr1((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; cj=(false, true))
        @test test_contr1((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; cj=(true, true))
        @test test_contr1((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; perm = (5, 1, 3, 2, 4, 6))
        @test test_contr1((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; cj=(true, true), perm = (5, 1, 3, 2, 4, 6))
    end
    @timedtestset "test Fermionic contract operations" verbose=true begin
        # at least sign_tr=true should be enabled if sign_function=auto_sign is enabled
        # Float64
        @test test_contr1((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; sign_contr=true, sign_function=auto_sign)
        @test test_contr1((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; sign_contr=true, sign_conj1=true, sign_conj2=true, 
        cj=(true, true), sign_function=auto_sign)
        @test test_contr1((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; sign_contr=true, sign_perm=true, 
        perm = (5, 1, 3, 2, 4, 6), sign_function=auto_sign)
        @test test_contr1((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; sign_contr=true, sign_perm=true, sign_conj1=true, sign_conj2=true, 
        cj=(true, true), perm = (5, 1, 3, 2, 4, 6), sign_function=auto_sign)
        # ComplexF64
        @test test_contr1((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; sign_contr=true, sign_function=auto_sign)
        @test test_contr1((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; sign_contr=true, sign_conj1=true, sign_conj2=true, 
        cj=(true, true), sign_function=auto_sign)
        @test test_contr1((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; sign_contr=true, sign_perm=true, 
        perm = (5, 1, 3, 2, 4, 6), sign_function=auto_sign)
        @test test_contr1((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; sign_contr=true, sign_perm=true, sign_conj1=true, sign_conj2=true, 
        cj=(true, true), perm = (5, 1, 3, 2, 4, 6), sign_function=auto_sign)
    end
end

# ------------------------- contract two indices -------------------------

@timedtestset "test contract operations on two indices (even-even)" verbose=true begin
    @timedtestset "test ordinary contract operations" verbose=true begin
        # Float64
        @test test_contr2((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, Float64)
        @test test_contr2((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, Float64; cj=(true, false))
        @test test_contr2((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, Float64; cj=(false, true))
        @test test_contr2((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, Float64; cj=(true, true))
        @test test_contr2((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, Float64; perm=(3, 1, 4, 2))
        @test test_contr2((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, Float64; cj=(true, true), perm=(3, 1, 4, 2))
        # ComplexF64
        @test test_contr2((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, ComplexF64)
        @test test_contr2((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, ComplexF64; cj=(true, false))
        @test test_contr2((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, ComplexF64; cj=(false, true))
        @test test_contr2((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, ComplexF64; cj=(true, true))
        @test test_contr2((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, ComplexF64; perm=(3, 1, 4, 2))
        @test test_contr2((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, ComplexF64; cj=(true, true), perm=(3, 1, 4, 2))
    end
    @timedtestset "test Fermionic contract operations" verbose=true begin
        # at least sign_tr=true should be enabled if sign_function=auto_sign is enabled
        # Float64
        @test test_contr2((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, Float64; sign_contr=true, sign_function=auto_sign)
        @test test_contr2((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, Float64; sign_contr=true, sign_conj1=true, sign_conj2=true, 
        cj=(true, true), sign_function=auto_sign)
        @test test_contr2((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, Float64; sign_contr=true, sign_perm=true, 
        perm=(3, 1, 4, 2), sign_function=auto_sign)
        @test test_contr2((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, Float64; sign_contr=true, sign_perm=true, sign_conj1=true, sign_conj2=true, 
        cj=(true, true), perm=(3, 1, 4, 2), sign_function=auto_sign)
        # ComplexF64
        @test test_contr2((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, ComplexF64; sign_contr=true, sign_function=auto_sign)
        @test test_contr2((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, ComplexF64; sign_contr=true, sign_conj1=true, sign_conj2=true, 
        cj=(true, true), sign_function=auto_sign)
        @test test_contr2((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, ComplexF64; sign_contr=true, sign_perm=true, 
        perm=(3, 1, 4, 2), sign_function=auto_sign)
        @test test_contr2((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, ComplexF64; sign_contr=true, sign_perm=true, sign_conj1=true, sign_conj2=true, 
        cj=(true, true), perm=(3, 1, 4, 2), sign_function=auto_sign)
    end
end

@timedtestset "test contract operations on two indices (even-odd)" verbose=true begin
    @timedtestset "test ordinary contract operations" verbose=true begin
        # Float64
        @test test_contr2((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64)
        @test test_contr2((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; cj=(true, false))
        @test test_contr2((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; cj=(false, true))
        @test test_contr2((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; cj=(true, true))
        @test test_contr2((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; perm=(3, 1, 4, 2))
        @test test_contr2((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; cj=(true, true), perm=(3, 1, 4, 2))
        # ComplexF64
        @test test_contr2((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64)
        @test test_contr2((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; cj=(true, false))
        @test test_contr2((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; cj=(false, true))
        @test test_contr2((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; cj=(true, true))
        @test test_contr2((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; perm=(3, 1, 4, 2))
        @test test_contr2((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; cj=(true, true), perm=(3, 1, 4, 2))
    end
    @timedtestset "test Fermionic contract operations" verbose=true begin
        # at least sign_tr=true should be enabled if sign_function=auto_sign is enabled
        # Float64
        @test test_contr2((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; sign_contr=true, sign_function=auto_sign)
        @test test_contr2((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; sign_contr=true, sign_conj1=true, sign_conj2=true, 
        cj=(true, true), sign_function=auto_sign)
        @test test_contr2((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; sign_contr=true, sign_perm=true, 
        perm=(3, 1, 4, 2), sign_function=auto_sign)
        @test test_contr2((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; sign_contr=true, sign_perm=true, sign_conj1=true, sign_conj2=true, 
        cj=(true, true), perm=(3, 1, 4, 2), sign_function=auto_sign)
        # ComplexF64
        @test test_contr2((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; sign_contr=true, sign_function=auto_sign)
        @test test_contr2((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; sign_contr=true, sign_conj1=true, sign_conj2=true, 
        cj=(true, true), sign_function=auto_sign)
        @test test_contr2((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; sign_contr=true, sign_perm=true, 
        perm=(3, 1, 4, 2), sign_function=auto_sign)
        @test test_contr2((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; sign_contr=true, sign_perm=true, sign_conj1=true, sign_conj2=true, 
        cj=(true, true), perm=(3, 1, 4, 2), sign_function=auto_sign)
    end
end

@timedtestset "test contract operations on two indices (odd-odd)" verbose=true begin
    @timedtestset "test ordinary contract operations" verbose=true begin
        # Float64
        @test test_contr2((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64)
        @test test_contr2((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; cj=(true, false))
        @test test_contr2((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; cj=(false, true))
        @test test_contr2((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; cj=(true, true))
        @test test_contr2((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; perm=(3, 1, 4, 2))
        @test test_contr2((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; cj=(true, true), perm=(3, 1, 4, 2))
        # ComplexF64
        @test test_contr2((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64)
        @test test_contr2((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; cj=(true, false))
        @test test_contr2((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; cj=(false, true))
        @test test_contr2((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; cj=(true, true))
        @test test_contr2((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; perm=(3, 1, 4, 2))
        @test test_contr2((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; cj=(true, true), perm=(3, 1, 4, 2))
    end
    @timedtestset "test Fermionic contract operations" verbose=true begin
        # at least sign_tr=true should be enabled if sign_function=auto_sign is enabled
        # Float64
        @test test_contr2((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; sign_contr=true, sign_function=auto_sign)
        @test test_contr2((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; sign_contr=true, sign_conj1=true, sign_conj2=true, 
        cj=(true, true), sign_function=auto_sign)
        @test test_contr2((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; sign_contr=true, sign_perm=true, 
        perm=(3, 1, 4, 2), sign_function=auto_sign)
        @test test_contr2((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; sign_contr=true, sign_perm=true, sign_conj1=true, sign_conj2=true, 
        cj=(true, true), perm=(3, 1, 4, 2), sign_function=auto_sign)
        # ComplexF64
        @test test_contr2((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; sign_contr=true, sign_function=auto_sign)
        @test test_contr2((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; sign_contr=true, sign_conj1=true, sign_conj2=true, 
        cj=(true, true), sign_function=auto_sign)
        @test test_contr2((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; sign_contr=true, sign_perm=true, 
        perm=(3, 1, 4, 2), sign_function=auto_sign)
        @test test_contr2((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; sign_contr=true, sign_perm=true, sign_conj1=true, sign_conj2=true, 
        cj=(true, true), perm=(3, 1, 4, 2), sign_function=auto_sign)
    end
end

# ------------------------- contract all the indices -------------------------

@timedtestset "test contract operations on all the indices (even-even)" verbose=true begin
    @timedtestset "test ordinary contract operations" verbose=true begin
        # Float64
        @test test_contr3((3, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, Float64)
        @test test_contr3((3, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, Float64; cj=(true, false))
        @test test_contr3((3, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, Float64; cj=(false, true))
        @test test_contr3((3, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, Float64; cj=(true, true))
        # ComplexF64
        @test test_contr3((3, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, ComplexF64)
        @test test_contr3((3, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, ComplexF64; cj=(true, false))
        @test test_contr3((3, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, ComplexF64; cj=(false, true))
        @test test_contr3((3, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, ComplexF64; cj=(true, true))
    end
    @timedtestset "test Fermionic contract operations" verbose=true begin
        # at least sign_tr=true should be enabled if sign_function=auto_sign is enabled
        # Float64
        @test test_contr3((3, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, Float64; sign_contr=true, sign_function=auto_sign)
        @test test_contr3((3, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, Float64; sign_contr=true, sign_conj1=true, sign_conj2=true, 
        cj=(true, true), sign_function=auto_sign)
        @test test_contr3((3, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, Float64; sign_contr=true, sign_conj1=true, sign_conj2=true, 
        cj=(true, true), sign_function=auto_sign)
        # ComplexF64
        @test test_contr3((3, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, ComplexF64; sign_contr=true, sign_function=auto_sign)
        @test test_contr3((3, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, ComplexF64; sign_contr=true, sign_conj1=true, sign_conj2=true, 
        cj=(true, true), sign_function=auto_sign)
        @test test_contr3((3, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, ComplexF64; sign_contr=true, sign_conj1=true, sign_conj2=true, 
        cj=(true, true), sign_function=auto_sign)
    end
end

@timedtestset "test contract operations on all the indices (odd-odd)" verbose=true begin
    @timedtestset "test ordinary contract operations" verbose=true begin
        # Float64
        @test test_contr3((3, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64)
        @test test_contr3((3, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; cj=(true, false))
        @test test_contr3((3, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; cj=(false, true))
        @test test_contr3((3, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; cj=(true, true))
        # ComplexF64
        @test test_contr3((3, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64)
        @test test_contr3((3, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; cj=(true, false))
        @test test_contr3((3, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; cj=(false, true))
        @test test_contr3((3, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; cj=(true, true))
    end
    @timedtestset "test Fermionic contract operations" verbose=true begin
        # at least sign_tr=true should be enabled if sign_function=auto_sign is enabled
        # Float64
        @test test_contr3((3, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; sign_contr=true, sign_function=auto_sign)
        @test test_contr3((3, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; sign_contr=true, sign_conj1=true, sign_conj2=true, 
        cj=(true, true), sign_function=auto_sign)
        @test test_contr3((3, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; sign_contr=true, sign_conj1=true, sign_conj2=true, 
        cj=(true, true), sign_function=auto_sign)
        # ComplexF64
        @test test_contr3((3, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; sign_contr=true, sign_function=auto_sign)
        @test test_contr3((3, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; sign_contr=true, sign_conj1=true, sign_conj2=true, 
        cj=(true, true), sign_function=auto_sign)
        @test test_contr3((3, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; sign_contr=true, sign_conj1=true, sign_conj2=true, 
        cj=(true, true), sign_function=auto_sign)
    end
end

################################## helper functions for AD tests ##################################

# only focus on the elements of certain parity sector
function mask_parity_sector(t_array, even_size, parity_sector::Symbol)
    result = copy(t_array)
    for idx in CartesianIndices(t_array)
        inds = Tuple(idx)
        parity = sum(i -> inds[i] <= even_size[i] ? 0 : 1, 1:length(even_size)) % 2
        if (parity_sector == :even && parity != 0) || (parity_sector == :odd && parity == 0)
            result[idx] = 0.0
        end
    end
    return result
end

function test_tr1_ad(
    total_size::NTuple{6, Int}, 
    even_size::NTuple{6, Int}, 
    p_flag::Symbol,
    elemtype::Type; 
    sign_function::F=trivial_sign, 
    perm::NTuple{4, Int}=(1, 2, 3, 4),
    cj::Bool=false,
    pbc::Bool=true, 
    sign_conj::Bool=false, 
    sign_tr::Bool=false, 
    sign_perm::Bool=false, 
    sign_bc::Bool=false)  where {F}

    T = Grassmann(total_size, even_size, (:out, :in, :out, :in, :out, :in), elemtype; 
    init=:random, parity=p_flag)

    g = gradient(x -> abs(sum(trace(x, (2, 5); perm=perm, cj=cj, pbc=pbc, sign_function=sign_function))), T)[1]
    g_array = convert(Array, g)

    T_array = convert(Array, T)
    g_array_test = gradient(x -> abs(sum(manual_tr1(x, total_size, even_size; perm=perm, cj=cj, sign_conj=sign_conj, sign_tr=sign_tr, 
    sign_perm=sign_perm, sign_bc=sign_bc))), T_array)[1]
    
    (g_array ≈ mask_parity_sector(g_array_test, even_size, p_flag))
end

function test_tr2_ad(
    total_size::NTuple{6, Int}, 
    even_size::NTuple{6, Int}, 
    p_flag::Symbol,
    elemtype::Type; 
    sign_function::F=trivial_sign, 
    perm::NTuple{2, Int}=(1, 2),
    cj::Bool=false,
    pbc::NTuple{2, Bool}=(true, true), 
    sign_conj::Bool=false, 
    sign_tr::Bool=false, 
    sign_perm::Bool=false, 
    sign_bc::Bool=false) where {F}

    T = Grassmann(total_size, even_size, (:out, :in, :out, :in, :out, :in), elemtype; 
    init=:random, parity=p_flag)

    g = gradient(x -> abs(sum(trace(x, ((2, 3), (5, 4)); perm=perm, cj=cj, pbc=pbc, sign_function=sign_function))), T)[1]
    g_array = convert(Array, g)

    T_array = convert(Array, T)
    g_array_test = gradient(x -> abs(sum(manual_tr2(x, total_size, even_size; perm=perm, cj=cj, sign_conj=sign_conj, sign_tr=sign_tr, 
    sign_perm=sign_perm, sign_bc=sign_bc, pbc=pbc))), T_array)[1]
    
    (g_array ≈ mask_parity_sector(g_array_test, even_size, p_flag))
end

function test_tr3_ad(
    total_size::NTuple{6, Int}, 
    even_size::NTuple{6, Int}, 
    p_flag::Symbol,
    elemtype::Type; 
    sign_function::F=trivial_sign, 
    cj::Bool=false,
    pbc::NTuple{3, Bool}=(true, true, true), 
    sign_conj::Bool=false, 
    sign_tr::Bool=false, 
    sign_bc::Bool=false) where {F}

    T = Grassmann(total_size, even_size, (:out, :in, :out, :in, :out, :in), elemtype; 
    init=:random, parity=p_flag)

    g = gradient(x -> abs(sum(trace(x, ((2, 3, 1), (5, 4, 6)); cj=cj, pbc=pbc, sign_function=sign_function))), T)[1]
    g_array = convert(Array, g)

    T_array = convert(Array, T)
    g_array_test = gradient(x -> abs(sum(manual_tr3(x, total_size, even_size; cj=cj, sign_conj=sign_conj, sign_tr=sign_tr, 
    sign_bc=sign_bc, pbc=pbc))), T_array)[1]
    
    (g_array ≈ mask_parity_sector(g_array_test, even_size, p_flag))
end

function test_contr0_ad(
    total_size1::NTuple{4, Int}, 
    even_size1::NTuple{4, Int}, 
    p_flag1::Symbol, 
    total_size2::NTuple{4, Int}, 
    even_size2::NTuple{4, Int}, 
    p_flag2::Symbol,
    elemtype::Type; 
    sign_function::F=trivial_sign, 
    perm::NTuple{8, Int}=(1, 2, 3, 4, 5, 6, 7, 8),
    cj::NTuple{2, Bool}=(false, false), 
    sign_conj1::Bool=false, 
    sign_conj2::Bool=false, 
    sign_perm::Bool=false) where {F}

    T1 = Grassmann(total_size1, even_size1, (:out, :in, :out, :in), elemtype; 
    init=:random, parity=p_flag1)
    T2 = Grassmann(total_size2, even_size2, (:in, :in, :out, :out), elemtype; 
    init=:random, parity=p_flag2)

    T1_array = convert(Array, T1)
    T2_array = convert(Array, T2)

    # Test gradient w.r.t. T1
    g1 = gradient(x -> abs(sum(contract(x, T2; perm=perm, cj=cj, sign_function=sign_function))), T1)[1]
    g1_array = convert(Array, g1)
    g1_array_test = gradient(x -> abs(sum(manual_contr0(x, T2_array, total_size1, even_size1, total_size2, even_size2; 
    perm=perm, cj=cj, sign_conj1=sign_conj1, sign_conj2=sign_conj2, sign_perm=sign_perm))), T1_array)[1]

    # Test gradient w.r.t. T2
    g2 = gradient(x -> abs(sum(contract(T1, x; perm=perm, cj=cj, sign_function=sign_function))), T2)[1]
    g2_array = convert(Array, g2)
    g2_array_test = gradient(x -> abs(sum(manual_contr0(T1_array, x, total_size1, even_size1, total_size2, even_size2; 
    perm=perm, cj=cj, sign_conj1=sign_conj1, sign_conj2=sign_conj2, sign_perm=sign_perm))), T2_array)[1]
    
    (g1_array ≈ mask_parity_sector(g1_array_test, even_size1, p_flag1)) && (g2_array ≈ mask_parity_sector(g2_array_test, even_size2, p_flag2))
end

function test_contr1_ad(
    total_size1::NTuple{4, Int}, 
    even_size1::NTuple{4, Int}, 
    p_flag1::Symbol, 
    total_size2::NTuple{4, Int}, 
    even_size2::NTuple{4, Int}, 
    p_flag2::Symbol,
    elemtype::Type;
    sign_function::F=trivial_sign, 
    perm::NTuple{6, Int}=(1, 2, 3, 4, 5, 6),
    cj::NTuple{2, Bool}=(false, false), 
    sign_conj1::Bool=false, 
    sign_conj2::Bool=false, 
    sign_contr::Bool=false, 
    sign_perm::Bool=false) where {F}

    T1 = Grassmann(total_size1, even_size1, (:out, :in, :out, :in), elemtype; 
    init=:random, parity=p_flag1)
    T2 = Grassmann(total_size2, even_size2, (:in, :in, :out, :out), elemtype; 
    init=:random, parity=p_flag2)

    T1_array = convert(Array, T1)
    T2_array = convert(Array, T2)

    # Test gradient w.r.t. T1
    g1 = gradient(x -> abs(sum(contract(x, T2, (2, 4); perm=perm, cj=cj, sign_function=sign_function))), T1)[1]
    g1_array = convert(Array, g1)
    g1_array_test = gradient(x -> abs(sum(manual_contr1(x, T2_array, total_size1, even_size1, total_size2, even_size2; 
    perm=perm, cj=cj, sign_conj1=sign_conj1, sign_conj2=sign_conj2, sign_contr=sign_contr, sign_perm=sign_perm))), T1_array)[1]

    # Test gradient w.r.t. T2
    g2 = gradient(x -> abs(sum(contract(T1, x, (2, 4); perm=perm, cj=cj, sign_function=sign_function))), T2)[1]
    g2_array = convert(Array, g2)
    g2_array_test = gradient(x -> abs(sum(manual_contr1(T1_array, x, total_size1, even_size1, total_size2, even_size2; 
    perm=perm, cj=cj, sign_conj1=sign_conj1, sign_conj2=sign_conj2, sign_contr=sign_contr, sign_perm=sign_perm))), T2_array)[1]
    
    (g1_array ≈ mask_parity_sector(g1_array_test, even_size1, p_flag1)) && (g2_array ≈ mask_parity_sector(g2_array_test, even_size2, p_flag2))
end

function test_contr2_ad(
    total_size1::NTuple{4, Int}, 
    even_size1::NTuple{4, Int}, 
    p_flag1::Symbol, 
    total_size2::NTuple{4, Int}, 
    even_size2::NTuple{4, Int}, 
    p_flag2::Symbol,
    elemtype::Type; 
    sign_function::F=trivial_sign, 
    perm::NTuple{4, Int}=(1, 2, 3, 4),
    cj::NTuple{2, Bool}=(false, false), 
    sign_conj1::Bool=false, 
    sign_conj2::Bool=false, 
    sign_contr::Bool=false, 
    sign_perm::Bool=false) where {F}

    T1 = Grassmann(total_size1, even_size1, (:out, :in, :out, :in), elemtype; 
    init=:random, parity=p_flag1)
    T2 = Grassmann(total_size2, even_size2, (:in, :in, :out, :out), elemtype; 
    init=:random, parity=p_flag2)

    T1_array = convert(Array, T1)
    T2_array = convert(Array, T2)

    # Test gradient w.r.t. T1
    g1 = gradient(x -> abs(sum(contract(x, T2, ((2, 3), (4, 2)); perm=perm, cj=cj, sign_function=sign_function))), T1)[1]
    g1_array = convert(Array, g1)
    g1_array_test = gradient(x -> abs(sum(manual_contr2(x, T2_array, total_size1, even_size1, total_size2, even_size2; 
    perm=perm, cj=cj, sign_conj1=sign_conj1, sign_conj2=sign_conj2, sign_contr=sign_contr, sign_perm=sign_perm))), T1_array)[1]

    # Test gradient w.r.t. T2
    g2 = gradient(x -> abs(sum(contract(T1, x, ((2, 3), (4, 2)); perm=perm, cj=cj, sign_function=sign_function))), T2)[1]
    g2_array = convert(Array, g2)
    g2_array_test = gradient(x -> abs(sum(manual_contr2(T1_array, x, total_size1, even_size1, total_size2, even_size2; 
    perm=perm, cj=cj, sign_conj1=sign_conj1, sign_conj2=sign_conj2, sign_contr=sign_contr, sign_perm=sign_perm))), T2_array)[1]
    
    (g1_array ≈ mask_parity_sector(g1_array_test, even_size1, p_flag1)) && (g2_array ≈ mask_parity_sector(g2_array_test, even_size2, p_flag2))
end

function test_contr3_ad(
    total_size1::NTuple{4, Int}, 
    even_size1::NTuple{4, Int}, 
    p_flag1::Symbol, 
    total_size2::NTuple{4, Int}, 
    even_size2::NTuple{4, Int}, 
    p_flag2::Symbol,
    elemtype::Type;  
    sign_function::F=trivial_sign, 
    cj::NTuple{2, Bool}=(false, false), 
    sign_conj1::Bool=false, 
    sign_conj2::Bool=false, 
    sign_contr::Bool=false) where {F}

    T1 = Grassmann(total_size1, even_size1, (:out, :in, :out, :in), elemtype; 
    init=:random, parity=p_flag1)
    T2 = Grassmann(total_size2, even_size2, (:in, :in, :out, :out), elemtype; 
    init=:random, parity=p_flag2)

    T1_array = convert(Array, T1)
    T2_array = convert(Array, T2)

    # Test gradient w.r.t. T1
    g1 = gradient(x -> abs(sum(contract(x, T2, ((2, 3, 4, 1), (4, 2, 3, 1)); cj=cj, sign_function=sign_function))), T1)[1]
    g1_array = convert(Array, g1)
    g1_array_test = gradient(x -> abs(sum(manual_contr3(x, T2_array, total_size1, even_size1, total_size2, even_size2; 
    cj=cj, sign_conj1=sign_conj1, sign_conj2=sign_conj2, sign_contr=sign_contr))), T1_array)[1]

    # Test gradient w.r.t. T2
    g2 = gradient(x -> abs(sum(contract(T1, x, ((2, 3, 4, 1), (4, 2, 3, 1)); cj=cj, sign_function=sign_function))), T2)[1]
    g2_array = convert(Array, g2)
    g2_array_test = gradient(x -> abs(sum(manual_contr3(T1_array, x, total_size1, even_size1, total_size2, even_size2; 
    cj=cj, sign_conj1=sign_conj1, sign_conj2=sign_conj2, sign_contr=sign_contr))), T2_array)[1]
    
    (g1_array ≈ mask_parity_sector(g1_array_test, even_size1, p_flag1)) && (g2_array ≈ mask_parity_sector(g2_array_test, even_size2, p_flag2))
end

################################## AD tests ##################################

# ------------------------- trace a single index -------------------------

@timedtestset "AD test: test trace operations on one index for even-parity tensor" verbose=true begin
    @timedtestset "AD test: test ordinary trace operations" verbose=true begin
        # Float64
        @test test_tr1_ad((4, 5, 6, 3, 5, 4), (2, 2, 3, 2, 2, 2), :even, Float64)
        @test test_tr1_ad((4, 5, 6, 3, 5, 4), (2, 2, 3, 2, 2, 2), :even, Float64; perm=(4, 2, 3, 1))
        @test test_tr1_ad((4, 5, 6, 3, 5, 4), (2, 2, 3, 2, 2, 2), :even, Float64; cj=true)
        @test test_tr1_ad((4, 5, 6, 3, 5, 4), (2, 2, 3, 2, 2, 2), :even, Float64; perm=(4, 2, 3, 1), cj=true)
        # ComplexF64
        @test test_tr1_ad((4, 5, 6, 3, 5, 4), (2, 2, 3, 2, 2, 2), :even, ComplexF64)
        @test test_tr1_ad((4, 5, 6, 3, 5, 4), (2, 2, 3, 2, 2, 2), :even, ComplexF64; perm=(4, 2, 3, 1))
        @test test_tr1_ad((4, 5, 6, 3, 5, 4), (2, 2, 3, 2, 2, 2), :even, ComplexF64; cj=true)
        @test test_tr1_ad((4, 5, 6, 3, 5, 4), (2, 2, 3, 2, 2, 2), :even, ComplexF64; perm=(4, 2, 3, 1), cj=true)
    end
    @timedtestset "AD test: test Fermionic trace operations" verbose=true begin
        # at least sign_tr=true should be enabled if sign_function=auto_sign is enabled
        # Float64
        @test test_tr1_ad((4, 5, 6, 3, 5, 4), (2, 2, 3, 2, 2, 2), :even, Float64; sign_tr=true, sign_function=auto_sign)
        @test test_tr1_ad((4, 5, 6, 3, 5, 4), (2, 2, 3, 2, 2, 2), :even, Float64; sign_tr=true, sign_bc=true, pbc=false, sign_function=auto_sign)
        @test test_tr1_ad((4, 5, 6, 3, 5, 4), (2, 2, 3, 2, 2, 2), :even, Float64; sign_tr=true, sign_perm=true, perm=(4, 2, 3, 1), sign_function=auto_sign)
        @test test_tr1_ad((4, 5, 6, 3, 5, 4), (2, 2, 3, 2, 2, 2), :even, Float64; sign_tr=true, sign_conj=true, cj=true, sign_function=auto_sign)
        @test test_tr1_ad((4, 5, 6, 3, 5, 4), (2, 2, 3, 2, 2, 2), :even, Float64; sign_tr=true, sign_conj=true, sign_perm=true,
        cj=true, perm=(4, 2, 3, 1), sign_function=auto_sign)
        @test test_tr1_ad((4, 5, 6, 3, 5, 4), (2, 2, 3, 2, 2, 2), :even, Float64; sign_tr=true, sign_conj=true,
        sign_perm=true,
        sign_bc=true,
        pbc=false, cj=true, perm=(4, 2, 3, 1), sign_function=auto_sign)
        # ComplexF64
        @test test_tr1_ad((4, 5, 6, 3, 5, 4), (2, 2, 3, 2, 2, 2), :even, ComplexF64; sign_tr=true, sign_function=auto_sign)
        @test test_tr1_ad((4, 5, 6, 3, 5, 4), (2, 2, 3, 2, 2, 2), :even, ComplexF64; sign_tr=true, sign_bc=true, pbc=false, sign_function=auto_sign)
        @test test_tr1_ad((4, 5, 6, 3, 5, 4), (2, 2, 3, 2, 2, 2), :even, ComplexF64; sign_tr=true, sign_perm=true, perm=(4, 2, 3, 1), sign_function=auto_sign)
        @test test_tr1_ad((4, 5, 6, 3, 5, 4), (2, 2, 3, 2, 2, 2), :even, ComplexF64; sign_tr=true, sign_conj=true, cj=true, sign_function=auto_sign)
        @test test_tr1_ad((4, 5, 6, 3, 5, 4), (2, 2, 3, 2, 2, 2), :even, ComplexF64; sign_tr=true, sign_conj=true, sign_perm=true,
        cj=true, perm=(4, 2, 3, 1), sign_function=auto_sign)
        @test test_tr1_ad((4, 5, 6, 3, 5, 4), (2, 2, 3, 2, 2, 2), :even, ComplexF64; sign_tr=true, sign_conj=true,
        sign_perm=true,
        sign_bc=true,
        pbc=false, cj=true, perm=(4, 2, 3, 1), sign_function=auto_sign)
    end
end

@timedtestset "AD test: test trace operations on one index for odd-parity tensor" verbose=true begin
    @timedtestset "AD test: test ordinary trace operations" verbose=true begin
        # Float64
        @test test_tr1_ad((4, 5, 6, 3, 5, 4), (2, 2, 3, 2, 2, 2), :odd, Float64)
        @test test_tr1_ad((4, 5, 6, 3, 5, 4), (2, 2, 3, 2, 2, 2), :odd, Float64; perm=(4, 2, 3, 1))
        @test test_tr1_ad((4, 5, 6, 3, 5, 4), (2, 2, 3, 2, 2, 2), :odd, Float64; cj=true)
        @test test_tr1_ad((4, 5, 6, 3, 5, 4), (2, 2, 3, 2, 2, 2), :odd, Float64; perm=(4, 2, 3, 1), cj=true)
        # ComplexF64
        @test test_tr1_ad((4, 5, 6, 3, 5, 4), (2, 2, 3, 2, 2, 2), :odd, ComplexF64)
        @test test_tr1_ad((4, 5, 6, 3, 5, 4), (2, 2, 3, 2, 2, 2), :odd, ComplexF64; perm=(4, 2, 3, 1))
        @test test_tr1_ad((4, 5, 6, 3, 5, 4), (2, 2, 3, 2, 2, 2), :odd, ComplexF64; cj=true)
        @test test_tr1_ad((4, 5, 6, 3, 5, 4), (2, 2, 3, 2, 2, 2), :odd, ComplexF64; perm=(4, 2, 3, 1), cj=true)
    end
    @timedtestset "AD test: test Fermionic trace operations" verbose=true begin
        # at least sign_tr=true should be enabled if sign_function=auto_sign is enabled
        # Float64
        @test test_tr1_ad((4, 5, 6, 3, 5, 4), (2, 2, 3, 2, 2, 2), :odd, Float64; sign_tr=true, sign_function=auto_sign)
        @test test_tr1_ad((4, 5, 6, 3, 5, 4), (2, 2, 3, 2, 2, 2), :odd, Float64; sign_tr=true, sign_bc=true, pbc=false, sign_function=auto_sign)
        @test test_tr1_ad((4, 5, 6, 3, 5, 4), (2, 2, 3, 2, 2, 2), :odd, Float64; sign_tr=true, sign_perm=true, perm=(4, 2, 3, 1), sign_function=auto_sign)
        @test test_tr1_ad((4, 5, 6, 3, 5, 4), (2, 2, 3, 2, 2, 2), :odd, Float64; sign_tr=true, sign_conj=true, cj=true, sign_function=auto_sign)
        @test test_tr1_ad((4, 5, 6, 3, 5, 4), (2, 2, 3, 2, 2, 2), :odd, Float64; sign_tr=true, sign_conj=true, sign_perm=true,
        cj=true, perm=(4, 2, 3, 1), sign_function=auto_sign)
        @test test_tr1_ad((4, 5, 6, 3, 5, 4), (2, 2, 3, 2, 2, 2), :odd, Float64; sign_tr=true, sign_conj=true,
        sign_perm=true,
        sign_bc=true,
        pbc=false, cj=true, perm=(4, 2, 3, 1), sign_function=auto_sign)
        # ComplexF64
        @test test_tr1_ad((4, 5, 6, 3, 5, 4), (2, 2, 3, 2, 2, 2), :odd, ComplexF64; sign_tr=true, sign_function=auto_sign)
        @test test_tr1_ad((4, 5, 6, 3, 5, 4), (2, 2, 3, 2, 2, 2), :odd, ComplexF64; sign_tr=true, sign_bc=true, pbc=false, sign_function=auto_sign)
        @test test_tr1_ad((4, 5, 6, 3, 5, 4), (2, 2, 3, 2, 2, 2), :odd, ComplexF64; sign_tr=true, sign_perm=true, perm=(4, 2, 3, 1), sign_function=auto_sign)
        @test test_tr1_ad((4, 5, 6, 3, 5, 4), (2, 2, 3, 2, 2, 2), :odd, ComplexF64; sign_tr=true, sign_conj=true, cj=true, sign_function=auto_sign)
        @test test_tr1_ad((4, 5, 6, 3, 5, 4), (2, 2, 3, 2, 2, 2), :odd, ComplexF64; sign_tr=true, sign_conj=true, sign_perm=true,
        cj=true, perm=(4, 2, 3, 1), sign_function=auto_sign)
        @test test_tr1_ad((4, 5, 6, 3, 5, 4), (2, 2, 3, 2, 2, 2), :odd, ComplexF64; sign_tr=true, sign_conj=true,
        sign_perm=true,
        sign_bc=true,
        pbc=false, cj=true, perm=(4, 2, 3, 1), sign_function=auto_sign)
    end
end

# ------------------------- trace two indices -------------------------

@timedtestset "AD test: test trace operations on two indices for even-parity tensor" verbose=true begin
    @timedtestset "AD test: test ordinary trace operations" verbose=true begin
        # Float64
        @test test_tr2_ad((4, 5, 3, 3, 5, 4), (2, 2, 2, 2, 2, 2), :even, Float64)
        @test test_tr2_ad((4, 5, 3, 3, 5, 4), (2, 2, 2, 2, 2, 2), :even, Float64; perm=(2, 1))
        @test test_tr2_ad((4, 5, 3, 3, 5, 4), (2, 2, 2, 2, 2, 2), :even, Float64; cj=true)
        @test test_tr2_ad((4, 5, 3, 3, 5, 4), (2, 2, 2, 2, 2, 2), :even, Float64; perm=(2, 1), cj=true)
        # ComplexF64
        @test test_tr2_ad((4, 5, 3, 3, 5, 4), (2, 2, 2, 2, 2, 2), :even, ComplexF64)
        @test test_tr2_ad((4, 5, 3, 3, 5, 4), (2, 2, 2, 2, 2, 2), :even, ComplexF64; perm=(2, 1))
        @test test_tr2_ad((4, 5, 3, 3, 5, 4), (2, 2, 2, 2, 2, 2), :even, ComplexF64; cj=true)
        @test test_tr2_ad((4, 5, 3, 3, 5, 4), (2, 2, 2, 2, 2, 2), :even, ComplexF64; perm=(2, 1), cj=true)
    end
    @timedtestset "AD test: test Fermionic trace operations" verbose=true begin
        # at least sign_tr=true should be enabled if sign_function=auto_sign is enabled
        # Float64
        @test test_tr2_ad((4, 5, 3, 3, 5, 4), (2, 2, 2, 2, 2, 2), :even, Float64; sign_tr=true, sign_function=auto_sign)
        @test test_tr2_ad((4, 5, 3, 3, 5, 4), (2, 2, 2, 2, 2, 2), :even, Float64; sign_tr=true, sign_bc=true, pbc=(false, false), sign_function=auto_sign)
        @test test_tr2_ad((4, 5, 3, 3, 5, 4), (2, 2, 2, 2, 2, 2), :even, Float64; sign_tr=true, sign_perm=true, perm=(2, 1), sign_function=auto_sign)
        @test test_tr2_ad((4, 5, 3, 3, 5, 4), (2, 2, 2, 2, 2, 2), :even, Float64; sign_tr=true, sign_conj=true, cj=true, sign_function=auto_sign)
        @test test_tr2_ad((4, 5, 3, 3, 5, 4), (2, 2, 2, 2, 2, 2), :even, Float64; sign_tr=true, sign_conj=true, sign_perm=true,
        cj=true, perm=(2, 1), sign_function=auto_sign)
        @test test_tr2_ad((4, 5, 3, 3, 5, 4), (2, 2, 2, 2, 2, 2), :even, Float64; sign_tr=true,
        sign_conj=true,
        sign_perm=true,
        sign_bc=true,
        pbc=(false, false), cj=true, perm=(2, 1), sign_function=auto_sign)
        # ComplexF64
        @test test_tr2_ad((4, 5, 3, 3, 5, 4), (2, 2, 2, 2, 2, 2), :even, ComplexF64; sign_tr=true, sign_function=auto_sign)
        @test test_tr2_ad((4, 5, 3, 3, 5, 4), (2, 2, 2, 2, 2, 2), :even, ComplexF64; sign_tr=true, sign_bc=true, pbc=(false, false), sign_function=auto_sign)
        @test test_tr2_ad((4, 5, 3, 3, 5, 4), (2, 2, 2, 2, 2, 2), :even, ComplexF64; sign_tr=true, sign_perm=true, perm=(2, 1), sign_function=auto_sign)
        @test test_tr2_ad((4    , 5, 3, 3, 5, 4), (2, 2, 2, 2, 2, 2), :even,
        ComplexF64; sign_tr=true,
        sign_conj=true,
        cj=true,
        sign_function=auto_sign)
        @test test_tr2_ad((4 ,5 ,3 ,3 ,5 ,4) , (2 ,2 ,2 ,2 ,2 ,2) , :even ,ComplexF64 ;sign_tr=true ,
        sign_conj=true ,
        sign_perm=true ,
        cj=true, perm=(2, 1), sign_function=auto_sign)
        @test test_tr2_ad((4, 5, 3, 3, 5, 4), (2, 2, 2, 2, 2, 2), :even, ComplexF64; sign_tr=true,
        sign_conj=true,
        sign_perm=true,
        sign_bc=true,
        pbc=(false, false), cj=true, perm=(2, 1), sign_function=auto_sign)
    end
end

@timedtestset "AD test: test trace operations on two indices for odd-parity tensor" verbose=true begin
    @timedtestset "AD test: test ordinary trace operations" verbose=true begin
        # Float64
        @test test_tr2_ad((4, 5, 3, 3, 5, 4), (2, 2, 2, 2, 2, 2), :odd, Float64)
        @test test_tr2_ad((4, 5, 3, 3, 5, 4), (2, 2, 2, 2, 2, 2), :odd, Float64; perm=(2, 1))
        @test test_tr2_ad((4, 5, 3, 3, 5, 4), (2, 2, 2, 2, 2, 2), :odd, Float64; cj=true)
        @test test_tr2_ad((4, 5, 3, 3, 5, 4), (2, 2, 2, 2, 2, 2), :odd, Float64; perm=(2, 1), cj=true)
        # ComplexF64
        @test test_tr2_ad((4, 5, 3, 3, 5, 4), (2, 2, 2, 2, 2, 2), :odd, ComplexF64)
        @test test_tr2_ad((4, 5, 3, 3, 5, 4), (2, 2, 2, 2, 2, 2), :odd, ComplexF64; perm=(2, 1))
        @test test_tr2_ad((4, 5, 3, 3, 5, 4), (2, 2, 2, 2, 2, 2), :odd, ComplexF64; cj=true)
        @test test_tr2_ad((4, 5, 3, 3, 5, 4), (2, 2, 2, 2, 2, 2), :odd, ComplexF64; perm=(2, 1), cj=true)
    end
    @timedtestset "AD test: test Fermionic trace operations" verbose=true begin
        # at least sign_tr=true should be enabled if sign_function=auto_sign is enabled
        # Float64
        @test test_tr2_ad((4, 5, 3, 3, 5, 4), (2, 2, 2, 2, 2, 2), :odd, Float64; sign_tr=true, sign_function=auto_sign)
        @test test_tr2_ad((4, 5, 3, 3, 5, 4), (2, 2, 2, 2, 2, 2), :odd, Float64; sign_tr=true, sign_bc=true, pbc=(false, false), sign_function=auto_sign)
        @test test_tr2_ad((4, 5, 3, 3, 5, 4), (2, 2, 2, 2, 2, 2), :odd, Float64; sign_tr=true, sign_perm=true, perm=(2, 1), sign_function=auto_sign)
        @test test_tr2_ad((4, 5, 3, 3, 5, 4), (2, 2, 2, 2, 2, 2), :odd, Float64; sign_tr=true, sign_conj=true, cj=true, sign_function=auto_sign)
        @test test_tr2_ad((4, 5, 3, 3, 5, 4), (2, 2, 2, 2, 2, 2), :odd, Float64; sign_tr=true, sign_conj=true, sign_perm=true,
        cj=true, perm=(2, 1), sign_function=auto_sign)
        @test test_tr2_ad((4, 5, 3, 3, 5, 4), (2, 2, 2, 2, 2, 2), :odd, Float64; sign_tr=true,
        sign_conj=true,
        sign_perm=true,
        sign_bc=true,
        pbc=(false, false), cj=true, perm=(2, 1), sign_function=auto_sign)
        # ComplexF64
        @test test_tr2_ad((4, 5, 3, 3, 5, 4), (2, 2, 2, 2, 2, 2), :odd, ComplexF64; sign_tr=true, sign_function=auto_sign)
        @test test_tr2_ad((4, 5, 3, 3, 5, 4), (2, 2, 2, 2, 2, 2), :odd, ComplexF64; sign_tr=true, sign_bc=true, pbc=(false, false), sign_function=auto_sign)
        @test test_tr2_ad((4, 5, 3, 3, 5, 4), (2, 2, 2, 2, 2, 2), :odd, ComplexF64; sign_tr=true, sign_perm=true, perm=(2, 1), sign_function=auto_sign)
        @test test_tr2_ad((4    , 5, 3, 3, 5, 4), (2, 2, 2, 2, 2, 2), :odd,
        ComplexF64; sign_tr=true,
        sign_conj=true,
        cj=true,
        sign_function=auto_sign)
        @test test_tr2_ad((4 ,5 ,3 ,3 ,5 ,4) , (2 ,2 ,2 ,2 ,2 ,2) , :odd ,ComplexF64 ;sign_tr=true ,
        sign_conj=true ,
        sign_perm=true ,
        cj=true, perm=(2, 1), sign_function=auto_sign)
        @test test_tr2_ad((4, 5, 3, 3, 5, 4), (2, 2, 2, 2, 2, 2), :odd, ComplexF64; sign_tr=true,
        sign_conj=true,
        sign_perm=true,
        sign_bc=true,
        pbc=(false, false), cj=true, perm=(2, 1), sign_function=auto_sign)
    end
end

# ------------------------- trace all the indices -------------------------

@timedtestset "AD test: est trace operations on all indices for even-parity tensor" verbose=true begin
    @timedtestset "AD test: test ordinary trace operations" verbose=true begin
        # Float64
        @test test_tr3_ad((4, 5, 3, 3, 5, 4), (2, 2, 2, 2, 2, 2), :even, Float64)
        @test test_tr3_ad((4, 5, 3, 3, 5, 4), (2, 2, 2, 2, 2, 2), :even, Float64; cj=true)
        # ComplexF64
        @test test_tr3_ad((4, 5, 3, 3, 5, 4), (2, 2, 2, 2, 2, 2), :even, ComplexF64)
        @test test_tr3_ad((4, 5, 3, 3, 5, 4), (2, 2, 2, 2, 2, 2), :even, ComplexF64; cj=true)
    end
    @timedtestset "AD test: test Fermionic trace operations" verbose=true begin
        # at least sign_tr=true should be enabled if sign_function=auto_sign is enabled
        # Float64
        @test test_tr3_ad((4, 5, 3, 3, 5, 4), (2, 2, 2, 2, 2, 2), :even, Float64; sign_tr=true, sign_function=auto_sign)
        @test test_tr3_ad((4, 5, 3, 3, 5, 4), (2, 2, 2, 2, 2, 2), :even, Float64; sign_tr=true, sign_bc=true, pbc=(true, false, false), sign_function=auto_sign)
        @test test_tr3_ad((4, 5, 3, 3, 5, 4), (2, 2, 2, 2, 2, 2), :even, Float64; sign_tr=true, sign_bc=true, pbc=(false, true, false), sign_function=auto_sign)
        @test test_tr3_ad((4, 5, 3, 3, 5, 4), (2, 2, 2, 2, 2, 2), :even, Float64; sign_tr=true, sign_bc=true, pbc=(false, false, true), sign_function=auto_sign)
        @test test_tr3_ad((4, 5, 3, 3, 5, 4), (2, 2, 2, 2, 2, 2), :even, Float64; sign_tr=true, sign_bc=true, pbc=(false, false, false), sign_function=auto_sign)
        @test test_tr3_ad((4, 5, 3, 3, 5, 4), (2, 2, 2, 2, 2, 2), :even, Float64; sign_tr=true, sign_conj=true, cj=true, sign_function=auto_sign)
        @test test_tr3_ad((4, 5, 3, 3, 5, 4), (2, 2, 2, 2, 2, 2), :even, Float64; sign_tr=true, sign_conj=true, sign_bc=true, 
        pbc=(false, false, false), cj=true, sign_function=auto_sign)
        # ComplexF64
        @test test_tr3_ad((4, 5, 3, 3, 5, 4), (2, 2, 2, 2, 2, 2), :even, ComplexF64; sign_tr=true, sign_function=auto_sign)
        @test test_tr3_ad((4, 5, 3, 3, 5, 4), (2, 2, 2, 2, 2, 2), :even, ComplexF64; sign_tr=true, sign_bc=true, pbc=(true, false, false), sign_function=auto_sign)
        @test test_tr3_ad((4, 5, 3, 3, 5, 4), (2, 2, 2, 2, 2, 2), :even, ComplexF64; sign_tr=true, sign_bc=true, pbc=(false, true, false), sign_function=auto_sign)
        @test test_tr3_ad((4, 5, 3, 3, 5, 4), (2, 2, 2, 2, 2, 2), :even, ComplexF64; sign_tr=true, sign_bc=true, pbc=(false, false, true), sign_function=auto_sign)
        @test test_tr3_ad((4, 5, 3, 3, 5, 4), (2, 2, 2, 2, 2, 2), :even, ComplexF64; sign_tr=true, sign_bc=true, pbc=(false, false, false), sign_function=auto_sign)
        @test test_tr3_ad((4, 5, 3, 3, 5, 4), (2, 2, 2, 2, 2, 2), :even, ComplexF64; sign_tr=true, sign_conj=true, cj=true, sign_function=auto_sign)
        @test test_tr3_ad((4, 5, 3, 3, 5, 4), (2, 2, 2, 2, 2, 2), :even, ComplexF64; sign_tr=true, sign_conj=true, sign_bc=true, 
        pbc=(false, false, false), cj=true, sign_function=auto_sign)
    end
end

# ------------------------- contract 0 index(a.k.a direct product) -------------------------

@timedtestset "AD test: test contract operations on 0 index (even-even)" verbose=true begin
    @timedtestset "AD test: test ordinary contract operations" verbose=true begin
        # Float64
        @test test_contr0_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, Float64)
        @test test_contr0_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, Float64; cj=(true, false))
        @test test_contr0_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, Float64; cj=(false, true))
        @test test_contr0_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, Float64; cj=(true, true))
        @test test_contr0_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, Float64; perm=(3, 1, 2, 4, 7, 5, 6, 8))
        @test test_contr0_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, Float64; cj=(true, true), perm=(3, 1, 2, 4, 7, 5, 6, 8))
        # ComplexF64
        @test test_contr0_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, ComplexF64)
        @test test_contr0_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, ComplexF64; cj=(true, false))
        @test test_contr0_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, ComplexF64; cj=(false, true))
        @test test_contr0_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, ComplexF64; cj=(true, true))
        @test test_contr0_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, ComplexF64; perm=(3, 1, 2, 4, 7, 5, 6, 8))
        @test test_contr0_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, ComplexF64; cj=(true, true), perm=(3, 1, 2, 4, 7, 5, 6, 8))
    end
    @timedtestset "AD test: test Fermionic contract operations" verbose=true begin
        # Float64
        @test test_contr0_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, Float64; sign_function=auto_sign)
        @test test_contr0_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, Float64; sign_conj1=true, sign_conj2=true, 
        cj=(true, true), sign_function=auto_sign)
        @test test_contr0_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, Float64; sign_perm=true, 
        perm=(3, 1, 2, 4, 7, 5, 6, 8), sign_function=auto_sign)
        @test test_contr0_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, Float64; sign_perm=true, sign_conj1=true, sign_conj2=true, 
        cj=(true, true), perm=(3, 1, 2, 4, 7, 5, 6, 8), sign_function=auto_sign)
        # ComplexF64
        @test test_contr0_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, ComplexF64; sign_function=auto_sign)
        @test test_contr0_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, ComplexF64; sign_conj1=true, sign_conj2=true, 
        cj=(true, true), sign_function=auto_sign)
        @test test_contr0_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, ComplexF64; sign_perm=true, 
        perm=(3, 1, 2, 4, 7, 5, 6, 8), sign_function=auto_sign)
        @test test_contr0_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, ComplexF64; sign_perm=true, sign_conj1=true, sign_conj2=true, 
        cj=(true, true), perm=(3, 1, 2, 4, 7, 5, 6, 8), sign_function=auto_sign)
    end
end

@timedtestset "AD test: test contract operations on 0 index (even-odd)" verbose=true begin
    @timedtestset "AD test: test ordinary contract operations" verbose=true begin
        # Float64
        @test test_contr0_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64)
        @test test_contr0_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; cj=(true, false))
        @test test_contr0_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; cj=(false, true))
        @test test_contr0_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; cj=(true, true))
        @test test_contr0_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; perm=(3, 1, 2, 4, 7, 5, 6, 8))
        @test test_contr0_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; cj=(true, true), perm=(3, 1, 2, 4, 7, 5, 6, 8))
        # ComplexF64
        @test test_contr0_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64)
        @test test_contr0_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; cj=(true, false))
        @test test_contr0_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; cj=(false, true))
        @test test_contr0_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; cj=(true, true))
        @test test_contr0_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; perm=(3, 1, 2, 4, 7, 5, 6, 8))
        @test test_contr0_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; cj=(true, true), perm=(3, 1, 2, 4, 7, 5, 6, 8))
    end
    @timedtestset "AD test: test Fermionic contract operations" verbose=true begin
        # Float64
        @test test_contr0_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; sign_function=auto_sign)
        @test test_contr0_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; sign_conj1=true, sign_conj2=true, 
        cj=(true, true), sign_function=auto_sign)
        @test test_contr0_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; sign_perm=true, 
        perm=(3, 1, 2, 4, 7, 5, 6, 8), sign_function=auto_sign)
        @test test_contr0_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; sign_perm=true, sign_conj1=true, sign_conj2=true, 
        cj=(true, true), perm=(3, 1, 2, 4, 7, 5, 6, 8), sign_function=auto_sign)
        # ComplexF64
        @test test_contr0_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; sign_function=auto_sign)
        @test test_contr0_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; sign_conj1=true, sign_conj2=true, 
        cj=(true, true), sign_function=auto_sign)
        @test test_contr0_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; sign_perm=true, 
        perm=(3, 1, 2, 4, 7, 5, 6, 8), sign_function=auto_sign)
        @test test_contr0_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; sign_perm=true, sign_conj1=true, sign_conj2=true, 
        cj=(true, true), perm=(3, 1, 2, 4, 7, 5, 6, 8), sign_function=auto_sign)
    end
end

@timedtestset "AD test: test contract operations on 0 index (odd-odd)" verbose=true begin
    @timedtestset "AD test: test ordinary contract operations" verbose=true begin
        # Float64
        @test test_contr0_ad((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64)
        @test test_contr0_ad((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; cj=(true, false))
        @test test_contr0_ad((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; cj=(false, true))
        @test test_contr0_ad((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; cj=(true, true))
        @test test_contr0_ad((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; perm=(3, 1, 2, 4, 7, 5, 6, 8))
        @test test_contr0_ad((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; cj=(true, true), perm=(3, 1, 2, 4, 7, 5, 6, 8))
        # ComplexF64
        @test test_contr0_ad((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64)
        @test test_contr0_ad((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; cj=(true, false))
        @test test_contr0_ad((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; cj=(false, true))
        @test test_contr0_ad((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; cj=(true, true))
        @test test_contr0_ad((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; perm=(3, 1, 2, 4, 7, 5, 6, 8))
        @test test_contr0_ad((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; cj=(true, true), perm=(3, 1, 2, 4, 7, 5, 6, 8))
    end
    @timedtestset "AD test: test Fermionic contract operations" verbose=true begin
        # Float64
        @test test_contr0_ad((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; sign_function=auto_sign)
        @test test_contr0_ad((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; sign_conj1=true, sign_conj2=true, 
        cj=(true, true), sign_function=auto_sign)
        @test test_contr0_ad((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; sign_perm=true, 
        perm=(3, 1, 2, 4, 7, 5, 6, 8), sign_function=auto_sign)
        @test test_contr0_ad((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; sign_perm=true, sign_conj1=true, sign_conj2=true, 
        cj=(true, true), perm=(3, 1, 2, 4, 7, 5, 6, 8), sign_function=auto_sign)
        # ComplexF64
        @test test_contr0_ad((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; sign_function=auto_sign)
        @test test_contr0_ad((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; sign_conj1=true, sign_conj2=true, 
        cj=(true, true), sign_function=auto_sign)
        @test test_contr0_ad((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; sign_perm=true, 
        perm=(3, 1, 2, 4, 7, 5, 6, 8), sign_function=auto_sign)
        @test test_contr0_ad((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; sign_perm=true, sign_conj1=true, sign_conj2=true, 
        cj=(true, true), perm=(3, 1, 2, 4, 7, 5, 6, 8), sign_function=auto_sign)
    end
end

# ------------------------- contract a single index -------------------------

@timedtestset "AD test: test contract operations on a single index (even-even)" verbose=true begin
    @timedtestset "AD test: test ordinary contract operations" verbose=true begin
        # Float64
        @test test_contr1_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, Float64)
        @test test_contr1_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, Float64; cj=(true, false))
        @test test_contr1_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, Float64; cj=(false, true))
        @test test_contr1_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, Float64; cj=(true, true))
        @test test_contr1_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, Float64; perm = (5, 1, 3, 2, 4, 6))
        @test test_contr1_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, Float64; cj=(true, true), perm = (5, 1, 3, 2, 4, 6))
        # ComplexF64
        @test test_contr1_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, ComplexF64)
        @test test_contr1_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, ComplexF64; cj=(true, false))
        @test test_contr1_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, ComplexF64; cj=(false, true))
        @test test_contr1_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, ComplexF64; cj=(true, true))
        @test test_contr1_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, ComplexF64; perm = (5, 1, 3, 2, 4, 6))
        @test test_contr1_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, ComplexF64; cj=(true, true), perm = (5, 1, 3, 2, 4, 6))
    end
    @timedtestset "AD test: test Fermionic contract operations" verbose=true begin
        # at least sign_tr=true should be enabled if sign_function=auto_sign is enabled
        # Float64
        @test test_contr1_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, Float64; sign_contr=true, sign_function=auto_sign)
        @test test_contr1_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, Float64; sign_contr=true, sign_conj1=true, sign_conj2=true, 
        cj=(true, true), sign_function=auto_sign)
        @test test_contr1_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, Float64; sign_contr=true, sign_perm=true, 
        perm = (5, 1, 3, 2, 4, 6), sign_function=auto_sign)
        @test test_contr1_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, Float64; sign_contr=true, sign_perm=true, sign_conj1=true, sign_conj2=true, 
        cj=(true, true), perm = (5, 1, 3, 2, 4, 6), sign_function=auto_sign)
        # ComplexF64
        @test test_contr1_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, ComplexF64; sign_contr=true, sign_function=auto_sign)
        @test test_contr1_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, ComplexF64; sign_contr=true, sign_conj1=true, sign_conj2=true, 
        cj=(true, true), sign_function=auto_sign)
        @test test_contr1_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, ComplexF64; sign_contr=true, sign_perm=true, 
        perm = (5, 1, 3, 2, 4, 6), sign_function=auto_sign)
        @test test_contr1_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, ComplexF64; sign_contr=true, sign_perm=true, sign_conj1=true, sign_conj2=true, 
        cj=(true, true), perm = (5, 1, 3, 2, 4, 6), sign_function=auto_sign)
    end
end

@timedtestset "AD test: test contract operations on a single index (even-odd)" verbose=true begin
    @timedtestset "AD test: test ordinary contract operations" verbose=true begin
        # Float64
        @test test_contr1_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64)
        @test test_contr1_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; cj=(true, false))
        @test test_contr1_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; cj=(false, true))
        @test test_contr1_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; cj=(true, true))
        @test test_contr1_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; perm = (5, 1, 3, 2, 4, 6))
        @test test_contr1_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; cj=(true, true), perm = (5, 1, 3, 2, 4, 6))
        # ComplexF64
        @test test_contr1_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64)
        @test test_contr1_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; cj=(true, false))
        @test test_contr1_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; cj=(false, true))
        @test test_contr1_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; cj=(true, true))
        @test test_contr1_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; perm = (5, 1, 3, 2, 4, 6))
        @test test_contr1_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; cj=(true, true), perm = (5, 1, 3, 2, 4, 6))
    end
    @timedtestset "AD test: test Fermionic contract operations" verbose=true begin
        # at least sign_tr=true should be enabled if sign_function=auto_sign is enabled
        # Float64
        @test test_contr1_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; sign_contr=true, sign_function=auto_sign)
        @test test_contr1_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; sign_contr=true, sign_conj1=true, sign_conj2=true, 
        cj=(true, true), sign_function=auto_sign)
        @test test_contr1_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; sign_contr=true, sign_perm=true, 
        perm = (5, 1, 3, 2, 4, 6), sign_function=auto_sign)
        @test test_contr1_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; sign_contr=true, sign_perm=true, sign_conj1=true, sign_conj2=true, 
        cj=(true, true), perm = (5, 1, 3, 2, 4, 6), sign_function=auto_sign)
        # ComplexF64
        @test test_contr1_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; sign_contr=true, sign_function=auto_sign)
        @test test_contr1_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; sign_contr=true, sign_conj1=true, sign_conj2=true, 
        cj=(true, true), sign_function=auto_sign)
        @test test_contr1_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; sign_contr=true, sign_perm=true, 
        perm = (5, 1, 3, 2, 4, 6), sign_function=auto_sign)
        @test test_contr1_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; sign_contr=true, sign_perm=true, sign_conj1=true, sign_conj2=true, 
        cj=(true, true), perm = (5, 1, 3, 2, 4, 6), sign_function=auto_sign)
    end
end

@timedtestset "AD test: test contract operations on a single index (odd-odd)" verbose=true begin
    @timedtestset "AD test: test ordinary contract operations" verbose=true begin
        # Float64
        @test test_contr1_ad((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64)
        @test test_contr1_ad((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; cj=(true, false))
        @test test_contr1_ad((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; cj=(false, true))
        @test test_contr1_ad((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; cj=(true, true))
        @test test_contr1_ad((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; perm = (5, 1, 3, 2, 4, 6))
        @test test_contr1_ad((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; cj=(true, true), perm = (5, 1, 3, 2, 4, 6))
        # ComplexF64
        @test test_contr1_ad((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64)
        @test test_contr1_ad((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; cj=(true, false))
        @test test_contr1_ad((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; cj=(false, true))
        @test test_contr1_ad((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; cj=(true, true))
        @test test_contr1_ad((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; perm = (5, 1, 3, 2, 4, 6))
        @test test_contr1_ad((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; cj=(true, true), perm = (5, 1, 3, 2, 4, 6))
    end
    @timedtestset "AD test: test Fermionic contract operations" verbose=true begin
        # at least sign_tr=true should be enabled if sign_function=auto_sign is enabled
        # Float64
        @test test_contr1_ad((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; sign_contr=true, sign_function=auto_sign)
        @test test_contr1_ad((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; sign_contr=true, sign_conj1=true, sign_conj2=true, 
        cj=(true, true), sign_function=auto_sign)
        @test test_contr1_ad((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; sign_contr=true, sign_perm=true, 
        perm = (5, 1, 3, 2, 4, 6), sign_function=auto_sign)
        @test test_contr1_ad((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; sign_contr=true, sign_perm=true, sign_conj1=true, sign_conj2=true, 
        cj=(true, true), perm = (5, 1, 3, 2, 4, 6), sign_function=auto_sign)
        # ComplexF64
        @test test_contr1_ad((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; sign_contr=true, sign_function=auto_sign)
        @test test_contr1_ad((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; sign_contr=true, sign_conj1=true, sign_conj2=true, 
        cj=(true, true), sign_function=auto_sign)
        @test test_contr1_ad((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; sign_contr=true, sign_perm=true, 
        perm = (5, 1, 3, 2, 4, 6), sign_function=auto_sign)
        @test test_contr1_ad((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; sign_contr=true, sign_perm=true, sign_conj1=true, sign_conj2=true, 
        cj=(true, true), perm = (5, 1, 3, 2, 4, 6), sign_function=auto_sign)
    end
end

# ------------------------- contract two indices -------------------------

@timedtestset "AD test: test contract operations on two indices (even-even)" verbose=true begin
    @timedtestset "AD test: test ordinary contract operations" verbose=true begin
        # Float64
        @test test_contr2_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, Float64)
        @test test_contr2_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, Float64; cj=(true, false))
        @test test_contr2_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, Float64; cj=(false, true))
        @test test_contr2_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, Float64; cj=(true, true))
        @test test_contr2_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, Float64; perm=(3, 1, 4, 2))
        @test test_contr2_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, Float64; cj=(true, true), perm=(3, 1, 4, 2))
        # ComplexF64
        @test test_contr2_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, ComplexF64)
        @test test_contr2_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, ComplexF64; cj=(true, false))
        @test test_contr2_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, ComplexF64; cj=(false, true))
        @test test_contr2_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, ComplexF64; cj=(true, true))
        @test test_contr2_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, ComplexF64; perm=(3, 1, 4, 2))
        @test test_contr2_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, ComplexF64; cj=(true, true), perm=(3, 1, 4, 2))
    end
    @timedtestset "AD test: test Fermionic contract operations" verbose=true begin
        # at least sign_tr=true should be enabled if sign_function=auto_sign is enabled
        # Float64
        @test test_contr2_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, Float64; sign_contr=true, sign_function=auto_sign)
        @test test_contr2_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, Float64; sign_contr=true, sign_conj1=true, sign_conj2=true, 
        cj=(true, true), sign_function=auto_sign)
        @test test_contr2_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, Float64; sign_contr=true, sign_perm=true, 
        perm=(3, 1, 4, 2), sign_function=auto_sign)
        @test test_contr2_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, Float64; sign_contr=true, sign_perm=true, sign_conj1=true, sign_conj2=true, 
        cj=(true, true), perm=(3, 1, 4, 2), sign_function=auto_sign)
        # ComplexF64
        @test test_contr2_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, ComplexF64; sign_contr=true, sign_function=auto_sign)
        @test test_contr2_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, ComplexF64; sign_contr=true, sign_conj1=true, sign_conj2=true, 
        cj=(true, true), sign_function=auto_sign)
        @test test_contr2_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, ComplexF64; sign_contr=true, sign_perm=true, 
        perm=(3, 1, 4, 2), sign_function=auto_sign)
        @test test_contr2_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, ComplexF64; sign_contr=true, sign_perm=true, sign_conj1=true, sign_conj2=true, 
        cj=(true, true), perm=(3, 1, 4, 2), sign_function=auto_sign)
    end
end

@timedtestset "AD test: test contract operations on two indices (even-odd)" verbose=true begin
    @timedtestset "AD test: test ordinary contract operations" verbose=true begin
        # Float64
        @test test_contr2_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64)
        @test test_contr2_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; cj=(true, false))
        @test test_contr2_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; cj=(false, true))
        @test test_contr2_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; cj=(true, true))
        @test test_contr2_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; perm=(3, 1, 4, 2))
        @test test_contr2_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; cj=(true, true), perm=(3, 1, 4, 2))
        # ComplexF64
        @test test_contr2_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64)
        @test test_contr2_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; cj=(true, false))
        @test test_contr2_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; cj=(false, true))
        @test test_contr2_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; cj=(true, true))
        @test test_contr2_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; perm=(3, 1, 4, 2))
        @test test_contr2_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; cj=(true, true), perm=(3, 1, 4, 2))
    end
    @timedtestset "AD test: test Fermionic contract operations" verbose=true begin
        # at least sign_tr=true should be enabled if sign_function=auto_sign is enabled
        # Float64
        @test test_contr2_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; sign_contr=true, sign_function=auto_sign)
        @test test_contr2_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; sign_contr=true, sign_conj1=true, sign_conj2=true, 
        cj=(true, true), sign_function=auto_sign)
        @test test_contr2_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; sign_contr=true, sign_perm=true, 
        perm=(3, 1, 4, 2), sign_function=auto_sign)
        @test test_contr2_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; sign_contr=true, sign_perm=true, sign_conj1=true, sign_conj2=true, 
        cj=(true, true), perm=(3, 1, 4, 2), sign_function=auto_sign)
        # ComplexF64
        @test test_contr2_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; sign_contr=true, sign_function=auto_sign)
        @test test_contr2_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; sign_contr=true, sign_conj1=true, sign_conj2=true, 
        cj=(true, true), sign_function=auto_sign)
        @test test_contr2_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; sign_contr=true, sign_perm=true, 
        perm=(3, 1, 4, 2), sign_function=auto_sign)
        @test test_contr2_ad((4, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; sign_contr=true, sign_perm=true, sign_conj1=true, sign_conj2=true, 
        cj=(true, true), perm=(3, 1, 4, 2), sign_function=auto_sign)
    end
end

@timedtestset "AD test: test contract operations on two indices (odd-odd)" verbose=true begin
    @timedtestset "AD test: test ordinary contract operations" verbose=true begin
        # Float64
        @test test_contr2_ad((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64)
        @test test_contr2_ad((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; cj=(true, false))
        @test test_contr2_ad((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; cj=(false, true))
        @test test_contr2_ad((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; cj=(true, true))
        @test test_contr2_ad((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; perm=(3, 1, 4, 2))
        @test test_contr2_ad((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; cj=(true, true), perm=(3, 1, 4, 2))
        # ComplexF64
        @test test_contr2_ad((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64)
        @test test_contr2_ad((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; cj=(true, false))
        @test test_contr2_ad((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; cj=(false, true))
        @test test_contr2_ad((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; cj=(true, true))
        @test test_contr2_ad((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; perm=(3, 1, 4, 2))
        @test test_contr2_ad((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; cj=(true, true), perm=(3, 1, 4, 2))
    end
    @timedtestset "AD test: test Fermionic contract operations" verbose=true begin
        # at least sign_tr=true should be enabled if sign_function=auto_sign is enabled
        # Float64
        @test test_contr2_ad((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; sign_contr=true, sign_function=auto_sign)
        @test test_contr2_ad((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; sign_contr=true, sign_conj1=true, sign_conj2=true, 
        cj=(true, true), sign_function=auto_sign)
        @test test_contr2_ad((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; sign_contr=true, sign_perm=true, 
        perm=(3, 1, 4, 2), sign_function=auto_sign)
        @test test_contr2_ad((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; sign_contr=true, sign_perm=true, sign_conj1=true, sign_conj2=true, 
        cj=(true, true), perm=(3, 1, 4, 2), sign_function=auto_sign)
        # ComplexF64
        @test test_contr2_ad((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; sign_contr=true, sign_function=auto_sign)
        @test test_contr2_ad((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; sign_contr=true, sign_conj1=true, sign_conj2=true, 
        cj=(true, true), sign_function=auto_sign)
        @test test_contr2_ad((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; sign_contr=true, sign_perm=true, 
        perm=(3, 1, 4, 2), sign_function=auto_sign)
        @test test_contr2_ad((4, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; sign_contr=true, sign_perm=true, sign_conj1=true, sign_conj2=true, 
        cj=(true, true), perm=(3, 1, 4, 2), sign_function=auto_sign)
    end
end

# ------------------------- contract all the indices -------------------------

@timedtestset "AD test: test contract operations on all the indices (even-even)" verbose=true begin
    @timedtestset "AD test: test ordinary contract operations" verbose=true begin
        # Float64
        @test test_contr3_ad((3, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, Float64)
        @test test_contr3_ad((3, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, Float64; cj=(true, false))
        @test test_contr3_ad((3, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, Float64; cj=(false, true))
        @test test_contr3_ad((3, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, Float64; cj=(true, true))
        # ComplexF64
        @test test_contr3_ad((3, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, ComplexF64)
        @test test_contr3_ad((3, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, ComplexF64; cj=(true, false))
        @test test_contr3_ad((3, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, ComplexF64; cj=(false, true))
        @test test_contr3_ad((3, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, ComplexF64; cj=(true, true))
    end
    @timedtestset "AD test: test Fermionic contract operations" verbose=true begin
        # at least sign_tr=true should be enabled if sign_function=auto_sign is enabled
        # Float64
        @test test_contr3_ad((3, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, Float64; sign_contr=true, sign_function=auto_sign)
        @test test_contr3_ad((3, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, Float64; sign_contr=true, sign_conj1=true, sign_conj2=true, 
        cj=(true, true), sign_function=auto_sign)
        @test test_contr3_ad((3, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, Float64; sign_contr=true, sign_conj1=true, sign_conj2=true, 
        cj=(true, true), sign_function=auto_sign)
        # ComplexF64
        @test test_contr3_ad((3, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, ComplexF64; sign_contr=true, sign_function=auto_sign)
        @test test_contr3_ad((3, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, ComplexF64; sign_contr=true, sign_conj1=true, sign_conj2=true, 
        cj=(true, true), sign_function=auto_sign)
        @test test_contr3_ad((3, 3, 4, 2), (2, 2, 2, 2), :even, (3, 4, 2, 3), (2, 2, 2, 2), :even, ComplexF64; sign_contr=true, sign_conj1=true, sign_conj2=true, 
        cj=(true, true), sign_function=auto_sign)
    end
end

@timedtestset "AD test: test contract operations on all the indices (odd-odd)" verbose=true begin
    @timedtestset "AD test: test ordinary contract operations" verbose=true begin
        # Float64
        @test test_contr3_ad((3, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64)
        @test test_contr3_ad((3, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; cj=(true, false))
        @test test_contr3_ad((3, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; cj=(false, true))
        @test test_contr3_ad((3, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; cj=(true, true))
        # ComplexF64
        @test test_contr3_ad((3, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64)
        @test test_contr3_ad((3, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; cj=(true, false))
        @test test_contr3_ad((3, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; cj=(false, true))
        @test test_contr3_ad((3, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; cj=(true, true))
    end
    @timedtestset "AD test: test Fermionic contract operations" verbose=true begin
        # at least sign_tr=true should be enabled if sign_function=auto_sign is enabled
        # Float64
        @test test_contr3_ad((3, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; sign_contr=true, sign_function=auto_sign)
        @test test_contr3_ad((3, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; sign_contr=true, sign_conj1=true, sign_conj2=true, 
        cj=(true, true), sign_function=auto_sign)
        @test test_contr3_ad((3, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, Float64; sign_contr=true, sign_conj1=true, sign_conj2=true, 
        cj=(true, true), sign_function=auto_sign)
        # ComplexF64
        @test test_contr3_ad((3, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; sign_contr=true, sign_function=auto_sign)
        @test test_contr3_ad((3, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; sign_contr=true, sign_conj1=true, sign_conj2=true, 
        cj=(true, true), sign_function=auto_sign)
        @test test_contr3_ad((3, 3, 4, 2), (2, 2, 2, 2), :odd, (3, 4, 2, 3), (2, 2, 2, 2), :odd, ComplexF64; sign_contr=true, sign_conj1=true, sign_conj2=true, 
        cj=(true, true), sign_function=auto_sign)
    end
end

