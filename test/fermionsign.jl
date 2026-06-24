
####################### helper functions #######################

"""
e.g.  QN = (a, b, c, d, e, f, g, h, i, j)
          index = (3, 8)
          index_type = (:out, :in)

    sign = p(h) × (p(g) + p(f) + p(e) + p(d) + p(c))
"""

function sign_func_tr1(QN::NTuple{10, Int})
    sign_flag = QN[8] * (QN[7] + QN[6] + QN[5] + QN[4] + QN[3])
    mod(sign_flag, 2) == 0 ? 1 : -1 
end

"""
e.g.  QN = (a, b, c, d, e, f, g, h, i, j)
      index = ((1, 3), (9, 5))
      index_type = ((:out, :in), (:in, :out))

      sign = p(i) × (p(h) + p(g) + p(f) + p(e) + p(d) + p(c) + p(b) + p(a)) + p(e) × p(d)
"""

function sign_func_tr2(QN::NTuple{10, Int})
    sign_flag = QN[9] * (QN[8] + QN[7] + QN[6] + QN[5] + QN[4] + QN[3] + QN[2] + QN[1]) + QN[5] * QN[4]
    mod(sign_flag, 2) == 0 ? 1 : -1 
end

"""
e.g.  QN = (a, b, c, d, e, f, g, h, i, j)
      index = ((1, 3, 2), (9, 5, 8))
      index_type = ((:out, :in, :in), (:in, :out, :out))

      sign = p(i) × (p(h) + p(g) + p(f) + p(e) + p(d) + p(c) + p(b) + p(a)) + p(e) × p(d) + p(h) × (p(g) + p(f) + p(d))
"""

function sign_func_tr3(QN::NTuple{10, Int})
    sign_flag = QN[9] * (QN[8] + QN[7] + QN[6] + QN[5] + QN[4] + QN[3] + QN[2] + QN[1]) + QN[5] * QN[4] + QN[8] * (QN[7] + QN[6] + QN[4])
    mod(sign_flag, 2) == 0 ? 1 : -1 
end

"""
e.g.  QN1 = (a, b, c, d, e, f)
      QN2 = (g, h, i, j, k)
      index = (3, 4)
      index_type = (:in, :out)

    sign = p(j) × (p(i) + p(h) + p(g) + p(f) + p(e) + p(d))
"""

function sign_func_contract1(QN1::NTuple{6, Int}, QN2::NTuple{5, Int})
    sign_flag = QN2[4] * (QN2[3] + QN2[2] + QN2[1] + QN1[6] + QN1[5] + QN1[4])
    mod(sign_flag, 2) == 0 ? 1 : -1 
end

"""
e.g.  QN1 = (a, b, c, d, e, f)
      QN2 = (g, h, i, j, k)
      index = ((3, 1), (4, 5))
      index_type = ((:in, :out), (:out, :in))

    sign = p(j) × (p(i) + p(h) + p(g) + p(f) + p(e) + p(d)) + p(k) × (p(i) + p(h) + p(g) + p(f) + p(e) + p(d) + p(b) + p(a))
"""

function sign_func_contract2(QN1::NTuple{6, Int}, QN2::NTuple{5, Int})
    sign_flag = QN2[4] * (QN2[3] + QN2[2] + QN2[1] + QN1[6] + QN1[5] + QN1[4]) + 
    QN2[5] * (QN2[3] + QN2[2] + QN2[1] + QN1[6] + QN1[5] + QN1[4] + QN1[2] + QN1[1])
    mod(sign_flag, 2) == 0 ? 1 : -1 
end

"""
e.g.  QN1 = (a, b, c, d, e, f)
      QN2 = (g, h, i, j, k)
      index = ((3, 1, 2), (4, 5, 1))
      index_type = ((:in, :out, :out), (:out, :in, :in))

    sign = p(j) × (p(i) + p(h) + p(g) + p(f) + p(e) + p(d)) + p(k) × (p(i) + p(h) + p(g) + p(f) + p(e) + p(d) + p(b) + p(a))
         + p(g) × (p(f) + p(e) + p(d) + p(b))
"""

function sign_func_contract3(QN1::NTuple{6, Int}, QN2::NTuple{5, Int})
    sign_flag = QN2[4] * (QN2[3] + QN2[2] + QN2[1] + QN1[6] + QN1[5] + QN1[4]) + 
    QN2[5] * (QN2[3] + QN2[2] + QN2[1] + QN1[6] + QN1[5] + QN1[4] + QN1[2] + QN1[1]) +
    QN2[1] * (QN1[6] + QN1[5] + QN1[4] + QN1[2])
    mod(sign_flag, 2) == 0 ? 1 : -1 
end

"""
e.g.  QN = (a, b, c, d, e, f, h, i, j, k, l) 
           (k, a, b, c, d, e, f, h, i, j, l) --- (k to 1)
           (k, a, c, b, d, e, f, h, i, j, l) --- (c to 3)
           (k, a, c, b, j, d, e, f, h, i, l) --- (j to 5)
           (k, a, c, b, j, l, d, e, f, h, i) --- (l to 6)
           (k, a, c, b, j, l, f, d, e, h, i) --- (f to 7)
           (k, a, c, b, j, l, f, d, i, e, h) --- (i to 9)

      dst = (10, 1, 3, 2, 9, 11, 6, 4, 8, 5, 7)

    sign = p(k) × (p(j) + p(i) + p(h) + p(f) + p(e) + p(d) + p(c) + p(b) + p(a)) 
         + p(c) × p(b)  
         + p(j) × (p(i) + p(h) + p(f) + p(e) + p(d)) 
         + p(l) × (p(i) + p(h) + p(f) + p(e) + p(d)) 
         + p(f) × (p(e) + p(d)) +
         + p(i) × (p(h) + p(e)) 

"""

function sign_func_perm(QN::NTuple{11, Int}, dst::NTuple{11, Int})
    sign_flag = QN[10] * (QN[9] + QN[8] + QN[7] + QN[6] + QN[5] + QN[4] + QN[3] + QN[2] + QN[1]) + 
                QN[3] * QN[2] +
                QN[9] * (QN[8] + QN[7] + QN[6] + QN[5] + QN[4]) +
                QN[11] * (QN[8] + QN[7] + QN[6] + QN[5] + QN[4]) +
                QN[6] * (QN[5] + QN[4]) +
                QN[8] * (QN[7] + QN[5])
    mod(sign_flag, 2) == 0 ? 1 : -1 
end

####################### test functions #######################

function test_sign_func_tr1()
    Bool_vec = []
    for qn in Iterators.product(ntuple(i->0:1, 10)...)
        flag = sign_func_tr1(qn) == auto_sign(qn, (3, 8), (:out, :in))
        push!(Bool_vec, flag)
    end
    return !(false in Bool_vec)
end

function test_sign_func_tr2()
    Bool_vec = []
    for qn in Iterators.product(ntuple(i->0:1, 10)...)
        flag = sign_func_tr2(qn) == auto_sign(qn, ((1, 3), (9, 5)), ((:out, :in), (:in, :out)))
        push!(Bool_vec, flag)
    end
    return !(false in Bool_vec)
end

function test_sign_func_tr3()
    Bool_vec = []
    for qn in Iterators.product(ntuple(i->0:1, 10)...)
        flag = sign_func_tr3(qn) == auto_sign(qn, ((1, 3, 2), (9, 5, 8)), ((:out, :in, :in), (:in, :out, :out)))
        push!(Bool_vec, flag)
    end
    return !(false in Bool_vec)
end

function test_sign_func_contract1()
    Bool_vec = []
    for qn1 in Iterators.product(ntuple(i->0:1, 6)...), qn2 in Iterators.product(ntuple(i->0:1, 5)...)
        flag = sign_func_contract1(qn1, qn2) == auto_sign(qn1, qn2, (3, 4), (:in, :out))
        push!(Bool_vec, flag)
    end
    return !(false in Bool_vec)
end

function test_sign_func_contract2()
    Bool_vec = []
    for qn1 in Iterators.product(ntuple(i->0:1, 6)...), qn2 in Iterators.product(ntuple(i->0:1, 5)...)
        flag = sign_func_contract2(qn1, qn2) == auto_sign(qn1, qn2, ((3, 1), (4, 5)), ((:in, :out), (:out, :in)))
        push!(Bool_vec, flag)
    end
    return !(false in Bool_vec)
end

function test_sign_func_contract3()
    Bool_vec = []
    for qn1 in Iterators.product(ntuple(i->0:1, 6)...), qn2 in Iterators.product(ntuple(i->0:1, 5)...)
        flag = sign_func_contract3(qn1, qn2) == auto_sign(qn1, qn2, ((3, 1, 2), (4, 5, 1)), ((:in, :out, :out), (:out, :in, :in)))
        push!(Bool_vec, flag)
    end
    return !(false in Bool_vec)
end

function test_sign_func_perm()
    Bool_vec = []
    for qn in Iterators.product(ntuple(i->0:1, 11)...)
        flag = sign_func_perm(qn, (10, 1, 3, 2, 9, 11, 6, 4, 8, 5, 7)) == auto_sign(qn, (10, 1, 3, 2, 9, 11, 6, 4, 8, 5, 7))
        push!(Bool_vec, flag)
    end
    return !(false in Bool_vec)
end

####################### testing #######################

@timedtestset "test the fermionsign.jl" begin
    @timedtestset "test sign factor from boundary conditions" begin
        @test 1 == auto_sign((0, 1, 0, 1, 0, 1), (true, true, true, true, true, true))
        @test -1 == auto_sign((1, 1, 0, 1, 1, 1), (false, true, true, true, true, true))
        @test 1 == auto_sign((0, 1, 0, 1, 1, 1), (false, true, true, true, true, true))
        @test -1 == auto_sign((1, 1, 0, 1, 1, 1), (true, false, true, true, true, true))
        @test 1 == auto_sign((1, 0, 0, 1, 1, 1), (true, false, true, true, true, true))
        @test -1 == auto_sign((1, 0, 1, 1, 1, 1), (true, true, false, true, true, true))
        @test 1 == auto_sign((1, 0, 0, 1, 1, 1), (true, true, false, true, true, true))
        @test -1 == auto_sign((1, 0, 1, 1, 1, 1), (true, true, true, false, true, true))
        @test 1 == auto_sign((1, 0, 1, 0, 1, 1), (true, true, true, false, true, true))
        @test -1 == auto_sign((1, 0, 1, 0, 1, 1), (true, true, true, true, false, true))
        @test 1 == auto_sign((1, 0, 1, 0, 0, 1), (true, true, true, true, false, true))
        @test -1 == auto_sign((1, 0, 1, 0, 0, 1), (true, true, true, true, true, false))
        @test 1 == auto_sign((1, 0, 1, 0, 0, 0), (true, true, true, true, true, false))
        @test -1 == auto_sign((0, 1, 0, 1, 0, 1), (false, false, false, false, false, false))
        @test 1 == auto_sign((0, 1, 1, 1, 0, 1), (false, false, false, false, false, false))
    end
    @timedtestset "test sign factor from tracing a single index" begin
        @test test_sign_func_tr1()
    end
    @timedtestset "test sign factor from tracing two indices" begin
        @test test_sign_func_tr2()
    end
    @timedtestset "test sign factor from tracing three indices" begin
        @test test_sign_func_tr3()
    end
    @timedtestset "test sign factor from contracting a single index" begin
        @test test_sign_func_contract1()
    end
    @timedtestset "test sign factor from contracting two indices" begin
        @test test_sign_func_contract2()
    end
    @timedtestset "test sign factor from contracting three indices" begin
        @test test_sign_func_contract3()
    end
    @timedtestset "test sign factor from permutation" begin
        @test test_sign_func_perm()
    end
    @timedtestset "test add_parity_sign function" begin
        total_size = (4, 8, 6, 4)
        even_size = (2, 4, 3, 2)
        index_types = (:in, :out, :in, :out)
        T = Grassmann(total_size, even_size, index_types, Float64; init=:random, parity=:even)
        T_new = add_parity_sign(T, 2)
        @test T[(0, 0, 0, 0)] == T_new[(0, 0, 0, 0)]
        @test T[(1, 1, 0, 0)] == -T_new[(1, 1, 0, 0)]
        @test T[(1, 0, 1, 0)] == T_new[(1, 0, 1, 0)]
        @test T[(1, 0, 0, 1)] == T_new[(1, 0, 0, 1)]
        @test T[(0, 1, 1, 0)] == -T_new[(0, 1, 1, 0)]
        @test T[(0, 1, 0, 1)] == -T_new[(0, 1, 0, 1)]
        @test T[(0, 0, 1, 1)] == T_new[(0, 0, 1, 1)]
        @test T[(1, 1, 1, 1)] == -T_new[(1, 1, 1, 1)]
    end
    @timedtestset "test add_perm_sign function" begin
        total_size = (4, 8, 6, 4)
        even_size = (2, 4, 3, 2)
        index_types = (:in, :out, :in, :out)
        T = Grassmann(total_size, even_size, index_types, Float64; init=:random, parity=:even)
        T_new = add_perm_sign(T, (2, 3, 4, 1)) # sign = qn[1] * (qn[2] + qn[3] + qn[4])
        @test T[(0, 0, 0, 0)] == T_new[(0, 0, 0, 0)]
        @test T[(1, 1, 0, 0)] == -T_new[(1, 1, 0, 0)] 
        @test T[(1, 0, 1, 0)] == -T_new[(1, 0, 1, 0)]
        @test T[(1, 0, 0, 1)] == -T_new[(1, 0, 0, 1)]
        @test T[(0, 1, 1, 0)] == T_new[(0, 1, 1, 0)]
        @test T[(0, 1, 0, 1)] == T_new[(0, 1, 0, 1)]
        @test T[(0, 0, 1, 1)] == T_new[(0, 0, 1, 1)]
        @test T[(1, 1, 1, 1)] == -T_new[(1, 1, 1, 1)]
    end
end
