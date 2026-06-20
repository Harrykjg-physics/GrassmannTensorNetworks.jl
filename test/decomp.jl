
########################### auxiliary functions ###########################

# A high-evel but less efficient version of truncation function
function truncation(
    S_sorted::Vector{Float64}, 
    S_even::Vector{Float64}, 
    Dcut::Int)

    S_sort_trunc = S_sorted[1:Dcut]
    S_even_trunc_test = []
    S_odd_trunc_test = []

    for val in S_sort_trunc
        if val in S_even
            push!(S_even_trunc_test, val)
        else
            push!(S_odd_trunc_test, val)
        end
    end

    size_even_trunc = length(S_even_trunc_test)
    size_odd_trunc = length(S_odd_trunc_test)
    
    S_trunc_test = cat(S_even_trunc_test, S_odd_trunc_test; dims=1)

    return S_trunc_test, size_even_trunc, size_odd_trunc 
end

function generator(
    De::Int, 
    Do::Int)
    
    Se = rand(Float64, De)
    Se_sorted = sort(Se; rev=true)
    So = rand(Float64, Do)
    So_sorted = sort(So; rev=true)
    S = cat(Se_sorted, So_sorted; dims=1)

    return S, Se
end

########################### test functions ###########################

function test_GSVD_rank2_1(total_size, even_size, Dcut)

    T = Grassmann(total_size, even_size, ComplexF64; init=:random, parity=:even)

    Uout, Sout, Vout, _  = gsvd(T, Dcut)

    Uout_array = convert2array(Uout)
    Vout_array = convert2array(Vout)
    Sout_array = convert2array(Sout)

    T_array = convert2array(T)
    U_array, S_vec, V_array = svd(T_array)
    U_array_out = U_array[:, 1:Dcut]
    V_array_out = V_array[:, 1:Dcut]
    S_mat_out = Diagonal(S_vec[1:Dcut])

    return Uout_array, Sout_array, Vout_array, U_array_out, S_mat_out, V_array_out
end

function test_GSVD_rank2_2(total_size, even_size, Dcut)

    T = Grassmann(total_size, even_size, ComplexF64; init=:random, parity=:even)
    Uout, Sout, Vout, _ = gsvd(T, Dcut)
    T_00 = T[(0, 0)]; T_11 = T[(1, 1)]
    U_00, S_00, V_00 = svd(T_00)
    U_11, S_11, V_11 = svd(T_11)

    return Uout, Sout, Vout, U_00, S_00, V_00, U_11, S_11, V_11
end

function test_GSVD_rank2_3(total_size, even_size, Dcut)

    T = Grassmann(total_size, even_size, ComplexF64; init=:random, parity=:even)
    Uout, Sout, Vout, _ = gsvd(T, Dcut)
    T1 = contract(Uout, Sout, (2, 1))
    T2 = contract(T1, Vout, (2, 2); cj=(false, true))
    T2_array = convert2array(T2)
    T_array = convert2array(T)
    T_array ≈ T2_array
end

function test_GSVD_rank2_4(total_size, even_size, Dcut)

    T = Grassmann(total_size, even_size, ComplexF64; init=:random, parity=:even)
    Uout, Sout, Vout, _ = gsvd(T, Dcut)
    T1 = contract(Uout, Sout, (2, 1))
    T2 = contract(T1, Vout, (2, 2); cj=(false, true))
    T2_array = convert2array(T2)
    T_array = convert2array(T)
    Uout_array, Sout_vec, Vout_array = svd(T_array)
    Uout_array_trunc = Uout_array[:, 1:Dcut]
    Vout_array_trunc = Vout_array[:, 1:Dcut]
    Sout_vec_trunc = Sout_vec[1:Dcut]
    T_array_trunc = Uout_array_trunc * Diagonal(Sout_vec_trunc) * Vout_array_trunc'
    T2_array ≈ T_array_trunc
end

function test_GSVD_rank2_5(total_size, even_size, Dcut)

    T = Grassmann(total_size, even_size, ComplexF64; init=:random, parity=:even)
    Uout, Sout, Vout, _ = gsvd(T, Dcut)
    I_U = contract(Uout, Uout, (1, 1); cj=(false, true))
    I_V = contract(Vout, Vout, (1, 1); cj=(false, true))
    I_U_array = convert2array(I_U)
    I_V_array = convert2array(I_V)
    return I_U_array, I_V_array
end

function test_GSVD_rank2_6(total_size, even_size, Dcut)

    T = Grassmann(total_size, even_size, ComplexF64; init=:random, parity=:even)
    Uout, Sout, Vout, _  = gsvd(T, Dcut)

    I_U1 = contract(Uout, Uout, (1, 1); cj=(false, true))
    I_V1 = contract(Vout, Vout, (1, 1); cj=(false, true))
    I_U1_array = convert2array(I_U1)
    I_V1_array = convert2array(I_V1)

    I_U2 = contract(Uout, Uout, (2, 2); cj=(false, true))
    I_V2 = contract(Vout, Vout, (2, 2); cj=(false, true))
    I_U2_array = convert2array(I_U2)
    I_V2_array = convert2array(I_V2)
    return I_U1_array, I_V1_array, I_U2_array, I_V2_array
end

function test_GSVD_rank2_7(total_size, even_size, Dcut)

    T = Grassmann(total_size, even_size, ComplexF64; init=:random, parity=:even)
    Uout1, Sout1, Vout1, _  = gsvd(T, Dcut; trunc=false)
    Sout1, dim_even_trunc1, dim_odd_trunc1 = truncation(Sout1, Dcut)
    Uout1 = Uout1[2, (dim_even_trunc1, dim_odd_trunc1)]
    Vout1 = Vout1[2, (dim_even_trunc1, dim_odd_trunc1)]

    Uout2, Sout2, Vout2, _  = gsvd(T, Dcut)

    return Uout1, Sout1, Vout1, Uout2, Sout2, Vout2
end

function test_GSVD_rank4_1(
    total_size, even_size, 
    indsrow::NTuple{N1, Int}, 
    indscol::NTuple{N2, Int}; 
    sign_function=trivial_sign) where {N1, N2}

    T = Grassmann(total_size, even_size, ComplexF64; init=:random, parity=:even)
    U_out, S_out, V_out, _ = gsvd(T, indsrow, indscol, 10000; sign_function=sign_function)
    T1 = contract(U_out, S_out, (N1+1, 1); sign_function=sign_function)
    T2 = contract(T1, V_out, (N1+1, N2+1); cj=(false, true), sign_function=sign_function)
    T ≈ T2
end

function test_GSVD_rank4_2(
    total_size, 
    even_size, 
    indsrow::NTuple{N1, Int}, 
    indscol::NTuple{N2, Int}) where {N1, N2}

    T = Grassmann(total_size, even_size, ComplexF64; init=:random, parity=:even)
    Uout, Sout, Vout, _  = gsvd(T, indsrow, indscol, 10000)
    inds_U = ntuple(i->i, N1)
    inds_V = ntuple(i->i, N2)
    I_U = contract(Uout, Uout, (inds_U, inds_U); cj=(false, true))
    I_V = contract(Vout, Vout, (inds_V, inds_V); cj=(false, true))
    I_U_array = convert2array(I_U)
    I_V_array = convert2array(I_V)
    return I_U_array, I_V_array
end

function test_GSVD_rank4_3(
    total_size, even_size, Dcut, 
    indsrow::NTuple{N1, Int}, 
    indscol::NTuple{N2, Int}) where {N1, N2}

    T = Grassmann(total_size, even_size, ComplexF64; init=:random, parity=:even)
    Uout, Sout, Vout, _  = gsvd(T, indsrow, indscol, Dcut)
    inds_U = ntuple(i->i, N1)
    inds_V = ntuple(i->i, N2)
    I_U = contract(Uout, Uout, (inds_U, inds_U); cj=(false, true))
    I_V = contract(Vout, Vout, (inds_V, inds_V); cj=(false, true))
    I_U_array = convert2array(I_U)
    I_V_array = convert2array(I_V)
    return I_U_array, I_V_array
end

const test_GSVD_rank6_1 = test_GSVD_rank4_1

const test_GSVD_rank6_2 = test_GSVD_rank4_2

const test_GSVD_rank6_3 = test_GSVD_rank4_3

function test_GEVD_rank2_non_trunc1(total_size, even_size, Dcut)

    T = Grassmann(total_size, even_size, ComplexF64; init=:random, parity=:even)
    Uout, Λout, _ = gevd(T, Dcut)
    TU = contract(T, Uout, (2, 1))
    UΛ = contract(Uout, Λout, (2, 1))
    TU ≈ UΛ
end

function test_GEVD_rank2_non_trunc2(total_size, even_size, Dcut)

    T_coef = rand(ComplexF64, total_size)
    T_coef = (T_coef + T_coef')/2
    T = Grassmann(T_coef, total_size, even_size)
    Uout, Λout, _ = gevd(T, Dcut)
    T1 = contract(Uout, Λout, (2, 1))
    T2 = contract(T1, Uout, (2, 2); cj=(false, true))
    T ≈ T2
end

function test_GEVD_rank2_non_trunc3(total_size, even_size, Dcut)

    T_coef1 = rand(ComplexF64, total_size)
    T1 = Grassmann(T_coef1, total_size, even_size)
    Uout1, Λout1, _ = gevd(T1, Dcut; symflag=true)
    Ta1 = contract(Uout1, Λout1, (2, 1))
    Tb1 = contract(Ta1, Uout1, (2, 2); cj=(false, true))

    T_coef2 = (T_coef1 + T_coef1')/2
    T2 = Grassmann(T_coef2, total_size, even_size)
    Uout2, Λout2, _ = gevd(T2, Dcut)
    Ta2 = contract(Uout2, Λout2, (2, 1))
    Tb2 = contract(Ta2, Uout2, (2, 2); cj=(false, true))
    Tb1 ≈ Tb2
end

function test_GEVD_rank4_non_trunc1(
    total_size, 
    even_size, 
    indsrow::NTuple{N1, Int}, 
    indscol::NTuple{N2, Int}
    ) where {N1, N2}

    T = Grassmann(total_size, even_size, ComplexF64; init=:random, parity=:even)
    U, Λ, _  =  gevd(T, indsrow, indscol, 10000)
    TU = contract(T, U, (indscol, indsrow))
    UΛ = contract(U, Λ, (N1+1, 1))
    TU ≈ UΛ
end

function test_HOSVD_1(
    total_size::NTuple{4, Int}, 
    even_size::NTuple{4, Int})

    # T1[l1, r1, u1, d1]
    T1 = Grassmann(total_size, even_size, ComplexF64; init=:random, parity=:even)
    # T2[l2, r2, u2, d2]
    T2 = copy(T1)
    # Ml[(l1, l2), r1, u1, r2, d2] <-- Ml[l1, r1, u1, l2, r2, d2] = T1[l1, r1, u1, dum] * T2[l2, r2, dum, d2]
    Ml = contract(T1, T2, (4, 3); perm=(1, 4, 2, 3, 5, 6))
    # Mr[l1, u1, l2, d2, (r1, r2)] <-- Mr[l1, r1, u1, l2, r2, d2] = T1[l1, r1, u1, dum] * T2[l2, r2, dum, d2]
    Mr = contract(T1, T2, (4, 3); perm=(1, 3, 4, 6, 2, 5))

    Ul, Sl, _, _ = gsvd(Ml, (1, 2), (3, 4, 5, 6), 10000)
    _, Sr, Vr, _ = gsvd(Mr, (1, 2, 3, 4), (5, 6), 10000)

    Nl = contract(Ml, Ml, ((3, 4, 5, 6), (3, 4, 5, 6)); cj=(false, true))
    Ult, Λlt, _ = gevd(Nl, (1, 2), (3, 4), 10000)
    Nr = contract(Mr, Mr, ((1, 2, 3, 4), (1, 2, 3, 4)); cj=(true, false))
    Vrt, Λrt, _ = gevd(Nr, (1, 2), (3, 4), 10000)

    return Ul, Sl, Vr, Sr, Ult, Λlt, Vrt, Λrt
end

function test_GQR_rank2_1(total_size, even_size)

    T = Grassmann(total_size, even_size, ComplexF64; init=:random, parity=:even)
    Q, R = gortho(T; alg=LinearAlgebra.qr)
    T_test = contract(Q, R, (2, 1))
    T ≈ T_test
end

function test_GQR_rank2_2(total_size, even_size)

    T = Grassmann(total_size, even_size, ComplexF64; init=:random, parity=:even)
    Q, R = gortho(T; alg=LinearAlgebra.qr)
    I_test = contract(Q, Q, (1, 1); cj=(true, false))
    I_test_array = convert2array(I_test)
    I_array = diagm(ones(size(I_test_array)[1]))
    I_test_array ≈ I_array
end

function test_GLQ_rank2_1(total_size, even_size)

    T = Grassmann(total_size, even_size, ComplexF64; init=:random, parity=:even)
    L, Q = gortho(T; alg=LinearAlgebra.lq)
    T_test = contract(L, Q, (2, 1))
    T ≈ T_test
end

function test_GLQ_rank2_2(total_size, even_size)

    T = Grassmann(total_size, even_size, ComplexF64; init=:random, parity=:even)
    L, Q = gortho(T; alg=LinearAlgebra.lq)
    I_test = contract(Q, Q, (1, 1); cj=(true, false))
    I_test_array = convert2array(I_test)
    I_array = diagm(ones(size(I_test_array)[1]))
    I_test_array ≈ I_array
end

function test_GQR_rank4_1(total_size, even_size, indsrow::NTuple{N1, Int64}, indscol::NTuple{N2, Int64}) where {N1, N2}

    T = Grassmann(total_size, even_size, ComplexF64; init=:random, parity=:even)
    Q, R = gortho(T, indsrow, indscol; alg=LinearAlgebra.qr)
    T_test = contract(Q, R, (N1+1, 1))
    T ≈ T_test
end

function test_GLQ_rank4_1(total_size, even_size, indsrow::NTuple{N1, Int64}, indscol::NTuple{N2, Int64}) where {N1, N2}

    T = Grassmann(total_size, even_size, ComplexF64; init=:random, parity=:even)
    L, Q = gortho(T, indsrow, indscol; alg=LinearAlgebra.lq)
    T_test = contract(L, Q, (N1+1, 1))
    T ≈ T_test
end

########################### testing ###########################

@timedtestset "Test Z2 truncation" verbose=true begin
    @timedtestset "Test Dcut > D, no truncation" begin
        De = 10; Do = 10; Dcut = 30
        S, Se = generator(De, Do)
        S_trunc, s_even_trunc, s_odd_trunc = truncation(S, De, Dcut)
        @test s_even_trunc == De
        @test s_odd_trunc == Do
        @test S == S_trunc
    end
    @timedtestset "Test Dcut < D, contain both even and odd-parity states" begin
        De = 10; Do = 10; Dcut = 10
        S, Se = generator(De, Do)
        s_trunc, s_even_trunc, s_odd_trunc = truncation(S, De, Dcut)
        S_sorted = sort(S; rev=true)
        s_trunc_test, s_even_trunc_test, s_odd_trunc_test = truncation(S_sorted, Se, Dcut)
        @test s_even_trunc == s_even_trunc_test
        @test s_odd_trunc == s_odd_trunc_test
        @test s_trunc ≈ s_trunc_test
    end
    @timedtestset "Test Dcut < D, contain only even-parity states" begin
        De = 20; Do = 0; Dcut = 10
        S, Se = generator(De, Do)
        s_trunc, s_even_trunc, s_odd_trunc = truncation(S, De, Dcut)
        s_sorted = sort(Se; rev=true)
        @test s_even_trunc == Dcut
        @test s_odd_trunc == 0
        @test s_trunc ≈ s_sorted[1:Dcut]
    end
end

@timedtestset "Test GSVD on Grassmann matrix" begin
    @timedtestset "Only (even, even) sector exist" begin
        total_size = (16, 24); even_size = (16, 24); Dcut = 12
        Uout_array, Sout_array, Vout_array, Uout_array_test, Sout_array_test, Vout_array_test = test_GSVD_rank2_1(total_size, even_size, Dcut)
        @test Uout_array ≈ Uout_array_test
        @test Sout_array ≈ Sout_array_test
        @test Vout_array ≈ Vout_array_test
    end
    @timedtestset "Only (odd, odd) sector exist" begin
        total_size = (16, 24); even_size = (0, 0); Dcut = 12
        Uout_array, Sout_array, Vout_array, Uout_array_test, Sout_array_test, Vout_array_test = test_GSVD_rank2_1(total_size, even_size, Dcut)
        @test Uout_array ≈ Uout_array_test
        @test Sout_array ≈ Sout_array_test
        @test Vout_array ≈ Vout_array_test
    end
    @timedtestset "Both sectors exist and no truncation" begin
        total_size = (16, 24); even_size = (8, 12); Dcut = 30    
        Uout, Sout, Vout, U_00, S_00, V_00, U_11, S_11, V_11 = test_GSVD_rank2_2(total_size, even_size, Dcut)
        @test U_00 ≈ Uout[(0, 0)]
        @test U_11 ≈ Uout[(1, 1)]
        @test V_00 ≈ Vout[(0, 0)]
        @test V_11 ≈ Vout[(1, 1)]
        @test Diagonal(S_00) ≈ Sout[(0, 0)]
        @test Diagonal(S_11) ≈ Sout[(1, 1)]
    end
    @timedtestset "Test T = USV+ , both sectors exist and no truncation" begin
        total_size = (16, 24); even_size = (8, 12); Dcut = 30
        @test test_GSVD_rank2_3(total_size, even_size, Dcut)
    end
    @timedtestset "Test T = USV+ , only (even, even) sector exist and no truncation" begin
        total_size = (16, 24); even_size = (16, 24); Dcut = 30
        @test test_GSVD_rank2_3(total_size, even_size, Dcut)
    end
    @timedtestset "Test T = USV+ , only (odd, odd) sector exist and no truncation" begin
        total_size = (16, 24); even_size = (0, 0); Dcut = 30
        @test test_GSVD_rank2_3(total_size, even_size, Dcut)
    end
    @timedtestset "Test T = USV+ , both sectors exist and with truncation" begin
        total_size = (16, 24); even_size = (8, 12); Dcut = 8
        @test test_GSVD_rank2_4(total_size, even_size, Dcut)
    end
    @timedtestset "Test T = USV+ , only (even, even) exist and with truncation" begin
        total_size = (16, 24); even_size = (16, 24); Dcut = 8
        @test test_GSVD_rank2_4(total_size, even_size, Dcut)
    end
    @timedtestset "Test T = USV+ , only (odd, odd) exist and with truncation" begin
        total_size = (16, 24); even_size = (0, 0); Dcut = 8
        @test test_GSVD_rank2_4(total_size, even_size, Dcut)
    end
    @timedtestset "Check the unitarity of U and V, both sectors exist and no truncation" begin
        total_size = (16, 24); even_size = (8, 12); Dcut = 30
        min_dim = minimum(total_size)
        I_U_array, I_V_array = test_GSVD_rank2_5(total_size, even_size, Dcut)
        @test I_U_array ≈ Diagonal(ones(min_dim))
        @test I_V_array ≈ Diagonal(ones(min_dim))
    end
    @timedtestset "Check the unitarity of U and V, only (even, even) exist and no truncation" begin
        total_size = (16, 24); even_size = (16, 24); Dcut = 30
        min_dim = minimum(total_size)
        I_U_array, I_V_array = test_GSVD_rank2_5(total_size, even_size, Dcut)
        @test I_U_array ≈ Diagonal(ones(min_dim))
        @test I_V_array ≈ Diagonal(ones(min_dim))
    end
    @timedtestset "Test the unitarity of U and V, both sectors exist and with truncation" begin
        total_size = (16, 24); even_size = (8, 12); Dcut = 12
        I_U1_array, I_V1_array, I_U2_array, I_V2_array = test_GSVD_rank2_6(total_size, even_size, Dcut)
        @test I_U1_array ≈ Diagonal(ones(Dcut))
        @test I_V1_array ≈ Diagonal(ones(Dcut))
        @test !(I_U2_array ≈ Diagonal(ones(total_size[1])))
        @test !(I_V2_array ≈ Diagonal(ones(total_size[2])))
    end
    @timedtestset "Test the unitarity of U and V, only (even, even) sectors exist and with truncation" begin
        total_size = (16, 24); even_size = (16, 24); Dcut = 12
        I_U1_array, I_V1_array, I_U2_array, I_V2_array = test_GSVD_rank2_6(total_size, even_size, Dcut)
        @test I_U1_array ≈ Diagonal(ones(Dcut))
        @test I_V1_array ≈ Diagonal(ones(Dcut))
        @test !(I_U2_array ≈ Diagonal(ones(total_size[1])))
        @test !(I_V2_array ≈ Diagonal(ones(total_size[2])))
    end
    @timedtestset "Test performing SVD truncation afterwards(truncation(S::Grassmann{Float64, 2}, Dcut::Int))" begin
        total_size = (16, 24); even_size = (8, 12); Dcut = 12
        U1, S1, V1, U2, S2, V2 = test_GSVD_rank2_7(total_size, even_size, Dcut)
        @test U1 ≈ U2
        @test S1 ≈ S2
        @test V1 ≈ V2
    end
end

@timedtestset "Test GSVD on rank-4 Grassmann tensor" begin
    @timedtestset "Test using contract and no truncation" begin
        total_size = (8, 8, 12, 12); even_size = (4, 4, 6, 6)
        @test test_GSVD_rank4_1(total_size, even_size, (1, 2), (3, 4))
        @test test_GSVD_rank4_1(total_size, even_size, (1, 2, 3), (4, ))
        @test test_GSVD_rank4_1(total_size, even_size, (1, ), (2, 3, 4, ))
    end
    @timedtestset "Test using contract with auto_sign function and no truncation" begin
        total_size = (8, 8, 10, 6); even_size = (4, 4, 4, 4)
        @test test_GSVD_rank4_1(total_size, even_size, (1, 2), (3, 4); sign_function=auto_sign)
        @test test_GSVD_rank4_1(total_size, even_size, (1, 2, 3), (4, ); sign_function=auto_sign)
        @test test_GSVD_rank4_1(total_size, even_size, (1, ), (2, 3, 4, ); sign_function=auto_sign)
    end
    @timedtestset "Test the unitarity of U and V with no truncation" begin
        total_size = (4, 4, 4, 4); even_size = (2, 2, 2, 2)
        I_U_array, I_V_array = test_GSVD_rank4_2(total_size, even_size, (1, 2), (3, 4))
        @test I_U_array ≈ diagm(ones(16))
        @test I_V_array ≈ diagm(ones(16))
    end
    @timedtestset "Test the unitarity of U and V with truncation" begin
        total_size = (4, 4, 4, 4); even_size = (2, 2, 2, 2); Dcut = 8
        I_U_array, I_V_array = test_GSVD_rank4_3(total_size, even_size, Dcut, (1, 2), (3, 4))
        @test I_U_array ≈ diagm(ones(Dcut))
        @test I_V_array ≈ diagm(ones(Dcut))
    end
end

@timedtestset "Test GSVD on rank-6 Grassmann tensor" begin
    @timedtestset "Test using contract and no truncation" begin
        total_size = (4, 4, 4, 4, 4, 4); even_size = (2, 2, 2, 2, 2, 2)
        @test test_GSVD_rank6_1(total_size, even_size, (1, 2, 3), (4, 5, 6))
        @test test_GSVD_rank6_1(total_size, even_size, (1, 2, ), (3, 4, 5, 6))
        @test test_GSVD_rank6_1(total_size, even_size, (1, ), (2, 3, 4, 5, 6))
        @test test_GSVD_rank6_1(total_size, even_size, (1, 2, 3, 4,), (5, 6))
        @test test_GSVD_rank6_1(total_size, even_size, (1, 2, 3, 4, 5), (6, ))
    end
    @timedtestset "Test using contract with auto_sign function and no truncation" begin
        total_size = (4, 4, 4, 4, 4, 4); even_size = (2, 2, 2, 2, 2, 2)
        @test test_GSVD_rank6_1(total_size, even_size, (1, 2, 3), (4, 5, 6); sign_function=auto_sign)
        @test test_GSVD_rank6_1(total_size, even_size, (1, 2, ), (3, 4, 5, 6); sign_function=auto_sign)
        @test test_GSVD_rank6_1(total_size, even_size, (1, ), (2, 3, 4, 5, 6); sign_function=auto_sign)
        @test test_GSVD_rank6_1(total_size, even_size, (1, 2, 3, 4,), (5, 6); sign_function=auto_sign)
        @test test_GSVD_rank6_1(total_size, even_size, (1, 2, 3, 4, 5), (6, ); sign_function=auto_sign)
    end
    @timedtestset "Test the unitarity of U and V with no truncation" begin
        total_size = (4, 4, 4, 4, 4, 4); even_size = (2, 2, 2, 2, 2, 2)
        I_U_array, I_V_array = test_GSVD_rank6_2(total_size, even_size, (1, 2, 3), (4, 5, 6))
        @test I_U_array ≈ diagm(ones(64))
        @test I_V_array ≈ diagm(ones(64))
    end
    @timedtestset "Test the unitarity of U and V with truncation" begin
        total_size = (4, 4, 4, 4, 4, 4); even_size = (2, 2, 2, 2, 2, 2); Dcut = 16
        I_U_array, I_V_array = test_GSVD_rank6_3(total_size, even_size, Dcut, (1, 2, 3), (4, 5, 6))
        @test I_U_array ≈ diagm(ones(Dcut))
        @test I_V_array ≈ diagm(ones(Dcut))
    end
end

@timedtestset "Test GEVD on Grassmann matrix, no truncation" begin
    @timedtestset "Test TU = UΛ, both sector exist" begin
        total_size = (16, 16); even_size = (8, 8); Dcut = 20
        @test test_GEVD_rank2_non_trunc1(total_size, even_size, Dcut)
    end
    @timedtestset "Test TU = UΛ, only (0, 0) exist" begin
        total_size = (16, 16); even_size = (16, 16); Dcut = 20
        @test test_GEVD_rank2_non_trunc1(total_size, even_size, Dcut)
    end
    @timedtestset "Test TU = UΛ, only (1, 1) exist" begin
        total_size = (16, 16); even_size = (0, 0); Dcut = 20
        @test test_GEVD_rank2_non_trunc1(total_size, even_size, Dcut)
    end
    @timedtestset "Test T = UΛU†, both sector exist" begin
        total_size = (16, 16); even_size = (8, 8); Dcut = 20
        @test test_GEVD_rank2_non_trunc2(total_size, even_size, Dcut)
    end
    @timedtestset "Test T = UΛU†, only (0, 0) sector exist" begin
        total_size = (16, 16); even_size = (16, 16); Dcut = 20
        @test test_GEVD_rank2_non_trunc2(total_size, even_size, Dcut)
    end
    @timedtestset "Test T = UΛU†, only (1, 1) sector exist" begin
        total_size = (16, 16); even_size = (0, 0); Dcut = 20
        @test test_GEVD_rank2_non_trunc2(total_size, even_size, Dcut)
    end
    @timedtestset "Test symflag=true, both sectors exist" begin
        total_size = (16, 16); even_size = (8, 8); Dcut = 20
        @test test_GEVD_rank2_non_trunc3(total_size, even_size, Dcut)
    end
    @timedtestset "Test symflag=true, only (0, 0) exist" begin
        total_size = (16, 16); even_size = (16, 16); Dcut = 20
        @test test_GEVD_rank2_non_trunc3(total_size, even_size, Dcut)
    end
    @timedtestset "Test symflag=true, only (1, 1) exist" begin
        total_size = (16, 16); even_size = (0, 0); Dcut = 20
        @test test_GEVD_rank2_non_trunc3(total_size, even_size, Dcut)
    end
end

@timedtestset "Test GEVD on rank-4 Grassmann tensor, no truncation" begin
    @timedtestset "Test TU = UΛ" begin
        @test test_GEVD_rank4_non_trunc1((4, 4, 4, 4), (2, 2, 2, 2), (1, 2), (3, 4))
        @test test_GEVD_rank4_non_trunc1((4, 4, 4, 4), (4, 4, 4, 4), (1, 2), (3, 4))
        @test test_GEVD_rank4_non_trunc1((4, 4, 4, 4), (0, 0, 0, 0), (1, 2), (3, 4))
        @test test_GEVD_rank4_non_trunc1((4, 4, 4, 4), (4, 0, 4, 0), (1, 2), (3, 4))
        @test test_GEVD_rank4_non_trunc1((4, 4, 4, 4), (0, 4, 0, 4), (1, 2), (3, 4))
    end
end

@timedtestset "Test GHOSVD using GSVD and GEVD" begin
    Ul, Sl, Vr, Sr, Ult, Λlt, Vrt, Λrt = test_HOSVD_1((4, 4, 4, 4), (2, 2, 2, 2))
    Ul_array = convert2array(Ul)
    Sl_array = convert2array(Sl)
    Vr_array = convert2array(Vr)
    Sr_array = convert2array(Sr)
    Ult_array = convert2array(Ult)
    Λlt_array = convert2array(Λlt)
    Vrt_array = convert2array(Vrt)
    Λrt_array = convert2array(Λrt)
    errl1 = (diag(Sl_array).^2 - diag(abs.(Λlt_array))) ./ diag(Sl_array).^2
    errr1 = (diag(Sr_array).^2 - diag(abs.(Λrt_array))) ./ diag(Sr_array).^2
    errl2 = (abs.(Ul_array) - abs.(Ult_array)) 
    errr2 = (abs.(Vr_array) - abs.(Vrt_array)) 
    @test maximum(errl1) < 1e-13
    @test maximum(errr1) < 1e-13
    @test maximum(errl2) < 1e-13
    @test maximum(errr2) < 1e-13
end

@timedtestset "Test GQR/GLQ on Grassmann matrix" begin
    @timedtestset "Test T = QR" begin
        @test test_GQR_rank2_1((8, 8), (4, 4))
        @test test_GQR_rank2_1((8, 8), (8, 8))
        @test test_GQR_rank2_1((8, 8), (0, 0))
    end
    @timedtestset "Test unitarity of Q" begin
        @test test_GQR_rank2_2((8, 8), (4, 4))
        @test test_GQR_rank2_2((8, 8), (8, 8))
        @test test_GQR_rank2_2((8, 8), (0, 0))
    end
    @timedtestset "Test T = LQ" begin
        @test test_GLQ_rank2_1((8, 8), (4, 4))
        @test test_GLQ_rank2_1((8, 8), (8, 8))
        @test test_GLQ_rank2_1((8, 8), (0, 0))
    end
    @timedtestset "Test unitarity of Q" begin
        @test test_GLQ_rank2_2((8, 8), (4, 4))
        @test test_GLQ_rank2_2((8, 8), (8, 8))
        @test test_GLQ_rank2_2((8, 8), (0, 0))
    end
end

@timedtestset "Test GQR/GLQ on rank-4 Grassmann tensor" begin
    @timedtestset "Test T = QR" begin
        @test test_GQR_rank4_1((8, 8, 8, 8), (4, 4, 4, 4), (1, 2), (3, 4))
        @test test_GQR_rank4_1((8, 8, 8, 8), (4, 4, 4, 4), (1, 2, 3), (4, ))
        @test test_GQR_rank4_1((8, 8, 8, 8), (4, 4, 4, 4), (1, ), (2, 3, 4))
    end
    @timedtestset "Test T = LQ" begin
        @test test_GLQ_rank4_1((8, 8, 8, 8), (4, 4, 4, 4), (1, 2), (3, 4))
        @test test_GLQ_rank4_1((8, 8, 8, 8), (4, 4, 4, 4), (1, 2, 3), (4, ))
        @test test_GLQ_rank4_1((8, 8, 8, 8), (4, 4, 4, 4), (1, ), (2, 3, 4))
    end
end

############################## tests AD ##############################


