
@timedtestset "test the linalg.jl" begin
    @timedtestset "test the norm function" begin
        T = Grassmann((4, 8), (2, 4), (:in, :out), ComplexF64; init=:random, parity=:even)
        norm1 = norm(T)
        T_array = convert(Array, T)
        norm2 = norm(T_array)
        @test norm1 ≈ norm2
    end
    @timedtestset "test the diag function 1" begin
        T = Grassmann((16, 16), (8, 8), (:in, :out), ComplexF64; init=:random, parity=:even)
        T_diag = diag(T)
        T_array = convert(Array, T)
        T_array_diag = diag(T_array)
        @test T_diag ≈ T_array_diag
    end
    @timedtestset "test the diag function 2" begin
        T = Grassmann((16, 16), (0, 0), (:in, :out), ComplexF64; init=:random, parity=:even)
        T_diag = diag(T)
        T_array = convert(Array, T)
        T_array_diag = diag(T_array)
        @test T_diag ≈ T_array_diag
    end
    @timedtestset "test the transpose function" begin
        T = Grassmann((16, 16), (8, 8), (:in, :out), ComplexF64; init=:random, parity=:even)
        T_transpose = transpose(T)
        T_transpose_array = convert(Array, T_transpose)
        T_array = convert(Array, T)
        T_array_transpose = transpose(T_array)
        @test T_transpose_array ≈ T_array_transpose
        @test index_type(T_transpose) == (:out, :in)
    end
    @timedtestset "test the transpose function with non-trivial sign functions" begin
        T = Grassmann((4, 8), (2, 4), (:in, :out), ComplexF64; init=:random, parity=:even)
        T_transpose = transpose(T; sign_function=auto_sign)
        T_transpose_array = convert(Array, T_transpose)
        T00_t = transpose(T[(0, 0)])
        T11_t = transpose(T[(1, 1)])
        @test T00_t ≈ T_transpose[(0, 0)]
        @test T11_t ≈ - T_transpose[(1, 1)]
    end
    @timedtestset "test the inv function" begin
        T = Grassmann((16, 16), (8, 8), (:in, :out), ComplexF64; init=:random, parity=:even)
        T_inv = inv(T)
        T_inv_array = convert(Array, T_inv)
        T_array = convert(Array, T)
        T_array_inv = inv(T_array)
        @test T_inv_array ≈ T_array_inv
    end
    @timedtestset "test the dot function" begin
        tot_size = (4, 8, 4); even_size = (2, 4, 2)
        T1 = Grassmann(tot_size, even_size, (:out, :in, :in), ComplexF64; init=:random, parity=:even)
        T2 = Grassmann(tot_size, even_size, (:out, :in, :in), ComplexF64; init=:random, parity=:even)
        dot_res1 = dot(T1, T2)
        T1_array = convert(Array, T1)
        T2_array = convert(Array, T2)
        dot_res2 = dot(T1_array, T2_array)
        @test dot_res1 ≈ dot_res2
    end
end

# ------------------------------------- ad tests -------------------------------------

function linalg_basis_like(A::Grassmann{T, N}, sector::NTuple{N, Int}, index::CartesianIndex{N}) where {T, N}

    data_dict = Dict{NTuple{N, Int}, Array{T, N}}()
    for (current_sector, block) in nonzero_pairs(A)
        data_dict[current_sector] = zeros(T, size(block))
    end
    data_dict[sector][index] = one(T)

    return Grassmann(size(A), even(A), index_type(A), data_dict)
end

function finite_difference_linalg_gradient(f, A::Grassmann{T, N}) where {T, N}

    fdm = central_fdm(5, 1)
    data_dict = Dict{NTuple{N, Int}, Array{T, N}}()

    for (sector, block) in nonzero_pairs(A)
        gradient = zeros(T, size(block))
        for index in CartesianIndices(block)
            basis = linalg_basis_like(A, sector, index)
            gradient[index] = fdm(step -> f(A + step * basis), zero(T))
        end
        data_dict[sector] = gradient
    end

    return Grassmann(size(A), even(A), index_type(A), data_dict)
end

function positive_definite_grassmann_matrix()

    A = Grassmann((4, 4), (2, 2), (:out, :in), Float64; init=:random, parity=:even)
    data_dict = Dict{NTuple{2, Int}, Matrix{Float64}}()
    for (sector, block) in nonzero_pairs(A)
        data_dict[sector] = block * block' + 2.0I
    end

    return Grassmann(size(A), even(A), index_type(A), data_dict)
end

function test_linalg_gradient(f, A; atol=1e-6, rtol=1e-6)

    gradient_ad = gradient(f, A)[1]
    gradient_fd = finite_difference_linalg_gradient(f, A)
    return isapprox(gradient_ad, gradient_fd; atol=atol, rtol=rtol)
end

@testset "linear algebra AD" begin

    A = positive_definite_grassmann_matrix()
    B = Grassmann((4, 4), (2, 2), (:out, :in), Float64; init=:random, parity=:even)

    @testset "norm" begin
        @test test_linalg_gradient(norm, A)
        @test test_linalg_gradient(x -> norm(x, 3), A)
    end

    @testset "diag" begin
        @test test_linalg_gradient(x -> sum(abs2, diag(x)), A)
    end

    @testset "transpose" begin
        @test test_linalg_gradient(x -> sum(transpose(x)), A)
        @test test_linalg_gradient(x -> sum(transpose(x; sign_function=auto_sign)), A)
    end

    @testset "inv" begin
        @test test_linalg_gradient(x -> sum(inv(x)), A)
    end

    @testset "dot" begin
        @test test_linalg_gradient(x -> dot(x, B), A)
    end

    @testset "log" begin
        @test test_linalg_gradient(x -> sum(abs2, log(x)), A; atol=2e-5, rtol=2e-5)
    end
end
