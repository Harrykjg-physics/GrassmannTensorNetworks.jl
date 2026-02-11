
####################### helper functions #######################

function sign_permutedims!(T)

    # (L, K, J, I) <-- (I, J, K, L)
    # S = p(L) × (p(K) + p(J) + p(I)) + p(K) × (p(J) + p(I)) + p(J) × p(I)
    # T[L, K, J, I]

    data_pairs = nonzero_pairs(T)

    for (inds, block) in data_pairs

        S = inds[1] * (inds[2] + inds[3] + inds[4]) + inds[2] * (inds[3] + inds[4]) + inds[3] * inds[4]

        flag = mod(S, 2)

        if flag == 1
            T[inds] = - block
        end

    end

    return T
end

function sign_conj!(T)

    # (L, K, J, I) <-- (I, J, K, L)
    # S = p(L) × (p(K) + p(J) + p(I)) + p(K) × (p(J) + p(I)) + p(J) × p(I)
    # T[L, K, J, I]

    data_pairs = nonzero_pairs(T)

    for (inds, block) in data_pairs

        S = inds[1] * (inds[2] + inds[3] + inds[4]) + inds[2] * (inds[3] + inds[4]) + inds[3] * inds[4]

        flag = mod(S, 2)

        if flag == 1
            T[inds] = -block
        end

    end

    return T
end

function get_index_test(total_size, even_size, even_dim_trunc, odd_dim_trunc, idx)

    odd_size = total_size .- even_size
    T = Grassmann(total_size, even_size, (:in, :in, :in, :out), ComplexF64; init=:random, parity=:even)
    T_trunc = T[idx, (even_dim_trunc, odd_dim_trunc)]
    T_trunc_array = convert(Array, T_trunc)
    T_array = convert(Array, T)
    index_range_even = [
        (1:even_dim_trunc, :, :, :), (:, 1:even_dim_trunc, :, :), 
        (:, :, 1:even_dim_trunc, :), (:, :, :, 1:even_dim_trunc)]
    index_range_odd = [
        (even_size[1]+1:even_size[1]+odd_dim_trunc, :, :, :), (:, even_size[2]+1:even_size[2]+odd_dim_trunc, :, :), 
        (:, :, even_size[3]+1:even_size[3]+odd_dim_trunc, :), (:, :, :, even_size[4]+1:even_size[4]+odd_dim_trunc)]
    T_array_even_test = T_array[index_range_even[idx]...]
    T_array_odd_test = T_array[index_range_odd[idx]...]
    T_trunc_test = cat(T_array_even_test, T_array_odd_test; dims=idx)

    T_trunc_array ≈ T_trunc_test
end

####################### testing #######################

@timedtestset "test the base.jl" verbose=true begin

    @timedtestset "test the similar and copy function" begin
        total_size = (4, 8, 6, 4); even_size = (2, 4, 3, 2)
        index_types = (:in, :in, :in, :out)
        type = rand([Float64, ComplexF64])
        T = Grassmann(total_size, even_size, index_types, type; init=:random, parity=:even)
        T_sim = similar(T)
        @test size(T_sim) == total_size
        @test even(T_sim) == even_size
        @test eltype(T) == type
        @test index_type(T_sim) == index_types
        @test tensor_parity(T_sim) == tensor_parity(T)
        T_copy = copy(T)
        @test T_copy ≈ T
    end

    @timedtestset "test the +/- function" begin
        total_size = (4, 8, 6, 4); even_size = (2, 4, 3, 2)
        index_types = (:in, :in, :in, :out)
        T1 = Grassmann(total_size, even_size, index_types, ComplexF64; init=:random, parity=:even)
        T2 = Grassmann(total_size, even_size, index_types, ComplexF64; init=:random, parity=:even)
        T1_array = convert(Array, T1)
        T2_array = convert(Array, T2)
        T_array_p = T1_array + T2_array
        T_array_m = T1_array - T2_array
        T_p = T1 + T2
        T_m = T1 - T2
        T_p_array = convert(Array, T_p)
        T_m_array = convert(Array, T_m)
        @test T_array_p ≈ T_p_array
        @test T_array_m ≈ T_m_array
    end

    @timedtestset "test the * function (number multiplication)" begin
        total_size = (4, 8, 6, 4); even_size = (2, 4, 3, 2)
        index_types = (:in, :in, :in, :out)
        T = Grassmann(total_size, even_size, index_types, ComplexF64; init=:random, parity=:even)
        a = rand(1)[]
        T_out1 = T * a
        T_out1_array = convert(Array, T_out1)
        T_out2 = a * T
        T_out2_array = convert(Array, T_out2)
        T_array = convert(Array, T)
        T_array_out = T_array * a
        @test T_out1_array ≈ T_array_out
        @test T_out2_array ≈ T_array_out
    end

    @timedtestset "test the / function (number division)" begin
        total_size = (4, 8, 6, 4); even_size = (2, 4, 3, 2)
        index_types = (:in, :in, :in, :out)
        T = Grassmann(total_size, even_size, index_types, ComplexF64; init=:random, parity=:even)
        a = rand(1)[]
        T_out = T/a
        T_out_array = convert(Array, T_out)
        T_array = convert(Array, T)
        T_array_out = T_array/a
        @test T_out_array ≈ T_array_out
    end

    @timedtestset "test the maximum function" begin
        total_size = (4, 8, 6, 4); even_size = (2, 4, 3, 2)
        index_types = (:in, :in, :in, :out)
        T = Grassmann(total_size, even_size, index_types, Float64; init=:random, parity=:even)
        val1 = maximum(T)
        T_array = convert(Array, T)
        val2 = maximum(T_array)
        @test val1 == val2
    end

    @timedtestset "test the real function" begin
        total_size = (4, 8, 6, 4); even_size = (2, 4, 3, 2)
        index_types = (:in, :in, :in, :out)
        T = Grassmann(total_size, even_size, index_types, ComplexF64; init=:random, parity=:even)
        T_real = real(T)
        T_real_array = convert(Array, T_real)
        T_array = convert(Array, T)
        T_array_real = real(T_array)
        @test T_real_array == T_array_real
    end

    @timedtestset "test the abs function" begin
        total_size = (4, 8, 6, 4); even_size = (2, 4, 3, 2)
        index_types = (:in, :in, :in, :out)
        T = Grassmann(total_size, even_size, index_types, ComplexF64; init=:random, parity=:even)
        T_abs = abs(T)
        T_abs_array = convert(Array, T_abs)
        T_array = convert(Array, T)
        T_array_abs = abs.(T_array)
        @test T_abs_array ≈ T_array_abs
    end

    @timedtestset "test the abs2 function" begin
        total_size = (4, 8, 6, 4); even_size = (2, 4, 3, 2)
        index_types = (:in, :in, :in, :out)
        T = Grassmann(total_size, even_size, index_types, ComplexF64; init=:random, parity=:even)
        T_abs2 = abs2(T)
        T_abs2_array = convert(Array, T_abs2)
        T_array = convert(Array, T)
        T_array_abs2 = abs2.(T_array)
        @test T_abs2_array ≈ T_array_abs2
    end

    @timedtestset "test the sqrt function" begin
        total_size = (4, 8, 6, 4); even_size = (2, 4, 3, 2)
        index_types = (:in, :in, :in, :out)
        T = Grassmann(total_size, even_size, index_types, Float64; init=:random, parity=:even)
        T_sqrt = sqrt(T)
        T_sqrt_array = convert(Array, T_sqrt)
        T_array = convert(Array, T)
        T_array_sqrt = sqrt.(T_array)
        @test T_sqrt_array ≈ T_array_sqrt
    end

    @timedtestset "test the sum function" begin
        total_size = (4, 8, 6, 4); even_size = (2, 4, 3, 2)
        index_types = (:in, :in, :in, :out)
        T = Grassmann(total_size, even_size, index_types, Float64; init=:random, parity=:even)
        T_sum = sum(T)
        T_array = convert(Array, T)
        T_array_sum = sum(T_array)
        @test T_sum ≈ T_array_sum
    end

    @timedtestset "test the sum(f, ) function" begin
        total_size = (4, 8, 6, 4); even_size = (2, 4, 3, 2)
        index_types = (:in, :in, :in, :out)
        T = Grassmann(total_size, even_size, index_types, Float64; init=:random, parity=:even)
        T_sum = sum(abs2, T)
        T_array = convert(Array, T)
        T_array_sum = sum(abs2, T_array)
        @test T_sum ≈ T_array_sum
    end

    @timedtestset "test the permutedims function with trivial fermion sign functions" begin
        total_size = (4, 8, 6, 4); even_size = (2, 4, 3, 2)
        index_types = (:in, :in, :out, :out)
        T = Grassmann(total_size, even_size, index_types, Float64; init=:random, parity=:even)
        dst = Tuple(shuffle(collect(1:4)))
        T_out = permutedims(T, dst)
        T_out_array = convert(Array, T_out)
        T_array = convert(Array, T)
        T_array_test = permutedims(T_array, dst)
        index_types_new = TupleTools.permute(index_types, dst)
        @test T_out_array ≈ T_array_test
        @test index_types_new == index_type(T_out)
    end

    @timedtestset "test the permutedims function with non-trivial fermion sign functions" begin
        total_size = (4, 8, 6, 4); even_size = (2, 4, 3, 2)
        index_types = (:in, :in, :out, :out)
        T = Grassmann(total_size, even_size, index_types, ComplexF64; init=:random, parity=:even)
        T_out = permutedims(T, (4, 3, 2, 1); sign_function=auto_sign)
        T_out_array = convert(Array, T_out)
        T_test = permutedims(T, (4, 3, 2, 1))
        sign_permutedims!(T_test)
        T_test_array = convert(Array, T_test)
        @test T_out_array ≈ T_test_array
        @test (:out, :out, :in, :in) == index_type(T_out)
    end

    @timedtestset "test the conj function with trivial fermion sign functions" begin
        total_size = (4, 8, 6, 4); even_size = (2, 4, 3, 2)
        index_types = (:in, :in, :in, :in)
        T = Grassmann(total_size, even_size, index_types, ComplexF64; init=:random, parity=:even)
        T_conj = conj(T)
        T_conj_array = convert(Array, T_conj)
        T_array = convert(Array, T)
        T_array_conj = conj(T_array)
        @test T_conj_array ≈ T_array_conj
        @test (:out, :out, :out, :out) == index_type(T_conj)
    end

    @timedtestset "test the conj function with non-trivial fermion sign" begin
        total_size = (4, 8, 6, 4); even_size = (2, 4, 3, 2)
        index_types = (:in, :in, :in, :in)
        T = Grassmann(total_size, even_size, index_types, ComplexF64; init=:random, parity=:even)
        T_conj = conj(T; sign_function=auto_sign)
        index_type_conj = index_type(T_conj)
        T_conj_array = convert(Array, T_conj)
        sign_conj!(T)
        T_array = convert(Array, T)
        T_array_conj = conj(T_array)
        @test T_conj_array ≈ T_array_conj
        @test index_type_conj == (:out, :out, :out, :out)
    end

    @timedtestset "test the getindex function for Grassmann tensors" begin
        total_size = (4, 8, 6, 4); even_size = (2, 4, 3, 2); odd_size = total_size .- even_size
        idx = rand(1:length(total_size))
        @timedtestset "even_dim_trunc != 0 && odd_dim_trunc != 0" begin
            @test get_index_test(total_size, even_size, rand(1:even_size[idx]), rand(1:odd_size[idx]), idx)
        end
    end
end

####################### AD tests for base.jl operations #######################

using Zygote

@timedtestset "AD tests for base.jl operations" verbose=true begin

    @timedtestset "AD: maximum" begin
        total_size = (4, 4, 4, 4); even_size = (2, 2, 2, 2)
        index_types = (:in, :in, :out, :out)
        T = Grassmann(total_size, even_size, index_types, Float64; init=:random, parity=:even)
        g = gradient(x -> maximum(x), T)[1]
        g_array = convert(Array, g)
        T_array = convert(Array, T)
        g_test_array = gradient(x -> maximum(x), T_array)[1]

        @test g_array ≈ g_test_array
        # Analytic: gradient is one-hot at argmax position
        best_key = nothing
        best_i = nothing
        best_val = typemin(Float64)
        for (k, v) in nonzero_pairs(T)
            block_max, block_i = findmax(v)
            if block_max > best_val
                best_val = block_max
                best_key = k
                best_i = block_i
            end
        end
        ga = similar(T)
        for (k, v) in nonzero_pairs(ga)
            fill!(v, 0.0)
            if k == best_key
                v[best_i] = 1.0
            end
        end
        @test g ≈ ga
    end

    @timedtestset "AD: abs2" begin
        total_size = (4, 4, 4, 4); even_size = (2, 2, 2, 2)
        index_types = (:in, :in, :out, :out)
        T = Grassmann(total_size, even_size, index_types, Float64; init=:random, parity=:even)
        g = gradient(x -> sum(abs2(x)), T)[1]
        g_array = convert(Array, g)
        T_array = convert(Array, T)
        g_test_array = gradient(x -> sum(abs2.(x)), T_array)[1]

        @test g_array ≈ g_test_array
        # Analytic: d/dx sum(abs2(x)) = 2*conj(x)
        @test g ≈ (2 * conj(T))
    end

    @timedtestset "AD: sum" begin
        total_size = (4, 4, 4, 4); even_size = (2, 2, 2, 2)
        index_types = (:in, :in, :out, :out)
        T = Grassmann(total_size, even_size, index_types, Float64; init=:random, parity=:even)
        g = gradient(x -> sum(x), T)[1]
        # Analytic: gradient of sum(x) is ones
        ga = similar(T)
        for (k, v) in nonzero_pairs(ga)
            fill!(v, 1.0)
        end
        @test g ≈ ga
    end

    @timedtestset "AD: copy" begin
        total_size = (4, 4, 4, 4); even_size = (2, 2, 2, 2)
        index_types = (:in, :in, :out, :out)
        T = Grassmann(total_size, even_size, index_types, Float64; init=:random, parity=:even)
        g = gradient(x -> sum(abs2, copy(x)), T)[1]
        g_array = convert(Array, g)
        T_array = convert(Array, T)
        g_test_array = gradient(x -> sum(abs2, copy(x)), T_array)[1]

        @test g_array ≈ g_test_array
        # Analytic: copy is identity — gradient = 2*T
        @test g ≈ T * 2
    end

    @timedtestset "AD: + (addition)" begin
        total_size = (4, 4, 4, 4); even_size = (2, 2, 2, 2)
        index_types = (:in, :in, :out, :out)
        T1 = Grassmann(total_size, even_size, index_types, Float64; init=:random, parity=:even)
        T2 = Grassmann(total_size, even_size, index_types, Float64; init=:random, parity=:even)

        g1 = gradient(x -> sum(abs2, x + T2), T1)[1]
        g2 = gradient(x -> sum(abs2, T1 + x), T2)[1]
        g1_array = convert(Array, g1)
        g2_array = convert(Array, g2)
        T1_array = convert(Array, T1)
        T2_array = convert(Array, T2)
        g1_test_array = gradient(x -> sum(abs2, x + T2_array), T1_array)[1]
        g2_test_array = gradient(x -> sum(abs2, T1_array + x), T2_array)[1]

        @test g1_array ≈ g1_test_array
        @test g2_array ≈ g2_test_array
        # Analytic: gradient = 2*(T1+T2)
        @test g1 ≈ (T1 + T2) * 2
        @test g2 ≈ (T1 + T2) * 2
    end

    @timedtestset "AD: - (subtraction)" begin
        total_size = (4, 4, 4, 4); even_size = (2, 2, 2, 2)
        index_types = (:in, :in, :out, :out)
        T1 = Grassmann(total_size, even_size, index_types, Float64; init=:random, parity=:even)
        T2 = Grassmann(total_size, even_size, index_types, Float64; init=:random, parity=:even)

        g1 = gradient(x -> sum(abs2, x - T2), T1)[1]
        g2 = gradient(x -> sum(abs2, T1 - x), T2)[1]
        g1_array = convert(Array, g1)
        g2_array = convert(Array, g2)
        T1_array = convert(Array, T1)
        T2_array = convert(Array, T2)
        g1_test_array = gradient(x -> sum(abs2, x - T2_array), T1_array)[1]
        g2_test_array = gradient(x -> sum(abs2, T1_array - x), T2_array)[1]

        @test g1_array ≈ g1_test_array
        @test g2_array ≈ g2_test_array
        # Analytic: g1 = 2*(T1-T2), g2 = -2*(T1-T2)
        @test g1 ≈ (T1 - T2) * 2
        @test g2 ≈ (T1 - T2) * (-2)
    end

    @timedtestset "AD: * (right scalar multiplication)" begin
        total_size = (4, 4, 4, 4); even_size = (2, 2, 2, 2)
        index_types = (:in, :in, :out, :out)
        T = Grassmann(total_size, even_size, index_types, Float64; init=:random, parity=:even)
        a = 2.5

        g_t = gradient(x -> sum(abs2, x * a), T)[1]
        g_array = convert(Array, g_t)
        T_array = convert(Array, T)
        g_test_array = gradient(x -> sum(abs2, x * a), T_array)[1]

        @test g_array ≈ g_test_array
        # Analytic: d/dx sum(abs2, x*a) = 2*a^2*x
        @test g_t ≈ T * (2 * a^2)

        g_s = gradient(s -> sum(abs2, T * s), a)[1]
        g_s_test = gradient(s -> sum(abs2, T_array * s), a)[1]
        @test g_s ≈ g_s_test
    end

    @timedtestset "AD: * (left scalar multiplication)" begin
        total_size = (4, 4, 4, 4); even_size = (2, 2, 2, 2)
        index_types = (:in, :in, :out, :out)
        T = Grassmann(total_size, even_size, index_types, Float64; init=:random, parity=:even)
        a = 3.0

        g_t = gradient(x -> sum(abs2, a * x), T)[1]
        g_array = convert(Array, g_t)
        T_array = convert(Array, T)
        g_test_array = gradient(x -> sum(abs2, a * x), T_array)[1]

        @test g_array ≈ g_test_array
        # Left and right multiply give same gradient for real scalar
        g_t_right = gradient(x -> sum(abs2, x * a), T)[1]
        @test g_t ≈ g_t_right
    end

    @timedtestset "AD: / (scalar division)" begin
        total_size = (4, 4, 4, 4); even_size = (2, 2, 2, 2)
        index_types = (:in, :in, :out, :out)
        T = Grassmann(total_size, even_size, index_types, Float64; init=:random, parity=:even)
        a = 2.5

        g_t = gradient(x -> sum(abs2, x / a), T)[1]
        g_array = convert(Array, g_t)
        T_array = convert(Array, T)
        g_test_array = gradient(x -> sum(abs2, x / a), T_array)[1]

        @test g_array ≈ g_test_array
        # Analytic: d/dx sum(abs2, x/a) = 2*x/a^2
        @test g_t ≈ T * (2 / a^2)

        g_s = gradient(s -> sum(abs2, T / s), a)[1]
        g_s_test = gradient(s -> sum(abs2, T_array / s), a)[1]
        @test g_s ≈ g_s_test
    end

    @timedtestset "AD: real" begin
        total_size = (4, 4, 4, 4); even_size = (2, 2, 2, 2)
        index_types = (:in, :in, :out, :out)
        T = Grassmann(total_size, even_size, index_types, ComplexF64; init=:random, parity=:even)

        g = gradient(x -> sum(abs2, real(x)), T)[1]
        g_array = convert(Array, g)
        T_array = convert(Array, T)
        g_test_array = gradient(x -> sum(abs2, real(x)), T_array)[1]

        @test g_array ≈ g_test_array
        # Analytic: d/dx sum(real(x).^2) = 2*real(x)
        for k in nonzero_keys(T)
            @test g[k] ≈ 2 .* real.(T[k])
        end
    end

    @timedtestset "AD: conj (trivial_sign)" begin
        total_size = (4, 4, 4, 4); even_size = (2, 2, 2, 2)
        index_types = (:in, :in, :out, :out)
        T = Grassmann(total_size, even_size, index_types, Float64; init=:random, parity=:even)

        g = gradient(x -> sum(abs2, conj(x)), T)[1]
        g_array = convert(Array, g)
        T_array = convert(Array, T)
        g_test_array = gradient(x -> sum(abs2, conj(x)), T_array)[1]

        @test g_array ≈ g_test_array
        # For real tensors, conj is identity — gradient = 2*T
        @test g ≈ T * 2
    end

    @timedtestset "AD: permutedims (trivial_sign)" begin
        total_size = (4, 4, 4, 4); even_size = (2, 2, 2, 2)
        index_types = (:in, :in, :out, :out)
        T = Grassmann(total_size, even_size, index_types, Float64; init=:random, parity=:even)
        dst = (2, 3, 4, 1)

        g = gradient(x -> sum(abs2, permutedims(x, dst)), T)[1]
        g_array = convert(Array, g)
        T_array = convert(Array, T)
        g_test_array = gradient(x -> sum(abs2, permutedims(x, dst)), T_array)[1]

        @test g_array ≈ g_test_array
        # Analytic: permutedims gradient is inverse permutedims of output gradient
        out = permutedims(T, dst)
        inv_dst = ntuple(i -> findfirst(==(i), dst), Val(4))
        ga = permutedims(out * 2, inv_dst)
        @test g ≈ ga
    end

    @timedtestset "AD: sqrt" begin
        total_size = (4, 4, 4, 4); even_size = (2, 2, 2, 2)
        index_types = (:in, :in, :out, :out)
        T = Grassmann(total_size, even_size, index_types, Float64; init=:random, parity=:even)
        for (k, v) in nonzero_pairs(T)
            T[k] = abs.(v) .+ 0.1
        end

        g = gradient(x -> sum(sqrt, x), T)[1]
        #g_array = convert(Array, g)
        #T_array = convert(Array, T)
        #g_test_array = gradient(x -> sum(sqrt.(x)), T_array)[1]

        #@test g_array ≈ g_test_array
        # Analytic: d/dx sum(sqrt(x)) = 1/(2*sqrt(x))
        for k in nonzero_keys(T)
            @test g[k] ≈ 1.0 ./ (2.0 .* sqrt.(T[k]))
        end
    end

    @timedtestset "AD: chained operations (copy + permutedims + scalar multiply)" begin
        total_size = (4, 4, 4, 4); even_size = (2, 2, 2, 2)
        index_types = (:in, :in, :out, :out)
        T = Grassmann(total_size, even_size, index_types, Float64; init=:random, parity=:even)
        a = 1.5
        dst = (3, 1, 4, 2)

        g = gradient(x -> sum(abs2, permutedims(copy(x), dst) * a), T)[1]
        g_array = convert(Array, g)
        T_array = convert(Array, T)
        g_test_array = gradient(x -> sum(abs2, permutedims(copy(x), dst) * a), T_array)[1]

        @test g_array ≈ g_test_array
        # Analytic: y = permutedims(x, dst) * a, dL/dy = 2y, g = inv_perm(2a*out)
        out = permutedims(copy(T), dst) * a
        inv_dst = ntuple(i -> findfirst(==(i), dst), Val(4))
        ga = permutedims(out * (2 * a), inv_dst)
        @test g ≈ ga
    end

end
