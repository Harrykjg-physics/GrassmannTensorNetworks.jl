
############################## test functions #################################

function test_Z2fusion_norm(
    total_size::NTuple{N1, Int}, 
    even_size::NTuple{N1, Int}, 
    index_type::NTuple{N1, Symbol}, 
    p_flag::Symbol, 
    inds::NTuple{N2, Int}) where {N1, N2}

    t = Grassmann(total_size, even_size, index_type, ComplexF64; init=:random, parity=p_flag)
    norm1 = sqrt(sum(abs2, t))
    norm2 = sqrt(sum(abs2, fuse(t, inds)))

    return (norm1 ≈ norm2)
end

function test_Z2fusion_maximum(
    total_size::NTuple{N1, Int}, 
    even_size::NTuple{N1, Int}, 
    index_type::NTuple{N1, Symbol}, 
    p_flag::Symbol, 
    inds::NTuple{N2, Int}) where {N1, N2}

    t = Grassmann(total_size, even_size, index_type, ComplexF64; init=:random, parity=p_flag)
    max1 = maximum(abs(t))
    max2 = maximum(abs((fuse(t, inds))))

    return (max1 ≈ max2)
end

function test_Z2fusion_contract(
    total_size1::NTuple{N1, Int}, 
    even_size1::NTuple{N1, Int}, 
    index_type1::NTuple{N1, Symbol}, 
    p_flag1::Symbol, 
    total_size2::NTuple{N2, Int}, 
    even_size2::NTuple{N2, Int}, 
    index_type2::NTuple{N2, Symbol}, 
    p_flag2::Symbol, 
    contr_inds::Tuple{NTuple{N3, Int}, NTuple{N3, Int}}) where {N1, N2, N3}

    t1 = Grassmann(total_size1, even_size1, index_type1, ComplexF64; init=:random, parity=p_flag1)
    t2 = Grassmann(total_size2, even_size2, index_type2, ComplexF64; init=:random, parity=p_flag2)
    t3 = contract(t1, t2, contr_inds)
    t1_fused = fuse(t1, contr_inds[1])
    t2_fused = fuse(t2, contr_inds[2])
    min_ind1 = minimum(contr_inds[1])
    min_ind2 = minimum(contr_inds[2])
    t3_fused = contract(t1_fused, t2_fused, (min_ind1, min_ind2))

    return (t3 ≈ t3_fused)
end

function test_Z2fusion_Z2split(
    total_size::NTuple{N1, Int}, 
    even_size::NTuple{N1, Int}, 
    index_type::NTuple{N1, Symbol}, 
    p_flag::Symbol, 
    inds::NTuple{N2, Int}) where {N1, N2}

    t = Grassmann(total_size, even_size, index_type, ComplexF64; init=:random, parity=p_flag)
    t_fused = fuse(t, inds)
    min_ind = minimum(inds)
    t_fused_split = split(t_fused, min_ind, total_size, even_size, index_type)

    return (t ≈ t_fused_split)
end

############################## testing fusion.jl ##############################

@timedtestset "Test Z2 fusion / splitting" begin
    @timedtestset "test norm invariance(even-parity)" begin
        @test test_Z2fusion_norm((4, 3, 4, 6, 5), (2, 2, 2, 3, 3), (:out, :out, :in, :in, :in), :even, (2, 3))
        @test test_Z2fusion_norm((4, 3, 4, 6, 5), (2, 2, 2, 3, 3), (:out, :out, :in, :in, :in), :even, (2, 3, 4))
        @test test_Z2fusion_norm((4, 3, 4, 6, 5), (2, 2, 2, 3, 3), (:out, :out, :in, :in, :in), :even, (1, 2, 3, 4))
    end
    @timedtestset "test norm invariance(odd-parity)" begin
        @test test_Z2fusion_norm((4, 3, 4, 6, 5), (2, 2, 2, 3, 3), (:out, :out, :in, :in, :in), :odd, (2, 3))
        @test test_Z2fusion_norm((4, 3, 4, 6, 5), (2, 2, 2, 3, 3), (:out, :out, :in, :in, :in), :odd, (2, 3, 4))
        @test test_Z2fusion_norm((4, 3, 4, 6, 5), (2, 2, 2, 3, 3), (:out, :out, :in, :in, :in), :odd, (1, 2, 3, 4))
    end
    @timedtestset "test maximum invariance(even-parity)" begin
        @test test_Z2fusion_maximum((4, 3, 4, 6, 5), (2, 2, 2, 3, 3), (:out, :out, :in, :in, :in), :even, (2, 3))
        @test test_Z2fusion_maximum((4, 3, 4, 6, 5), (2, 2, 2, 3, 3), (:out, :out, :in, :in, :in), :even, (2, 3, 4))
        @test test_Z2fusion_maximum((4, 3, 4, 6, 5), (2, 2, 2, 3, 3), (:out, :out, :in, :in, :in), :even, (1, 2, 3, 4))
    end
    @timedtestset "test maximum invariance(odd-parity)" begin
        @test test_Z2fusion_maximum((4, 3, 4, 6, 5), (2, 2, 2, 3, 3), (:out, :out, :in, :in, :in), :odd, (2, 3))
        @test test_Z2fusion_maximum((4, 3, 4, 6, 5), (2, 2, 2, 3, 3), (:out, :out, :in, :in, :in), :odd, (2, 3, 4))
        @test test_Z2fusion_maximum((4, 3, 4, 6, 5), (2, 2, 2, 3, 3), (:out, :out, :in, :in, :in), :odd, (1, 2, 3, 4))
    end
    @timedtestset "test contraction invariance (even-even)" begin
        @test test_Z2fusion_contract(
            (4, 3, 4, 6, 5), (2, 2, 2, 3, 3), (:out, :out, :in, :in, :in), :even, 
            (5, 4, 3, 4, 6), (3, 2, 2, 2, 3), (:in, :in, :in, :out, :out), :even,
            ((1, 2), (2, 3))
        )
        @test test_Z2fusion_contract(
            (4, 3, 4, 6, 5), (2, 2, 2, 3, 3), (:out, :out, :in, :in, :in), :even, 
            (5, 4, 3, 4, 6), (3, 2, 2, 2, 3), (:in, :in, :in, :out, :out), :even,
            ((1, 2, 3), (2, 3, 4))
        )
        @test test_Z2fusion_contract(
            (4, 3, 4, 6, 5), (2, 2, 2, 3, 3), (:out, :out, :in, :in, :in), :even, 
            (5, 4, 3, 4, 6), (3, 2, 2, 2, 3), (:in, :in, :in, :out, :out), :even,
            ((1, 2, 3, 4), (2, 3, 4, 5))
        )
    end
    @timedtestset "test contraction invariance (even-odd)" begin
        @test test_Z2fusion_contract(
            (4, 3, 4, 6, 5), (2, 2, 2, 3, 3), (:out, :out, :in, :in, :in), :even, 
            (5, 4, 3, 4, 6), (3, 2, 2, 2, 3), (:in, :in, :in, :out, :out), :odd,
            ((1, 2), (2, 3))
        )
        @test test_Z2fusion_contract(
            (4, 3, 4, 6, 5), (2, 2, 2, 3, 3), (:out, :out, :in, :in, :in), :even, 
            (5, 4, 3, 4, 6), (3, 2, 2, 2, 3), (:in, :in, :in, :out, :out), :odd,
            ((1, 2, 3), (2, 3, 4))
        )
        @test test_Z2fusion_contract(
            (4, 3, 4, 6, 5), (2, 2, 2, 3, 3), (:out, :out, :in, :in, :in), :even, 
            (5, 4, 3, 4, 6), (3, 2, 2, 2, 3), (:in, :in, :in, :out, :out), :odd,
            ((1, 2, 3, 4), (2, 3, 4, 5))
        )
    end
    @timedtestset "test contraction invariance (odd-even)" begin
        @test test_Z2fusion_contract(
            (4, 3, 4, 6, 5), (2, 2, 2, 3, 3), (:out, :out, :in, :in, :in), :odd, 
            (5, 4, 3, 4, 6), (3, 2, 2, 2, 3), (:in, :in, :in, :out, :out), :even,
            ((1, 2), (2, 3))
        )
        @test test_Z2fusion_contract(
            (4, 3, 4, 6, 5), (2, 2, 2, 3, 3), (:out, :out, :in, :in, :in), :odd, 
            (5, 4, 3, 4, 6), (3, 2, 2, 2, 3), (:in, :in, :in, :out, :out), :even,
            ((1, 2, 3), (2, 3, 4))
        )
        @test test_Z2fusion_contract(
            (4, 3, 4, 6, 5), (2, 2, 2, 3, 3), (:out, :out, :in, :in, :in), :odd, 
            (5, 4, 3, 4, 6), (3, 2, 2, 2, 3), (:in, :in, :in, :out, :out), :even,
            ((1, 2, 3, 4), (2, 3, 4, 5))
        )
    end
    @timedtestset "test contraction invariance (odd-odd)" begin
        @test test_Z2fusion_contract(
            (4, 3, 4, 6, 5), (2, 2, 2, 3, 3), (:out, :out, :in, :in, :in), :odd, 
            (5, 4, 3, 4, 6), (3, 2, 2, 2, 3), (:in, :in, :in, :out, :out), :odd,
            ((1, 2), (2, 3))
        )
        @test test_Z2fusion_contract(
            (4, 3, 4, 6, 5), (2, 2, 2, 3, 3), (:out, :out, :in, :in, :in), :odd, 
            (5, 4, 3, 4, 6), (3, 2, 2, 2, 3), (:in, :in, :in, :out, :out), :odd,
            ((1, 2, 3), (2, 3, 4))
        )
        @test test_Z2fusion_contract(
            (4, 3, 4, 6, 5), (2, 2, 2, 3, 3), (:out, :out, :in, :in, :in), :odd, 
            (5, 4, 3, 4, 6), (3, 2, 2, 2, 3), (:in, :in, :in, :out, :out), :odd,
            ((1, 2, 3, 4), (2, 3, 4, 5))
        )
    end
    @timedtestset "test fusion and splitting (even-parity)" begin
        @test test_Z2fusion_Z2split((4, 3, 4, 6, 5), (2, 2, 2, 3, 3), (:out, :out, :in, :in, :in), :even, (2, 3))
        @test test_Z2fusion_Z2split((4, 3, 4, 6, 5), (2, 2, 2, 3, 3), (:out, :out, :in, :in, :in), :even, (2, 3, 4))
        @test test_Z2fusion_Z2split((4, 3, 4, 6, 5), (2, 2, 2, 3, 3), (:out, :out, :in, :in, :in), :even, (1, 2, 3, 4))
    end
    @timedtestset "test fusion and splitting (odd-parity)" begin
        @test test_Z2fusion_Z2split((4, 3, 4, 6, 5), (2, 2, 2, 3, 3), (:out, :out, :in, :in, :in), :odd, (2, 3))
        @test test_Z2fusion_Z2split((4, 3, 4, 6, 5), (2, 2, 2, 3, 3), (:out, :out, :in, :in, :in), :odd, (2, 3, 4))
        @test test_Z2fusion_Z2split((4, 3, 4, 6, 5), (2, 2, 2, 3, 3), (:out, :out, :in, :in, :in), :odd, (1, 2, 3, 4))
    end
end

############################## AD test functions #################################

function test_Z2fusion_ad1(
    total_size::NTuple{N1, Int}, 
    even_size::NTuple{N1, Int}, 
    index_type::NTuple{N1, Symbol}, 
    p_flag::Symbol, 
    inds::NTuple{N2, Int}) where {N1, N2}

    t = Grassmann(total_size, even_size, index_type, ComplexF64; init=:random, parity=p_flag)
    g_fuse = gradient(x -> sum(abs2, fuse(x, inds)), t)[1]
    g_test = gradient(x -> sum(abs2, x), t)[1]

    return (g_fuse ≈ g_test)
end

function test_Z2fusion_ad2(
    total_size1::NTuple{N1, Int}, 
    even_size1::NTuple{N1, Int}, 
    index_type1::NTuple{N1, Symbol}, 
    p_flag1::Symbol, 
    total_size2::NTuple{N2, Int}, 
    even_size2::NTuple{N2, Int}, 
    index_type2::NTuple{N2, Symbol}, 
    p_flag2::Symbol, 
    contr_inds::Tuple{NTuple{N3, Int}, NTuple{N3, Int}}) where {N1, N2, N3}

    t1 = Grassmann(total_size1, even_size1, index_type1, ComplexF64; init=:random, parity=p_flag1)
    t2 = Grassmann(total_size2, even_size2, index_type2, ComplexF64; init=:random, parity=p_flag2)

    g_t1_test = gradient(x -> abs(sum(contract(x, t2, contr_inds))), t1)[1]
    g_t2_test = gradient(x -> abs(sum(contract(t1, x, contr_inds))), t2)[1]

    min_ind1 = minimum(contr_inds[1])
    min_ind2 = minimum(contr_inds[2])

    g_t1 = gradient(x -> abs(sum(contract(fuse(x, contr_inds[1]), fuse(t2, contr_inds[2]), (min_ind1, min_ind2)))), t1)[1]
    g_t2 = gradient(x -> abs(sum(contract(fuse(t1, contr_inds[1]), fuse(x, contr_inds[2]), (min_ind1, min_ind2)))), t2)[1]

    return (g_t1 ≈ g_t1_test) && (g_t2 ≈ g_t2_test)
end

function test_Z2split_ad1(
    total_size::NTuple{N1, Int}, 
    even_size::NTuple{N1, Int}, 
    index_type::NTuple{N1, Symbol}, 
    p_flag::Symbol, 
    inds::NTuple{N2, Int}) where {N1, N2}

    t = Grassmann(total_size, even_size, index_type, ComplexF64; init=:random, parity=p_flag)
    tf = fuse(t, inds)

    min_ind = minimum(inds)
    g_split = gradient(x -> sum(abs2, split(x, min_ind, total_size, even_size, index_type)), tf)[1]
    g_split_test = gradient(x -> sum(abs2, x), tf)[1]
    
    return g_split ≈ g_split_test
end

function test_Z2split_ad2(
    total_size1::NTuple{N1, Int}, 
    even_size1::NTuple{N1, Int}, 
    index_type1::NTuple{N1, Symbol}, 
    p_flag1::Symbol, 
    total_size2::NTuple{N2, Int}, 
    even_size2::NTuple{N2, Int}, 
    index_type2::NTuple{N2, Symbol}, 
    p_flag2::Symbol, 
    contr_inds::Tuple{NTuple{N3, Int}, NTuple{N3, Int}}) where {N1, N2, N3}

    t1 = Grassmann(total_size1, even_size1, index_type1, ComplexF64; init=:random, parity=p_flag1)
    t2 = Grassmann(total_size2, even_size2, index_type2, ComplexF64; init=:random, parity=p_flag2)
    tf1 = fuse(t1, contr_inds[1])
    tf2 = fuse(t2, contr_inds[2])

    min_ind1 = minimum(contr_inds[1])
    min_ind2 = minimum(contr_inds[2])

    g_t1 = gradient(x -> abs(sum(contract(x, tf2, (min_ind1, min_ind2)))), tf1)[1]
    g_t2 = gradient(x -> abs(sum(contract(tf1, x, (min_ind1, min_ind2)))), tf2)[1]

    g_t1_test = gradient(x -> abs(sum(contract(
        split(x, min_ind1, total_size1, even_size1, index_type1), 
        split(tf2, min_ind2, total_size2, even_size2, index_type2), contr_inds))), tf1)[1]

    g_t2_test = gradient(x -> abs(sum(contract(
        split(tf1, min_ind1, total_size1, even_size1, index_type1), 
        split(x, min_ind2, total_size2, even_size2, index_type2), contr_inds))), tf2)[1]

    return (g_t1 ≈ g_t1_test) && (g_t2 ≈ g_t2_test)
end

############################## tests AD ##############################

@timedtestset "AD test: Z2fusion, sum invariance" begin
    @test test_Z2fusion_ad1((4, 3, 4, 6, 5), (2, 2, 2, 3, 3), (:out, :out, :in, :in, :in), :even, (2, 3))
    @test test_Z2fusion_ad1((4, 3, 4, 6, 5), (2, 2, 2, 3, 3), (:out, :out, :in, :in, :in), :odd, (2, 3))
    @test test_Z2fusion_ad1((4, 3, 4, 6, 5), (2, 2, 2, 3, 3), (:out, :out, :in, :in, :in), :even, (2, 3, 4))
    @test test_Z2fusion_ad1((4, 3, 4, 6, 5), (2, 2, 2, 3, 3), (:out, :out, :in, :in, :in), :odd, (2, 3, 4))
    @test test_Z2fusion_ad1((4, 3, 4, 6, 5), (2, 2, 2, 3, 3), (:out, :out, :in, :in, :in), :even, (1, 2, 3, 4))
    @test test_Z2fusion_ad1((4, 3, 4, 6, 5), (2, 2, 2, 3, 3), (:out, :out, :in, :in, :in), :odd, (1, 2, 3, 4))
end

@timedtestset "AD test: Z2fusion, contraction invariance (even-even)" begin
        @test test_Z2fusion_ad2(
            (4, 3, 4, 6, 5), (2, 2, 2, 3, 3), (:out, :out, :in, :in, :in), :even, 
            (5, 4, 3, 4, 6), (3, 2, 2, 2, 3), (:in, :in, :in, :out, :out), :even,
            ((1, 2), (2, 3))
        )
        @test test_Z2fusion_ad2(
            (4, 3, 4, 6, 5), (2, 2, 2, 3, 3), (:out, :out, :in, :in, :in), :even, 
            (5, 4, 3, 4, 6), (3, 2, 2, 2, 3), (:in, :in, :in, :out, :out), :even,
            ((1, 2, 3), (2, 3, 4))
        )
        @test test_Z2fusion_ad2(
            (4, 3, 4, 6, 5), (2, 2, 2, 3, 3), (:out, :out, :in, :in, :in), :even, 
            (5, 4, 3, 4, 6), (3, 2, 2, 2, 3), (:in, :in, :in, :out, :out), :even,
            ((1, 2, 3, 4), (2, 3, 4, 5))
        )
end

@timedtestset "AD test: Z2fusion, contraction invariance (even-odd)" begin
        @test test_Z2fusion_ad2(
            (4, 3, 4, 6, 5), (2, 2, 2, 3, 3), (:out, :out, :in, :in, :in), :even, 
            (5, 4, 3, 4, 6), (3, 2, 2, 2, 3), (:in, :in, :in, :out, :out), :odd,
            ((1, 2), (2, 3))
        )
        @test test_Z2fusion_ad2(
            (4, 3, 4, 6, 5), (2, 2, 2, 3, 3), (:out, :out, :in, :in, :in), :even, 
            (5, 4, 3, 4, 6), (3, 2, 2, 2, 3), (:in, :in, :in, :out, :out), :odd,
            ((1, 2, 3), (2, 3, 4))
        )
        @test test_Z2fusion_ad2(
            (4, 3, 4, 6, 5), (2, 2, 2, 3, 3), (:out, :out, :in, :in, :in), :even, 
            (5, 4, 3, 4, 6), (3, 2, 2, 2, 3), (:in, :in, :in, :out, :out), :odd,
            ((1, 2, 3, 4), (2, 3, 4, 5))
        )
end

@timedtestset "AD test: Z2fusion, contraction invariance (odd-odd)" begin
        @test test_Z2fusion_ad2(
            (4, 3, 4, 6, 5), (2, 2, 2, 3, 3), (:out, :out, :in, :in, :in), :odd, 
            (5, 4, 3, 4, 6), (3, 2, 2, 2, 3), (:in, :in, :in, :out, :out), :odd,
            ((1, 2), (2, 3))
        )
        @test test_Z2fusion_ad2(
            (4, 3, 4, 6, 5), (2, 2, 2, 3, 3), (:out, :out, :in, :in, :in), :odd, 
            (5, 4, 3, 4, 6), (3, 2, 2, 2, 3), (:in, :in, :in, :out, :out), :odd,
            ((1, 2, 3), (2, 3, 4))
        )
        @test test_Z2fusion_ad2(
            (4, 3, 4, 6, 5), (2, 2, 2, 3, 3), (:out, :out, :in, :in, :in), :odd, 
            (5, 4, 3, 4, 6), (3, 2, 2, 2, 3), (:in, :in, :in, :out, :out), :odd,
            ((1, 2, 3, 4), (2, 3, 4, 5))
        )
end

@timedtestset "AD test: Z2split, sum invariance" begin
    @test test_Z2split_ad1((4, 3, 4, 6, 5), (2, 2, 2, 3, 3), (:out, :out, :in, :in, :in), :even, (2, 3))
    @test test_Z2split_ad1((4, 3, 4, 6, 5), (2, 2, 2, 3, 3), (:out, :out, :in, :in, :in), :odd, (2, 3))
    @test test_Z2split_ad1((4, 3, 4, 6, 5), (2, 2, 2, 3, 3), (:out, :out, :in, :in, :in), :even, (2, 3, 4))
    @test test_Z2split_ad1((4, 3, 4, 6, 5), (2, 2, 2, 3, 3), (:out, :out, :in, :in, :in), :odd, (2, 3, 4))
    @test test_Z2split_ad1((4, 3, 4, 6, 5), (2, 2, 2, 3, 3), (:out, :out, :in, :in, :in), :even, (1, 2, 3, 4))
    @test test_Z2split_ad1((4, 3, 4, 6, 5), (2, 2, 2, 3, 3), (:out, :out, :in, :in, :in), :odd, (1, 2, 3, 4))
end

@timedtestset "AD test: Z2split, contraction invariance (even-even)" begin
        @test test_Z2split_ad2(
            (4, 3, 4, 6, 5), (2, 2, 2, 3, 3), (:out, :out, :in, :in, :in), :even, 
            (5, 4, 3, 4, 6), (3, 2, 2, 2, 3), (:in, :in, :in, :out, :out), :even,
            ((1, 2), (2, 3))
        )
        @test test_Z2split_ad2(
            (4, 3, 4, 6, 5), (2, 2, 2, 3, 3), (:out, :out, :in, :in, :in), :even, 
            (5, 4, 3, 4, 6), (3, 2, 2, 2, 3), (:in, :in, :in, :out, :out), :even,
            ((1, 2, 3), (2, 3, 4))
        )
        @test test_Z2split_ad2(
            (4, 3, 4, 6, 5), (2, 2, 2, 3, 3), (:out, :out, :in, :in, :in), :even, 
            (5, 4, 3, 4, 6), (3, 2, 2, 2, 3), (:in, :in, :in, :out, :out), :even,
            ((1, 2, 3, 4), (2, 3, 4, 5))
        )
end

@timedtestset "AD test: Z2split, contraction invariance (even-odd)" begin
        @test test_Z2split_ad2(
            (4, 3, 4, 6, 5), (2, 2, 2, 3, 3), (:out, :out, :in, :in, :in), :even, 
            (5, 4, 3, 4, 6), (3, 2, 2, 2, 3), (:in, :in, :in, :out, :out), :odd,
            ((1, 2), (2, 3))
        )
        @test test_Z2split_ad2(
            (4, 3, 4, 6, 5), (2, 2, 2, 3, 3), (:out, :out, :in, :in, :in), :even, 
            (5, 4, 3, 4, 6), (3, 2, 2, 2, 3), (:in, :in, :in, :out, :out), :odd,
            ((1, 2, 3), (2, 3, 4))
        )
        @test test_Z2split_ad2(
            (4, 3, 4, 6, 5), (2, 2, 2, 3, 3), (:out, :out, :in, :in, :in), :even, 
            (5, 4, 3, 4, 6), (3, 2, 2, 2, 3), (:in, :in, :in, :out, :out), :odd,
            ((1, 2, 3, 4), (2, 3, 4, 5))
        )
end

@timedtestset "AD test: Z2split, contraction invariance (odd-odd)" begin
        @test test_Z2split_ad2(
            (4, 3, 4, 6, 5), (2, 2, 2, 3, 3), (:out, :out, :in, :in, :in), :odd, 
            (5, 4, 3, 4, 6), (3, 2, 2, 2, 3), (:in, :in, :in, :out, :out), :odd,
            ((1, 2), (2, 3))
        )
        @test test_Z2split_ad2(
            (4, 3, 4, 6, 5), (2, 2, 2, 3, 3), (:out, :out, :in, :in, :in), :odd, 
            (5, 4, 3, 4, 6), (3, 2, 2, 2, 3), (:in, :in, :in, :out, :out), :odd,
            ((1, 2, 3), (2, 3, 4))
        )
        @test test_Z2split_ad2(
            (4, 3, 4, 6, 5), (2, 2, 2, 3, 3), (:out, :out, :in, :in, :in), :odd, 
            (5, 4, 3, 4, 6), (3, 2, 2, 2, 3), (:in, :in, :in, :out, :out), :odd,
            ((1, 2, 3, 4), (2, 3, 4, 5))
        )
end
