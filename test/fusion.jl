
############################## test functions #################################

function test_Z2fusion_norm(
    total_size::NTuple{N1, Int}, 
    even_size::NTuple{N1, Int}, 
    index_type::NTuple{N1, Symbol}, 
    p_flag::Symbol, 
    inds::NTuple{N2, Int}
    ) where {N1, N2}

    t = Grassmann(total_size, even_size, index_type, ComplexF64; init=:random, parity=p_flag)
    norm1 = norm(t)
    norm2 = norm(fuse(t, inds))

    return (norm1 ≈ norm2)
end

############################## AD and inference tests ##############################

using Zygote

@timedtestset "test type inference for fusion" begin verbose=true
    @test (@inferred calculate_sectors((4, 3), (2, 2))) isa Vector{NTuple{2, Int}}
    @test (@inferred calculate_fused_size((4, 3), (2, 2))) == (12, 6)

    total_size = (4, 3, 4, 6, 5)
    even_size = (2, 2, 2, 3, 3)
    index_type_in = (:out, :out, :in, :in, :in)
    t = Grassmann(total_size, even_size, index_type_in, Float64; init=:random, parity=:even)

    fused_info = @inferred prepare_fused_info(total_size, even_size, index_type_in, :in, (2, 3))
    @test fused_info == ((4, 12, 6, 5), (2, 6, 3, 3), (:out, :in, :in, :in))

    t_fused = @inferred fuse(t, (2, 3))
    @test t_fused isa Grassmann{Float64, 4}

    t_split = @inferred split(t_fused, 2, total_size, even_size, index_type_in)
    @test t_split isa Grassmann{Float64, 5}
end

@timedtestset "test error branches for fusion/splitting" begin verbose=true
    total_size = (4, 3, 4, 6, 5)
    even_size = (2, 2, 2, 3, 3)
    index_type_in = (:out, :out, :in, :in, :in)
    t = Grassmann(total_size, even_size, index_type_in, Float64; init=:random, parity=:even)

    @test_throws ArgumentError fuse(t, (1, 3))
    @test_throws ArgumentError fuse(t, (3, 2))

    t_fused = fuse(t, (2, 3))
    wrong_total = (4, 4, 4, 6, 5)
    wrong_even = (2, 1, 2, 3, 3)
    wrong_index_type = (:out, :in, :in, :in, :in)
    @test_throws ArgumentError split(t_fused, 2, wrong_total, even_size, index_type_in)
    @test_throws ArgumentError split(t_fused, 2, total_size, wrong_even, index_type_in)
    @test_throws ArgumentError split(t_fused, 2, total_size, even_size, wrong_index_type)
    @test_throws ArgumentError split(t_fused, 2, (4, 3, 6, 5), (2, 2, 3, 3), (:out, :in, :in, :in))
end

@timedtestset "AD test: fusion/splitting" begin verbose=true
    total_size = (4, 3, 4, 6, 5)
    even_size = (2, 2, 2, 3, 3)
    index_type_in = (:out, :out, :in, :in, :in)
    t = Grassmann(total_size, even_size, index_type_in, Float64; init=:random, parity=:odd)

    loss_direct = x -> real(sum(abs2, convert2array(x)))
    loss_fuse = x -> real(sum(abs2, convert2array(fuse(x, (2, 3)))))

    g_direct = gradient(loss_direct, t)[1]
    g_fuse = gradient(loss_fuse, t)[1]
    @test g_fuse ≈ g_direct

    t_fused = fuse(t, (2, 3))
    loss_split = x -> real(sum(abs2, convert2array(split(x, 2, total_size, even_size, index_type_in))))
    loss_fused_direct = x -> real(sum(abs2, convert2array(x)))

    g_fused_direct = gradient(loss_fused_direct, t_fused)[1]
    g_split = gradient(loss_split, t_fused)[1]
    @test g_split ≈ g_fused_direct

    loss_roundtrip = x -> real(sum(abs2, convert2array(split(fuse(x, (2, 3)), 2, size(x), even(x), index_type(x)))))
    g_roundtrip = gradient(loss_roundtrip, t)[1]
    @test g_roundtrip ≈ g_direct
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
    contr_inds::Tuple{NTuple{N3, Int}, NTuple{N3, Int}}
    ) where {N1, N2, N3}

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

@timedtestset "Test Z2 fusion / splitting" begin verbose=true
    @timedtestset "test norm invariance(even-parity)" begin verbose=true
        @test test_Z2fusion_norm((4, 3, 4, 6, 5), (2, 2, 2, 3, 3), (:out, :out, :in, :in, :in), :even, (2, 3))
        @test test_Z2fusion_norm((4, 3, 4, 6, 5), (2, 2, 2, 3, 3), (:out, :out, :in, :in, :in), :even, (2, 3, 4))
        @test test_Z2fusion_norm((4, 3, 4, 6, 5), (2, 2, 2, 3, 3), (:out, :out, :in, :in, :in), :even, (1, 2, 3, 4))
    end
    @timedtestset "test norm invariance(odd-parity)" begin verbose=true
        @test test_Z2fusion_norm((4, 3, 4, 6, 5), (2, 2, 2, 3, 3), (:out, :out, :in, :in, :in), :odd, (2, 3))
        @test test_Z2fusion_norm((4, 3, 4, 6, 5), (2, 2, 2, 3, 3), (:out, :out, :in, :in, :in), :odd, (2, 3, 4))
        @test test_Z2fusion_norm((4, 3, 4, 6, 5), (2, 2, 2, 3, 3), (:out, :out, :in, :in, :in), :odd, (1, 2, 3, 4))
    end
    @timedtestset "test maximum invariance(even-parity)" begin verbose=true
        @test test_Z2fusion_maximum((4, 3, 4, 6, 5), (2, 2, 2, 3, 3), (:out, :out, :in, :in, :in), :even, (2, 3))
        @test test_Z2fusion_maximum((4, 3, 4, 6, 5), (2, 2, 2, 3, 3), (:out, :out, :in, :in, :in), :even, (2, 3, 4))
        @test test_Z2fusion_maximum((4, 3, 4, 6, 5), (2, 2, 2, 3, 3), (:out, :out, :in, :in, :in), :even, (1, 2, 3, 4))
    end
    @timedtestset "test maximum invariance(odd-parity)" begin verbose=true
        @test test_Z2fusion_maximum((4, 3, 4, 6, 5), (2, 2, 2, 3, 3), (:out, :out, :in, :in, :in), :odd, (2, 3))
        @test test_Z2fusion_maximum((4, 3, 4, 6, 5), (2, 2, 2, 3, 3), (:out, :out, :in, :in, :in), :odd, (2, 3, 4))
        @test test_Z2fusion_maximum((4, 3, 4, 6, 5), (2, 2, 2, 3, 3), (:out, :out, :in, :in, :in), :odd, (1, 2, 3, 4))
    end
    @timedtestset "test contraction invariance (even-even)" begin verbose=true
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
    @timedtestset "test contraction invariance (even-odd)" begin verbose=true
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
    @timedtestset "test contraction invariance (odd-even)" begin verbose=true
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
    @timedtestset "test contraction invariance (odd-odd)" begin verbose=true
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
    @timedtestset "test fusion and splitting (even-parity)" begin verbose=true
        @test test_Z2fusion_Z2split((4, 3, 4, 6, 5), (2, 2, 2, 3, 3), (:out, :out, :in, :in, :in), :even, (2, 3))
        @test test_Z2fusion_Z2split((4, 3, 4, 6, 5), (2, 2, 2, 3, 3), (:out, :out, :in, :in, :in), :even, (2, 3, 4))
        @test test_Z2fusion_Z2split((4, 3, 4, 6, 5), (2, 2, 2, 3, 3), (:out, :out, :in, :in, :in), :even, (1, 2, 3, 4))
    end
    @timedtestset "test fusion and splitting (odd-parity)" begin verbose=true
        @test test_Z2fusion_Z2split((4, 3, 4, 6, 5), (2, 2, 2, 3, 3), (:out, :out, :in, :in, :in), :odd, (2, 3))
        @test test_Z2fusion_Z2split((4, 3, 4, 6, 5), (2, 2, 2, 3, 3), (:out, :out, :in, :in, :in), :odd, (2, 3, 4))
        @test test_Z2fusion_Z2split((4, 3, 4, 6, 5), (2, 2, 2, 3, 3), (:out, :out, :in, :in, :in), :odd, (1, 2, 3, 4))
    end
end

@timedtestset "AD test: fusion and split" begin verbose=true
    total_size = (4, 3, 4, 6, 5)
    even_size = (2, 2, 2, 3, 3)
    index_types = (:out, :out, :in, :in, :in)
    inds = (2, 3)
    min_ind = minimum(inds)
    t = Grassmann(total_size, even_size, index_types, Float64; init=:random, parity=:even)
    tf = fuse(t, inds)

    g_fuse = gradient(x -> sum(abs2, fuse(x, inds)), t)[1]
    g_ref = gradient(x -> sum(abs2, x), t)[1]
    @test g_fuse ≈ g_ref

    g_split = gradient(
        x -> sum(abs2, split(x, min_ind, total_size, even_size, index_types)),
        tf,
    )[1]
    g_split_ref = gradient(x -> sum(abs2, x), tf)[1]
    @test g_split ≈ g_split_ref
end

