
####################### testing #######################

@timedtestset "test the grassmann.jl" begin
    @timedtestset "test basic methods for the Grassman tensor" begin
        total_size = (4, 8, 6, 4)
        even_size = (2, 4, 3, 2)
        index_types = (:in, :in, :in, :out)
        type_list = [Int64, Float64, ComplexF64]
        eltype_choose = rand(type_list)
        parity = [:even, :odd]
        parity_choose = rand(parity)
        T = Grassmann(total_size, even_size, index_types, eltype_choose; init=:random, parity=parity_choose)
        @test size(T) == total_size
        @test even(T) == even_size
        @test data(T) == T.data
        @test eltype(T) == eltype_choose
        @test index_type(T) == index_types
        @test tensor_parity(T) == (parity_choose == :even ? 0 : 1)
        @test tensor_rank(T) == length(total_size)
    end
    @timedtestset "test index type conjugation" begin
        total_size = (4, 8, 6, 4)
        even_size = (2, 4, 3, 2)
        index_types = (:in, :out, :in, :out)
        T = Grassmann(total_size, even_size, index_types, Float64; init=:random, parity=:odd)
        T_new1 = index_conjugation(T, 2)
        T_new2 = index_conjugation(T, (3, 4))
        @test index_type(T_new1) == (:in, :in, :in, :out)
        @test index_type(T_new2) == (:in, :out, :out, :in)
    end
    @timedtestset "test convert function" begin
        total_size = (4, 8, 6, 4)
        even_size = (2, 4, 3, 2)
        index_types = (:in, :in, :in, :out)
        T = Grassmann(total_size, even_size, index_types, Float64; init=:random, parity=:even)
        Tc = convert(T, ComplexF64)
        @test eltype(Tc) == ComplexF64
        @test index_type(T) == index_type(Tc)
        @test data(T) == data(Tc)
    end
end
