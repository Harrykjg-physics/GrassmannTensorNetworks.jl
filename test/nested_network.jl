using Test
using GrassmannTensorNetworks

@testset "Nested layout" begin
    layout = NestedLayout((2, 3))
    @test size(layout) == (4, 6)
    @test layout.source_size == (2, 3)
    @test layout.ket_sites[2, 3] == CartesianIndex(3, 5)
    @test layout.y_sites[2, 3] == CartesianIndex(3, 6)
    @test layout.x_sites[2, 3] == CartesianIndex(4, 5)
    @test layout.bra_sites[2, 3] == CartesianIndex(4, 6)
    @test_throws ArgumentError NestedLayout((0, 3))

    peps = Square_GPEPS(2, 1, 2, 1, 2, Float64, false)
    @test NestedLayout(peps).source_size == (1, 2)
end

@testset "Nested network wrapper" begin
    tensor = Grassmann((1, 1, 1, 1), (1, 1, 1, 1), (:in, :out, :in, :out), Float64; init=:zeros)
    network = Matrix{Grassmann{Float64, 4}}(undef, 2, 3)
    fill!(network, tensor)
    nested = NestedNetwork(network, NestedLayout((1, 1)), trues(1, 1))

    @test size(nested) == (2, 3)
    @test size(nested, 2) == 3
    @test axes(nested) == axes(network)
    @test nested[2, 3] === network[2, 3]
end
