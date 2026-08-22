@testset "Model helpers" begin
    @test isdefined(GrassmannTensorNetworks, :InteractingSpinlessFermion)

    model = InteractingSpinlessFermion(2, 3.0, 5)
    @test model.t === 2.0
    @test model.μ === 3.0
    @test model.V === 5.0

    @test GrassmannTensorNetworks.n_site_Fock_basis(model) == [0.0 0.0; 0.0 1.0]

    H = GrassmannTensorNetworks.nn_bond_Fock_basis(model)
    expected = zeros(Float64, 2, 2, 2, 2)
    expected[2, 1, 1, 2] = -2.0
    expected[1, 2, 2, 1] = -2.0
    expected[2, 1, 2, 1] = -3.0 / 4
    expected[1, 2, 1, 2] = -3.0 / 4
    expected[2, 2, 2, 2] = 5.0 - 3.0 / 2
    @test H == expected

    @test size(n_site(model)) == (2, 2)
    @test size(nn_bond(model)) == (2, 2, 2, 2)
    @test size(gate(model, 0.1)) == (2, 2, 2, 2)
end

@testset "t-J model helpers" begin
    @test isdefined(GrassmannTensorNetworks, :TJModel)

    model = TJModel(2, 6.0, 4)
    @test model.t === 2.0
    @test model.J === 6.0
    @test model.μ === 4.0

    @test GrassmannTensorNetworks.n_site_Fock_basis(model) == [
        0.0 0.0 0.0
        0.0 1.0 0.0
        0.0 0.0 1.0
    ]

    H = GrassmannTensorNetworks.nn_bond_Fock_basis(model)
    expected = zeros(Float64, 3, 3, 3, 3)

    expected[2, 1, 1, 2] = -2.0
    expected[1, 2, 2, 1] = -2.0
    expected[3, 1, 1, 3] = -2.0
    expected[1, 3, 3, 1] = -2.0

    expected[2, 1, 2, 1] = -1.0
    expected[3, 1, 3, 1] = -1.0
    expected[1, 2, 1, 2] = -1.0
    expected[1, 3, 1, 3] = -1.0

    expected[2, 2, 2, 2] = -2.0
    expected[3, 3, 3, 3] = -2.0
    expected[2, 3, 2, 3] = -5.0
    expected[3, 2, 3, 2] = -5.0
    expected[3, 2, 2, 3] = 3.0
    expected[2, 3, 3, 2] = 3.0
    @test H == expected

    @test size(n_site(model)) == (3, 3)
    @test size(nn_bond(model)) == (3, 3, 3, 3)
    @test size(gate(model, 0.1)) == (3, 3, 3, 3)
end
