using Test
using Random
using GrassmannTensorNetworks

function direct_bond_measurement_fixture()
    Random.seed!(0x48424f4e44)
    peps = Square_GPEPS(2, 1, 2, 2, 2, Float64, false)
    Tbulk = reduced_tensor(peps)
    H_bond = nn_bond(SpinlessFermionModel(1.0, 0.7, 0.3))
    env = CTMRGEnv(Tbulk, 4, 2)
    T_vbond_imp, T_hbond_imp = reduced_tensor(peps, H_bond)

    return peps, Tbulk, H_bond, env, T_hbond_imp, T_vbond_imp
end

@testset "Direct bond measurements match explicit bond impurities" begin
    peps, Tbulk, H_bond, env, T_hbond_imp, T_vbond_imp =
    direct_bond_measurement_fixture()

    Z_h_explicit, Eh_explicit = compute_exp_hbond(Tbulk, T_hbond_imp, env)
    Z_v_explicit, Ev_explicit = compute_exp_vbond(Tbulk, T_vbond_imp, env)

    Z_h_direct, Eh_direct = compute_exp_hbond(Tbulk, peps, H_bond, env)
    Z_v_direct, Ev_direct = compute_exp_vbond(Tbulk, peps, H_bond, env)

    @test Z_h_direct ≈ Z_h_explicit rtol=1e-10 atol=1e-10
    @test Eh_direct ≈ Eh_explicit rtol=1e-10 atol=1e-10
    @test Z_v_direct ≈ Z_v_explicit rtol=1e-10 atol=1e-10
    @test Ev_direct ≈ Ev_explicit rtol=1e-10 atol=1e-10
end
