using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

using GrassmannTensorNetworks

function run_CTMRG_Square_Hubbard(
    t::Float64, 
    U::Float64, 
    μ::Float64,
    peps_filename::String,
    peps_param_str::String,  
    χ::Int, 
    ctmrg_iter::Int; 
    load_env::String="random")

    ##################### Load the Square GPEPS from the ground state optimization #####################

    wpeps = load(peps_filename, peps_param_str, Square_GPEPS)
    peps = absorb_Schmidt_weights(wpeps)

    ##################### Introduce the bond Hamiltonian  #####################

    model = HubbardModel(t, U, μ)
    H_nn_bond = nn_bond(model)

    ##################### Construct reduced tensors and impurity tensors from Square GPEPS #####################

    T_square_mat = reduced_tensor(peps)
    T_x_bond_imp_mat, T_y_bond_imp_mat = reduced_tensor(peps, H_nn_bond)

    ##################### Running Grassmann CTMRG to compute environment tensors #####################

    load_env == "random" ? ctmrg_env = CTMRGEnv(T_square_mat, χ, Int(χ/2)) : ctmrg_env = load("ctmrg_env", load_env, CTMRGEnv)

    run_GCTMRG!(T_square_mat, ctmrg_env, χ; 
    ctmrg_iter=ctmrg_iter, ctmrg_tol=1e-12, average_trunc=true, 
    verbosity=1, save_iter=20, save_filename="ctmrg_env")

    _, Eh = compute_exp_hbond(T_square_mat, T_y_bond_imp_mat, ctmrg_env)
    _, Ev = compute_exp_vbond(T_square_mat, T_x_bond_imp_mat, ctmrg_env)
    Es_avg = sum(Eh, Ev)/(size(Eh, 1) * size(Eh, 2))
    println("Average ground energy per site: $Es_avg at U = $U, μ = $μ, χ = $χ")
    save("exp_ctmrg", χ, "Eh", Eh, "Ev", Ev, "Es_avg", Es_avg)
end

t = 1.0
U = 4.0
μ = 0.0
peps_filename = "tensor_file"
peps_param_str = "iter3000"*"_δτ0.0001"
ctmrg_iter = 100
load_env = "random"

GrassmannTensorNetworks.global_sign = auto_sign

run_CTMRG_Square_Hubbard(t, U, μ, peps_filename, peps_param_str, 16, ctmrg_iter; load_env=load_env)
run_CTMRG_Square_Hubbard(t, U, μ, peps_filename, peps_param_str, 32, ctmrg_iter; load_env=load_env)
run_CTMRG_Square_Hubbard(t, U, μ, peps_filename, peps_param_str, 48, ctmrg_iter; load_env=load_env)
