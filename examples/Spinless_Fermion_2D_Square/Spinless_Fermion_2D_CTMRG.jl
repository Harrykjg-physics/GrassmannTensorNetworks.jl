using Pkg
Pkg.activate(joinpath(@__DIR__, "../.."))
Pkg.instantiate()

using GrassmannTensorNetworks

function run_CTMRG_Square_SpinlessFermion(
    t::Float64, 
    γ::Float64,
    λ::Float64,
    peps_filename::String,
    peps_param_str::String,  
    χ::Int, 
    ctmrg_iter::Int; 
    load_env::String="random")

    ##################### Load the Square GPEPS from the ground state optimization #####################

    wpeps = load(peps_filename, peps_param_str, Square_GPEPS)
    peps = absorb_Schmidt_weights(wpeps)

    ##################### Introduce the bond Hamiltonian  #####################

    model = SpinlessFermionModel(t, γ, λ)
    H_nn_bond = nn_bond(model)
    N_site = n_site(model)

    ##################### Construct reduced tensors and impurity tensors from Square GPEPS #####################

    T_square_mat = reduced_tensor(peps)
    T_n_imp_mat = reduced_tensor(peps, N_site)
    T_vbond_imp_mat, T_hbond_imp_mat = reduced_tensor(peps, H_nn_bond)

    ##################### Running Grassmann CTMRG to compute environment tensors #####################

    ctmrg_env = (load_env == "random" ? CTMRGEnv(T_square_mat, χ, Int(χ/2)) : load("ctmrg_env", load_env, CTMRGEnv))
    run_GCTMRG!(T_square_mat, T_n_imp_mat, ctmrg_env, χ; 
    ctmrg_iter=ctmrg_iter, ctmrg_tol=1e-12, average_trunc=true, 
    verbosity=1, save_iter=20, save_filename="ctmrg_env")

    _, ns = compute_exp_site(T_square_mat, T_n_imp_mat, ctmrg_env)
    ns_avg = sum(ns)/(size(ns, 1) * size(ns, 2))
    _, Eh = compute_exp_hbond(T_square_mat, T_hbond_imp_mat, ctmrg_env)
    _, Ev = compute_exp_vbond(T_square_mat, T_vbond_imp_mat, ctmrg_env)
    Es_avg = (sum(Eh) + sum(Ev))/(size(Eh, 1) * size(Eh, 2))
    println("Average ground energy per site: $Es_avg at t = $t, γ = $γ, λ = $λ")
    save("exp_ctmrg", "χ$χ", "ns", ns, "ns_avg", ns_avg, "Eh", Eh, "Ev", Ev, "Es_avg", Es_avg)
end

t = -1.0
γ = 1.0
λ = 3.0
peps_filename = "tensor_file"
peps_param_str = "iter3000"*"_δτ0.0001"
ctmrg_iter = 100
load_env = "random"

GrassmannTensorNetworks.global_sign = auto_sign

run_CTMRG_Square_SpinlessFermion(t, γ, λ, peps_filename, peps_param_str, 16, ctmrg_iter; load_env=load_env)
run_CTMRG_Square_SpinlessFermion(t, γ, λ, peps_filename, peps_param_str, 32, ctmrg_iter; load_env=load_env)
run_CTMRG_Square_SpinlessFermion(t, γ, λ, peps_filename, peps_param_str, 48, ctmrg_iter; load_env=load_env)
