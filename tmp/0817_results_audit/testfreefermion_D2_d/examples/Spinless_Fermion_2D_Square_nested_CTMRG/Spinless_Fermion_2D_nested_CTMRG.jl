include("../../src/GrassmannTensorNetworks.jl")

using .GrassmannTensorNetworks
using Printf
using Random

function spinless_exact_energy(t::Real, gamma::Real, lambda::Real; nk::Int=1024)

    nk > 0 || throw(ArgumentError("nk must be positive"))
    delta = 2pi / nk
    integral = 0.0
    for ix in 0:(nk - 1), iy in 0:(nk - 1)
        kx = -pi + (ix + 0.5) * delta
        ky = -pi + (iy + 0.5) * delta
        normal = t * (cos(kx) + cos(ky)) - lambda
        pairing = gamma * (sin(kx) + sin(ky))
        integral += hypot(normal, pairing)
    end
    return -lambda - integral / nk^2
end

function run_nested_CTMRG_Square_SpinlessFermion(
    t::Float64,
    gamma::Float64,
    lambda::Float64,
    peps_filename::String,
    peps_param_str::String,
    chi::Int,
    ctmrg_iter::Int;
    load_env::String="random",
    ctmrg_tol::Float64=1e-12,
    save_iter::Int=20,
    verbosity::Int=1)

    wpeps = load(peps_filename, peps_param_str, Square_GPEPS)
    peps = absorb_Schmidt_weights(wpeps)

    model = SpinlessFermionModel(t, gamma, lambda)
    H_nn_bond = nn_bond(model)
    N_site = n_site(model)
    H_nn_bonds = fill(H_nn_bond, size(peps))
    N_sites = fill(N_site, size(peps))

    nested = adapt_CTMRG(nested_network(peps))

    env = (load_env == "random" ? 
    initialize_nested_environment(nested, chi, div(chi, 2)) : 
    load("ctmrg_nested_env", load_env, CTMRGEnv))

    run_nested_GCTMRG!(
        nested, 
        env, 
        chi; 
        ctmrg_iter=ctmrg_iter, 
        ctmrg_tol=ctmrg_tol, 
        average_trunc=true, 
        verbosity=verbosity, 
        save_iter=save_iter, 
        save_filename="ctmrg_nested_env")

    _, ns = compute_nested_exp_site(nested, peps, N_sites, env)
    ns_avg = sum(ns) / length(ns)
    _, Eh = compute_nested_exp_hbond(nested, peps, H_nn_bonds, env)
    _, Ev = compute_nested_exp_vbond(nested, peps, H_nn_bonds, env)
    Es_avg = (sum(Eh) + sum(Ev)) / length(Eh)

    println("Average ground energy per site: $Es_avg at t = $t, gamma = $gamma, λ = $lambda")
    save("exp_nested_ctmrg", "chi$chi", "ns", ns, "ns_avg", ns_avg, "Eh", Eh, "Ev", Ev, "Es_avg", Es_avg)
end

t = -1.0
gamma = 4.0
lambda = 5.0  
ctmrg_iter = 100
peps_filename = "tensor_file"
peps_param_str = "iter4000"*"_δτ0.0001"
load_env = "random"
verbosity = 1
seed = 1234

Random.seed!(seed)
GrassmannTensorNetworks.global_sign = auto_sign

run_nested_CTMRG_Square_SpinlessFermion(t, gamma, lambda, peps_filename, peps_param_str, 16, ctmrg_iter; load_env=load_env, verbosity=verbosity)
run_nested_CTMRG_Square_SpinlessFermion(t, gamma, lambda, peps_filename, peps_param_str, 32, ctmrg_iter; load_env=load_env, verbosity=verbosity)
run_nested_CTMRG_Square_SpinlessFermion(t, gamma, lambda, peps_filename, peps_param_str, 48, ctmrg_iter; load_env=load_env, verbosity=verbosity)
