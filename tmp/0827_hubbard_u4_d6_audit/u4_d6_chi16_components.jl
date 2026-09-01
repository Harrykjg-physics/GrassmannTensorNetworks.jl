using Pkg

const RESDIR = "/home/jkkong/work/2026/0822/Hubbard_U4_D6_half_filled/examples/Fermi_Hubbard_2D_Square_su_1e-2_CTMRG"
Pkg.activate(joinpath(RESDIR, "../.."))

using GrassmannTensorNetworks
using LinearAlgebra
using Printf

const PARAM = "iter5861_δτ0.01"
const CHI_KEY = "\u03c716iter100"

function measure_energy(Tbulk, peps, env, model)
    H = nn_bond(model)
    _, Eh = compute_exp_hbond(Tbulk, peps, H, env)
    _, Ev = compute_exp_vbond(Tbulk, peps, H, env)
    Es = (sum(Eh) + sum(Ev)) / length(Tbulk)
    return Es, Eh, Ev
end

println("Loading PEPS and chi=16 CTMRG environment")
wpeps = load(joinpath(RESDIR, "tensor_file"), PARAM, Square_GPEPS)
peps = absorb_Schmidt_weights(wpeps)
Tbulk = reduced_tensor(peps)
N_site = n_site(HubbardModel(1.0, 4.0, 2.0))
Tn = reduced_tensor(peps, N_site)
env = load(joinpath(RESDIR, "ctmrg_env"), CHI_KEY, CTMRGEnv)

_, ns = compute_exp_site(Tbulk, Tn, env)
navg = sum(ns) / length(ns)
println("n_avg = ", repr(navg))
println("ns = ", repr(ns))

models = [
    ("full_mu2", HubbardModel(1.0, 4.0, 2.0)),
    ("canonical_mu0", HubbardModel(1.0, 4.0, 0.0)),
    ("hop_only", HubbardModel(1.0, 0.0, 0.0)),
    ("U_only", HubbardModel(0.0, 4.0, 0.0)),
    ("mu_only", HubbardModel(0.0, 0.0, 2.0)),
]

results = Dict{String, Float64}()
for (name, model) in models
    Es, Eh, Ev = measure_energy(Tbulk, peps, env, model)
    results[name] = Es
    @printf("%s Es %.16f Eh_mean %.16f Ev_mean %.16f\n", name, Es, sum(Eh)/length(Eh), sum(Ev)/length(Ev))
    println(name, " Eh = ", repr(Eh))
    println(name, " Ev = ", repr(Ev))
end

println("checks:")
@printf("full_mu2 + 2*n_avg = %.16f\n", results["full_mu2"] + 2.0 * navg)
@printf("canonical_mu0 - (full_mu2 + 2*n_avg) = %.16e\n", results["canonical_mu0"] - (results["full_mu2"] + 2.0 * navg))
@printf("canonical_mu0 - (hop_only + U_only) = %.16e\n", results["canonical_mu0"] - (results["hop_only"] + results["U_only"]))
@printf("mu_only + 2*n_avg = %.16e\n", results["mu_only"] + 2.0 * navg)
