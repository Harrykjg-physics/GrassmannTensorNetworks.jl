using Pkg
Pkg.activate(joinpath(@__DIR__, "../.."))
Pkg.instantiate()

using GrassmannTensorNetworks

function run_CTMRG_Square_GNW(
    m::Float64, 
    g2::Float64, 
    μ::Float64, 
    χ::Int, 
    max_iter::Int, 
    tol::Float64; 
    load_env::String="random")

    ##################### Construct the local Grassmann tensor corresponding to the Gross-Neveu-Wilson model's partition function #####################

    model = Nf1_Gross_Neveu_Wilson_model(m, g2, μ)
    T_coef_gnw = PartitionFunctionTensor(model)

    T_bulk = Matrix{Grassmann{ComplexF64, 4}}(undef, 1, 1)
    T_gnw = Grassmann(T_coef_gnw, (4, 4, 4, 4), (2, 2, 2, 2), (:in, :in, :out, :out))
    # T_bulk[l, r, u, d] <-- T_gnw[r, u, l, d]
    T_bulk[1, 1] = permutedims(T_gnw, (3, 1, 2, 4); sign_function=auto_sign)

    ##################### Running Grassmann CTMRG to compute environment tensors #####################

    ctmrg_env = (load_env == "random" ? CTMRGEnv(T_bulk, χ, Int(χ/2)) : load("ctmrg_env", load_env, CTMRGEnv))

    run_GCTMRG!(T_bulk, T_bulk, ctmrg_env, χ; 
    ctmrg_iter=ctmrg_iter, ctmrg_tol=1e-14, 
    average_trunc=true, verbosity=1, 
    save_iter=20, save_filename="ctmrg_env")

    ##################### Running Grassmann CTMRG to compute environment tensors #####################

    Z = compute_exp_site(T_bulk, ctmrg_env)
    logZ_avg = log.(Z[1, 1])

    # lnZ/V = 0.84112502060056715 at m=-1
    println("***** μ : ", μ,  "  m : ", m,  "  g2 : ", g2, "  logZ/V : ", logZ_avg)
    # save("GN_CTMRG", χ, "μ_m_g2", [μ, m, g2], "logZ", logZ_avg)

end

m = -1.0
g2 = 0.0
μ = 0.0
ctmrg_iter = 500
load_env = "random"

GrassmannTensorNetworks.global_sign = auto_sign

run_CTMRG_Square_GNW(m, g2, μ, 16, ctmrg_iter, 1e-14; load_env=load_env)
run_CTMRG_Square_GNW(m, g2, μ, 32, ctmrg_iter, 1e-14; load_env=load_env)
run_CTMRG_Square_GNW(m, g2, μ, 48, ctmrg_iter, 1e-14; load_env=load_env)
