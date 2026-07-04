using Pkg
Pkg.activate(joinpath(@__DIR__, "../.."))
Pkg.instantiate()

using GrassmannTensorNetworks

function run_SU_Square_SpinlessFermion(
    t::Float64, 
    γ::Float64,
    λ::Float64,
    Dbond::Int64, 
    Lx::Int, 
    Ly::Int, 
    iter_vec::Vector{Int}, 
    tol_vec::Vector{Float64}, 
    dτ_vec::Vector{Float64})

    peps = Square_GPEPS(2, 1, Dbond, Lx, Ly, Float64, true)
    model = SpinlessFermionModel(t, γ, λ)

    for (dτ, iter, tol) in zip(dτ_vec, iter_vec, tol_vec)
        G = gate(model, dτ)
        peps = Grassmann_SU(G, peps, dτ, Dbond; su_iter=iter, su_tol=tol, save_iter=100, average_trunc=true, start=0)
    end

    return peps
end

t = -1.0
γ = 1.0
λ = 3.0
Dbond = 2
Lx = 1
Ly = 1

GrassmannTensorNetworks.global_sign = auto_sign

run_SU_Square_SpinlessFermion(t, γ, λ, Dbond, Lx, Ly, [100, 100, 100], [1e-6, 1e-10, 1e-12], [1e-2, 1e-3, 1e-4])
