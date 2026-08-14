include("../../src/GrassmannTensorNetworks.jl")

using .GrassmannTensorNetworks
using Printf
using Random

function run_SU_Square_SpinlessFermion(
    t::Float64,
    gamma::Float64,
    lambda::Float64,
    Dbond::Int,
    Lx::Int,
    Ly::Int,
    iter_vec::Vector{Int},
    tol_vec::Vector{Float64},
    dt_vec::Vector{Float64};
    save_filename::String="tensor_file",
    save_key::String="D$(Dbond)_final",
    save_iter::Int=100)

    length(iter_vec) == length(tol_vec) == length(dt_vec) ||
    throw(DimensionMismatch("iter_vec, tol_vec, and dt_vec must have the same length"))

    peps = Square_GPEPS(2, 1, Dbond, Lx, Ly, Float64, true)
    model = SpinlessFermionModel(t, gamma, lambda)

    for (dt, iter, tol) in zip(dt_vec, iter_vec, tol_vec)
        @printf("Simple Update stage: D=%d dt=%.6g iter=%d tol=%.3e\n", Dbond, dt, iter, tol)
        flush(stdout)
        G = gate(model, dt)
        peps = Grassmann_SU(G, peps, dt, Dbond; su_iter=iter, su_tol=tol, save_iter=save_iter, average_trunc=true, start=0)
    end

    save(peps, save_filename, save_key)
    @printf("Saved final weighted PEPS to %s.h5 with key %s\n", save_filename, save_key)
    flush(stdout)

    return peps
end

t = -1.0
gamma = 1.0
lambda = 3.0
Dbond = 2
Lx = 2
Ly = 2
iter_vec = [600, 2000, 4000]
tol_vec = [1e-6, 1e-10, 1e-12]
dt_vec = [1e-2, 1e-3, 1e-4]
save_key = "D$(Dbond)_final"
save_iter = 100
seed = 1234

Random.seed!(seed)
Main.GrassmannTensorNetworks.global_sign = auto_sign
run_SU_Square_SpinlessFermion(t, gamma, lambda, Dbond, Lx, Ly, iter_vec, tol_vec, dt_vec;
save_key=save_key, save_iter=save_iter)
