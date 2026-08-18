"""
using Pkg
const PROJECT_ROOT = normpath(joinpath(@__DIR__, "../.."))
const IS_MAIN = abspath(PROGRAM_FILE) == abspath(@__FILE__)
Pkg.activate(PROJECT_ROOT)
Pkg.instantiate()
cd(@__DIR__)
"""

include("../../src/GrassmannTensorNetworks.jl")

using GrassmannTensorNetworks
using Printf
using Random

"""
function _env_int(name::String, default::Int)
    return parse(Int, get(ENV, name, string(default)))
end

function _env_float(name::String, default::Float64)
    return parse(Float64, get(ENV, name, string(default)))
end

function _env_int_vector(name::String, default::Vector{Int})
    value = get(ENV, name, "")
    isempty(strip(value)) && return default
    return [parse(Int, strip(item)) for item in split(value, ",") if !isempty(strip(item))]
end

function _env_float_vector(name::String, default::Vector{Float64})
    value = get(ENV, name, "")
    isempty(strip(value)) && return default
    return [parse(Float64, strip(item)) for item in split(value, ",") if !isempty(strip(item))]
end
"""

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

"""
if IS_MAIN
    @eval GrassmannTensorNetworks global global_sign = auto_sign

    t = _env_float("SPINLESS_T", -1.0)
    gamma = _env_float("SPINLESS_GAMMA", 1.0)
    lambda = _env_float("SPINLESS_LAMBDA", 3.0)
    Dbond = _env_int("D", 2)
    Lx = _env_int("LX", 2)
    Ly = _env_int("LY", 2)
    iter_vec = _env_int_vector("SU_ITERS", [600, 2000, 4000])
    tol_vec = _env_float_vector("SU_TOLS", [1e-6, 1e-10, 1e-12])
    dt_vec = _env_float_vector("SU_DTS", [1e-2, 1e-3, 1e-4])
    save_key = get(ENV, "PEPS_KEY", "D$(Dbond)_final")
    save_iter = _env_int("SU_SAVE_ITER", 100)
    seed = _env_int("RANDOM_SEED", 1234)

    Random.seed!(seed)

    run_SU_Square_SpinlessFermion(
        t,
        gamma,
        lambda,
        Dbond,
        Lx,
        Ly,
        iter_vec,
        tol_vec,
        dt_vec;
        save_key=save_key,
        save_iter=save_iter,
    )
end
"""

t = -1.0
gamma = 1.0
lambda = 3.0
Dbond = 2
Lx = 2
Ly = 2
iter_vec = [1, 1, 1]
tol_vec = [1e-6, 1e-10, 1e-12]
dt_vec = [1e-2, 1e-3, 1e-4]
save_key = "D$(Dbond)_smoke_review"
save_iter = 0
seed = 1234

Random.seed!(seed)
run_SU_Square_SpinlessFermion(t, gamma, lambda, Dbond, Lx, Ly, iter_vec, tol_vec, dt_vec; 
save_key=save_key, save_iter=save_iter)
