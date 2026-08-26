using Pkg
Pkg.activate(joinpath(@__DIR__, "../.."))
Pkg.instantiate()

using GrassmannTensorNetworks
using LinearAlgebra
using Optim
using Printf
using Random

function normalize_params!(params::AbstractVector{<:Real})
    scale = norm(params) / sqrt(length(params))
    scale > eps(eltype(params)) && (params ./= scale)
    return params
end

function normalized_params(params::AbstractVector{<:Real})
    scale = norm(params) / sqrt(length(params))
    return collect(params) ./ (scale + eps(Float64))
end

function convert_grassmann_grid(mat::Matrix{<:Grassmann{S, N}}, ::Type{T}) where {S, N, T}
    converted = [convert(mat[i], T) for i in eachindex(mat)]
    return Matrix{Grassmann{T, N}}(reshape(converted, size(mat)))
end

function convert_ctmrg_env(env::CTMRGEnv{T}, ::Type{T}) where {T}
    return env
end

function convert_ctmrg_env(env::CTMRGEnv, ::Type{T}) where {T}
    return CTMRGEnv{T}(
        convert_grassmann_grid(env.El, T),
        convert_grassmann_grid(env.Er, T),
        convert_grassmann_grid(env.Eu, T),
        convert_grassmann_grid(env.Ed, T),
        convert_grassmann_grid(env.Clu, T),
        convert_grassmann_grid(env.Cru, T),
        convert_grassmann_grid(env.Cld, T),
        convert_grassmann_grid(env.Crd, T))
end

function compute_reduced_tensors(
    h_bond::Grassmann{TH, 4},
    Dbond::Int,
    Lx::Int,
    Ly::Int,
    params::AbstractVector{T}) where {TH, T<:Real}

    peps = Square_GPEPS(2, 1, Dbond, Lx, Ly, params, false)
    h_bond_t = (TH === T ? h_bond : convert(h_bond, T))
    t_bulk = Matrix{Grassmann{T, 4}}([
        reduced_tensor(peps.A[r, c]) for r in 1:Lx, c in 1:Ly])
    t_vbond = Matrix{Grassmann{T, 6}}([
        reduced_tensor_vbond(peps.A[r, c], peps.A[Nmod(r + 1, Lx), c], h_bond_t) 
        for r in 1:Lx, c in 1:Ly])
    t_hbond = Matrix{Grassmann{T, 6}}([
        reduced_tensor_hbond(peps.A[r, c], peps.A[r, Nmod(c + 1, Ly)], h_bond_t) 
        for r in 1:Lx, c in 1:Ly])

    return t_bulk, t_vbond, t_hbond
end

function update_ctmrg_env!(
    h_bond::Grassmann{T, 4},
    Dbond::Int,
    Lx::Int,
    Ly::Int,
    params::AbstractVector{T},
    chi::Int,
    ctmrg_iter::Int;
    env::Union{Nothing, CTMRGEnv}=nothing,
    ctmrg_tol::Float64=1e-10,
    average_trunc::Bool=true,
    verbosity::Int=0) where {T<:Real}

    t_bulk, _, _ = compute_reduced_tensors(h_bond, Dbond, Lx, Ly, params)
    env = (env === nothing ? CTMRGEnv(t_bulk, chi, div(chi, 2)) : env)

    run_GCTMRG!(
        t_bulk,
        t_bulk,
        env,
        chi;
        ctmrg_iter=ctmrg_iter,
        ctmrg_tol=ctmrg_tol,
        average_trunc=average_trunc,
        verbosity=verbosity,
        save_iter=0)

    return env
end

function compute_energy(
    h_bond::Grassmann{TH, 4},
    Dbond::Int,
    Lx::Int,
    Ly::Int,
    params::AbstractVector{T},
    env::CTMRGEnv) where {TH, T<:Real}

    t_bulk, t_vbond, t_hbond = compute_reduced_tensors(h_bond, Dbond, Lx, Ly, params)
    env_t = convert_ctmrg_env(env, T)
    energy = zero(T)

    for c in 1:Ly, r in 1:Lx

        c_p1 = Nmod(c + 1, Ly)
        r_p1 = Nmod(r + 1, Lx)

        _, eh = compute_exp_hbond(
            t_bulk[r, c],
            t_bulk[r, c_p1],
            t_hbond[r, c],
            env_t.El[r, c],
            env_t.Er[r, c_p1],
            env_t.Eu[r, c],
            env_t.Eu[r, c_p1],
            env_t.Ed[r, c],
            env_t.Ed[r, c_p1],
            env_t.Clu[r, c],
            env_t.Cru[r, c_p1],
            env_t.Cld[r, c],
            env_t.Crd[r, c_p1])

        _, ev = compute_exp_vbond(
            t_bulk[r_p1, c],
            t_bulk[r, c],
            t_vbond[r, c],
            env_t.Ed[r_p1, c],
            env_t.Eu[r, c],
            env_t.Er[r_p1, c],
            env_t.Er[r, c],
            env_t.El[r_p1, c],
            env_t.El[r, c],
            env_t.Crd[r_p1, c],
            env_t.Cru[r, c],
            env_t.Cld[r_p1, c],
            env_t.Clu[r, c])

        energy += eh + ev
    end

    energy /= Lx * Ly

    return real(energy)
end

function optimize_fixed_env!(
    h_bond::Grassmann{Float64, 4},
    Dbond::Int,
    Lx::Int,
    Ly::Int,
    params::Vector{Float64},
    env::CTMRGEnv;
    iterations::Int=5,
    method=Optim.LBFGS(),
    energy_bound::Float64=100.0,
    max_step_norm::Float64=1.0,
    step_shrink::Float64=0.5,
    min_step::Float64=1e-4,
    max_validation_trials::Int=2,
    acceptance_energy::Union{Nothing, Float64}=nothing,
    acceptance_tol::Float64=0.0,
    validation_objective::Union{Nothing, Function}=nothing)

    params_before = copy(params)
    objective = x -> compute_energy(h_bond, Dbond, Lx, Ly, normalized_params(x), env)
    energy_before = objective(params)
    energy_target = acceptance_energy === nothing ? energy_before : acceptance_energy
    options = Optim.Options(iterations=iterations, show_trace=false)
    result = Optim.optimize(objective, params, method, options; autodiff=:forward)
    raw_candidate = normalized_params(Optim.minimizer(result))
    raw_candidate_energy = objective(raw_candidate)

    candidate = raw_candidate
    candidate_step_norm = norm(candidate - params_before)
    if isfinite(max_step_norm) && max_step_norm > 0 && candidate_step_norm > max_step_norm
        candidate = normalized_params(@. params_before + (max_step_norm / candidate_step_norm) * (candidate - params_before))
        candidate_step_norm = norm(candidate - params_before)
    end

    candidate_energy = objective(candidate)

    accepted = false
    accepted_step = 0.0
    energy_after = energy_before
    validated_energy_after = NaN
    validation_trials = 0
    alpha = 1.0

    while alpha >= min_step
        trial = normalized_params(@. (1 - alpha) * params_before + alpha * candidate)
        trial_energy = objective(trial)

        fixed_env_ok = isfinite(trial_energy) &&
            abs(trial_energy) <= energy_bound &&
            trial_energy <= energy_before

        if !fixed_env_ok
            alpha *= step_shrink
            continue
        end

        if validation_objective === nothing
            validation_energy = trial_energy
        elseif validation_trials < max_validation_trials
            validation_trials += 1
            validation_energy = validation_objective(trial)
        else
            break
        end

        if isfinite(validation_energy) &&
            abs(validation_energy) <= energy_bound &&
            validation_energy <= energy_target + acceptance_tol

            copyto!(params, trial)
            energy_after = trial_energy
            validated_energy_after = validation_energy
            accepted = true
            accepted_step = alpha
            break
        end

        alpha *= step_shrink
    end

    accepted || copyto!(params, params_before)

    return energy_before, energy_after, result, accepted,
        raw_candidate_energy, candidate_energy, candidate_step_norm,
        validated_energy_after, accepted_step, validation_trials
end

function run_Square_SpinlessFermion_AD(
    t::Float64, gamma::Float64, lambda::Float64,
    Dbond::Int,
    Lx::Int, Ly::Int, 
    chi::Int,
    ctmrg_iter::Int;
    outer_ad_iter::Int=8, 
    inner_optim_iter::Int=2,
    ad_tol::Float64=1e-6,
    seed::Int=1234,
    energy_bound::Float64=100.0,
    max_step_norm::Float64=1.0,
    step_shrink::Float64=0.5,
    min_step::Float64=1e-4,
    validate_with_ctmrg::Bool=true,
    validation_ctmrg_iter::Int=ctmrg_iter,
    max_validation_trials::Int=2,
    monotone_best::Bool=true,
    acceptance_tol::Float64=0.0,
    average_trunc::Bool=true,
    ctmrg_tol::Float64=1e-10,
    verbosity::Int=0)

    Random.seed!(seed)

    nparams = square_gpeps_parameter_count(2, 1, Dbond, Lx, Ly)
    params = normalize_params!(randn(nparams))
    best_params = copy(params)
    best_energy = Inf
    history = NamedTuple[]
    env = nothing
    h_bond = nn_bond(SpinlessFermionModel(t, gamma, lambda))

    for ad_iter in 1:outer_ad_iter

        @printf("Updating CTMRG environment %d/%d...\n", ad_iter, outer_ad_iter)
        flush(stdout)

        env = update_ctmrg_env!(
            h_bond, Dbond, Lx, Ly, params, chi, ctmrg_iter;
            env=env,
            ctmrg_tol=ctmrg_tol,
            average_trunc=average_trunc,
            verbosity=verbosity)

        validation_objective = if validate_with_ctmrg
            trial_params -> begin
                trial_env = update_ctmrg_env!(
                    h_bond, Dbond, Lx, Ly, trial_params, chi, validation_ctmrg_iter;
                    env=nothing,
                    ctmrg_tol=ctmrg_tol,
                    average_trunc=average_trunc,
                    verbosity=verbosity)
                compute_energy(h_bond, Dbond, Lx, Ly, trial_params, trial_env)
            end
        else
            nothing
        end

        params_at_outer_start = copy(params)
        energy_before, energy_after, opt_result, accepted,
            raw_candidate_energy, candidate_energy, candidate_step_norm,
            validated_energy_after, accepted_step, validation_trials = optimize_fixed_env!(
            h_bond, Dbond, Lx, Ly, params, env;
            iterations=inner_optim_iter,
            energy_bound=energy_bound,
            max_step_norm=max_step_norm,
            step_shrink=step_shrink,
            min_step=min_step,
            max_validation_trials=max_validation_trials,
            acceptance_energy=monotone_best && isfinite(best_energy) ? best_energy : nothing,
            acceptance_tol=acceptance_tol,
            validation_objective=validation_objective)

        grad_norm = accepted ? Optim.g_residual(opt_result) : Inf
        reported_energy = validate_with_ctmrg && accepted ? validated_energy_after : energy_after
        if energy_before < best_energy
            best_energy = energy_before
            copyto!(best_params, params_at_outer_start)
        end
        if accepted && reported_energy < best_energy
            best_energy = reported_energy
            copyto!(best_params, params)
        end

        push!(history, (
            ad_iter=ad_iter,
            iterations=Optim.iterations(opt_result),
            energy=energy_before,
            new_energy=energy_after,
            raw_candidate_energy=raw_candidate_energy,
            candidate_energy=candidate_energy,
            candidate_step_norm=candidate_step_norm,
            validated_energy=validated_energy_after,
            reported_energy=reported_energy,
            validation_trials=validation_trials,
            grad_norm=grad_norm,
            accepted_step=accepted_step,
            accepted=accepted,
            converged=Optim.converged(opt_result)))

        @printf(
            "Optim outer %2d | iters %3d | E %.12f -> %.12f | E_valid = %.12f | 
            E_raw = %.12f | E_candidate = %.12f | dx = %.3e | step = %.3e | validations = %d | |g| = %.6e%s%s\n",
            ad_iter, Optim.iterations(opt_result), energy_before, energy_after,
            validated_energy_after, raw_candidate_energy, candidate_energy, candidate_step_norm,
            accepted_step, validation_trials, grad_norm,
            accepted ? "" : " (rejected)",
            Optim.converged(opt_result) ? " (converged)" : "")
        flush(stdout)

        accepted && grad_norm < ad_tol && return (
            params=params,
            peps=Square_GPEPS(2, 1, Dbond, Lx, Ly, params, false),
            env=env,
            history=history,
            best_energy=best_energy,
            final_energy=energy_after)
    end

    if isfinite(best_energy)
        copyto!(params, best_params)
        @printf("Restoring best validated parameters with E = %.12f\n", best_energy)
        flush(stdout)
    end

    env = update_ctmrg_env!(
        h_bond, Dbond, Lx, Ly, params, chi, ctmrg_iter;
        env=env,
        ctmrg_tol=ctmrg_tol,
        average_trunc=average_trunc,
        verbosity=verbosity)
        
    final_energy = compute_energy(h_bond, Dbond, Lx, Ly, params, env)

    @printf(
        "Final refreshed CTMRG energy: %.12f\n",
        final_energy)

    return (
        params=params,
        peps=Square_GPEPS(2, 1, Dbond, Lx, Ly, params, false),
        env=env,
        history=history,
        best_energy=best_energy,
        final_energy=final_energy)
end

if abspath(PROGRAM_FILE) == @__FILE__

    GrassmannTensorNetworks.global_sign = auto_sign

    t = 1.0
    gamma = 1.0
    lambda = 3.0
    Dbond = 2
    Lx = 2
    Ly = 2
    chi = 20
    ctmrg_iter = 100

    result = run_Square_SpinlessFermion_AD(
        t,
        gamma,
        lambda,
        Dbond,
        Lx,
        Ly,
        chi,
        ctmrg_iter;
        outer_ad_iter=2,
        inner_optim_iter=2,
        verbosity=0)

    @printf("Stored %d optimization records.\n", length(result.history))
end
