using Pkg

const GTN_ROOT = normpath(joinpath(@__DIR__, "../.."))

if abspath(PROGRAM_FILE) == @__FILE__
    Pkg.activate(GTN_ROOT)
    Pkg.instantiate()
    Pkg.activate(@__DIR__)
    Pkg.instantiate()
    GTN_ROOT in LOAD_PATH || push!(LOAD_PATH, GTN_ROOT)
end

using GrassmannTensorNetworks
using LinearAlgebra
using Printf
using Random
using Zygote

function spinless_exact_energy(
    t::Real,
    gamma::Real,
    lambda::Real;
    nk::Int=1024,
)
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

function normalized_params(params::AbstractVector{<:Real})
    scale = norm(params) / sqrt(length(params))
    return collect(params) ./ (scale + eps(Float64))
end

function convert_grassmann_grid(
    grid::Matrix{<:Grassmann{S, N}},
    ::Type{T},
) where {S, N, T}
    converted = [convert(tensor, T) for tensor in grid]
    return Matrix{Grassmann{T, N}}(reshape(converted, size(grid)))
end

convert_ctmrg_env(env::CTMRGEnv{T}, ::Type{T}) where {T} = env

function convert_ctmrg_env(env::CTMRGEnv, ::Type{T}) where {T}
    return CTMRGEnv{T}(
        convert_grassmann_grid(env.El, T),
        convert_grassmann_grid(env.Er, T),
        convert_grassmann_grid(env.Eu, T),
        convert_grassmann_grid(env.Ed, T),
        convert_grassmann_grid(env.Clu, T),
        convert_grassmann_grid(env.Cru, T),
        convert_grassmann_grid(env.Cld, T),
        convert_grassmann_grid(env.Crd, T),
    )
end

function compute_nested_energy(
    h::Grassmann{TH, 4},
    peps::Square_GPEPS{T},
    nested::NestedNetwork,
    env::CTMRGEnv,
) where {TH, T}
    h_t = TH === T ? h : convert(h, T)
    env_t = convert_ctmrg_env(env, T)
    energy = zero(promote_type(T, TH))
    for site in CartesianIndices(peps.A)
        _, horizontal = compute_nested_exp_hbond(
            nested, peps, h_t, env_t, site
        )
        _, vertical = compute_nested_exp_vbond(
            nested, peps, h_t, env_t, site
        )
        energy += horizontal + vertical
    end
    return real(energy / length(peps.A))
end

function update_nested_environment!(
    D::Int,
    Lx::Int,
    Ly::Int,
    params::AbstractVector,
    chi::Int,
    ctmrg_iter::Int;
    env::Union{Nothing, CTMRGEnv}=nothing,
    ctmrg_tol::Float64=1e-10,
    average_trunc::Bool=true,
    verbosity::Int=0,
)
    peps = Square_GPEPS(2, 1, D, Lx, Ly, params, false)
    nested = nested_network(peps)
    env = env === nothing ?
        initialize_nested_environment(nested, chi) : env
    run_nested_GCTMRG!(
        nested,
        env,
        chi;
        ctmrg_iter=ctmrg_iter,
        ctmrg_tol=ctmrg_tol,
        average_trunc=average_trunc,
        verbosity=verbosity,
        save_iter=0,
    )
    return peps, nested, env
end

function _normalized_gradient_candidate(
    objective,
    params::Vector{Float64};
    inner_optim_iter::Int=2,
    max_step_norm::Float64=0.25,
)
    inner_optim_iter >= 0 ||
        throw(ArgumentError("inner_optim_iter must be nonnegative"))
    isfinite(max_step_norm) && max_step_norm >= 0 ||
        throw(ArgumentError(
            "max_step_norm must be finite and nonnegative"
        ))

    origin = normalized_params(params)
    initial_value, gradient_tuple = Zygote.withgradient(objective, origin)
    initial_gradient = only(gradient_tuple)
    initial_gradient === nothing &&
        (initial_gradient = zeros(eltype(origin), length(origin)))
    current = origin
    current_value = initial_value
    current_gradient = initial_gradient
    status = :no_step
    attempted_iterations = 0
    accepted_inner_steps = 0
    backtracking_trials = 0

    if !isfinite(initial_value)
        status = :nonfinite_value
    elseif !all(isfinite, initial_gradient)
        status = :nonfinite_gradient
    elseif inner_optim_iter > 0 && max_step_norm > 0
        for iteration in 1:inner_optim_iter
            attempted_iterations = iteration
            gradient_norm = norm(current_gradient)
            if gradient_norm <=
               sqrt(eps(Float64)) * max(1, norm(origin))
                status = :zero_gradient
                break
            end
            direction = -current_gradient / gradient_norm
            alpha = max_step_norm
            accepted = false
            for _ in 1:12
                backtracking_trials += 1
                trial = normalized_params(current + alpha * direction)
                if norm(trial - origin) <=
                   max_step_norm + 64eps(Float64)
                    displacement = trial - current
                    slope = dot(current_gradient, displacement)
                    trial_value = objective(trial)
                    if slope < 0 && isfinite(trial_value) &&
                       trial_value <= current_value + 1e-4 * slope
                        current = trial
                        current_value = trial_value
                        accepted_inner_steps += 1
                        accepted = true
                        status = :accepted
                        break
                    end
                end
                alpha *= 0.5
            end
            if !accepted
                status = :line_search_failed
                break
            elseif iteration < inner_optim_iter
                current_value, gradient_tuple =
                    Zygote.withgradient(objective, current)
                current_gradient = only(gradient_tuple)
                current_gradient === nothing &&
                    (current_gradient =
                        zeros(eltype(origin), length(origin)))
                if !isfinite(current_value)
                    status = :nonfinite_value
                    break
                elseif !all(isfinite, current_gradient)
                    status = :nonfinite_gradient
                    break
                end
            end
        end
    end

    return (
        candidate=current,
        energy=initial_value,
        gradient=initial_gradient,
        candidate_energy=current_value,
        optim_iterations=attempted_iterations,
        optim_converged=status == :zero_gradient,
        optim_status=status,
        accepted_inner_steps=accepted_inner_steps,
        backtracking_trials=backtracking_trials,
    )
end

function fixed_environment_candidate(
    h,
    D::Int,
    Lx::Int,
    Ly::Int,
    params::Vector{Float64},
    env::CTMRGEnv;
    inner_optim_iter::Int=2,
    max_step_norm::Float64=0.25,
)
    objective = x -> begin
        peps = Square_GPEPS(2, 1, D, Lx, Ly, x, false)
        nested = nested_network(peps)
        compute_nested_energy(h, peps, nested, env)
    end
    return _normalized_gradient_candidate(
        objective,
        params;
        inner_optim_iter=inner_optim_iter,
        max_step_norm=max_step_norm,
    )
end

function run_Square_SpinlessFermion_AD_nested(
    t::Float64,
    gamma::Float64,
    lambda::Float64,
    D::Int,
    Lx::Int,
    Ly::Int,
    chi::Int,
    ctmrg_iter::Int;
    ad_iter::Int=20,
    inner_optim_iter::Int=2,
    seed::Int=1234,
    step_shrink::Float64=0.5,
    min_step::Float64=1e-4,
    max_step_norm::Float64=0.25,
    ctmrg_tol::Float64=1e-10,
    average_trunc::Bool=true,
    verbosity::Int=0,
)
    ad_iter > 0 || throw(ArgumentError("ad_iter must be positive"))
    isfinite(step_shrink) && 0 < step_shrink < 1 ||
        throw(ArgumentError(
            "step_shrink must be finite and in (0, 1)"
        ))
    isfinite(min_step) && min_step > 0 ||
        throw(ArgumentError("min_step must be finite and positive"))
    Random.seed!(seed)
    started = time()
    parameter_count = square_gpeps_parameter_count(2, 1, D, Lx, Ly)
    params = normalized_params(randn(parameter_count))
    h = nn_bond(SpinlessFermionModel(t, gamma, lambda))
    exact = spinless_exact_energy(t, gamma, lambda)
    history = NamedTuple[]
    env = nothing

    for iteration in 1:ad_iter
        peps, nested, env = update_nested_environment!(
            D,
            Lx,
            Ly,
            params,
            chi,
            ctmrg_iter;
            env=env,
            ctmrg_tol=ctmrg_tol,
            average_trunc=average_trunc,
            verbosity=verbosity,
        )
        current = compute_nested_energy(h, peps, nested, env)
        proposal = fixed_environment_candidate(
            h,
            D,
            Lx,
            Ly,
            params,
            env;
            inner_optim_iter=inner_optim_iter,
            max_step_norm=max_step_norm,
        )

        accepted = false
        accepted_step = 0.0
        validated = current
        trial_params = params
        trial_env = env
        proposal_usable = proposal.optim_status ∉ (
            :nonfinite_value,
            :nonfinite_gradient,
            :no_step,
        ) && norm(proposal.candidate - params) > sqrt(eps(Float64))
        alpha = proposal_usable ? 1.0 : 0.0
        validation_trials = 0
        while alpha >= min_step && validation_trials < 2
            candidate = normalized_params(
                params + alpha * (proposal.candidate - params)
            )
            candidate_peps = Square_GPEPS(
                2, 1, D, Lx, Ly, candidate, false
            )
            candidate_nested = nested_network(candidate_peps)
            fixed_energy = compute_nested_energy(
                h, candidate_peps, candidate_nested, env
            )
            if isfinite(fixed_energy) && fixed_energy <= current
                validation_trials += 1
                validated_peps, validated_nested, candidate_env =
                    update_nested_environment!(
                        D,
                        Lx,
                        Ly,
                        candidate,
                        chi,
                        ctmrg_iter;
                        env=nothing,
                        ctmrg_tol=ctmrg_tol,
                        average_trunc=average_trunc,
                        verbosity=verbosity,
                    )
                candidate_energy = compute_nested_energy(
                    h, validated_peps, validated_nested, candidate_env
                )
                if isfinite(candidate_energy) && candidate_energy <= current
                    accepted = true
                    accepted_step = alpha
                    validated = candidate_energy
                    trial_params = candidate
                    trial_env = candidate_env
                    break
                end
            end
            alpha *= step_shrink
        end

        params = copy(trial_params)
        env = trial_env
        gradient_norm = norm(proposal.gradient)
        push!(history, (
            iteration=iteration,
            energy=current,
            validated_energy=validated,
            gradient_norm=gradient_norm,
            accepted=accepted,
            accepted_step=accepted_step,
            validation_trials=validation_trials,
            optim_iterations=proposal.optim_iterations,
            optim_converged=proposal.optim_converged,
            optim_status=proposal.optim_status,
            accepted_inner_steps=proposal.accepted_inner_steps,
            backtracking_trials=proposal.backtracking_trials,
        ))
        @printf(
            "AD %2d/%2d E %.12f -> %.12f |g| %.4e step %.3e%s\n",
            iteration,
            ad_iter,
            current,
            validated,
            gradient_norm,
            accepted_step,
            accepted ? "" : " (rejected)",
        )
        flush(stdout)
    end

    peps, nested, env = update_nested_environment!(
        D,
        Lx,
        Ly,
        params,
        chi,
        ctmrg_iter;
        env=env,
        ctmrg_tol=ctmrg_tol,
        average_trunc=average_trunc,
        verbosity=verbosity,
    )
    energy = compute_nested_energy(h, peps, nested, env)
    environment_max_norm = maximum(
        maximum(norm, grid) for grid in (
            env.El,
            env.Er,
            env.Eu,
            env.Ed,
            env.Clu,
            env.Cru,
            env.Cld,
            env.Crd,
        )
    )
    report = (
        chi=chi,
        D=D,
        ctmrg_iter=ctmrg_iter,
        ad_iter=ad_iter,
        energy=energy,
        exact_energy=exact,
        signed_error=energy - exact,
        absolute_error=abs(energy - exact),
        relative_error=abs((energy - exact) / exact),
        elapsed_seconds=time() - started,
        accepted_steps=count(record -> record.accepted, history),
        final_gradient_norm=last(history).gradient_norm,
        environment_max_norm=environment_max_norm,
        environment_finite=isfinite(environment_max_norm),
    )
    return (
        params=params,
        peps=peps,
        nested=nested,
        env=env,
        history=history,
        report=report,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    @eval GrassmannTensorNetworks global global_sign = auto_sign
    chi = parse(Int, get(ENV, "NESTED_CHI", "4"))
    result = run_Square_SpinlessFermion_AD_nested(
        1.0,
        1.0,
        3.0,
        2,
        2,
        2,
        chi,
        20;
        ad_iter=20,
        seed=parse(Int, get(ENV, "NESTED_SEED", "1234")),
        verbosity=parse(Int, get(ENV, "NESTED_VERBOSITY", "0")),
    )
    println("NESTED_RESULT=", repr(result.report))
end
