abstract type AbstractCTMRGGaugeFix end

struct CTMRGNoGaugeFix <: AbstractCTMRGGaugeFix end
struct CTMRGPhaseGaugeFix <: AbstractCTMRGGaugeFix end

_ctmrg_gauge_algorithm(::Nothing) = CTMRGNoGaugeFix()
_ctmrg_gauge_algorithm(flag::Bool) = flag ? CTMRGPhaseGaugeFix() : CTMRGNoGaugeFix()
_ctmrg_gauge_algorithm(alg::AbstractCTMRGGaugeFix) = alg

function _ctmrg_grouped_matrix(C::GrassmannMatrix)
    return C
end

# El[:in, :out, :in], Ed[:in, :out, :in]
# Er[:out, :in, :out], Eu[:out, :in, :out]
function _ctmrg_grouped_matrix(Edge::Grassmann{Q, 3}) where {Q}

    if index_type(Edge) == (:in, :out, :in)
        Edge_fused = fuse(Edge, (1, 2); index_type_fused=:out)
    elseif index_type(Edge) == (:out, :in, :out)
        Edge_fused = fuse(Edge, (1, 2); index_type_fused=:in)
    else
        throw(ArgumentError("Unsupported edge tensor index types: $(index_type(Edge)). Expected (:in, :out, :in) or (:out, :in, :out)."))
    end

    return Edge_fused
end

function _normalize_spectrum(spectrum::AbstractVector)

    scale = norm(spectrum)
    scale == 0 && return collect(real.(spectrum))
    return collect(real.(spectrum ./ scale))
end

"""
    ctmrg_spectrum(t; normalize=true)

Return the singular-value spectrum used for convergence diagnostics.
The tensor is first viewed as a matrix.
The default normalization removes arbitrary scalar normalization of environment tensors.
"""

function ctmrg_spectrum(t::Grassmann; normalize::Bool=true)

    _, s, _, _ = gsvd(_ctmrg_grouped_matrix(t), 100000; trunc=false)
    spectrum = sort(diag(s); rev=true)

    return normalize ? _normalize_spectrum(spectrum) : collect(real.(spectrum))
end

function _singular_value_spectrum_distance(
    spectrum1::AbstractVector,
    spectrum2::AbstractVector;
    normalize::Bool=true)

    s1 = normalize ? _normalize_spectrum(spectrum1) : collect(real.(spectrum1))
    s2 = normalize ? _normalize_spectrum(spectrum2) : collect(real.(spectrum2))
    n = max(length(s1), length(s2))
    v1 = zeros(Float64, n)
    v2 = zeros(Float64, n)
    v1[1:length(s1)] .= s1
    v2[1:length(s2)] .= s2

    return norm(v1 - v2)
end

function singular_value_spectrum_distance(
    tensor1::Grassmann,
    tensor2::Grassmann;
    normalize::Bool=true)

    return _singular_value_spectrum_distance(
        ctmrg_spectrum(tensor1; normalize=false),
        ctmrg_spectrum(tensor2; normalize=false);
        normalize=normalize)
end

function _ctmrg_corner_tensors(env::CTMRGEnv)
    return (env.Clu, env.Cru, env.Cld, env.Crd)
end

function _ctmrg_edge_tensors(env::CTMRGEnv)
    return (env.El, env.Er, env.Eu, env.Ed)
end

function ctmrg_corner_spectra(env::CTMRGEnv; normalize::Bool=true)
    return map(
        grid -> map(t -> ctmrg_spectrum(t; normalize=normalize), grid), _ctmrg_corner_tensors(env))
end

function ctmrg_edge_spectra(env::CTMRGEnv; normalize::Bool=true)
    return map(
        grid -> map(t -> ctmrg_spectrum(t; normalize=normalize), grid), _ctmrg_edge_tensors(env))
end

function ctmrg_spectra(env::CTMRGEnv; normalize::Bool=true)

    return (corners=ctmrg_corner_spectra(env; normalize=normalize),
            edges=ctmrg_edge_spectra(env; normalize=normalize))
end

function _spectrum_grid_distance(grid1, grid2; normalize::Bool=true)

    size(grid1) == size(grid2) || throw(DimensionMismatch("CTMRG tensor grids must have the same size"))
    distances = Matrix{Float64}(undef, size(grid1))
    for I in eachindex(grid1, grid2)
        distances[I] = singular_value_spectrum_distance(grid1[I], grid2[I]; normalize=normalize)
    end

    return distances
end

function ctmrg_spectrum_distance(env1::CTMRGEnv, env2::CTMRGEnv; normalize::Bool=true)

    size(env1) == size(env2) || throw(DimensionMismatch("CTMRG environments must have the same unit cell size"))

    corner_distances = map(
        (grid1, grid2) -> _spectrum_grid_distance(grid1, grid2; normalize=normalize),
        _ctmrg_corner_tensors(env1),
        _ctmrg_corner_tensors(env2))

    edge_distances = map(
        (grid1, grid2) -> _spectrum_grid_distance(grid1, grid2; normalize=normalize),
        _ctmrg_edge_tensors(env1),
        _ctmrg_edge_tensors(env2))

    corner_max = maximum(maximum, corner_distances)
    edge_max = maximum(maximum, edge_distances)

    return (max=max(corner_max, edge_max),
            corner=corner_max,
            edge=edge_max,
            corners=corner_distances,
            edges=edge_distances)
end

function _alignment_phase(C_ref::GrassmannMatrix{Q}, C::GrassmannMatrix{Q}) where {Q}

    overlap = scalar(contract(C_ref, C, ((1, 2), (1, 2)); cj=(true, false), sign_function=global_sign))
    magnitude = abs(overlap)
    magnitude == 0 && return one(overlap)

    return conj(overlap) / magnitude
end

function _alignment_phase(E_ref::Grassmann{Q, 3}, E::Grassmann{Q, 3}) where {Q}

    overlap = scalar(contract(E_ref, E, ((1, 2, 3), (1, 2, 3)); cj=(true, false), sign_function=global_sign))
    magnitude = abs(overlap)
    magnitude == 0 && return one(overlap)

    return conj(overlap) / magnitude
end

function _fixgauge_tensor(
    tensor::Grassmann{T, N},
    reference::Grassmann{T, N},
    ::CTMRGPhaseGaugeFix) where {T, N}

    index_type(tensor) == index_type(reference) || throw(ArgumentError("Tensors must have the same index types for gauge fixing."))

    return tensor * _alignment_phase(reference, tensor)
end

_fixgauge_tensor(tensor::Grassmann, reference::Grassmann, ::CTMRGNoGaugeFix) = tensor

function _fixgauge_grid!(grid, reference_grid, alg::AbstractCTMRGGaugeFix)

    size(grid) == size(reference_grid) || throw(DimensionMismatch("CTMRG tensor grids must have the same size"))
    for I in eachindex(grid, reference_grid)
        grid[I] = _fixgauge_tensor(grid[I], reference_grid[I], alg)
    end
    return grid
end

"""
    fixgauge!(env, reference; alg=CTMRGPhaseGaugeFix())

Align the scalar phase/sign gauge of every corner and edge tensor in `env` with `reference`.
it removes arbitrary tensor phases before elementwise convergence is inspected.
"""

function fixgauge!(
    env::CTMRGEnv,
    reference::CTMRGEnv;
    alg::AbstractCTMRGGaugeFix=CTMRGPhaseGaugeFix())

    size(env) == size(reference) || throw(DimensionMismatch("CTMRG environments must have the same unit cell size"))
    _fixgauge_grid!(env.El, reference.El, alg)
    _fixgauge_grid!(env.Er, reference.Er, alg)
    _fixgauge_grid!(env.Eu, reference.Eu, alg)
    _fixgauge_grid!(env.Ed, reference.Ed, alg)
    _fixgauge_grid!(env.Clu, reference.Clu, alg)
    _fixgauge_grid!(env.Cru, reference.Cru, alg)
    _fixgauge_grid!(env.Cld, reference.Cld, alg)
    _fixgauge_grid!(env.Crd, reference.Crd, alg)
    return env
end

function fixgauge(
    env::CTMRGEnv,
    reference::CTMRGEnv;
    alg::AbstractCTMRGGaugeFix=CTMRGPhaseGaugeFix())

    fixed = deepcopy(env)
    return fixgauge!(fixed, reference; alg=alg)
end

function ctmrg_elementwise_distance(env1::CTMRGEnv, env2::CTMRGEnv)

    size(env1) == size(env2) || throw(DimensionMismatch("CTMRG environments must have the same unit cell size"))

    max_distance = 0.0
    for (grids1, grids2) in ((_ctmrg_corner_tensors(env1), _ctmrg_corner_tensors(env2)), (_ctmrg_edge_tensors(env1), _ctmrg_edge_tensors(env2)))
        for (grid1, grid2) in zip(grids1, grids2), I in eachindex(grid1, grid2)
            max_distance = max(max_distance, norm(grid1[I] - grid2[I], Inf))
        end
    end
    return max_distance
end

function _ctmrg_convergence_error(metric::Symbol, spectrum_error::Real, elementwise_error::Real)

    _check_ctmrg_convergence_metric(metric)
    metric == :spectrum && return spectrum_error
    metric == :elementwise && return elementwise_error
    
    return max(spectrum_error, elementwise_error)
end

function _check_ctmrg_convergence_metric(metric::Symbol)
    metric in (:spectrum, :elementwise, :both) && return nothing
    throw(ArgumentError("Unsupported CTMRG convergence_metric: $metric. Use :spectrum, :elementwise, or :both."))
end
