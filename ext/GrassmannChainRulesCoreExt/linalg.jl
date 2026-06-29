################################ AD rules for linalg.jl operations ################################

function _matrix_log_pullback(A::AbstractMatrix, G::AbstractMatrix)

    n = LinearAlgebra.checksquare(A)
    size(G) == size(A) || throw(DimensionMismatch("matrix-log cotangent has the wrong size"))

    Q = promote_type(eltype(A), eltype(G))
    frechet_arg = zeros(Q, 2n, 2n)
    frechet_arg[1:n, 1:n] .= adjoint(A)
    frechet_arg[1:n, n + 1:2n] .= adjoint(G)
    frechet_arg[n + 1:2n, n + 1:2n] .= adjoint(A)

    frechet_log = log(frechet_arg)
    gradient = adjoint(@view frechet_log[1:n, n + 1:2n])

    return ProjectTo(A)(gradient)
end

function ChainRulesCore.rrule(::typeof(log), t::GrassmannMatrix)

    y = log(t)

    function log_pullback(delta_y)
        delta_y isa AbstractZero && return (NoTangent(), ZeroTangent())
        delta_y = unthunk(delta_y)

        data_pairs = collect(nonzero_pairs(t))
        first_sector, first_block = first(data_pairs)
        first_gradient = _matrix_log_pullback(first_block, delta_y[first_sector])
        AT = typeof(first_gradient)
        data_dict = Dict{NTuple{2, Int}, AT}(first_sector => first_gradient)
        sizehint!(data_dict, length(data_pairs))

        for (i, (sector, block)) in enumerate(data_pairs)
            i == 1 && continue
            data_dict[sector] = _matrix_log_pullback(block, delta_y[sector])
        end

        delta_t = Grassmann(size(t), even(t), index_type(t), data_dict)
        return (NoTangent(), delta_t)
    end

    return y, log_pullback
end

function _norm_pullback(t::Grassmann{T, N, AT}, p::Real, y, delta_y) where {T, N, AT}

    delta_y = real(unthunk(delta_y))
    data_dict = Dict{NTuple{N, Int}, AT}()
    sizehint!(data_dict, length(data(t)))

    if p == 0 || iszero(y)
        for (sector, block) in nonzero_pairs(t)
            data_dict[sector] = zero(block)
        end
    else
        denominator = y^(p - 1)
        for (sector, block) in nonzero_pairs(t)
            gradient = similar(block)
            for index in eachindex(block)
                value = block[index]
                gradient[index] = iszero(value) ? zero(value) : delta_y * value * abs(value)^(p - 2) / denominator
            end
            data_dict[sector] = gradient
        end
    end

    return Grassmann(size(t), even(t), index_type(t), data_dict)
end

function _extreme_norm_pullback(t::Grassmann{T, N, AT}, p::Real, delta_y) where {T, N, AT}

    delta_y = real(unthunk(delta_y))
    best_value = p == Inf ? -Inf : Inf
    best_sector = nothing
    best_index = nothing

    for (sector, block) in nonzero_pairs(t)
        values = abs.(block)
        value, index = p == Inf ? findmax(values) : findmin(values)
        is_better = p == Inf ? value > best_value : value < best_value
        if is_better
            best_value = value
            best_sector = sector
            best_index = index
        end
    end

    data_dict = Dict{NTuple{N, Int}, AT}()
    sizehint!(data_dict, length(data(t)))
    for (sector, block) in nonzero_pairs(t)
        gradient = zero(block)
        if sector == best_sector && !iszero(best_value)
            value = block[best_index]
            gradient[best_index] = delta_y * value / abs(value)
        end
        data_dict[sector] = gradient
    end

    return Grassmann(size(t), even(t), index_type(t), data_dict)
end

function _norm_rrule(t::Grassmann, p::Real)

    y = norm(t, p)
    function norm_pullback(delta_y)
        delta_y isa AbstractZero && return ZeroTangent()
        return (p == Inf || p == -Inf) ?
               _extreme_norm_pullback(t, p, delta_y) :
               _norm_pullback(t, p, y, delta_y)
    end

    return y, norm_pullback
end

function ChainRulesCore.rrule(::typeof(norm), t::Grassmann)
    y, pullback = _norm_rrule(t, 2)
    return y, delta_y -> (NoTangent(), pullback(delta_y))
end

function ChainRulesCore.rrule(::typeof(norm), t::Grassmann, p::Real)
    y, pullback = _norm_rrule(t, p)
    return y, delta_y -> (NoTangent(), pullback(delta_y), NoTangent())
end

function ChainRulesCore.rrule(::typeof(diag), t::GrassmannMatrix{T}) where {T}

    y = diag(t)

    function diag_pullback(delta_y)
        delta_y isa AbstractZero && return (NoTangent(), ZeroTangent())
        delta_y = unthunk(delta_y)
        data_dict = Dict{NTuple{2, Int}, Matrix{T}}()
        sizehint!(data_dict, length(data(t)))
        offset = 0

        for sector in ((0, 0), (1, 1))
            haskey(t, sector) || continue
            block = t[sector]
            diagonal_length = min(size(block)...)
            gradient = zero(block)
            gradient[diagind(gradient)] .= @view delta_y[offset + 1:offset + diagonal_length]
            data_dict[sector] = gradient
            offset += diagonal_length
        end

        delta_t = Grassmann(size(t), even(t), index_type(t), data_dict)
        return (NoTangent(), delta_t)
    end

    return y, diag_pullback
end

function ChainRulesCore.rrule(::typeof(transpose), t::GrassmannMatrix; sign_function::Function=trivial_sign)

    y = transpose(t; sign_function=sign_function)

    function transpose_pullback(delta_y)
        delta_y isa AbstractZero && return (NoTangent(), ZeroTangent())
        delta_t = transpose(unthunk(delta_y); sign_function=sign_function)
        return (NoTangent(), delta_t)
    end

    return y, transpose_pullback
end

function ChainRulesCore.rrule(::typeof(inv), t::GrassmannMatrix{T}) where {T}

    y = inv(t)

    function inv_pullback(delta_y)
        delta_y isa AbstractZero && return (NoTangent(), ZeroTangent())
        delta_y = unthunk(delta_y)
        data_dict = Dict{NTuple{2, Int}, Matrix{T}}()
        sizehint!(data_dict, length(data(t)))

        for (sector, inverse_block) in nonzero_pairs(y)
            data_dict[sector] = ProjectTo(t[sector])(-adjoint(inverse_block) * delta_y[sector] * adjoint(inverse_block))
        end

        delta_t = Grassmann(size(t), even(t), index_type(t), data_dict)
        return (NoTangent(), delta_t)
    end

    return y, inv_pullback
end

function ChainRulesCore.rrule(::typeof(dot), t1::Grassmann, t2::Grassmann)
    
    y = dot(t1, t2)

    function dot_pullback(delta_y)
        delta_y isa AbstractZero && return (NoTangent(), ZeroTangent(), ZeroTangent())
        delta_y = unthunk(delta_y)

        data_1 = Dict(sector => block * conj(delta_y) for (sector, block) in nonzero_pairs(t2))
        data_2 = Dict(sector => block * delta_y for (sector, block) in nonzero_pairs(t1))
        delta_t1 = Grassmann(size(t1), even(t1), index_type(t1), data_1)
        delta_t2 = Grassmann(size(t2), even(t2), index_type(t2), data_2)
        return (NoTangent(), delta_t1, delta_t2)
    end

    return y, dot_pullback
end