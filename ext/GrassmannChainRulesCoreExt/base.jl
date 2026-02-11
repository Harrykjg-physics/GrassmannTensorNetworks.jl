################################ AD rules for base.jl operations ################################

# ─── copy: identity pullback ──────────────────────────────────────────────────

function ChainRulesCore.rrule(::typeof(Base.copy), t::Grassmann)
    y = copy(t)
    copy_pullback(Δy) = (NoTangent(), Δy)
    return y, copy_pullback
end

# ─── + : gradient distributes to both inputs ──────────────────────────────────

function ChainRulesCore.rrule(::typeof(+), t1::Grassmann, t2::Grassmann)
    y = t1 + t2
    plus_pullback(Δy) = (NoTangent(), Δy, Δy)
    return y, plus_pullback
end

# ─── - : first input unchanged, second negated ───────────────────────────────

function ChainRulesCore.rrule(::typeof(-), t1::Grassmann, t2::Grassmann)
    y = t1 - t2
    function minus_pullback(Δy)
        return (NoTangent(), Δy, -1 * Δy)
    end
    return y, minus_pullback
end

# ─── * (right scalar multiplication): standard product rule ───────────────────

function ChainRulesCore.rrule(::typeof(*), t::Grassmann, val::Number)
    y = t * val
    function times_r_pullback(Δy)
        Δt = Δy * conj(val)
        # Inner product ⟨conj(t), Δy⟩ for scalar gradient
        Δval = sum(sum(conj.(t[k]) .* Δy[k]) for k in nonzero_keys(t))
        return (NoTangent(), Δt, Δval)
    end
    return y, times_r_pullback
end

# ─── * (left scalar multiplication): standard product rule ────────────────────

function ChainRulesCore.rrule(::typeof(*), val::Number, t::Grassmann)
    y = val * t
    function times_l_pullback(Δy)
        Δt = Δy * conj(val)
        Δval = sum(sum(conj.(t[k]) .* Δy[k]) for k in nonzero_keys(t))
        return (NoTangent(), Δval, Δt)
    end
    return y, times_l_pullback
end

# ─── / (scalar division): standard quotient rule ─────────────────────────────

function ChainRulesCore.rrule(::typeof(/), t::Grassmann, val::Number)
    y = t / val
    function div_pullback(Δy)
        Δt = Δy / conj(val)
        # Δval = -⟨y, Δy⟩ / conj(val) = -⟨conj(t), Δy⟩ / |val|²
        inner = sum(sum(conj.(y[k]) .* Δy[k]) for k in nonzero_keys(y))
        Δval = -inner / conj(val)
        return (NoTangent(), Δt, Δval)
    end
    return y, div_pullback
end

# ─── real: gradient flows through real part only ──────────────────────────────

function ChainRulesCore.rrule(::typeof(Base.real), t::Grassmann)
    y = real(t)
    function real_pullback(Δy)
        # Gradient of real() w.r.t. input is identity on the real part.
        # Δy is real-valued; pass it through directly.
        # Type promotion handles embedding real gradient into complex space.
        return (NoTangent(), Δy)
    end
    return y, real_pullback
end

# ─── conj: reverse conjugation ────────────────────────────────────────────────
# Works for both trivial_sign and auto_sign because sign factors are ±1 (self-inverse).

function ChainRulesCore.rrule(::typeof(Base.conj), t::Grassmann; sign_function=trivial_sign)
    y = conj(t; sign_function=sign_function)
    function conj_pullback(Δy)
        # conj with the same sign_function reverses both value conjugation and sign factor
        Δt = conj(Δy; sign_function=sign_function)
        return (NoTangent(), Δt)
    end
    return y, conj_pullback
end

# ─── permutedims: inverse permutation ─────────────────────────────────────────
# Works for both trivial_sign and auto_sign because Koszul signs satisfy
# σ(k, dst) = σ(permute(k, dst), inv_dst).

function ChainRulesCore.rrule(
    ::typeof(Base.permutedims),
    t::Grassmann{T, N, AT},
    dst::NTuple{N, Int};
    sign_function=trivial_sign
) where {T, N, AT}
    y = permutedims(t, dst; sign_function=sign_function)
    function permutedims_pullback(Δy)
        # Compute inverse permutation
        inv_dst = ntuple(i -> findfirst(==(i), dst), Val(N))
        Δt = permutedims(Δy, inv_dst; sign_function=sign_function)
        return (NoTangent(), Δt, NoTangent())
    end
    return y, permutedims_pullback
end

# ─── sqrt: element-wise 1/(2√x) derivative ───────────────────────────────────

function ChainRulesCore.rrule(::typeof(Base.sqrt), t::Grassmann{T, N, AT}) where {T, N, AT}
    y = sqrt(t)
    function sqrt_pullback(Δy)
        data_dict = Dict{NTuple{N, Int}, AT}()
        data_pairs = nonzero_pairs(t)
        sizehint!(data_dict, length(data_pairs))
        @inbounds for (key, _) in data_pairs
            # d(√x)/dx = 1/(2√x)
            data_dict[key] = Δy[key] ./ (2 .* y[key])
        end
        Δt = Grassmann(size(t), even(t), index_type(t), data_dict)
        return (NoTangent(), Δt)
    end
    return y, sqrt_pullback
end

# ─── convert(::Type{Array}, t::Grassmann): dense array conversion ─────────────

function ChainRulesCore.rrule(::typeof(Base.convert), ::Type{Array}, t::Grassmann{T, N}) where {T, N}
    y = convert(Array, t)
    function convert_array_pullback(Δy)
        # Reconstruct Grassmann gradient from dense array gradient
        total_size = size(t)
        even_size = even(t)
        _, mask_range = _parity_mask(total_size, even_size)

        data_dict = Dict{NTuple{N, Int}, Array{T, N}}()
        data_pairs = nonzero_pairs(t)
        sizehint!(data_dict, length(data_pairs))

        inds_range = Vector{UnitRange}(undef, N)
        @inbounds for (sector, _) in data_pairs
            for (ind, p) in enumerate(sector)
                inds_range[ind] = mask_range[ind][p + 1]
            end
            data_dict[sector] = Δy[inds_range...]
        end
        Δt = Grassmann(total_size, even_size, index_type(t), data_dict)
        return (NoTangent(), NoTangent(), Δt)
    end
    return y, convert_array_pullback
end

# ─── maximum: gradient is one-hot at argmax ───────────────────────────────────

function ChainRulesCore.rrule(::typeof(Base.maximum), t::Grassmann{T, N, AT}) where {T, N, AT}
    y = maximum(t)
    function maximum_pullback(Δy)
        # Find global argmax across all blocks
        best_key = nothing
        best_i = nothing
        best_val = typemin(T)
        for (k, v) in nonzero_pairs(t)
            block_max, block_i = findmax(v)
            if block_max > best_val
                best_val = block_max
                best_key = k
                best_i = block_i
            end
        end
        # Gradient: one-hot at argmax position, scaled by upstream Δy
        data_dict = Dict{NTuple{N, Int}, AT}()
        Δy_unthunk = ChainRulesCore.unthunk(Δy)
        for (k, v) in nonzero_pairs(t)
            grad_block = zeros(T, size(v))
            if k == best_key
                grad_block[best_i] = Δy_unthunk
            end
            data_dict[k] = grad_block
        end
        Δt = Grassmann(size(t), even(t), index_type(t), data_dict)
        return (NoTangent(), Δt)
    end
    return y, maximum_pullback
end

# ─── abs2: gradient d|x|²/dx = 2*conj(x) ───────────────────────────────────────

function ChainRulesCore.rrule(::typeof(Base.abs2), t::Grassmann{T, N, AT}) where {T, N, AT}
    y = abs2(t)
    function abs2_pullback(Δy)
        data_dict = Dict{NTuple{N, Int}, AT}()
        data_pairs = nonzero_pairs(t)
        sizehint!(data_dict, length(data_pairs))
        Δy_unthunk = ChainRulesCore.unthunk(Δy)
        # Δy is Grassmann (from sum) or scalar
        if Δy_unthunk isa Grassmann
            @inbounds for (key, block) in data_pairs
                data_dict[key] = (2 * conj.(block)) .* Δy_unthunk[key]
            end
        else
            @inbounds for (key, block) in data_pairs
                data_dict[key] = (2 * conj.(block)) .* Δy_unthunk
            end
        end
        Δt = Grassmann(size(t), even(t), index_type(t), data_dict)
        return (NoTangent(), Δt)
    end
    return y, abs2_pullback
end

# ─── sum: gradient is ones scaled by upstream ──────────────────────────────────

function ChainRulesCore.rrule(::typeof(Base.sum), t::Grassmann{T, N, AT}) where {T, N, AT}
    y = sum(t)
    function sum_pullback(Δy)
        Δy_unthunk = ChainRulesCore.unthunk(Δy)
        data_dict = Dict{NTuple{N, Int}, AT}()
        data_pairs = nonzero_pairs(t)
        sizehint!(data_dict, length(data_pairs))
        @inbounds for (key, block) in data_pairs
            data_dict[key] = fill(convert(T, Δy_unthunk), size(block)...)
        end
        Δt = Grassmann(size(t), even(t), index_type(t), data_dict)
        return (NoTangent(), Δt)
    end
    return y, sum_pullback
end

# ─── sum(abs2, t): d/dx sum(abs2,x) = 2*conj(x) ─────────────────────────────────

function ChainRulesCore.rrule(::typeof(Base.sum), ::typeof(abs2), t::Grassmann{T, N, AT}) where {T, N, AT}
    y = sum(abs2, t)
    function sum_abs2_pullback(Δy)
        Δy_unthunk = ChainRulesCore.unthunk(Δy)
        data_dict = Dict{NTuple{N, Int}, AT}()
        data_pairs = nonzero_pairs(t)
        sizehint!(data_dict, length(data_pairs))
        @inbounds for (key, block) in data_pairs
            data_dict[key] = (2 * conj.(block)) .* Δy_unthunk
        end
        Δt = Grassmann(size(t), even(t), index_type(t), data_dict)
        return (NoTangent(), NoTangent(), Δt)
    end
    return y, sum_abs2_pullback
end

# ─── sum(sqrt, t): d/dx sum(sqrt,x) = 1/(2*sqrt(x)) ───────────────────────────────

function ChainRulesCore.rrule(::typeof(Base.sum), ::typeof(sqrt), t::Grassmann{T, N, AT}) where {T, N, AT}
    y = sum(sqrt, t)
    y_sqrt = sqrt(t)
    function sum_sqrt_pullback(Δy)
        Δy_unthunk = ChainRulesCore.unthunk(Δy)
        data_dict = Dict{NTuple{N, Int}, AT}()
        data_pairs = nonzero_pairs(t)
        sizehint!(data_dict, length(data_pairs))
        @inbounds for (key, block) in data_pairs
            data_dict[key] = (1.0 ./ (2.0 .* y_sqrt[key])) .* Δy_unthunk
        end
        Δt = Grassmann(size(t), even(t), index_type(t), data_dict)
        return (NoTangent(), NoTangent(), Δt)
    end
    return y, sum_sqrt_pullback
end

# ─── Mark non-differentiable operations ───────────────────────────────────────

@non_differentiable similar(::Grassmann)
@non_differentiable similar(::Grassmann, ::Type)
@non_differentiable abs(::Grassmann)
@non_differentiable Base.getindex(::Grassmann, ::Int, ::Tuple{Int, Int})
@non_differentiable prepare_range_dict(::Grassmann, ::Any, ::Any)
