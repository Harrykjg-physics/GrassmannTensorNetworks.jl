
@non_differentiable Grassmann(total_size, even_size, index_types, T)
@non_differentiable _parity_mask(total_size, even_parity_size)
# @non_differentiable _fixed_parity_blocks(total_size, even_parity_size; parity)

function ChainRulesCore.rrule(
                            ::Grassmann, 
                            total_size::NTuple{N, Int},
                            even_size::NTuple{N, Int},
                            index_types::NTuple{N, Symbol},
                            T::Type;
                            init::Symbol=:random,
                            parity::Symbol=:even) where {N}

    G = Grassmann(total_size, even_size, index_types, T; init=init, parity=parity)

    function Grassmann_pullback(ΔG)
        
        return (NoTangent(),
                NoTangent(),       
                NoTangent(),     
                NoTangent(),     
                NoTangent(),    
                NoTangent(),    
                NoTangent()) 
    end

    return G, Grassmann_pullback
end

Zygote.@adjoint function Square_GPEPS(
    Dphys::Int,
    Dphys_even::Int,
    Dvir::Int,
    Lx::Int,
    Ly::Int,
    params::AbstractVector{T},
    has_bond_weights::Bool=false;
    parity::Symbol=:even) where {T<:Number}

    peps = Square_GPEPS(Dphys, Dphys_even, Dvir, Lx, Ly, params, has_bond_weights; parity=parity)

    function square_gpeps_pullback(Δpeps)
        grad_params = zeros(T, length(params))
        Δpeps_unthunk = unthunk(Δpeps)
        Δpeps_unthunk isa AbstractZero && return (nothing, nothing, nothing, nothing, nothing, grad_params, nothing)

        ΔA = Δpeps_unthunk isa Square_GPEPS ? Δpeps_unthunk.A : getproperty(Δpeps_unthunk, :A)
        ΔA = unthunk(ΔA)
        ΔA isa AbstractZero && return (nothing, nothing, nothing, nothing, nothing, grad_params, nothing)

        Dvir_even = div(Dvir, 2)
        total_size = (Dphys, Dvir, Dvir, Dvir, Dvir)
        even_size = (Dphys_even, Dvir_even, Dvir_even, Dvir_even, Dvir_even)
        block_dict = _fixed_parity_blocks(total_size, even_size; parity=parity)
        sectors = sort(collect(keys(block_dict)))
        params_per_tensor = div(length(params), Lx * Ly)
        first_param = firstindex(params)

        @inbounds for site in 1:(Lx * Ly)
            row = mod(site - 1, Lx) + 1
            col = div(site - 1, Lx) + 1
            offset = first_param + (site - 1) * params_per_tensor
            Δtensor_raw = ΔA[row, col]
            Δtensor_raw = unthunk(Δtensor_raw)
            Δtensor_raw isa AbstractZero && continue
            Δtensor = _materialize_grassmann_tangent(peps.A[row, col], Δtensor_raw)

            for sector in sectors
                block_size = map(length, block_dict[sector])
                block_length = prod(block_size)
                block_range = offset:(offset + block_length - 1)
                grad_params[block_range] .= vec(Δtensor[sector])
                offset += block_length
            end
        end

        return (nothing, nothing, nothing, nothing, nothing, grad_params, nothing)
    end

    return peps, square_gpeps_pullback
end

# rrule for convert function
function ChainRulesCore.rrule(
    ::typeof(Base.convert), 
    t::Grassmann{Q1, N}, 
    Q2::Type) where {Q1, N}

    y = Base.convert(t, Q2)
    
    function convert_pullback(Δy)
        # The gradient flows back through type conversion
        # Convert the gradient back to the original type
        Δt = Base.convert(Δy, Q1)
        return (NoTangent(), Δt, NoTangent())
    end
    
    return y, convert_pullback
end

# rrule for index_conjugation (single index)
function ChainRulesCore.rrule(
    ::typeof(index_conjugation), 
    t::Grassmann{T, N}, 
    ind::Int) where {T, N}

    y = index_conjugation(t, ind)
    
    function index_conjugation_pullback(Δy)
        # Index conjugation is a linear operation, so the gradient flows back unchanged
        # except for the index type change which is reversed
        Δt = index_conjugation(Δy, ind)
        return (NoTangent(), Δt, NoTangent())
    end
    
    return y, index_conjugation_pullback
end

# rrule for index_conjugation (multiple indices)
function ChainRulesCore.rrule(
    ::typeof(index_conjugation), 
    t::Grassmann{T, N}, 
    inds::NTuple{NR, Int}) where {T, N, NR}

    y = index_conjugation(t, inds)
    
    function index_conjugation_pullback(Δy)
        # Reverse the conjugation in the gradient
        Δt = index_conjugation(Δy, inds)
        return (NoTangent(), Δt, NoTangent())
    end
    
    return y, index_conjugation_pullback
end
