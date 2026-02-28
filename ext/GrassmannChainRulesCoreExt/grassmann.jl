
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
