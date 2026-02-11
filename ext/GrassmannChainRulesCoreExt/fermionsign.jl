
@non_differentiable auto_sign(::Any...)
@non_differentiable trivial_sign(::Any...)

# rrule for add_parity_sign
function ChainRulesCore.rrule(
    ::typeof(add_parity_sign), 
    t::Grassmann, 
    ind::Int; 
    sign_function=auto_sign)

    y = add_parity_sign(t, ind; sign_function)
    
    function add_parity_sign_pullback(Δy)
        # The sign factor is self-inverse: (-1)^p * (-1)^p = 1
        # Therefore, applying the sign twice gives the identity
        Δt = add_parity_sign(Δy, ind; sign_function)
        return (NoTangent(), Δt, NoTangent())
    end
    
    return y, add_parity_sign_pullback
end

# rrule for add_perm_sign  
function ChainRulesCore.rrule(
    ::typeof(add_perm_sign), 
    t::Grassmann{T, N}, 
    dst::NTuple{N, Int}; 
    sign_function=auto_sign) where {T, N}

    y = add_perm_sign(t, dst; sign_function)
    
    function add_perm_sign_pullback(Δy)
        # The permutation sign factor is also self-inverse
        Δt = add_perm_sign(Δy, dst; sign_function)
        return (NoTangent(), Δt, NoTangent())
    end
    
    return y, add_perm_sign_pullback
end
