function ChainRulesCore.rrule(::typeof(ctmrg_spectrum), t::Grassmann; normalize::Bool=true)
    y = ctmrg_spectrum(t; normalize=normalize)
    return y, _ -> (NoTangent(), NoTangent())
end

function ChainRulesCore.rrule(
    ::typeof(singular_value_spectrum_distance),
    tensor1::Grassmann,
    tensor2::Grassmann;
    normalize::Bool=true)

    y = singular_value_spectrum_distance(tensor1, tensor2; normalize=normalize)
    return y, _ -> (NoTangent(), NoTangent(), NoTangent())
end

function ChainRulesCore.rrule(::typeof(ctmrg_spectrum_distance), env1::CTMRGEnv, env2::CTMRGEnv; normalize::Bool=true)
    y = ctmrg_spectrum_distance(env1, env2; normalize=normalize)
    return y, _ -> (NoTangent(), NoTangent(), NoTangent())
end

function ChainRulesCore.rrule(::typeof(ctmrg_elementwise_distance), env1::CTMRGEnv, env2::CTMRGEnv)
    y = ctmrg_elementwise_distance(env1, env2)
    return y, _ -> (NoTangent(), NoTangent(), NoTangent())
end

@non_differentiable fixgauge!(::CTMRGEnv, ::CTMRGEnv)
@non_differentiable fixgauge(::CTMRGEnv, ::CTMRGEnv)
