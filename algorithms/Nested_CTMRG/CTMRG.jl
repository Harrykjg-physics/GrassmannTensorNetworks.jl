
function run_nested_GCTMRG!(
    nested::NestedNetwork,
    env::CTMRGEnv,
    chi::Int; kwargs...)

    size(env) == size(nested) || throw(DimensionMismatch("nested environment and network sizes differ"))
    run_GCTMRG!(nested.network, nested.network, env, chi; kwargs...)

    return env
end
