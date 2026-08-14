function _nested_ket_explicit(A::Grassmann)
    # K_perm[l, r, p, u, d] <-- K[p, l, r, u, d]
    A_perm = permutedims(
        A, (2, 3, 1, 4, 5);
        sign_function=GrassmannTensorNetworks.global_sign,
    )
    # Ko1[l, r, p, u, d] = (-1)^p * K_perm[l, r, p, u, d]
    A_signed = add_parity_sign(
        A_perm, 3;
        sign_function=GrassmannTensorNetworks.global_sign,
    )
    # Ko2[l, r, U, d] = Ko1[l, r, (p, u), d]
    return fuse(A_signed, (3, 4); index_type_fused=:in)
end

function _nested_bra_explicit(A::Grassmann)
    A_conj = conj(
        A; sign_function=GrassmannTensorNetworks.global_sign
    )
    # B_perm[l, p, r, u, d] <-- B[p, l, r, u, d]
    A_perm = permutedims(
        A_conj, (2, 1, 3, 4, 5);
        sign_function=GrassmannTensorNetworks.global_sign,
    )
    # Bo1[l, p, r, u, d] = (-1)^r * B_perm[l, p, r, u, d]
    A_signed = add_parity_sign(
        A_perm, 3;
        sign_function=GrassmannTensorNetworks.global_sign,
    )
    # Bo2[l, R, u, d] = Bo1[l, (p, r), u, d]
    return fuse(A_signed, (2, 3); index_type_fused=:in)
end

function ChainRulesCore.rrule(
    config::RuleConfig{>:HasReverseMode},
    ::typeof(_nested_ket),
    A::Grassmann,
)
    y, raw_pullback = rrule_via_ad(config, _nested_ket_explicit, A)
    function pullback(delta)
        delta = unthunk(delta)
        delta isa AbstractZero && return NoTangent(), ZeroTangent()
        _, delta_A = raw_pullback(delta)
        return NoTangent(), delta_A
    end
    return y, pullback
end

function ChainRulesCore.rrule(
    config::RuleConfig{>:HasReverseMode},
    ::typeof(_nested_bra),
    A::Grassmann,
)
    y, raw_pullback = rrule_via_ad(config, _nested_bra_explicit, A)
    function pullback(delta)
        delta = unthunk(delta)
        delta isa AbstractZero && return NoTangent(), ZeroTangent()
        _, delta_A = raw_pullback(delta)
        return NoTangent(), delta_A
    end
    return y, pullback
end

function _materialize_nested_cotangent(delta, primal)
    delta = unthunk(delta)
    return delta isa Tangent ?
        ChainRulesCore.construct(
            typeof(primal), ChainRulesCore.backing(delta)
        ) : delta
end

function _add_nested_cotangents(first, second, primal)
    first = _materialize_nested_cotangent(first, primal)
    second = _materialize_nested_cotangent(second, primal)
    first isa AbstractZero && return second
    second isa AbstractZero && return first
    return first + second
end

function ChainRulesCore.rrule(
    config::RuleConfig{>:HasReverseMode},
    ::typeof(nested_network),
    peps::Square_GPEPS,
    layout::NestedLayout,
)
    ket_rules = map(A -> rrule_via_ad(config, _nested_ket, A), peps.A)
    bra_rules = map(A -> rrule_via_ad(config, _nested_bra, A), peps.A)
    nested = nested_network(peps, layout)

    function pullback(delta_nested)
        delta_nested = unthunk(delta_nested)
        delta_nested isa AbstractZero &&
            return NoTangent(), ZeroTangent(), NoTangent()
        delta_network = unthunk(getproperty(delta_nested, :network))
        delta_network isa AbstractZero &&
            return NoTangent(), ZeroTangent(), NoTangent()
        raw_gradients = map(CartesianIndices(peps.A)) do source
            _, delta_ket = last(ket_rules[source])(
                delta_network[_layout_ket_site(layout, source)]
            )
            _, delta_bra = last(bra_rules[source])(
                delta_network[_layout_bra_site(layout, source)]
            )
            return _add_nested_cotangents(
                delta_ket,
                delta_bra,
                peps.A[source],
            )
        end
        Tensor = eltype(peps.A)
        gradients = map(CartesianIndices(peps.A)) do source
            gradient = raw_gradients[source]
            primal = peps.A[source]
            gradient isa AbstractZero ?
                primal * zero(eltype(primal)) : gradient
        end
        delta_peps = Tangent{typeof(peps)}(
            ; A=Matrix{Tensor}(gradients),
              Λx=NoTangent(),
              Λy=NoTangent(),
        )
        return NoTangent(), delta_peps, NoTangent()
    end
    return nested, pullback
end

function ChainRulesCore.rrule(
    config::RuleConfig{>:HasReverseMode},
    ::typeof(nested_network),
    peps::Square_GPEPS,
)
    nested, pullback = rrule(
        config, nested_network, peps, NestedLayout(peps)
    )
    function two_argument_pullback(delta)
        _, delta_peps, _ = pullback(delta)
        return NoTangent(), delta_peps
    end
    return nested, two_argument_pullback
end

function ChainRulesCore.rrule(
    config::RuleConfig{>:HasReverseMode},
    ::typeof(nested_x_operator),
    nested::NestedNetwork,
    peps::Square_GPEPS,
    site,
    operator::Grassmann,
)
    y = nested_x_operator(nested, peps, site, operator)
    source = _source_site(site)
    A = peps.A[source]
    _, operator_pullback = rrule_via_ad(
        config,
        op -> _nested_x(
            op,
            size(A)[3], even(A)[3],
            size(A)[4], even(A)[4],
        ),
        operator,
    )
    function pullback(delta)
        delta = unthunk(delta)
        delta isa AbstractZero && return (
            NoTangent(), NoTangent(), NoTangent(),
            NoTangent(), ZeroTangent(),
        )
        _, delta_operator = operator_pullback(delta)
        return (
            NoTangent(), NoTangent(), NoTangent(),
            NoTangent(), delta_operator,
        )
    end
    return y, pullback
end

function ChainRulesCore.rrule(
    config::RuleConfig{>:HasReverseMode},
    ::typeof(nested_y_operator),
    nested::NestedNetwork,
    peps::Square_GPEPS,
    site,
    operator::Grassmann,
)
    return rrule(
        config,
        nested_x_operator,
        nested,
        peps,
        site,
        operator,
    )
end

function _nested_hbond_from_prepared(
    nested,
    env,
    source,
    closed_left,
    closed_right,
    left_x,
    right_x,
)
    denominator = _contract_nested_hpatch3(
        nested, env, source, closed_left, closed_right
    )
    denominator_value = _nested_scalar_or_zero(denominator)
    numerator_tensor = _contract_nested_hpatch3_alpha(
        nested, env, source, left_x, right_x
    )
    numerator = _nested_scalar_or_zero(numerator_tensor)
    return denominator, numerator / denominator_value
end

function _nested_vbond_from_prepared(
    nested,
    env,
    source,
    closed_top,
    closed_bottom,
    top_x,
    bottom_x,
)
    denominator = _contract_nested_vpatch3(
        nested, env, source, closed_top, closed_bottom
    )
    denominator_value = _nested_scalar_or_zero(denominator)
    numerator_tensor = _contract_nested_vpatch3_alpha(
        nested, env, source, top_x, bottom_x
    )
    numerator = _nested_scalar_or_zero(numerator_tensor)
    return denominator, numerator / denominator_value
end

function ChainRulesCore.rrule(
    config::RuleConfig{>:HasReverseMode},
    ::typeof(compute_nested_exp_hbond),
    nested::NestedNetwork,
    peps::Square_GPEPS,
    operator::Grassmann{<:Number, 4},
    env::CTMRGEnv,
    site,
)
    y = compute_nested_exp_hbond(nested, peps, operator, env, site)
    source = _source_site(site)
    neighbor = _layout_right_source(nested.layout, source)
    closed_left = nested[_layout_x_site(nested.layout, source)]
    closed_right = nested[_layout_x_site(nested.layout, neighbor)]
    left_op, right_op = _bond_operator_gsvd(operator)
    left_x = _nested_x_bond_operator(nested, peps, source, left_op)
    right_x = _nested_x_bond_operator(nested, peps, neighbor, right_op)
    right_x = add_parity_sign(
        right_x, 5; sign_function=GrassmannTensorNetworks.global_sign
    )
    _, prepared_pullback = rrule_via_ad(
        config,
        (network, boundary) -> _nested_hbond_from_prepared(
            network,
            boundary,
            source,
            closed_left,
            closed_right,
            left_x,
            right_x,
        ),
        nested,
        env,
    )
    function pullback(delta)
        delta = unthunk(delta)
        delta isa AbstractZero && return (
            NoTangent(), ZeroTangent(), NoTangent(),
            NoTangent(), ZeroTangent(), NoTangent(),
        )
        _, delta_nested, delta_env = prepared_pullback(delta)
        return (
            NoTangent(), delta_nested, NoTangent(),
            NoTangent(), delta_env, NoTangent(),
        )
    end
    return y, pullback
end

function ChainRulesCore.rrule(
    config::RuleConfig{>:HasReverseMode},
    ::typeof(compute_nested_exp_vbond),
    nested::NestedNetwork,
    peps::Square_GPEPS,
    operator::Grassmann{<:Number, 4},
    env::CTMRGEnv,
    site,
)
    y = compute_nested_exp_vbond(nested, peps, operator, env, site)
    source = _source_site(site)
    neighbor = _layout_down_source(nested.layout, source)
    closed_top = nested[_layout_x_site(nested.layout, source)]
    closed_bottom = nested[_layout_x_site(nested.layout, neighbor)]
    top_op, bottom_op = _bond_operator_gsvd(operator)
    top_x = _nested_x_bond_operator(nested, peps, source, top_op)
    bottom_x = _nested_x_bond_operator(nested, peps, neighbor, bottom_op)
    top_x = add_parity_sign(
        top_x, 5; sign_function=GrassmannTensorNetworks.global_sign
    )
    _, prepared_pullback = rrule_via_ad(
        config,
        (network, boundary) -> _nested_vbond_from_prepared(
            network,
            boundary,
            source,
            closed_top,
            closed_bottom,
            top_x,
            bottom_x,
        ),
        nested,
        env,
    )
    function pullback(delta)
        delta = unthunk(delta)
        delta isa AbstractZero && return (
            NoTangent(), ZeroTangent(), NoTangent(),
            NoTangent(), ZeroTangent(), NoTangent(),
        )
        _, delta_nested, delta_env = prepared_pullback(delta)
        return (
            NoTangent(), delta_nested, NoTangent(),
            NoTangent(), delta_env, NoTangent(),
        )
    end
    return y, pullback
end
