function _nested_ket_explicit(A::Grassmann)
    # A_perm[l, r, p, u, d] = A[p, l, r, u, d]
    A_perm = permutedims(
        A, (2, 3, 1, 4, 5);
        sign_function=GrassmannTensorNetworks.global_sign,
    )
    # A_signed[l, r, p, u, d] = (-1)^p A_perm[l, r, p, u, d]
    A_signed = add_parity_sign(
        A_perm, 3;
        sign_function=GrassmannTensorNetworks.global_sign,
    )
    # K0[l, r, U, d] = A_signed[l, r, (p, u), d]
    K0 = fuse(A_signed, (3, 4); index_type_fused=:in)
    # K1[l, r, U, d] = (-1)^l K0[l, r, U, d]
    K1 = add_parity_sign(
        K0, 1;
        sign_function=GrassmannTensorNetworks.global_sign,
    )
    # K2[l, r, U, d] = (-1)^r K1[l, r, U, d]
    K2 = add_parity_sign(
        K1, 2;
        sign_function=GrassmannTensorNetworks.global_sign,
    )
    # K[l, r, U, d] = (-1)^d K2[l, r, U, d]
    return add_parity_sign(
        K2, 4;
        sign_function=GrassmannTensorNetworks.global_sign,
    )
end

function _nested_bra_explicit(A::Grassmann)
    A_conj = conj(
        A; sign_function=GrassmannTensorNetworks.global_sign
    )
    # A_perm[l, p, r, u, d] = conj(A)[p, l, r, u, d]
    A_perm = permutedims(
        A_conj, (2, 1, 3, 4, 5);
        sign_function=GrassmannTensorNetworks.global_sign,
    )
    # A_signed[l, p, r, u, d] = (-1)^d A_perm[l, p, r, u, d]
    A_signed = add_parity_sign(
        A_perm, 5;
        sign_function=GrassmannTensorNetworks.global_sign,
    )
    # B0[l, R, u, d] = A_signed[l, (p, r), u, d]
    B0 = fuse(A_signed, (2, 3); index_type_fused=:in)
    # B1[l, R, u, d] = conjugation(B0[l, R, u, d], (l, u, d))
    B1 = index_conjugation(B0, (1, 3, 4))
    # B2[l, R, u, d] = (-1)^l B1[l, R, u, d]
    B2 = add_parity_sign(
        B1, 1;
        sign_function=GrassmannTensorNetworks.global_sign,
    )
    # B[l, R, u, d] = (-1)^u B2[l, R, u, d]
    return add_parity_sign(
        B2, 3;
        sign_function=GrassmannTensorNetworks.global_sign,
    )
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
    prepared_terms,
    numerator_zero,
)
    denominator = _contract_nested_hpatch3(
        nested, env, source, closed_left, closed_right
    )
    denominator_value = _nested_scalar_or_zero(denominator)
    numerator = numerator_zero
    for (left_x, right_x) in prepared_terms
        term = _contract_nested_hpatch3(
            nested, env, source, left_x, right_x
        )
        numerator += _nested_scalar_or_zero(term)
    end
    return denominator, numerator / denominator_value
end

function _nested_vbond_from_prepared(
    nested,
    env,
    source,
    closed_top,
    closed_bottom,
    prepared_terms,
    numerator_zero,
)
    denominator = _contract_nested_vpatch3(
        nested, env, source, closed_top, closed_bottom
    )
    denominator_value = _nested_scalar_or_zero(denominator)
    numerator = numerator_zero
    for (sign, top_x, bottom_x) in prepared_terms
        term = _contract_nested_vpatch3(
            nested, env, source, top_x, bottom_x
        )
        numerator += sign * _nested_scalar_or_zero(term)
    end
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
    prepared_terms = map(_operator_schmidt(operator)) do (left_op, right_op)
        left_x = nested_x_operator(nested, peps, source, left_op)
        right_x = nested_x_operator(nested, peps, neighbor, right_op)
        return left_x, right_x
    end
    denominator_value = _nested_scalar_or_zero(first(y))
    numerator_zero = zero(promote_type(
        typeof(denominator_value), eltype(operator)
    ))
    _, prepared_pullback = rrule_via_ad(
        config,
        (network, boundary) -> _nested_hbond_from_prepared(
            network,
            boundary,
            source,
            closed_left,
            closed_right,
            prepared_terms,
            numerator_zero,
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
    prepared_terms = map(_operator_schmidt(operator)) do (top_op, bottom_op)
        sign = (-one(eltype(operator)))^tensor_parity(top_op)
        top_x = nested_x_operator(nested, peps, source, top_op)
        bottom_x = nested_x_operator(nested, peps, neighbor, bottom_op)
        return sign, top_x, bottom_x
    end
    denominator_value = _nested_scalar_or_zero(first(y))
    numerator_zero = zero(promote_type(
        typeof(denominator_value), eltype(operator)
    ))
    _, prepared_pullback = rrule_via_ad(
        config,
        (network, boundary) -> _nested_vbond_from_prepared(
            network,
            boundary,
            source,
            closed_top,
            closed_bottom,
            prepared_terms,
            numerator_zero,
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
