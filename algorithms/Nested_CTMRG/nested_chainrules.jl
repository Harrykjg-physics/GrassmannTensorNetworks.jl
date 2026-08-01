function ChainRulesCore.rrule(
    ::typeof(_graded_pair_sign), t::Grassmann, i::Int, j::Int
)
    y = _graded_pair_sign(t, i, j)
    function pullback(delta)
        delta = unthunk(delta)
        delta isa AbstractZero && return (
            NoTangent(), ZeroTangent(), NoTangent(), NoTangent()
        )
        return (
            NoTangent(),
            _graded_pair_sign(delta, i, j),
            NoTangent(),
            NoTangent(),
        )
    end
    return y, pullback
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
    for (sign, left_y, right_y) in prepared_terms
        term = _contract_nested_hpatch3(
            nested, env, source, left_y, right_y
        )
        numerator += sign * _nested_scalar_or_zero(term)
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
    for (top_y, bottom_y) in prepared_terms
        term = _contract_nested_vpatch3(
            nested, env, source, top_y, bottom_y
        )
        numerator += _nested_scalar_or_zero(term)
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
    y = compute_nested_exp_hbond(
        nested, peps, operator, env, site
    )
    source = _source_site(site)
    neighbor = CartesianIndex(
        source[1], Nmod(source[2] + 1, size(peps)[2])
    )
    identity_left = _physical_identity(peps.A[source])
    identity_right = _physical_identity(peps.A[neighbor])
    closed_left = nested_y_operator(
        nested, peps, source, identity_left
    )
    closed_right = nested_y_operator(
        nested, peps, neighbor, identity_right
    )
    prepared_terms = map(_operator_schmidt(operator)) do (left_op, right_op)
        sign = (-one(eltype(operator)))^tensor_parity(left_op)
        left_y = nested_y_operator(nested, peps, source, left_op)
        right_y = nested_y_operator(nested, peps, neighbor, right_op)
        return sign, left_y, right_y
    end
    denominator_value = _nested_scalar_or_zero(first(y))
    numerator_type =
        promote_type(typeof(denominator_value), eltype(operator))
    numerator_zero = zero(numerator_type)
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

    # Fixed-observable contract: the operator decomposition and direct PEPS
    # geometry are static. This is not an operator-gradient API.
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
    y = compute_nested_exp_vbond(
        nested, peps, operator, env, site
    )
    source = _source_site(site)
    neighbor = CartesianIndex(
        Nmod(source[1] + 1, size(peps)[1]), source[2]
    )
    identity_top = _physical_identity(peps.A[source])
    identity_bottom = _physical_identity(peps.A[neighbor])
    closed_top = nested_y_operator(
        nested, peps, source, identity_top
    )
    closed_bottom = nested_y_operator(
        nested, peps, neighbor, identity_bottom
    )
    prepared_terms = map(_operator_schmidt(operator)) do (top_op, bottom_op)
        top_y = nested_y_operator(nested, peps, source, top_op)
        bottom_y = nested_y_operator(
            nested, peps, neighbor, bottom_op
        )
        return top_y, bottom_y
    end
    denominator_value = _nested_scalar_or_zero(first(y))
    numerator_type =
        promote_type(typeof(denominator_value), eltype(operator))
    numerator_zero = zero(numerator_type)
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

    # Fixed-observable contract: the operator decomposition and direct PEPS
    # geometry are static. This is not an operator-gradient API.
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
    ::typeof(_nested_ket),
    A::Grassmann,
)
    y, raw_pullback = rrule_via_ad(config, _nested_ket_raw, A)
    function pullback(delta)
        delta = unthunk(delta)
        delta isa AbstractZero && return NoTangent(), ZeroTangent()
        _, delta_A = raw_pullback(delta)
        return NoTangent(), delta_A
    end
    return y, pullback
end

function _nested_bra_explicit_bends(A::Grassmann)
    bra = conj(
        A; sign_function=GrassmannTensorNetworks.global_sign
    )
    signed = _graded_pair_sign(bra, 1, 4)
    routed = permutedims(
        signed, (2, 3, 1, 4, 5);
        sign_function=GrassmannTensorNetworks.global_sign,
    )
    fused = fuse(routed, (3, 4); index_type_fused=:in)
    bent_left = _bend_index(fused, 1)
    bent_right = _bend_index(bent_left, 2)
    return _bend_index(bent_right, 4)
end

function ChainRulesCore.rrule(
    config::RuleConfig{>:HasReverseMode},
    ::typeof(_nested_bra),
    A::Grassmann,
)
    y = _nested_bra_raw(A)
    _, raw_pullback = rrule_via_ad(
        config, _nested_bra_explicit_bends, A
    )
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
    ::typeof(_nested_ket_for_network),
    A::Grassmann,
)
    y, pullback_A = rrule_via_ad(
        config,
        input -> _nested_ket(
            add_parity_sign(
                input, 4;
                sign_function=GrassmannTensorNetworks.global_sign,
            )
        ),
        A,
    )
    function pullback(delta)
        delta = unthunk(delta)
        delta isa AbstractZero && return NoTangent(), ZeroTangent()
        _, delta_A = pullback_A(delta)
        return NoTangent(), delta_A
    end
    return y, pullback
end

function ChainRulesCore.rrule(
    config::RuleConfig{>:HasReverseMode},
    ::typeof(_nested_bra_for_network),
    A::Grassmann,
)
    y, pullback_A = rrule_via_ad(
        config,
        input -> _nested_bra(
            add_parity_sign(
                input, 4;
                sign_function=GrassmannTensorNetworks.global_sign,
            )
        ),
        A,
    )
    function pullback(delta)
        delta = unthunk(delta)
        delta isa AbstractZero && return NoTangent(), ZeroTangent()
        _, delta_A = pullback_A(delta)
        return NoTangent(), delta_A
    end
    return y, pullback
end

function ChainRulesCore.rrule(
    ::typeof(_nested_x_for_network), xraw::Grassmann
)
    y = _nested_x_for_network(xraw)
    function pullback(delta)
        delta = unthunk(delta)
        delta isa AbstractZero && return NoTangent(), ZeroTangent()
        return NoTangent(), _nested_x_for_network(delta)
    end
    return y, pullback
end

function ChainRulesCore.rrule(
    ::typeof(_nested_y_for_network), yraw::Grassmann
)
    y = _nested_y_for_network(yraw)
    function pullback(delta)
        delta = unthunk(delta)
        delta isa AbstractZero && return NoTangent(), ZeroTangent()
        return NoTangent(), _nested_y_for_network(delta)
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
    ket_rules = map(
        A -> rrule_via_ad(config, _nested_ket_for_network, A), peps.A
    )
    bra_rules = map(
        A -> rrule_via_ad(config, _nested_bra_for_network, A), peps.A
    )
    nested = nested_network(peps, layout)

    function pullback(delta_nested)
        delta_nested = unthunk(delta_nested)
        delta_nested isa AbstractZero &&
            return NoTangent(), ZeroTangent(), NoTangent()
        delta_network = unthunk(getproperty(delta_nested, :network))
        delta_network isa AbstractZero &&
            return NoTangent(), ZeroTangent(), NoTangent()
        raw_gradients = map(CartesianIndices(peps.A)) do source
            _, dket = last(ket_rules[source])(
                delta_network[layout.ket_sites[source]]
            )
            _, dbra = last(bra_rules[source])(
                delta_network[layout.bra_sites[source]]
            )
            return _add_nested_cotangents(
                unthunk(dket), unthunk(dbra), peps.A[source]
            )
        end
        Tensor = eltype(peps.A)
        gradients = map(CartesianIndices(peps.A)) do source
            gradient = raw_gradients[source]
            primal = peps.A[source]
            gradient isa AbstractZero ? primal * zero(eltype(primal)) : gradient
        end
        delta_peps = Tangent{typeof(peps)}(
            ; A=Matrix{Tensor}(gradients),
              Λx=NoTangent(), Λy=NoTangent()
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
    nested, pullback =
        rrule(config, nested_network, peps, NestedLayout(peps))
    function two_argument_pullback(delta)
        _, delta_peps, _ = pullback(delta)
        return NoTangent(), delta_peps
    end
    return nested, two_argument_pullback
end

function _nested_y_operator_from_crossing(
    operator::Grassmann,
    crossing::Grassmann,
)
    product = contract(
        operator, crossing;
        sign_function=GrassmannTensorNetworks.global_sign,
    )
    routed = permutedims(
        product, (1, 3, 4, 5, 2, 6);
        sign_function=GrassmannTensorNetworks.global_sign,
    )
    west_fused = fuse(routed, (1, 2); index_type_fused=:out)
    raw = fuse(west_fused, (4, 5); index_type_fused=:out)
    return _nested_y_for_network(raw)
end

function ChainRulesCore.rrule(
    config::RuleConfig{>:HasReverseMode},
    ::typeof(nested_y_operator),
    nested::NestedNetwork,
    peps::Square_GPEPS,
    site,
    operator::Grassmann,
)
    y = nested_y_operator(nested, peps, site, operator)
    source = _source_site(site)
    rows, cols = size(peps)
    east_source = CartesianIndex(source[1], Nmod(source[2] + 1, cols))
    north_source = CartesianIndex(Nmod(source[1] - 1, rows), source[2])
    east_ket = nested[nested.layout.ket_sites[east_source]]
    north_bra = nested[nested.layout.bra_sites[north_source]]
    crossing = _nested_x(
        size(east_ket)[1], even(east_ket)[1],
        size(north_bra)[4], even(north_bra)[4],
        eltype(operator),
    )

    _, operator_pullback = rrule_via_ad(
        config,
        op -> _nested_y_operator_from_crossing(op, crossing),
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
