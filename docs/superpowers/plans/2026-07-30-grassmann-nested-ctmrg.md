# Grassmann Nested CTMRG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a native Grassmann doubled-cell nested tensor network with one-site and nearest-neighbor measurements, reverse-mode AD, a spinless-fermion optimization example, and the requested `chi=4,8,12` server acceptance runs.

**Architecture:** Each `Square_GPEPS` tensor becomes a `2 x 2` K/Y/X/B checkerboard of four-leg `Grassmann` tensors. The doubled matrix is passed unchanged to the existing `CTMRGEnv` and `run_GCTMRG!`; measurements replace the physical identity in Y tensors and contract one-site or three-node nearest-neighbor patches. ChainRules rules propagate cotangents through the numerical K/B paths while treating layout and crossing geometry as static.

**Tech Stack:** Julia 1.9+, GrassmannTensorNetworks, TensorOperations, ChainRulesCore, Zygote, FiniteDifferences, HDF5, existing GCTMRG.

## Global Constraints

- Keep `ChainRulesCore` and `Zygote` as weak dependencies loaded through `GrassmannChainRulesCoreExt`.
- Do not add PEPSKit or TensorKit dependencies.
- Do not copy or fork the existing CTMRG move/projector implementation.
- Every axis reorder uses Grassmann-aware operations with `sign_function=global_sign`.
- X odd-odd exchange is exactly `-1`; all other parity combinations are `+1`.
- Support only one-site and horizontal/vertical nearest-neighbor source operators.
- Preserve existing APIs and avoid unrelated refactoring.
- Test Julia code only on `jkkong@172.23.26.248` through the `julia-server-test` workflow, using `julia_grassmann`, at most five concurrent tasks, and a 20 GB per-task memory guard.
- Acceptance parameters are `D=2`, `chi=4,8,12`, `ctmrg_iter=20`, and `ad_iter=20`.
- Numerical acceptance requires finite, error-free runs whose three-energy table shows an overall trend toward `-6.170521774015...`; no fixed relative-error threshold is imposed.

## Reference Inputs

- Compare geometry and notation against
  `D:\C 盘备份\Back up\Work_Save\coding\VS files\AI optimized Project\PEPSKit_nesting\src\network.jl`,
  `observables.jl`, and `chainrules.jl`; translate their TensorKit braiding
  into this repository's `Grassmann` sign operations rather than adding those
  packages.
- The derivation source is
  `D:\C 盘备份\Back up\Work_Save\latex file\2026\0730_Nested fTN\main.pdf`.
- The approved K/Y/X/B diagrams are the three attached files under
  `C:\Users\Harry\.codex\attachments\33ad0ec5-f7ee-426a-8cc9-10fae725a17d\`.

---

## File Map

**Create**

- `algorithms/Nested_CTMRG/nested_network.jl` - layout, graded crossing primitives, K/Y/X/B construction, network wrapper, and CTMRG adapters.
- `algorithms/Nested_CTMRG/measurements.jl` - operator-dressed Y tensors and normalized one-site/three-node two-site contractions.
- `algorithms/Nested_CTMRG/nested_chainrules.jl` - reverse rules for pair signs, nested nodes, network assembly, and operator-dressed Y tensors.
- `test/nested_network.jl` - layout, sign, link, factorization, and CTMRG smoke tests.
- `test/nested_measurements.jl` - one-site/two-site normalization and reduced-layer comparisons.
- `test/nested_chainrules.jl` - centered finite-difference checks.
- `examples/Spinless_Fermion_2D_Square_AD_nested/Project.toml` - example-only Zygote dependency.
- `examples/Spinless_Fermion_2D_Square_AD_nested/Spinless_Fermion_2D_Square_AD_nested.jl` - optimization and acceptance driver.

**Modify**

- `src/algorithms.jl` - include the network and measurement source files.
- `src/GrassmannTensorNetworks.jl` - export the public nested API.
- `ext/GrassmannChainRulesCoreExt/GrassmannChainRulesCoreExt.jl` - import nested internals and include nested reverse rules.
- `test/runtests.jl` - include the nested tests.
- `test/server_runtests.jl` - include the same nested tests in the isolated server harness.
- `docs/algorithms.md` - document the nested API and supported geometries.

---

### Task 1: Nested Layout, Wrapper, and Package Wiring

**Files:**

- Create: `algorithms/Nested_CTMRG/nested_network.jl`
- Create: `test/nested_network.jl`
- Modify: `src/algorithms.jl`
- Modify: `src/GrassmannTensorNetworks.jl`
- Modify: `test/runtests.jl`
- Modify: `test/server_runtests.jl`

**Interfaces:**

- Consumes: `Square_GPEPS`, `Grassmann`, `CTMRGEnv`, `run_GCTMRG!`.
- Produces:
  - `NestedLayout(source_size::Tuple{Int,Int})`
  - `NestedLayout(peps::Square_GPEPS)`
  - `NestedNetwork(network, layout, x_crossings)`

- [ ] **Step 1: Write the failing layout and export tests**

Add to `test/nested_network.jl`:

```julia
using Test
using GrassmannTensorNetworks

@testset "Nested layout" begin
    layout = NestedLayout((2, 3))
    @test size(layout) == (4, 6)
    @test layout.source_size == (2, 3)
    @test layout.ket_sites[2, 3] == CartesianIndex(3, 5)
    @test layout.y_sites[2, 3] == CartesianIndex(3, 6)
    @test layout.x_sites[2, 3] == CartesianIndex(4, 5)
    @test layout.bra_sites[2, 3] == CartesianIndex(4, 6)
end
```

Add `include("nested_network.jl")` inside the top-level testset in both test
entry points. Add assertions to the existing package-symbol smoke test:

```julia
@test isdefined(GrassmannTensorNetworks, :NestedLayout)
@test isdefined(GrassmannTensorNetworks, :NestedNetwork)
```

- [ ] **Step 2: Run the focused test to verify failure**

Run on the Julia server:

```bash
julia_grassmann --project=. test/server_runtests.jl
```

Expected: FAIL with `UndefVarError: NestedLayout not defined`.

- [ ] **Step 3: Implement layout and wrapper types**

Add to `algorithms/Nested_CTMRG/nested_network.jl`:

```julia
struct NestedLayout
    source_size::Tuple{Int, Int}
    nested_size::Tuple{Int, Int}
    ket_sites::Matrix{CartesianIndex{2}}
    y_sites::Matrix{CartesianIndex{2}}
    x_sites::Matrix{CartesianIndex{2}}
    bra_sites::Matrix{CartesianIndex{2}}
end

function NestedLayout(source_size::Tuple{Int, Int})
    rows, cols = source_size
    rows > 0 && cols > 0 ||
        throw(ArgumentError("source unit-cell dimensions must be positive"))
    ket = [CartesianIndex(2r - 1, 2c - 1) for r in 1:rows, c in 1:cols]
    y = [CartesianIndex(2r - 1, 2c) for r in 1:rows, c in 1:cols]
    x = [CartesianIndex(2r, 2c - 1) for r in 1:rows, c in 1:cols]
    bra = [CartesianIndex(2r, 2c) for r in 1:rows, c in 1:cols]
    return NestedLayout(source_size, (2rows, 2cols), ket, y, x, bra)
end

NestedLayout(peps::Square_GPEPS) = NestedLayout(size(peps))
Base.size(layout::NestedLayout) = layout.nested_size
Base.size(layout::NestedLayout, dim::Integer) = layout.nested_size[dim]

struct NestedNetwork{T<:Number, X<:AbstractMatrix}
    network::Matrix{Grassmann{T, 4}}
    layout::NestedLayout
    x_crossings::X
end

Base.size(nested::NestedNetwork, args...) = size(nested.network, args...)
Base.axes(nested::NestedNetwork, args...) = axes(nested.network, args...)
Base.getindex(nested::NestedNetwork, inds...) = getindex(nested.network, inds...)
```

Wire the new network source into `src/algorithms.jl` after the existing CTMRG
core includes:

```julia
include(joinpath(@__DIR__, "..", "algorithms", "Nested_CTMRG", "nested_network.jl"))
```

Export the types and entry points from `src/GrassmannTensorNetworks.jl`:

```julia
export NestedLayout, NestedNetwork
```

- [ ] **Step 4: Run the focused tests**

Run the same server command. Expected: layout and wrapper symbol tests PASS.

- [ ] **Step 5: Commit**

```bash
git add algorithms/Nested_CTMRG/nested_network.jl \
        src/algorithms.jl src/GrassmannTensorNetworks.jl \
        test/nested_network.jl test/runtests.jl test/server_runtests.jl
git commit -m "feat: add nested CTMRG layout"
```

---

### Task 2: Graded Crossing and K/Y/X/B Local Nodes

**Files:**

- Modify: `algorithms/Nested_CTMRG/nested_network.jl`
- Modify: `test/nested_network.jl`

**Interfaces:**

- Consumes: `Grassmann`, `Square_GPEPS`, `permutedims`, `fuse`,
  `index_conjugation`, `add_parity_sign`, `global_sign`.
- Produces:
  - `_graded_pair_sign(t, i, j)`
  - `_nested_ket(A)`
  - `_nested_bra(A)`
  - `_nested_x(horizontal_size, horizontal_even, vertical_size, vertical_even, T)`
  - `_placed_nested_x(x)`
  - `_nested_y(operator, horizontal_size, horizontal_even, vertical_size, vertical_even)`
  - `_physical_identity(A)`
  - `_nested_reduced_basis(ordered)`

- [ ] **Step 1: Write failing sign and node-structure tests**

Append:

```julia
import GrassmannTensorNetworks:
    _graded_pair_sign, _nested_ket, _nested_bra, _nested_x, _nested_y

@testset "Nested graded primitives" begin
    odd_crossing = _nested_x(1, 0, 1, 0, Float64)
    @test !haskey(odd_crossing, (0, 0, 0, 0))
    @test only(odd_crossing[(1, 1, 1, 1)]) == -1

    mixed_crossing = _nested_x(2, 1, 2, 1, Float64)
    dense = convert(Array, mixed_crossing)
    @test dense[1, 1, 1, 1] == 1
    @test dense[2, 2, 1, 1] == 1
    @test dense[1, 1, 2, 2] == 1
    @test dense[2, 2, 2, 2] == -1

    peps = Square_GPEPS(2, 1, 2, 1, 1, Float64, false)
    A = peps.A[1, 1]
    K = _nested_ket(A)
    B = _nested_bra(A)
    @test size(K) == (2, 4, 2, 2)
    @test size(B) == (2, 2, 4, 2)
    @test index_type(K) == (:out, :in, :in, :out)
    @test index_type(B) == (:out, :in, :in, :out)

    identity = Grassmann(Matrix{Float64}(I, 2, 2), (2, 2), (1, 1), (:out, :in))
    Y = _nested_y(identity, 2, 1, 2, 1)
    @test size(Y) == (4, 2, 2, 4)
    @test index_type(Y) == (:out, :in, :in, :out)
end
```

- [ ] **Step 2: Run and confirm missing-function failures**

Expected: FAIL at import because `_graded_pair_sign` and node functions do not
exist.

- [ ] **Step 3: Implement the pair sign and pure crossing**

Add:

```julia
function _graded_pair_sign(t::Grassmann{T, N, AT}, i::Int, j::Int) where {T, N, AT}
    1 <= i <= N && 1 <= j <= N && i != j ||
        throw(ArgumentError("graded-pair indices must be distinct and in bounds"))
    blocks = Dict{NTuple{N, Int}, AT}()
    for (sector, block) in nonzero_pairs(t)
        blocks[sector] = (-1)^(sector[i] * sector[j]) .* block
    end
    return Grassmann(size(t), even(t), index_type(t), blocks)
end

function _nested_x(
    horizontal_size::Int,
    horizontal_even::Int,
    vertical_size::Int,
    vertical_even::Int,
    ::Type{T}) where {T}

    dense = zeros(T, horizontal_size, horizontal_size, vertical_size, vertical_size)
    parity(index, even_dim) = index <= even_dim ? 0 : 1
    for h in 1:horizontal_size, v in 1:vertical_size
        dense[h, h, v, v] =
            (-one(T))^(parity(h, horizontal_even) * parity(v, vertical_even))
    end
    return Grassmann(
        dense,
        (horizontal_size, horizontal_size, vertical_size, vertical_size),
        (horizontal_even, horizontal_even, vertical_even, vertical_even),
        (:out, :in, :in, :out),
    )
end

_placed_nested_x(x::Grassmann) =
    add_parity_sign(x, 1; sign_function=global_sign)
```

`_nested_x` is the raw braid and its odd-odd entry must remain `-1`.
The west-leg twist belongs to the placed X node, not to the raw crossing.

- [ ] **Step 4: Implement K, B, identity, and operator-dressed Y**

Use one explicit arrow-bend helper so every bend carries the same parity
twist:

```julia
function _bend_index(t::Grassmann, index::Int)
    return add_parity_sign(
        index_conjugation(t, index), index; sign_function=global_sign
    )
end

function _nested_ket_raw(A::Grassmann{T, 5}) where {T}
    signed = _graded_pair_sign(A, 1, 3)
    routed = permutedims(signed, (2, 1, 3, 4, 5); sign_function=global_sign)
    return fuse(routed, (2, 3); index_type_fused=:in)
end
_nested_ket(A::Grassmann{T, 5}) where {T} = _nested_ket_raw(A)

function _nested_bra_raw(A::Grassmann{T, 5}) where {T}
    bra = conj(A; sign_function=global_sign)
    signed = _graded_pair_sign(bra, 1, 4)
    routed = permutedims(signed, (2, 3, 1, 4, 5); sign_function=global_sign)
    fused = fuse(routed, (3, 4); index_type_fused=:in)
    return foldl(_bend_index, (1, 2, 4); init=fused)
end
_nested_bra(A::Grassmann{T, 5}) where {T} = _nested_bra_raw(A)

function _physical_identity(A::Grassmann{T, 5}) where {T}
    p, pe = size(A, 1), even(A)[1]
    return Grassmann(Matrix{T}(I, p, p), (p, p), (pe, pe), (:out, :in))
end

function _nested_y(
    operator::Grassmann{T, 2},
    horizontal_size::Int,
    horizontal_even::Int,
    vertical_size::Int,
    vertical_even::Int) where {T}

    crossing = _nested_x(
        horizontal_size, horizontal_even, vertical_size, vertical_even, T
    )
    product = contract(operator, crossing; sign_function=global_sign)
    routed = permutedims(
        product, (1, 3, 4, 5, 2, 6); sign_function=global_sign
    )
    west_fused = fuse(routed, (1, 2); index_type_fused=:out)
    return fuse(west_fused, (4, 5); index_type_fused=:out)
end
```

- [ ] **Step 5: Add sector-oracle, anisotropic, and factorization tests**

Native `reduced_tensor` stores each external pair in bra-first order.  The
categorical-to-native comparison must include the derived eight-leg basis
isomorphism before applying the exact reduced-layer fusers:

```julia
function _nested_reduced_basis(ordered)
    corrected = add_perm_sign(
        ordered, (2, 3, 5, 4, 6, 7, 1, 8);
        sign_function=global_sign,
    )
    for index in (2, 4, 6)
        corrected = add_parity_sign(
            corrected, index; sign_function=global_sign
        )
    end
    return corrected
end

function contract_nested_tile(K, Y, X, B)
    ky = contract(K, Y, (2, 1); sign_function=global_sign)
    kx = contract(ky, X, (3, 3); sign_function=global_sign)
    tile = contract(kx, B, ((5, 7), (3, 1)); sign_function=global_sign)
    ordered = permutedims(
        tile, (5, 1, 7, 3, 4, 2, 8, 6); sign_function=global_sign
    )
    ordered = _nested_reduced_basis(ordered)

    ordered = add_parity_sign(
        ordered, 1; sign_function=global_sign
    )
    left = fuse(ordered, (1, 2); index_type_fused=:out)
    left = add_perm_sign(
        left, (1, 3, 2, 4, 5, 6, 7); sign_function=global_sign
    )
    right = fuse(left, (2, 3); index_type_fused=:in)
    right = add_perm_sign(
        right, (1, 2, 4, 3, 5, 6); sign_function=global_sign
    )
    up = fuse(right, (3, 4); index_type_fused=:in)
    up = add_parity_sign(up, 4; sign_function=global_sign)
    return fuse(up, (4, 5); index_type_fused=:out)
end

@testset "Local nested factorization" begin
    for T in (Float64, ComplexF64)
        peps = Square_GPEPS(2, 1, 2, 1, 1, T, false)
        A = peps.A[1, 1]
        K, B = _nested_ket(A), _nested_bra(A)
        Xraw = _nested_x(
            size(A)[2], even(A)[2], size(A)[5], even(A)[5], T
        )
        Y = _nested_y(
            _physical_identity(A),
            size(A)[3], even(A)[3],
            size(A)[4], even(A)[4],
        )
        @test contract_nested_tile(
            K, Y, _placed_nested_x(Xraw), B
        ) ≈ reduced_tensor(A) rtol=1e-10
    end
end
```

Also enumerate all 128 compatible parity sectors for the minimal
even-dimension-one tensor and assert zero corrected sign residual.  Add a
mixed-multiplicity anisotropic test so that horizontal X is sourced from
`B.W` and vertical X from `K.S`; the isotropic `D=2` case cannot detect that
argument reversal.

The exact contraction indices, bra-first order, placed-X twist, basis
isomorphism, and reduced-layer fusers are implementation invariants.  Do not
weaken this comparison or substitute a dense production contraction that
bypasses Grassmann signs.

- [ ] **Step 6: Run tests and commit**

Expected: odd-odd sign, node structure, and local factorization all PASS.

```bash
git add algorithms/Nested_CTMRG/nested_network.jl test/nested_network.jl
git commit -m "feat: construct graded nested tensor nodes"
```

---

### Task 3: Assemble Periodic Nested Networks and Reuse GCTMRG

> **Superseded:** Do not execute the steps in this Task 3 section. The
> approved universal existing-operations replacement is
> `docs/superpowers/plans/2026-07-31-grassmann-nested-existing-ops.md`.
> Complete and independently review both tasks in that supplement, then
> resume this parent plan at Task 4. The supplement preserves the `2m x 2n`
> checkerboard and replaces the non-composable comparison-map proposal with
> universal K/B/X/Y placement signs.

<!-- SUPERSEDED TASK 3 ARCHIVE: retained only for historical context.

**Files:**

- Modify: `algorithms/Nested_CTMRG/nested_network.jl`
- Modify: `test/nested_network.jl`

**Interfaces:**

- Consumes: all Task 2 primitives.
- Produces working `nested_network`, `initialize_nested_environment`, and
  `run_nested_GCTMRG!`.

- [ ] **Step 1: Write failing assembly and periodic-link tests**

```julia
@test isdefined(GrassmannTensorNetworks, :nested_network)
@test isdefined(GrassmannTensorNetworks, :initialize_nested_environment)
@test isdefined(GrassmannTensorNetworks, :run_nested_GCTMRG!)

@testset "Nested network assembly" begin
    peps = Square_GPEPS(2, 1, 2, 2, 2, Float64, false)
    nested = nested_network(peps)
    @test size(nested) == (4, 4)
    @test nested[nested.layout.ket_sites[1, 1]] ≈ _nested_ket(peps.A[1, 1])
    @test nested[nested.layout.bra_sites[2, 2]] ≈ _nested_bra(peps.A[2, 2])

    for r in axes(nested, 1), c in axes(nested, 2)
        below = Nmod(r + 1, size(nested, 1))
        right = Nmod(c + 1, size(nested, 2))
        @test size(nested[r, c], 2) == size(nested[r, right], 1)
        @test even(nested[r, c])[2] == even(nested[r, right])[1]
        @test size(nested[r, c], 4) == size(nested[below, c], 3)
        @test even(nested[r, c])[4] == even(nested[below, c])[3]
        @test index_type(nested[r, c])[2] != index_type(nested[r, right])[1]
        @test index_type(nested[r, c])[4] != index_type(nested[below, c])[3]
    end

    wrong = NestedLayout((1, 1))
    @test_throws ArgumentError nested_network(peps, wrong)
end
```

Export these Task 3 entry points from `src/GrassmannTensorNetworks.jl`:

```julia
export nested_network, initialize_nested_environment, run_nested_GCTMRG!
```

- [ ] **Step 2: Run and verify the assembly failure**

Expected: FAIL because `nested_network` has no assembly method.

- [ ] **Step 3: Implement periodic checkerboard assembly**

```julia
function nested_network(
    peps::Square_GPEPS{T},
    layout::NestedLayout=NestedLayout(peps)) where {T}

    layout.source_size == size(peps) ||
        throw(ArgumentError("layout source size does not match PEPS unit cell"))
    rows, cols = size(peps)
    ket = [_nested_ket(peps.A[r, c]) for r in 1:rows, c in 1:cols]
    bra = [_nested_bra(peps.A[r, c]) for r in 1:rows, c in 1:cols]
    x = [
        _nested_x(
            size(bra[r, c])[1], even(bra[r, c])[1],
            size(ket[r, c])[4], even(ket[r, c])[4], T,
        ) for r in 1:rows, c in 1:cols
    ]

    tensors = Matrix{Grassmann{T, 4}}(undef, size(layout)...)
    for r in 1:rows, c in 1:cols
        tensors[layout.ket_sites[r, c]] = ket[r, c]
        tensors[layout.bra_sites[r, c]] = bra[r, c]
        tensors[layout.x_sites[r, c]] = _placed_nested_x(x[r, c])
        north_bra = bra[Nmod(r - 1, rows), c]
        east_ket = ket[r, Nmod(c + 1, cols)]
        tensors[layout.y_sites[r, c]] = _nested_y(
            _physical_identity(peps.A[r, c]),
            size(east_ket, 1), even(east_ket)[1],
            size(north_bra, 4), even(north_bra)[4],
        )
    end
    nested = NestedNetwork(tensors, layout, x)
    _check_nested_links(nested)
    return nested
end
```

Before accepting assembly, add a `2x2` reduced / `4x4` nested periodic
contraction test for all four `(twist_x, twist_y)` spin structures.  Compose
the proven external basis maps explicitly across tile boundaries.  A naïve
periodic trace that omits those maps is not a valid comparison and must not
be used to alter the already proven local signs.

Add the link validator used above:

```julia
function _check_nested_link(
    left::Grassmann,
    left_axis::Int,
    right::Grassmann,
    right_axis::Int,
    left_site::CartesianIndex{2},
    right_site::CartesianIndex{2},
)
    size(left, left_axis) == size(right, right_axis) ||
        throw(DimensionMismatch(
            "nested link $left_site[$left_axis] -> " *
            "$right_site[$right_axis] has unequal dimensions",
        ))
    even(left)[left_axis] == even(right)[right_axis] ||
        throw(DimensionMismatch(
            "nested link $left_site[$left_axis] -> " *
            "$right_site[$right_axis] has unequal even sectors",
        ))
    index_type(left)[left_axis] != index_type(right)[right_axis] ||
        throw(DimensionMismatch(
            "nested link $left_site[$left_axis] -> " *
            "$right_site[$right_axis] has equal arrow directions",
        ))
    return nothing
end

function _check_nested_links(nested::NestedNetwork)
    for site in CartesianIndices(nested.network)
        below = CartesianIndex(
            Nmod(site[1] + 1, size(nested, 1)), site[2]
        )
        right = CartesianIndex(
            site[1], Nmod(site[2] + 1, size(nested, 2))
        )
        _check_nested_link(
            nested[site], 2, nested[right], 1, site, right
        )
        _check_nested_link(
            nested[site], 4, nested[below], 3, site, below
        )
    end
    return nested
end
```

- [ ] **Step 4: Implement CTMRG adapters and smoke test**

```julia
initialize_nested_environment(
    nested::NestedNetwork, chi::Int, chi_even::Int=div(chi, 2)
) = CTMRGEnv(nested.network, chi, chi_even)

function run_nested_GCTMRG!(
    nested::NestedNetwork,
    env::CTMRGEnv,
    chi::Int;
    kwargs...)

    size(env) == size(nested) ||
        throw(DimensionMismatch("nested environment and network sizes differ"))
    run_GCTMRG!(nested.network, nested.network, env, chi; kwargs...)
    return env
end
```

Add:

```julia
@testset "Nested CTMRG smoke" begin
    peps = Square_GPEPS(2, 1, 2, 1, 1, Float64, false)
    nested = nested_network(peps)
    env = initialize_nested_environment(nested, 4)
    @test size(env) == size(nested)
    @test run_nested_GCTMRG!(
        nested, env, 4; ctmrg_iter=1, verbosity=0, save_iter=0
    ) === env
end
```

- [ ] **Step 5: Run tests and commit**

```bash
git add algorithms/Nested_CTMRG/nested_network.jl test/nested_network.jl
git commit -m "feat: assemble nested network for GCTMRG"
```

-->

---

### Task 4: Operator-Dressed Y and One-Site Measurements

> **Testing amendment:** Keep the production steps in this task, but replace
> its CTMRG-based tests with Task 1 of
> `docs/superpowers/plans/2026-07-31-nested-exact-measurement-tests.md`.
> Measurement tests must compare exact nested and reduced closures without
> constructing or iterating a CTMRG environment.

**Files:**

- Create: `algorithms/Nested_CTMRG/measurements.jl`
- Create: `test/nested_measurements.jl`
- Modify: `src/algorithms.jl`
- Modify: `test/runtests.jl`
- Modify: `test/server_runtests.jl`
- Modify: `src/GrassmannTensorNetworks.jl`

**Interfaces:**

- Consumes: `NestedNetwork`, `_nested_y`, `_nested_y_for_network`, existing scalar
  `compute_exp_site(Tbulk, Timp, El, Er, Eu, Ed, Clu, Cru, Cld, Crd)`.
- Produces:
  - `nested_y_operator(nested, peps, site, operator)`
  - `compute_nested_exp_site(nested, peps, operator, env, site)`
  - `compute_nested_exp_site(nested, peps, operator_matrix, env)`

- [ ] **Step 1: Write exact one-site nested/reduced tests**

Execute Task 1 Steps 1-3 of
`docs/superpowers/plans/2026-07-31-nested-exact-measurement-tests.md`.
Those steps contain the complete environment-free identity, number,
validation, matrix-unit-cell, and four-spin-structure tests.

- [ ] **Step 2: Run and confirm missing measurement failure**

Use the guarded RED command in the exact-test companion. Expected: FAIL
because `nested_y_operator`, `compute_nested_exp_site`, or
`_check_nested_operator_unit_cell` is missing. No CTMRG iteration is run.

- [ ] **Step 3: Implement operator-dressed Y and scalar measurement**

Append to `src/algorithms.jl`:

```julia
include(joinpath(@__DIR__, "..", "algorithms", "Nested_CTMRG", "measurements.jl"))
```

Append to `src/GrassmannTensorNetworks.jl`:

```julia
export nested_y_operator
export compute_nested_exp_site, compute_nested_exp_hbond, compute_nested_exp_vbond
```

Create `algorithms/Nested_CTMRG/measurements.jl` with:

```julia
_source_site(site::CartesianIndex{2}) = site
_source_site(site::Tuple{Int, Int}) = CartesianIndex(site)

function _nested_y_operator_raw(
    nested::NestedNetwork,
    peps::Square_GPEPS,
    source::CartesianIndex{2},
    operator::Grassmann{T, 2},
) where {T}
    rows, cols = size(peps)
    east_source = CartesianIndex(source[1], Nmod(source[2] + 1, cols))
    north_source = CartesianIndex(Nmod(source[1] - 1, rows), source[2])
    east_ket = nested[nested.layout.ket_sites[east_source]]
    north_bra = nested[nested.layout.bra_sites[north_source]]
    raw = _nested_y(
        operator,
        size(east_ket)[1], even(east_ket)[1],
        size(north_bra)[4], even(north_bra)[4],
    )
    return _nested_y_for_network(raw)
end

function nested_y_operator(
    nested::NestedNetwork,
    peps::Square_GPEPS,
    site,
    operator::Grassmann{T, 2}) where {T}

    source = _source_site(site)
    checkbounds(Bool, peps.A, source) ||
        throw(ArgumentError("source site $source is outside the unit cell"))
    physical_size = size(peps.A[source])[1]
    physical_even = even(peps.A[source])[1]
    size(operator) == (physical_size, physical_size) ||
        throw(DimensionMismatch("operator physical dimensions do not match PEPS"))
    even(operator) == (physical_even, physical_even) ||
        throw(DimensionMismatch("operator physical parity split does not match PEPS"))
    index_type(operator) == (:out, :in) ||
        throw(ArgumentError("operator arrows must be (:out, :in)"))
    return _nested_y_operator_raw(nested, peps, source, operator)
end

function compute_nested_exp_site(
    nested::NestedNetwork,
    peps::Square_GPEPS,
    operator::Grassmann{<:Number, 2},
    env::CTMRGEnv,
    site)

    source = _source_site(site)
    ysite = nested.layout.y_sites[source]
    impurity = nested_y_operator(nested, peps, source, operator)
    return compute_exp_site(
        nested[ysite], impurity,
        env.El[ysite], env.Er[ysite], env.Eu[ysite], env.Ed[ysite],
        env.Clu[ysite], env.Cru[ysite], env.Cld[ysite], env.Crd[ysite],
    )
end
```

- [ ] **Step 4: Implement matrix overload and direct-layer comparison**

```julia
function compute_nested_exp_site(
    nested::NestedNetwork,
    peps::Square_GPEPS,
    operators::AbstractMatrix{<:Grassmann{Q, 2}},
    env::CTMRGEnv) where {Q}

    size(operators) == size(peps) ||
        throw(DimensionMismatch("operator and PEPS unit cells differ"))
    denominator = Matrix{promote_type(eltype(peps), Q)}(
        undef, size(peps)...
    )
    values = similar(denominator)
    for site in CartesianIndices(peps.A)
        denominator[site], values[site] =
            compute_nested_exp_site(nested, peps, operators[site], env, site)
    end
    return denominator, values
end
```

Add `_check_nested_operator_unit_cell` and the exact comparison tests from
Task 1 Steps 1-5 of the exact-test companion. The companion replaces the
old finite-iteration cross-representation smoke test with exact nested and
reduced denominator/numerator comparisons at `rtol=5e-12` and
`atol=1e-12`.

- [ ] **Step 5: Run tests and commit**

```bash
git add algorithms/Nested_CTMRG/measurements.jl \
        src/algorithms.jl test/nested_measurements.jl \
        test/runtests.jl test/server_runtests.jl \
        src/GrassmannTensorNetworks.jl
git commit -m "feat: measure nested one-site operators"
```

---

### Task 5: Horizontal and Vertical Three-Node Measurements

> **Testing amendment:** Keep the production steps in this task, but replace
> its CTMRG-based tests with Task 2 of
> `docs/superpowers/plans/2026-07-31-nested-exact-measurement-tests.md`.
> Measurement tests must compare exact horizontal and vertical nested and
> reduced bond closures without constructing or iterating a CTMRG
> environment.

**Files:**

- Modify: `algorithms/Nested_CTMRG/measurements.jl`
- Modify: `test/nested_measurements.jl`

**Interfaces:**

- Consumes: operator-dressed Y tensors and nested CTMRG environments.
- Produces:
  - `_check_nested_bond_operator(peps, operator, source, neighbor)`
  - `_contract_nested_hpatch3(nested, env, source, left_y, right_y)`
  - `_contract_nested_vpatch3(nested, env, source, top_y, bottom_y)`
  - `compute_nested_exp_hbond(...)`
  - `compute_nested_exp_vbond(...)`

- [ ] **Step 1: Write exact identity and Hamiltonian-bond tests**

Execute Task 2 Steps 1-3 of
`docs/superpowers/plans/2026-07-31-nested-exact-measurement-tests.md`.
Those steps contain the complete two-site identity, operator reconstruction,
rank-six reduced closure, horizontal/vertical comparison, nonzero-numerator,
and four-spin-structure tests.

- [ ] **Step 2: Run and verify missing-function failures**

Expected: FAIL because the two bond functions do not exist.

- [ ] **Step 3: Factor a two-site operator into parity-preserving one-site terms**

Implement an internal exact operator Schmidt decomposition. This converts one
two-site insertion into a sum of products of two operator-dressed Y tensors
and avoids inventing a separate rank-eight Grassmann impurity type:

```julia
function _check_nested_bond_operator(
    peps::Square_GPEPS,
    operator::Grassmann{<:Number, 4},
    source::CartesianIndex{2},
    neighbor::CartesianIndex{2},
)
    checkbounds(Bool, peps.A, source) ||
        throw(ArgumentError("source site $source is outside the unit cell"))
    checkbounds(Bool, peps.A, neighbor) ||
        throw(ArgumentError(
            "neighbor site $neighbor is outside the unit cell"
        ))
    source_physical = size(peps.A[source])[1]
    neighbor_physical = size(peps.A[neighbor])[1]
    expected_size = (
        source_physical,
        neighbor_physical,
        source_physical,
        neighbor_physical,
    )
    size(operator) == expected_size ||
        throw(DimensionMismatch(
            "bond-operator physical dimensions do not match PEPS"
        ))
    source_even = even(peps.A[source])[1]
    neighbor_even = even(peps.A[neighbor])[1]
    expected_even = (
        source_even,
        neighbor_even,
        source_even,
        neighbor_even,
    )
    even(operator) == expected_even ||
        throw(DimensionMismatch(
            "bond-operator parity splits do not match PEPS"
        ))
    index_type(operator) == (:out, :out, :in, :in) ||
        throw(ArgumentError(
            "bond-operator arrows must be (:out, :out, :in, :in)"
        ))
    tensor_parity(operator) == 0 ||
        throw(ArgumentError("bond operator must have even total parity"))
    return nothing
end

function _operator_schmidt(operator::Grassmann{T, 4}) where {T}
    dout1, dout2, din1, din2 = size(operator)
    grouped = permutedims(
        operator, (1, 3, 2, 4); sign_function=global_sign
    )
    matrix = reshape(
        convert(Array, grouped),
        dout1 * din1, dout2 * din2,
    )
    parity(index, even_dim) = index <= even_dim ? 0 : 1
    row_parity = [
        mod(
            parity(out, even(operator)[1]) +
            parity(input, even(operator)[3]),
            2,
        ) for out in 1:dout1, input in 1:din1
    ][:]
    col_parity = [
        mod(
            parity(out, even(operator)[2]) +
            parity(input, even(operator)[4]),
            2,
        ) for out in 1:dout2, input in 1:din2
    ][:]

    terms = Tuple{Grassmann, Grassmann}[]
    for sector in 0:1
        rows = findall(==(sector), row_parity)
        cols = findall(==(sector), col_parity)
        factor = svd(matrix[rows, cols])
        for alpha in eachindex(factor.S)
            factor.S[alpha] > eps(real(float(one(T)))) || continue
            left_vector = zeros(eltype(factor.U), dout1 * din1)
            right_vector = zeros(eltype(factor.Vt), dout2 * din2)
            left_vector[rows] =
                factor.U[:, alpha] * sqrt(factor.S[alpha])
            right_vector[cols] =
                factor.Vt[alpha, :] * sqrt(factor.S[alpha])
            parity_symbol = sector == 0 ? :even : :odd
            left = Grassmann(
                reshape(left_vector, dout1, din1),
                (dout1, din1),
                (even(operator)[1], even(operator)[3]),
                (:out, :in);
                parity=parity_symbol,
            )
            right = Grassmann(
                reshape(right_vector, dout2, din2),
                (dout2, din2),
                (even(operator)[2], even(operator)[4]),
                (:out, :in);
                parity=parity_symbol,
            )
            push!(terms, (left, right))
        end
    end
    return terms
end
```

Verify reconstruction before using the terms:

```julia
reconstructed = sum(
    contract(left, right; sign_function=global_sign) for (left, right) in terms
)
ordered = permutedims(reconstructed, (1, 3, 2, 4); sign_function=global_sign)
@test ordered ≈ operator rtol=1e-12
```

The two parity sectors are decomposed separately, so every Schmidt term has a
defined one-site operator parity and odd terms occur in odd-odd pairs.

- [ ] **Step 4: Implement normalized arbitrary-strip contractions**

Build each numerator as a sum over operator-Schmidt terms. For a horizontal
bond, replace the two endpoint Y tensors and retain the K tensor between them;
for a vertical bond retain the B tensor. Reuse the tested `left_move` and
`up_move` primitives from `algorithms/CTMRG/measurements.jl`; these absorb one
bulk column/row with the same fermionic axis order as ordinary correlation
functions. Add these complete strip helpers:

```julia
function _contract_horizontal_strip(
    bulks::NTuple{N, <:Grassmann},
    env::CTMRGEnv,
    sites::NTuple{N, CartesianIndex{2}},
) where {N}
    N > 0 || throw(ArgumentError("a strip must contain at least one tensor"))
    left_site = first(sites)
    right_site = last(sites)

    # L[top, bulk, bottom] from the left corners and edge.
    left_top = contract(
        env.Clu[left_site], env.El[left_site], (2, 1);
        sign_function=global_sign,
    )
    state = contract(
        left_top, env.Cld[left_site], (2, 2);
        sign_function=global_sign,
    )
    for (bulk, site) in zip(bulks, sites)
        state, _ = left_move(
            state, env.Ed[site], bulk, env.Eu[site]
        )
    end

    # R[top, bulk, bottom] from the right corners and edge.
    right_top = contract(
        env.Cru[right_site], env.Er[right_site], (2, 1);
        sign_function=global_sign,
    )
    right = contract(
        right_top, env.Crd[right_site], (2, 2);
        sign_function=global_sign,
    )
    return contract(
        state, right, ((1, 2, 3), (1, 2, 3));
        sign_function=global_sign,
    )
end

function _contract_vertical_strip(
    bulks::NTuple{N, <:Grassmann},
    env::CTMRGEnv,
    sites::NTuple{N, CartesianIndex{2}},
) where {N}
    N > 0 || throw(ArgumentError("a strip must contain at least one tensor"))
    top_site = first(sites)
    bottom_site = last(sites)

    # U[left, bulk, right] from the upper corners and edge.
    upper_left = contract(
        env.Clu[top_site], env.Eu[top_site], (1, 1);
        sign_function=global_sign,
    )
    state = contract(
        upper_left, env.Cru[top_site], (2, 1);
        sign_function=global_sign,
    )
    for (bulk, site) in zip(bulks, sites)
        state, _ = up_move(
            state, env.El[site], bulk, env.Er[site]
        )
    end

    # D[left, bulk, right] from the lower corners and edge.
    lower_left = contract(
        env.Cld[bottom_site], env.Ed[bottom_site], (1, 1);
        sign_function=global_sign,
    )
    lower = contract(
        lower_left, env.Crd[bottom_site], (2, 1);
        sign_function=global_sign,
    )
    return contract(
        state, lower, ((1, 2, 3), (1, 2, 3));
        sign_function=global_sign,
    )
end

function _contract_nested_hpatch3(
    nested::NestedNetwork,
    env::CTMRGEnv,
    source::CartesianIndex{2},
    left_y::Grassmann,
    right_y::Grassmann,
)
    y1 = nested.layout.y_sites[source]
    next_source = CartesianIndex(
        source[1],
        Nmod(source[2] + 1, nested.layout.source_size[2]),
    )
    y2 = nested.layout.y_sites[next_source]
    middle = CartesianIndex(
        y1[1], Nmod(y1[2] + 1, size(nested, 2))
    )
    return _contract_horizontal_strip(
        (left_y, nested[middle], right_y),
        env,
        (y1, middle, y2),
    )
end

function _contract_nested_vpatch3(
    nested::NestedNetwork,
    env::CTMRGEnv,
    source::CartesianIndex{2},
    top_y::Grassmann,
    bottom_y::Grassmann,
)
    y1 = nested.layout.y_sites[source]
    next_source = CartesianIndex(
        Nmod(source[1] + 1, nested.layout.source_size[1]),
        source[2],
    )
    y2 = nested.layout.y_sites[next_source]
    middle = CartesianIndex(
        Nmod(y1[1] + 1, size(nested, 1)), y1[2]
    )
    return _contract_vertical_strip(
        (top_y, nested[middle], bottom_y),
        env,
        (y1, middle, y2),
    )
end
```

Return the raw scalar patch contraction. Compute a denominator once using two
physical identities, and divide the summed operator-Schmidt numerator by that
same denominator.

- [ ] **Step 5: Implement public horizontal/vertical functions**

```julia
function compute_nested_exp_hbond(
    nested::NestedNetwork, peps::Square_GPEPS,
    operator::Grassmann{<:Number, 4}, env::CTMRGEnv, site)

    source = _source_site(site)
    neighbor = CartesianIndex(
        source[1], Nmod(source[2] + 1, size(peps)[2])
    )
    _check_nested_bond_operator(peps, operator, source, neighbor)
    identity_left = _physical_identity(peps.A[source])
    identity_right = _physical_identity(peps.A[neighbor])
    closed_left = nested_y_operator(nested, peps, source, identity_left)
    closed_right = nested_y_operator(nested, peps, neighbor, identity_right)
    denominator = _contract_nested_hpatch3(
        nested, env, source, closed_left, closed_right
    )
    denominator_value = _nested_scalar_or_zero(denominator)
    terms = _operator_schmidt(operator)
    numerator_type = promote_type(
        typeof(denominator_value), eltype(operator)
    )
    numerator = sum(terms; init=zero(numerator_type)) do (left_op, right_op)
        left_y = nested_y_operator(nested, peps, source, left_op)
        right_y = nested_y_operator(nested, peps, neighbor, right_op)
        term_sign =
            (-one(eltype(operator)))^tensor_parity(left_op)
        term_sign * _nested_scalar_or_zero(
            _contract_nested_hpatch3(
                nested, env, source, left_y, right_y
            )
        )
    end
    return denominator, numerator / denominator_value
end
```

Here `_nested_scalar_or_zero` is private to nested measurements and handles a
structurally empty rank-zero sector without invoking Grassmann tensor
addition:

```julia
_nested_scalar_or_zero(t::GrassmannScalar) =
    isempty(nonzero_keys(t)) ? zero(eltype(t)) : scalar(t)
```

The numeric accumulation is required because adding two rank-zero
`GrassmannScalar`s calls `tensor_parity` on the empty tuple sector key. Do not
change the global `Base.+` or `tensor_parity` implementations for this local
measurement issue.

Add the vertical method and explicit unit-cell overloads:

```julia
function compute_nested_exp_vbond(
    nested::NestedNetwork, peps::Square_GPEPS,
    operator::Grassmann{<:Number, 4}, env::CTMRGEnv, site)

    source = _source_site(site)
    neighbor = CartesianIndex(
        Nmod(source[1] + 1, size(peps)[1]), source[2]
    )
    _check_nested_bond_operator(peps, operator, source, neighbor)
    identity_top = _physical_identity(peps.A[source])
    identity_bottom = _physical_identity(peps.A[neighbor])
    closed_top = nested_y_operator(nested, peps, source, identity_top)
    closed_bottom = nested_y_operator(nested, peps, neighbor, identity_bottom)
    denominator = _contract_nested_vpatch3(
        nested, env, source, closed_top, closed_bottom
    )
    denominator_value = _nested_scalar_or_zero(denominator)
    terms = _operator_schmidt(operator)
    numerator_type = promote_type(
        typeof(denominator_value), eltype(operator)
    )
    numerator = sum(terms; init=zero(numerator_type)) do (top_op, bottom_op)
        top_y = nested_y_operator(nested, peps, source, top_op)
        bottom_y = nested_y_operator(nested, peps, neighbor, bottom_op)
        _nested_scalar_or_zero(
            _contract_nested_vpatch3(
                nested, env, source, top_y, bottom_y
            )
        )
    end
    return denominator, numerator / denominator_value
end

function compute_nested_exp_hbond(
    nested::NestedNetwork, peps::Square_GPEPS,
    operators::AbstractMatrix{<:Grassmann}, env::CTMRGEnv)

    size(operators) == size(peps) ||
        throw(DimensionMismatch("operator and PEPS unit cells differ"))
    results = map(CartesianIndices(peps.A)) do site
        compute_nested_exp_hbond(
            nested, peps, operators[site], env, site
        )
    end
    return first.(results), last.(results)
end

function compute_nested_exp_vbond(
    nested::NestedNetwork, peps::Square_GPEPS,
    operators::AbstractMatrix{<:Grassmann}, env::CTMRGEnv)

    size(operators) == size(peps) ||
        throw(DimensionMismatch("operator and PEPS unit cells differ"))
    results = map(CartesianIndices(peps.A)) do site
        compute_nested_exp_vbond(
            nested, peps, operators[site], env, site
        )
    end
    return first.(results), last.(results)
end
```

- [ ] **Step 6: Cross-check exact reduced-layer bond contractions**

Execute Task 2 Steps 1, 2, and 5 of the exact-test companion. Compare raw
denominators, raw numerators, and normalized horizontal/vertical
expectations at `rtol=5e-12` and `atol=1e-12`. Do not construct or iterate
a CTMRG environment in `test/nested_measurements.jl`.

As a regression-only exception for the public H/V API, construct a boundary
with `initialize_nested_environment(nested, 4)` but run zero CTMRG
iterations. Use a bond operator with more than one Schmidt term and compare
both public orientations against explicit per-term numeric accumulation.
This smoke test guards the rank-zero `GrassmannScalar` aggregation defect; it
does not replace the environment-free nested/reduced physics comparisons
above and must not be used as CTMRG convergence evidence.
Also pass a structurally valid zero `ComplexF64` bond operator against a
`Float64` PEPS/environment and assert that both public values are complex
zeros. This prevents the accumulator return type from depending on whether
the complex operator happens to have retained Schmidt terms.

- [ ] **Step 7: Run tests and commit**

```bash
git add algorithms/Nested_CTMRG/measurements.jl test/nested_measurements.jl
git commit -m "feat: measure nested nearest-neighbor operators"
```

---

### Task 6: Reverse Rules and Gradient Accuracy

**Files:**

- Create: `algorithms/Nested_CTMRG/nested_chainrules.jl`
- Create: `test/nested_chainrules.jl`
- Modify: `ext/GrassmannChainRulesCoreExt/GrassmannChainRulesCoreExt.jl`
- Modify: `test/runtests.jl`
- Modify: `test/server_runtests.jl`

**Interfaces:**

- Consumes: K/B/Y construction and existing Grassmann ChainRules.
- Produces `rrule`s for `_graded_pair_sign`, the four native placement
  helpers, `nested_network`, `nested_y_operator`, and fixed-observable
  horizontal/vertical measurements.

- [ ] **Step 1: Write a reusable centered directional-derivative test**

```julia
using ChainRulesCore
using FiniteDifferences
using LinearAlgebra
using Random
using Test
using Zygote

function directional_error(f, x, direction; step=1e-6)
    value, pullback = Zygote.pullback(f, x)
    gradient = only(pullback(one(value)))
    finite = (f(x + step * direction) - f(x - step * direction)) / (2step)
    analytic = real(dot(gradient, direction))
    relative =
        abs(analytic - finite) /
        max(abs(analytic), abs(finite), eps(Float64))
    return relative, analytic, finite
end

@testset "Nested reverse rules" begin
    Random.seed!(0x4e455354)
    count = square_gpeps_parameter_count(2, 1, 2, 1, 1)
    params = randn(count)
    direction = randn(count)
    direction ./= norm(direction)

    nested_norm2(x) = begin
        peps = Square_GPEPS(2, 1, 2, 1, 1, x, false)
        nested = nested_network(peps)
        sum(t -> sum(abs2, t), nested.network)
    end
    relative, analytic, finite =
        directional_error(nested_norm2, params, direction)
    @info "nested network directional derivative" relative analytic finite
    @test relative <= 1e-4
end
```

- [ ] **Step 2: Run and confirm reverse-mode failure**

Expected: Zygote fails on dictionary mutation, matrix assignment, or a missing
pullback in nested construction.

- [ ] **Step 3: Include nested rules only from the extension**

Add a second import statement after the existing
`import GrassmannTensorNetworks: ...` block:

```julia
import GrassmannTensorNetworks:
    NestedLayout, NestedNetwork,
    nested_network, nested_y_operator,
    compute_nested_exp_hbond, compute_nested_exp_vbond,
    _source_site, _graded_pair_sign,
    _nested_ket, _nested_ket_raw,
    _nested_bra, _nested_bra_raw, _bend_index,
    _nested_x,
    _nested_ket_for_network, _nested_bra_for_network,
    _nested_x_for_network, _nested_y_for_network,
    _nested_y_operator_raw,
    _physical_identity, _operator_schmidt,
    _nested_scalar_or_zero,
    _contract_nested_hpatch3, _contract_nested_vpatch3
```

Append:

```julia
include(joinpath(
    @__DIR__, "..", "..", "algorithms", "Nested_CTMRG", "nested_chainrules.jl"
))
```

Do not include this file from `src/algorithms.jl`.

- [ ] **Step 4: Implement linear pair-sign and node pullbacks**

```julia
function ChainRulesCore.rrule(
    ::typeof(_graded_pair_sign), t::Grassmann, i::Int, j::Int)

    y = _graded_pair_sign(t, i, j)
    function pullback(delta)
        delta = unthunk(delta)
        delta isa AbstractZero && return (
            NoTangent(), ZeroTangent(), NoTangent(), NoTangent()
        )
        return (
            NoTangent(), _graded_pair_sign(delta, i, j),
            NoTangent(), NoTangent(),
        )
    end
    return y, pullback
end
```

Define `_nested_ket` and `_nested_bra` rules around their raw pipelines:

```julia
function ChainRulesCore.rrule(
    config::RuleConfig{>:HasReverseMode},
    ::typeof(_nested_ket),
    A::Grassmann,
)
    y, raw_pullback = rrule_via_ad(config, _nested_ket_raw, A)
    function pullback(delta)
        _, delta_A = raw_pullback(unthunk(delta))
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
        _, delta_A = raw_pullback(unthunk(delta))
        return NoTangent(), delta_A
    end
    return y, pullback
end
```

The bra primal remains `_nested_bra_raw(A)`. Its raw implementation uses
`foldl` for three bends, and Julia 1.12 drops the init cotangent through that
`foldl`. The extension-local `_nested_bra_explicit_bends` pipeline is
forward-equivalent but keeps all three bend cotangents. This also keeps the
static integer permutations out of the returned tangent tuple while reusing
existing `fuse`, sign, bend, and conjugation rules.

Every linear pullback in this step must short-circuit an unthunked
`AbstractZero` output cotangent to the correctly shaped tuple containing
`ZeroTangent()` for its Grassmann input. This applies to raw K/B, placed K/B,
placed X/Y, and the operator-dressed Y rule; never pass `ZeroTangent()` into a
method whose primal argument must be `Grassmann`.

Define the placement-helper rules. The K/B rules include the north-input
twist; the X/Y placement maps are diagonal signs and therefore self-adjoint:

```julia
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
        _, delta_A = pullback_A(unthunk(delta))
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
        _, delta_A = pullback_A(unthunk(delta))
        return NoTangent(), delta_A
    end
    return y, pullback
end

function ChainRulesCore.rrule(
    ::typeof(_nested_x_for_network), xraw::Grassmann
)
    y = _nested_x_for_network(xraw)
    pullback(delta) =
        (NoTangent(), _nested_x_for_network(unthunk(delta)))
    return y, pullback
end

function ChainRulesCore.rrule(
    ::typeof(_nested_y_for_network), yraw::Grassmann
)
    y = _nested_y_for_network(yraw)
    pullback(delta) =
        (NoTangent(), _nested_y_for_network(unthunk(delta)))
    return y, pullback
end
```

- [ ] **Step 5: Implement the network assembly pullback**

Materialize `Tangent`-backed Grassmann cotangents before adding the K and B
paths:

```julia
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
```

Then use the approved reference pattern:

```julia
function ChainRulesCore.rrule(
    config::RuleConfig{>:HasReverseMode},
    ::typeof(nested_network),
    peps::Square_GPEPS,
    layout::NestedLayout)

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
            gradient isa AbstractZero ?
                primal * zero(eltype(primal)) : gradient
        end
        delta_peps = Tangent{typeof(peps)}(
            ; A=Matrix{Tensor}(gradients),
              Λx=NoTangent(), Λy=NoTangent()
        )
        return NoTangent(), delta_peps, NoTangent()
    end
    return nested, pullback
end
```

Add the two-argument rule by delegating to `NestedLayout(peps)`:

```julia
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
```

Add the operator-dressed Y rule. Geometry and PEPS dimensions are static;
only the operator is a numerical input to this rule. Construct the static
crossing before entering AD because `_nested_x` uses matrix assignment that
Zygote cannot differentiate. The extension-local helper must therefore accept
the crossing as a constant and reproduce only the operator-dependent
contraction, graded routing, fusion, and final network placement:

```julia
function _nested_y_operator_from_crossing(
    operator::Grassmann,
    crossing::Grassmann,
)
    product = contract(
        operator, crossing;
        sign_function=GrassmannTensorNetworks.global_sign,
    )
    routed = permutedims(
        product, (1,3,4,5,2,6);
        sign_function=GrassmannTensorNetworks.global_sign,
    )
    west_fused = fuse(routed, (1,2); index_type_fused=:out)
    raw = fuse(west_fused, (4,5); index_type_fused=:out)
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
        _, delta_operator = operator_pullback(unthunk(delta))
        return (
            NoTangent(), NoTangent(), NoTangent(),
            NoTangent(), delta_operator,
        )
    end
    return y, pullback
end
```

Use the actual `Square_GPEPS` field names `A`, `Λx`, and `Λy`.

- [ ] **Step 6: Add fixed-observable H/V measurement pullbacks**

The PEPS optimization differentiates a fixed Hamiltonian. Keep the Schmidt
decomposition, operator-dressed Y tensors, and horizontal endpoint signs
outside AD. Extension-local prepared helpers receive only the numerical
`nested` network and `env`; every rank-zero contraction is scalarized before
numeric accumulation:

```julia
function _nested_hbond_from_prepared(
    nested, env, source,
    closed_left, closed_right,
    prepared_terms, numerator_zero,
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
    nested, env, source,
    closed_top, closed_bottom,
    prepared_terms, numerator_zero,
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
```

Define `rrule`s for the scalar-site public H/V methods. First compute the
ordinary public primal, then prepare identities, Schmidt pairs, dressed Y
tensors, horizontal signs, and the promoted numeric zero. Use
`rrule_via_ad(config, (n, e) -> prepared_helper(...), nested, env)` so the
pullback returns
`(NoTangent(), delta_nested, NoTangent(), NoTangent(), delta_env,
NoTangent())` for function, nested, PEPS, operator, environment, and site.

The direct PEPS argument is structural in these measurement methods; all
PEPS numerical dependence flows through `nested_network(peps)`, so returning
a second direct PEPS cotangent would double-count that path. The environment
is numerical and must retain its cotangent. The operator cotangent is
`NoTangent()` under this explicitly fixed-observable optimization contract.
Do not present these rules as an operator-differentiation API: a future
operator gradient requires the adjoint of the original rank-four
operator-to-numerator linear map, not a zero cotangent disguised as such.

Add direct zero-seed pullback tests for the pair-sign, raw K/B, placed K/B,
placed X/Y, `nested_network` (including a structured tangent whose `network`
field alone is `ZeroTangent()`), and operator-dressed Y rules. Also assert
that the Y rule primal exactly matches `nested_y_operator` and preserves the
public validation errors for both an invalid operator and an out-of-bounds
site. The public primal must run before any crossing/layout indexing so the
rrule cannot replace the documented `ArgumentError` with an internal
`BoundsError`.
For a high-level Zygote objective that reads only constant crossing metadata,
accept either `nothing` (Zygote's no-gradient representation) or an explicit
numeric zero; keep the direct `rrule` structured-cotangent assertion strict.

- [ ] **Step 7: Add one-site and bond-energy gradient tests**

Freeze a deterministic one-iteration environment and define the three
objectives:

```julia
function convert_nested_env(env::CTMRGEnv, ::Type{T}) where {T}
    convert_grid(grid) = Matrix{Grassmann{T, tensor_rank(first(grid))}}(
        reshape([convert(tensor, T) for tensor in grid], size(grid))
    )
    return CTMRGEnv{T}(
        convert_grid(env.El), convert_grid(env.Er),
        convert_grid(env.Eu), convert_grid(env.Ed),
        convert_grid(env.Clu), convert_grid(env.Cru),
        convert_grid(env.Cld), convert_grid(env.Crd),
    )
end

initial_peps = Square_GPEPS(2, 1, 2, 1, 1, params, false)
initial_nested = nested_network(initial_peps)
frozen_env = initialize_nested_environment(initial_nested, 4)
run_nested_GCTMRG!(
    initial_nested, frozen_env, 4;
    ctmrg_iter=1, verbosity=0, save_iter=0,
)
model = SpinlessFermionModel(1.0, 1.0, 3.0)
number = n_site(model)
bond = nn_bond(model)

function nested_inputs(x)
    peps = Square_GPEPS(2, 1, 2, 1, 1, x, false)
    nested = nested_network(peps)
    return peps, nested, convert_nested_env(frozen_env, eltype(x))
end

site_operator_energy(operator) = begin
    _, value =
        compute_nested_exp_site(
            initial_nested, initial_peps, operator, frozen_env, (1, 1)
        )
    real(value)
end
horizontal_energy(x) = begin
    peps, nested, env = nested_inputs(x)
    _, value =
        compute_nested_exp_hbond(nested, peps, bond, env, (1, 1))
    real(value)
end
vertical_energy(x) = begin
    peps, nested, env = nested_inputs(x)
    _, value =
        compute_nested_exp_vbond(nested, peps, bond, env, (1, 1))
    real(value)
end

site_direction = _physical_identity(initial_peps.A[1, 1])
site_direction = site_direction / norm(site_direction)
site_relative, site_analytic, site_finite = directional_error(
    site_operator_energy, number, site_direction
)
@test max(abs(site_analytic), abs(site_finite)) > 1e-8
@test site_relative <= 1e-4

for (name, objective) in (
    ("horizontal", horizontal_energy),
    ("vertical", vertical_energy),
)
    relative, analytic, finite =
        directional_error(objective, params, direction)
    @info "nested measurement directional derivative" name relative analytic finite
    @test relative <= 1e-4
end
```

The one-site fixed-boundary closure is structurally independent of PEPS
values: its K/B dependence is already frozen into the environment, while the
dynamic Y tensor depends only on the operator and geometry. Therefore test a
nonzero public-site derivative with respect to the one-site operator, using a
normalized physical-identity direction. H/V continue to test PEPS parameter
derivatives. Assert `max(abs(analytic), abs(finite)) > 1e-8` before accepting
each relative error. The spinless optimization uses `nn_bond`, which already
contains the distributed chemical-potential term, so its fixed-environment
energy path does not require a separate site-to-PEPS derivative.

The logged values let a server failure distinguish a sign error from
numerical noise.

In addition to the three directional errors, call the public H/V `rrule`s
directly and verify that their pullback tuple has six entries, keeps
non-`AbstractZero` nested and environment cotangents for a nonzero output
seed, and returns `NoTangent()` for direct PEPS, fixed operator, and site.

- [ ] **Step 8: Run tests and commit**

```bash
git add algorithms/Nested_CTMRG/nested_chainrules.jl \
        ext/GrassmannChainRulesCoreExt/GrassmannChainRulesCoreExt.jl \
        test/nested_chainrules.jl test/runtests.jl test/server_runtests.jl
git commit -m "feat: differentiate nested Grassmann networks"
```

---

### Task 7: Spinless-Fermion Nested AD Driver and Exact Energy

**Files:**

- Create: `examples/Spinless_Fermion_2D_Square_AD_nested/Project.toml`
- Create: `examples/Spinless_Fermion_2D_Square_AD_nested/Spinless_Fermion_2D_Square_AD_nested.jl`
- Create: `examples/Spinless_Fermion_2D_Square_AD_nested/test/runtests.jl`

**Interfaces:**

- Consumes: all nested public APIs and `Zygote`.
- Produces:
  - `spinless_exact_energy(t, gamma, lambda; nk=1024)`
  - `compute_nested_energy(h, peps, nested, env)`
  - `run_Square_SpinlessFermion_AD_nested(...)`
  - JSON-compatible result named tuple/dictionary.

- [ ] **Step 1: Add exact-energy regression test**

```julia
const EXAMPLE_ROOT = normpath(joinpath(@__DIR__, ".."))
const GTN_ROOT = normpath(joinpath(EXAMPLE_ROOT, "..", ".."))
GTN_ROOT in LOAD_PATH || push!(LOAD_PATH, GTN_ROOT)
include(joinpath(EXAMPLE_ROOT, "Spinless_Fermion_2D_Square_AD_nested.jl"))

@test spinless_exact_energy(1.0, 1.0, 3.0; nk=512) ≈
    -6.170521774015 atol=2e-6
```

- [ ] **Step 2: Create the example environment**

`Project.toml`:

```toml
[deps]
Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[compat]
Zygote = "0.7"

[extras]
Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[targets]
test = ["Test"]
```

- [ ] **Step 3: Implement exact Brillouin-zone quadrature**

Use a periodic midpoint rule that is deterministic and allocation-light:

```julia
function spinless_exact_energy(
    t::Real, gamma::Real, lambda::Real; nk::Int=1024)

    nk > 0 || throw(ArgumentError("nk must be positive"))
    delta = 2pi / nk
    integral = 0.0
    for ix in 0:(nk - 1), iy in 0:(nk - 1)
        kx = -pi + (ix + 0.5) * delta
        ky = -pi + (iy + 0.5) * delta
        normal = t * (cos(kx) + cos(ky)) - lambda
        pairing = gamma * (sin(kx) + sin(ky))
        integral += hypot(normal, pairing)
    end
    return -lambda - integral / nk^2
end
```

- [ ] **Step 4: Adapt the existing trust-region optimization**

Start the example with the same isolated environment setup as the ordinary AD
example, then import Zygote explicitly. Do not depend on Optim: the required
server environment cannot resolve that uncached dependency offline, and a
small deterministic Armijo search is sufficient for this fixed-environment
candidate step.

```julia
using Pkg

const GTN_ROOT = normpath(joinpath(@__DIR__, "../.."))
if abspath(PROGRAM_FILE) == @__FILE__
    Pkg.activate(GTN_ROOT)
    Pkg.instantiate()
    Pkg.activate(@__DIR__)
    Pkg.instantiate()
    GTN_ROOT in LOAD_PATH || push!(LOAD_PATH, GTN_ROOT)
end

using GrassmannTensorNetworks
using LinearAlgebra
using Printf
using Random
using Zygote

function normalized_params(params::AbstractVector{<:Real})
    scale = norm(params) / sqrt(length(params))
    return collect(params) ./ (scale + eps(Float64))
end

function convert_grassmann_grid(
    grid::Matrix{<:Grassmann{S, N}}, ::Type{T}
) where {S, N, T}
    converted = [convert(tensor, T) for tensor in grid]
    return Matrix{Grassmann{T, N}}(reshape(converted, size(grid)))
end

convert_ctmrg_env(env::CTMRGEnv{T}, ::Type{T}) where {T} = env
function convert_ctmrg_env(env::CTMRGEnv, ::Type{T}) where {T}
    return CTMRGEnv{T}(
        convert_grassmann_grid(env.El, T),
        convert_grassmann_grid(env.Er, T),
        convert_grassmann_grid(env.Eu, T),
        convert_grassmann_grid(env.Ed, T),
        convert_grassmann_grid(env.Clu, T),
        convert_grassmann_grid(env.Cru, T),
        convert_grassmann_grid(env.Cld, T),
        convert_grassmann_grid(env.Crd, T),
    )
end
```

Define energy as the unit-cell average. Pass the already-built bond operator
instead of rebuilding the model inside differentiated code:

```julia
function compute_nested_energy(
    h::Grassmann{TH, 4},
    peps::Square_GPEPS{T},
    nested::NestedNetwork,
    env::CTMRGEnv,
) where {TH, T}
    h_t = TH === T ? h : convert(h, T)
    env_t = convert_ctmrg_env(env, T)
    energy = zero(promote_type(T, TH))
    for site in CartesianIndices(peps.A)
        _, eh =
            compute_nested_exp_hbond(nested, peps, h_t, env_t, site)
        _, ev =
            compute_nested_exp_vbond(nested, peps, h_t, env_t, site)
        energy += eh + ev
    end
    return real(energy / length(peps.A))
end
```

Add environment refresh and a Zygote-only fixed-environment candidate:

```julia
function update_nested_environment!(
    D::Int, Lx::Int, Ly::Int, params::AbstractVector,
    chi::Int, ctmrg_iter::Int;
    env::Union{Nothing, CTMRGEnv}=nothing,
    ctmrg_tol::Float64=1e-10,
    average_trunc::Bool=true,
    verbosity::Int=0,
)
    peps = Square_GPEPS(2, 1, D, Lx, Ly, params, false)
    nested = nested_network(peps)
    env = env === nothing ?
        initialize_nested_environment(nested, chi) : env
    run_nested_GCTMRG!(
        nested, env, chi;
        ctmrg_iter=ctmrg_iter,
        ctmrg_tol=ctmrg_tol,
        average_trunc=average_trunc,
        verbosity=verbosity,
        save_iter=0,
    )
    return peps, nested, env
end

function _normalized_gradient_candidate(
    objective,
    params::Vector{Float64};
    inner_optim_iter::Int=2,
    max_step_norm::Float64=0.25,
)
    inner_optim_iter >= 0 ||
        throw(ArgumentError("inner_optim_iter must be nonnegative"))
    isfinite(max_step_norm) && max_step_norm >= 0 ||
        throw(ArgumentError("max_step_norm must be finite and nonnegative"))

    origin = normalized_params(params)
    initial_value, gradient_tuple = Zygote.withgradient(objective, origin)
    initial_gradient = only(gradient_tuple)
    initial_gradient === nothing &&
        (initial_gradient = zeros(eltype(origin), length(origin)))
    current = origin
    current_value = initial_value
    current_gradient = initial_gradient
    status = :no_step
    attempted_iterations = 0
    accepted_inner_steps = 0
    backtracking_trials = 0

    if !isfinite(initial_value)
        status = :nonfinite_value
    elseif !all(isfinite, initial_gradient)
        status = :nonfinite_gradient
    elseif inner_optim_iter > 0 && max_step_norm > 0
        for iteration in 1:inner_optim_iter
            attempted_iterations = iteration
            gradient_norm = norm(current_gradient)
            if gradient_norm <= sqrt(eps(Float64)) * max(1, norm(origin))
                status = :zero_gradient
                break
            end
            direction = -current_gradient / gradient_norm
            alpha = max_step_norm
            accepted = false
            for _ in 1:12
                backtracking_trials += 1
                trial = normalized_params(current + alpha * direction)
                if norm(trial - origin) <= max_step_norm + 64eps(Float64)
                    displacement = trial - current
                    slope = dot(current_gradient, displacement)
                    trial_value = objective(trial)
                    if slope < 0 && isfinite(trial_value) &&
                       trial_value <= current_value + 1e-4 * slope
                        current = trial
                        current_value = trial_value
                        accepted_inner_steps += 1
                        accepted = true
                        status = :accepted
                        break
                    end
                end
                alpha *= 0.5
            end
            if !accepted
                status = :line_search_failed
                break
            elseif iteration < inner_optim_iter
                current_value, gradient_tuple =
                    Zygote.withgradient(objective, current)
                current_gradient = only(gradient_tuple)
                current_gradient === nothing &&
                    (current_gradient = zeros(eltype(origin), length(origin)))
                if !isfinite(current_value) || !all(isfinite, current_gradient)
                    status = :nonfinite_gradient
                    break
                end
            end
        end
    end

    return (
        candidate=current,
        energy=initial_value,
        gradient=initial_gradient,
        candidate_energy=current_value,
        optim_iterations=attempted_iterations,
        optim_converged=status == :zero_gradient,
        optim_status=status,
        accepted_inner_steps=accepted_inner_steps,
        backtracking_trials=backtracking_trials,
    )
end

function fixed_environment_candidate(
    h, D::Int, Lx::Int, Ly::Int,
    params::Vector{Float64}, env::CTMRGEnv;
    inner_optim_iter::Int=2,
    max_step_norm::Float64=0.25,
)
    objective = x -> begin
        peps = Square_GPEPS(2, 1, D, Lx, Ly, x, false)
        nested = nested_network(peps)
        compute_nested_energy(h, peps, nested, env)
    end
    return _normalized_gradient_candidate(
        objective, params;
        inner_optim_iter=inner_optim_iter,
        max_step_norm=max_step_norm,
    )
end
```

Implement the complete outer AD loop. Each candidate first passes the cheap
fixed-environment check, then passes a fresh-CTMRG validation before it is
accepted:

```julia
function run_Square_SpinlessFermion_AD_nested(
    t::Float64, gamma::Float64, lambda::Float64,
    D::Int, Lx::Int, Ly::Int, chi::Int, ctmrg_iter::Int;
    ad_iter::Int=20,
    inner_optim_iter::Int=2,
    seed::Int=1234,
    step_shrink::Float64=0.5,
    min_step::Float64=1e-4,
    max_step_norm::Float64=0.25,
    ctmrg_tol::Float64=1e-10,
    average_trunc::Bool=true,
    verbosity::Int=0,
)
    ad_iter > 0 || throw(ArgumentError("ad_iter must be positive"))
    isfinite(step_shrink) && 0 < step_shrink < 1 ||
        throw(ArgumentError("step_shrink must be finite and in (0, 1)"))
    isfinite(min_step) && min_step > 0 ||
        throw(ArgumentError("min_step must be finite and positive"))
    Random.seed!(seed)
    started = time()
    parameter_count = square_gpeps_parameter_count(2, 1, D, Lx, Ly)
    params = normalized_params(randn(parameter_count))
    h = nn_bond(SpinlessFermionModel(t, gamma, lambda))
    exact = spinless_exact_energy(t, gamma, lambda)
    history = NamedTuple[]
    env = nothing

    for iteration in 1:ad_iter
        peps, nested, env = update_nested_environment!(
            D, Lx, Ly, params, chi, ctmrg_iter;
            env=env, ctmrg_tol=ctmrg_tol,
            average_trunc=average_trunc, verbosity=verbosity,
        )
        current = compute_nested_energy(h, peps, nested, env)
        proposal = fixed_environment_candidate(
            h, D, Lx, Ly, params, env;
            inner_optim_iter=inner_optim_iter,
            max_step_norm=max_step_norm,
        )

        accepted = false
        accepted_step = 0.0
        validated = current
        trial_params = params
        trial_env = env
        proposal_usable = proposal.optim_status ∉ (
            :nonfinite_value, :nonfinite_gradient, :no_step,
        ) && norm(proposal.candidate - params) > sqrt(eps(Float64))
        alpha = proposal_usable ? 1.0 : 0.0
        validation_trials = 0
        while alpha >= min_step && validation_trials < 2
            candidate = normalized_params(
                params + alpha * (proposal.candidate - params)
            )
            candidate_peps = Square_GPEPS(
                2, 1, D, Lx, Ly, candidate, false
            )
            candidate_nested = nested_network(candidate_peps)
            fixed_energy = compute_nested_energy(
                h, candidate_peps, candidate_nested, env
            )
            if isfinite(fixed_energy) && fixed_energy <= current
                validation_trials += 1
                validated_peps, validated_nested, candidate_env =
                    update_nested_environment!(
                        D, Lx, Ly, candidate, chi, ctmrg_iter;
                        env=nothing, ctmrg_tol=ctmrg_tol,
                        average_trunc=average_trunc,
                        verbosity=verbosity,
                    )
                candidate_energy = compute_nested_energy(
                    h, validated_peps, validated_nested, candidate_env
                )
                if isfinite(candidate_energy) && candidate_energy <= current
                    accepted = true
                    accepted_step = alpha
                    validated = candidate_energy
                    trial_params = candidate
                    trial_env = candidate_env
                    break
                end
            end
            alpha *= step_shrink
        end

        params = copy(trial_params)
        env = trial_env
        gradient_norm = norm(proposal.gradient)
        push!(history, (
            iteration=iteration,
            energy=current,
            validated_energy=validated,
            gradient_norm=gradient_norm,
            accepted=accepted,
            accepted_step=accepted_step,
            validation_trials=validation_trials,
            optim_iterations=proposal.optim_iterations,
            optim_converged=proposal.optim_converged,
            optim_status=proposal.optim_status,
            accepted_inner_steps=proposal.accepted_inner_steps,
            backtracking_trials=proposal.backtracking_trials,
        ))
        @printf(
            "AD %2d/%2d E %.12f -> %.12f |g| %.4e step %.3e%s\n",
            iteration, ad_iter, current, validated, gradient_norm,
            accepted_step, accepted ? "" : " (rejected)",
        )
        flush(stdout)
    end

    peps, nested, env = update_nested_environment!(
        D, Lx, Ly, params, chi, ctmrg_iter;
        env=env, ctmrg_tol=ctmrg_tol,
        average_trunc=average_trunc, verbosity=verbosity,
    )
    energy = compute_nested_energy(h, peps, nested, env)
    environment_max_norm = maximum(
        maximum(norm, grid) for grid in
        (env.El, env.Er, env.Eu, env.Ed,
         env.Clu, env.Cru, env.Cld, env.Crd)
    )
    report = (
        chi=chi,
        D=D,
        ctmrg_iter=ctmrg_iter,
        ad_iter=ad_iter,
        energy=energy,
        exact_energy=exact,
        signed_error=energy - exact,
        absolute_error=abs(energy - exact),
        relative_error=abs((energy - exact) / exact),
        elapsed_seconds=time() - started,
        accepted_steps=count(record -> record.accepted, history),
        final_gradient_norm=last(history).gradient_norm,
        environment_max_norm=environment_max_norm,
        environment_finite=isfinite(environment_max_norm),
    )
    return (
        params=params, peps=peps, nested=nested, env=env,
        history=history, report=report,
    )
end
```

- [ ] **Step 5: Add environment-variable script entry point**

```julia
if abspath(PROGRAM_FILE) == @__FILE__
    GrassmannTensorNetworks.global_sign = auto_sign
    chi = parse(Int, get(ENV, "NESTED_CHI", "4"))
    result = run_Square_SpinlessFermion_AD_nested(
        1.0, 1.0, 3.0,
        2, 2, 2, chi, 20;
        ad_iter=20,
        seed=parse(Int, get(ENV, "NESTED_SEED", "1234")),
        verbosity=parse(Int, get(ENV, "NESTED_VERBOSITY", "0")),
    )
    println("NESTED_RESULT=", repr(result.report))
end
```

The report contains `chi`, `D`, `ctmrg_iter`, `ad_iter`, `energy`,
`exact_energy`, signed/absolute/relative errors, elapsed seconds, accepted
steps, and final CTMRG diagnostics.

- [ ] **Step 6: Run exact-energy and one-step example smoke tests**

Add this smoke test to the example-local `test/runtests.jl` after including
the example as in Step 1:

```julia
@testset "Nested AD example smoke" begin
    smoke = run_Square_SpinlessFermion_AD_nested(
        1.0, 1.0, 3.0, 2, 1, 1, 4, 1;
        ad_iter=1,
        inner_optim_iter=1,
        seed=1234,
        verbosity=0,
    )
    @test length(smoke.history) == 1
    @test isfinite(smoke.report.energy)
    @test isfinite(smoke.report.final_gradient_norm)
end
```

Run:

```bash
julia_grassmann \
  --project=examples/Spinless_Fermion_2D_Square_AD_nested \
  examples/Spinless_Fermion_2D_Square_AD_nested/test/runtests.jl
```

Expected: exact-energy and nested-example smoke testsets PASS with finite
energy and gradient and no exception.

Before the nested smoke, test `_normalized_gradient_candidate` on pure
numeric objectives: deterministic repeatability, quadratic descent,
normalization and total trust radius, constant-objective zero-gradient/no
move, and nonfinite-objective no move. Check the smoke history contains finite
candidate diagnostics and the Armijo status/trial counters.
Add driver validation tests for nonpositive/nonfinite `min_step` and for
`step_shrink` outside the finite open interval `(0, 1)`; these values must
throw before CTMRG so the outer backtracking loop cannot become infinite.

Do not add this CTMRG/AD smoke to `test/nested_measurements.jl`: those
tests remain dedicated to the approved environment-free nested/reduced
measurement comparisons (plus the separately approved zero-iteration public
aggregation regression). The example-local project owns its Zygote dependency
and prevents the root offline test harness from acquiring a new dependency.

- [ ] **Step 7: Commit**

```bash
git add examples/Spinless_Fermion_2D_Square_AD_nested
git commit -m "feat: add nested spinless fermion AD example"
```

---

### Task 8: Documentation and Full Local-Equivalent Verification

**Files:**

- Modify: `docs/algorithms.md`
- Modify: `README.md`

**Interfaces:**

- Documents all public Task 1/4/5 APIs and limitations.

- [ ] **Step 1: Add API documentation**

Document:

```markdown
## Nested CTMRG

`nested_network(peps)` maps an `Lx x Ly` Grassmann PEPS to a
`2Lx x 2Ly` K/Y/X/B network. Use `initialize_nested_environment` and
`run_nested_GCTMRG!` to reuse the package CTMRG implementation.

`compute_nested_exp_site`, `compute_nested_exp_hbond`, and
`compute_nested_exp_vbond` support one-site and horizontal/vertical
nearest-neighbor operators. All crossing, fusion, permutation, and arrow
operations preserve the Z2 fermionic sign convention.
```

Add the new example path to the README example list.

```markdown
- `examples/Spinless_Fermion_2D_Square_AD_nested/Spinless_Fermion_2D_Square_AD_nested.jl`:
  Grassmann nested-CTMRG optimization for the square-lattice spinless fermion.
```

- [ ] **Step 2: Run formatting and static checks**

```bash
git diff --check
git grep -n -E 'TO[D]O|TB[D]|FIX[M]E|broken[=]true' -- \
    algorithms/Nested_CTMRG \
    test/nested_*.jl examples/Spinless_Fermion_2D_Square_AD_nested
```

Expected: `git diff --check` exits 0 and the scan returns no matches.

- [ ] **Step 3: Run the complete isolated server test suite**

Use the Julia server skill to run:

```bash
julia_grassmann --project=. test/server_runtests.jl
```

Expected: all legacy and nested testsets PASS; no memory kill.

- [ ] **Step 4: Commit**

```bash
git add docs/algorithms.md README.md
git commit -m "docs: document nested CTMRG"
```

---

### Task 9: Requested Three-Chi Acceptance

**Files:**

- Create locally through the server skill:
  `D:\C 盘备份\Back up\Work_Save\coding\codex_server_logs\2026\0730\<task>.md`
- Do not commit runtime logs unless the repository documentation explicitly
  links a concise result table.

**Interfaces:**

- Consumes the completed example.
- Produces authoritative exit codes, energy/error rows, timings, convergence
  diagnostics, and memory outcomes for all requested χ values.

- [ ] **Step 1: Prepare three isolated server tasks**

Use separate remote task directories and the exact commands:

```bash
NESTED_CHI=4 NESTED_SEED=1234 julia_grassmann \
  examples/Spinless_Fermion_2D_Square_AD_nested/Spinless_Fermion_2D_Square_AD_nested.jl
NESTED_CHI=8 NESTED_SEED=1234 julia_grassmann \
  examples/Spinless_Fermion_2D_Square_AD_nested/Spinless_Fermion_2D_Square_AD_nested.jl
NESTED_CHI=12 NESTED_SEED=1234 julia_grassmann \
  examples/Spinless_Fermion_2D_Square_AD_nested/Spinless_Fermion_2D_Square_AD_nested.jl
```

Each task must use `D=2`, `Lx=Ly=2`, `ctmrg_iter=20`, `ad_iter=20`, the
20 GB memory monitor, and a distinct local Markdown log.

- [ ] **Step 2: Run at most three tasks concurrently**

The three tasks are independent and remain below the five-task server limit.
Monitor each with the server-skill polling interval. Do not run Julia locally.

- [ ] **Step 3: Parse and verify each result**

For each χ require:

```julia
@assert exit_code == 0
@assert !memory_killed
@assert isfinite(energy)
@assert isfinite(absolute_error)
@assert ctmrg_iter == 20
@assert ad_iter == 20
```

- [ ] **Step 4: Evaluate the χ trend**

Create a table:

```text
chi | energy | signed error | absolute error | relative error | seconds | peak RSS
4   | ...    | ...          | ...            | ...            | ...     | ...
8   | ...    | ...          | ...            | ...            | ...     | ...
12  | ...    | ...          | ...            | ...            | ...     | ...
```

Acceptance requires `abs_error(12) < abs_error(4)` and the three diagnostics
to support an overall approach toward the exact value. Small χ=8
nonmonotonicity is reported rather than hidden.

- [ ] **Step 5: If the trend fails, diagnose before changing tolerances**

Inspect accepted-step history, CTMRG residuals, candidate-validation energy,
and gradient norms. Fix a demonstrated algorithmic or sign problem, rerun the
focused tests, and repeat all affected acceptance tasks. Do not relax the
user-approved trend criterion.

---

### Task 10: Completion Review, Push, and Pull Request

**Files:**

- Review all files changed since `origin/main`.

**Interfaces:**

- Produces the final verified branch and GitHub pull request.

- [ ] **Step 1: Run the verification-before-completion checklist**

Re-run:

```bash
git diff --check origin/main...HEAD
julia_grassmann --project=. test/server_runtests.jl
```

Confirm all three Task 9 logs correspond to the current commit, not an older
source snapshot.

- [ ] **Step 2: Review scope and repository state**

```bash
git status --short --branch
git diff --stat origin/main...HEAD
git log --oneline --decorate origin/main..HEAD
```

Expected: only approved nested implementation, tests, example, and docs;
working tree clean except known ignored/local artifacts.

- [ ] **Step 3: Request code review and address findings**

Use the `requesting-code-review` skill against the full diff. Resolve every
correctness or acceptance finding, rerun the affected tests, and commit fixes
as Codex.

- [ ] **Step 4: Push the branch**

```bash
git push -u origin codex/grassmann-nested-ctmrg
```

- [ ] **Step 5: Open a ready GitHub pull request as Codex**

Title:

```text
Add Grassmann nested CTMRG and spinless-fermion AD example
```

The body includes:

- K/Y/X/B construction and fermionic sign invariants;
- one-site and nearest-neighbor measurement support;
- reverse-mode gradient-check maximum error;
- full test command and result;
- exact acceptance commands;
- the `chi=4,8,12` result table;
- exact energy `-6.170521774015...`;
- limitations and any nonmonotonic χ behavior.

Open the PR only after every acceptance condition above is supported by
current logs.
