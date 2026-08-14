# Universal Native Nested Placement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Assemble an exact periodic Grassmann K/Y/X/B nested network using
only existing sign, permutation, conjugation, fusion, and contraction
operations.

**Architecture:** Keep Task 2 raw categorical constructors unchanged. Apply
four universal native placement phases—`kU`, `bU`, `bL*kD`, and `kR*bU`—in
private placement helpers, then assemble the ordinary `2m x 2n` rank-four
checkerboard and reuse the existing GCTMRG implementation.

**Tech Stack:** Julia, GrassmannTensorNetworks native block tensors,
ChainRules-compatible Grassmann operations, Test, required SSH Julia server.

## Global Constraints

- The approved design is
  `docs/superpowers/specs/2026-07-31-grassmann-nested-existing-ops-design.md`.
- Do not add an MPO, auxiliary correction leg, new tensor type, or new
  contraction semantics.
- Preserve raw `_nested_x` odd-odd coefficient `-1`.
- The production correction must work for arbitrary positive source unit-cell
  dimensions; do not use the even-row X/Y-only specialization.
- X dimensions are always `horizontal=B.W` first and `vertical=K.S` second.
- Y dimensions are always `horizontal=east K.W` first and
  `vertical=north B.S` second.
- Every Julia test runs only on `jkkong@172.23.26.248` through
  `julia_grassmann`, under the 20 GB guard.
- Store local test logs under
  `D:\C 盘备份\Back up\Work_Save\coding\codex_server_logs\2026\0731`.
- Do not change dependency versions.
- Use TDD: record a missing-behaviour RED before production edits.
- Task 1 runs in
  `/home/jkkong/work/2026/0731/nested-existing-ops-placement-019fb340`;
  Task 2 runs in
  `/home/jkkong/work/2026/0731/nested-existing-ops-assembly-019fb340`.

---

### Task 1: Add Universal Native Placement Helpers

**Files:**

- Modify: `algorithms/Nested_CTMRG/nested_network.jl`
- Modify: `test/nested_network.jl`

**Interfaces:**

- Consumes: `_nested_ket`, `_nested_bra`, `_nested_x`,
  `_placed_nested_x`, `_nested_y`, `_graded_pair_sign`,
  `add_parity_sign`, `add_perm_sign`, and `global_sign`.
- Produces:
  - `_nested_ket_for_network(A)`
  - `_nested_bra_for_network(A)`
  - `_nested_x_for_network(xraw)`
  - `_nested_y_for_network(yraw)`
  - `_nested_network_reduced_basis(ordered)`

- [ ] **Step 1: Write failing placement and corrected-factorization tests**

Import the five new private helpers and append:

```julia
import GrassmannTensorNetworks:
    _nested_ket_for_network,
    _nested_bra_for_network,
    _nested_x_for_network,
    _nested_y_for_network,
    _nested_network_reduced_basis

function contract_corrected_nested_tile(K, Y, X, B)
    sign = GrassmannTensorNetworks.global_sign
    ky = contract(K, Y, (2, 1); sign_function=sign)
    kx = contract(ky, X, (3, 3); sign_function=sign)
    tile = contract(kx, B, ((5, 7), (3, 1)); sign_function=sign)
    ordered = permutedims(
        tile, (5, 1, 7, 3, 4, 2, 8, 6); sign_function=sign
    )
    ordered = _nested_network_reduced_basis(ordered)
    ordered = add_parity_sign(ordered, 1; sign_function=sign)
    left = fuse(ordered, (1, 2); index_type_fused=:out)
    left = add_perm_sign(
        left, (1, 3, 2, 4, 5, 6, 7); sign_function=sign
    )
    right = fuse(left, (2, 3); index_type_fused=:in)
    right = add_perm_sign(
        right, (1, 2, 4, 3, 5, 6); sign_function=sign
    )
    up = fuse(right, (3, 4); index_type_fused=:in)
    up = add_parity_sign(up, 4; sign_function=sign)
    return fuse(up, (4, 5); index_type_fused=:out)
end

function corrected_nested_tile(A)
    K = _nested_ket_for_network(A)
    B = _nested_bra_for_network(A)
    Xraw = _nested_x(
        size(B)[1], even(B)[1], size(K)[4], even(K)[4], eltype(A)
    )
    X = _nested_x_for_network(Xraw)
    Yraw = _nested_y(
        _physical_identity(A),
        size(A)[3], even(A)[3],
        size(A)[4], even(A)[4],
    )
    Y = _nested_y_for_network(Yraw)
    return contract_corrected_nested_tile(K, Y, X, B)
end

@testset "Universal native nested placement signs" begin
    A = deterministic_nested_tensor(
        ComplexF64, (2, 2, 2, 2, 2), (1, 1, 1, 1, 1)
    )
    xraw = _nested_x(2, 1, 2, 1, ComplexF64)
    @test only(xraw[(1, 1, 1, 1)]) == -1
    @test index_type(_nested_x_for_network(xraw)) == index_type(xraw)
    @test size(_nested_x_for_network(xraw)) == size(xraw)
    @test corrected_nested_tile(A) ≈ reduced_tensor(A) rtol=1e-12
end

@testset "Universal native placement preserves mixed bases" begin
    sizes = (3, 3, 4, 3, 4)
    evens = (2, 1, 3, 2, 1)
    for T in (Float64, ComplexF64)
        A = deterministic_nested_tensor(T, sizes, evens)
        @test corrected_nested_tile(A) ≈ reduced_tensor(A) rtol=5e-13
    end
end
```

- [ ] **Step 2: Run the focused test and verify RED**

Run only `test/nested_network.jl` with the Task 2 layout:

```bash
cd /home/jkkong/work/2026/0731/nested-existing-ops-placement-019fb340
./run_with_memory_guard.sh \
  --limit-gb 20 \
  --log log/red.log \
  -- bash -ic \
  'export PATH=/home/jkkong/bin:/usr/local/bin:/usr/bin:/bin;
   export JULIA_PKG_OFFLINE=true;
   julia_grassmann focused_runtests.jl'
```

Expected: import failure because `_nested_ket_for_network` and the other
placement helpers do not exist. Existing Task 2 tests before that import
remain green.

- [ ] **Step 3: Implement the universal placement helpers**

Add:

```julia
function _nested_input_north_twist(A::Grassmann)
    return add_parity_sign(A, 4; sign_function=global_sign)
end

_nested_ket_for_network(A::Grassmann{T, 5}) where {T} =
    _nested_ket(_nested_input_north_twist(A))

_nested_bra_for_network(A::Grassmann{T, 5}) where {T} =
    _nested_bra(_nested_input_north_twist(A))

function _nested_x_for_network(xraw::Grassmann{T, 4}) where {T}
    placed = _placed_nested_x(xraw)
    return add_perm_sign(
        placed, (1, 3, 2, 4); sign_function=global_sign
    )
end

function _nested_y_for_network(yraw::Grassmann{T, 4}) where {T}
    return add_perm_sign(
        yraw, (1, 3, 2, 4); sign_function=global_sign
    )
end

function _nested_network_reduced_basis(
    ordered::Grassmann{T, 8}
) where {T}
    corrected = ordered
    for axis in (1, 2, 4, 5)
        corrected = add_parity_sign(
            corrected, axis; sign_function=global_sign
        )
    end
    return corrected
end
```

Do not replace or weaken `_nested_reduced_basis`; it remains the comparison
map for the raw Task 2 primitives.

- [ ] **Step 4: Run focused tests and verify GREEN**

```bash
cd /home/jkkong/work/2026/0731/nested-existing-ops-placement-019fb340
./run_with_memory_guard.sh \
  --limit-gb 20 \
  --log log/green.log \
  -- bash -ic \
  'export PATH=/home/jkkong/bin:/usr/local/bin:/usr/bin:/bin;
   export JULIA_PKG_OFFLINE=true;
   julia_grassmann focused_runtests.jl'
```

Expected:

- raw X odd-odd remains `-1`;
- minimal `Float64` and `ComplexF64` corrected closure passes;
- nonuniform mixed-coordinate closure passes;
- all prior Task 1 and Task 2 assertions pass.

- [ ] **Step 5: Commit**

```bash
git add algorithms/Nested_CTMRG/nested_network.jl test/nested_network.jl
git commit -m "fix: place native nested fermion signs"
```

---

### Task 2: Assemble the Periodic Network and Reuse GCTMRG

**Files:**

- Modify: `algorithms/Nested_CTMRG/nested_network.jl`
- Modify: `test/nested_network.jl`
- Modify: `src/GrassmannTensorNetworks.jl`

**Interfaces:**

- Consumes: all Task 1 placement helpers, `NestedLayout`, `NestedNetwork`,
  `CTMRGEnv`, and `run_GCTMRG!`.
- Produces:
  - `nested_network(peps, layout=NestedLayout(peps))`
  - `_check_nested_links(nested)`
  - `initialize_nested_environment(nested, chi, chi_even=div(chi,2))`
  - `run_nested_GCTMRG!(nested, env, chi; kwargs...)`
- Preserves: `NestedNetwork.x_crossings` contains raw X tensors, while the
  checkerboard matrix contains corrected placed X tensors.

- [ ] **Step 1: Write failing assembly, odd-row, spin, and link tests**

Start with:

```julia
using Random

@test isdefined(GrassmannTensorNetworks, :nested_network)
@test isdefined(GrassmannTensorNetworks, :initialize_nested_environment)
@test isdefined(GrassmannTensorNetworks, :run_nested_GCTMRG!)
```

Add the exact native periodic test helpers:

```julia
function close_nested_test_row(row; twist_x=false)
    sign = GrassmannTensorNetworks.global_sign
    cols = length(row)
    current = permutedims(row[1], (1, 3, 4, 2); sign_function=sign)
    for c in 2:cols
        j = c - 1
        rank = 2j + 2
        perm = (
            1,
            (2:(j + 1))...,
            2j + 3,
            ((j + 2):(2j + 1))...,
            2j + 4,
            2j + 2,
        )
        current = contract(
            current, row[c], (rank, 1);
            perm, sign_function=sign,
        )
    end
    return trace(
        current, (1, 2cols + 2);
        pbc=!twist_x, sign_function=sign,
    )
end

function nested_test_torus_scalar(tensors; twist_x=false, twist_y=false)
    sign = GrassmannTensorNetworks.global_sign
    rows, cols = size(tensors)
    row_tensors = [
        close_nested_test_row(collect(tensors[r, :]); twist_x)
        for r in 1:rows
    ]
    current = row_tensors[1]
    for r in 2:rows
        current = contract(
            current,
            row_tensors[r],
            (ntuple(i -> cols + i, cols), ntuple(identity, cols));
            sign_function=sign,
        )
    end
    closed = trace(
        current,
        (ntuple(identity, cols), ntuple(i -> cols + i, cols));
        pbc=ntuple(_ -> !twist_y, cols),
        sign_function=sign,
    )
    return scalar(closed)
end
```

Add assembly and universal-shape tests:

```julia
@testset "Corrected nested network assembly" begin
    peps = Square_GPEPS(2, 1, 2, 2, 2, Float64, false)
    nested = nested_network(peps)
    @test size(nested) == (4, 4)
    @test nested[nested.layout.ket_sites[1, 1]] ≈
        _nested_ket_for_network(peps.A[1, 1])
    @test nested[nested.layout.bra_sites[2, 2]] ≈
        _nested_bra_for_network(peps.A[2, 2])

    for r in axes(nested, 1), c in axes(nested, 2)
        below = Nmod(r + 1, size(nested, 1))
        right = Nmod(c + 1, size(nested, 2))
        @test size(nested[r, c])[2] == size(nested[r, right])[1]
        @test even(nested[r, c])[2] == even(nested[r, right])[1]
        @test size(nested[r, c])[4] == size(nested[below, c])[3]
        @test even(nested[r, c])[4] == even(nested[below, c])[3]
        @test index_type(nested[r, c])[2] != index_type(nested[r, right])[1]
        @test index_type(nested[r, c])[4] != index_type(nested[below, c])[3]
    end

    @test_throws ArgumentError nested_network(peps, NestedLayout((1, 1)))
end

@testset "Universal odd-row and even-row periodic signs" begin
    spin_structures =
        ((false, false), (true, false), (false, true), (true, true))
    for (rows, cols) in ((1, 1), (1, 2), (2, 1), (2, 2))
        Random.seed!(10_000rows + cols)
        peps = Square_GPEPS(2, 1, 2, rows, cols, ComplexF64, false)
        nested = nested_network(peps)
        reduced = reduced_tensor(peps)
        for (twist_x, twist_y) in spin_structures
            nested_value = nested_test_torus_scalar(
                nested.network; twist_x, twist_y
            )
            reduced_value = nested_test_torus_scalar(
                reduced; twist_x, twist_y
            )
            @test nested_value ≈ reduced_value rtol=5e-13 atol=1e-13
        end
    end
end
```

Add a `2 x 2` nonuniform mixed-multiplicity fixture whose left/right and
up/down sizes are derived from shared periodic edge tables:

```julia
function mixed_periodic_peps(::Type{T}) where {T}
    horizontal = [(2, 1) (3, 2); (4, 3) (2, 1)]
    vertical = [(3, 2) (2, 1); (2, 1) (4, 3)]
    tensors = Matrix{Grassmann{T, 5}}(undef, 2, 2)
    for r in 1:2, c in 1:2
        left = horizontal[r, Nmod(c - 1, 2)]
        right = horizontal[r, c]
        up = vertical[Nmod(r - 1, 2), c]
        down = vertical[r, c]
        sizes = (3, left[1], right[1], up[1], down[1])
        evens = (2, left[2], right[2], up[2], down[2])
        tensors[r, c] =
            deterministic_nested_tensor(T, sizes, evens)
    end
    return Square_GPEPS{T}(tensors, missing, missing)
end

@testset "Mixed periodic nested algebra" begin
    spin_structures =
        ((false, false), (true, false), (false, true), (true, true))
    for T in (Float64, ComplexF64)
        peps = mixed_periodic_peps(T)
        nested = nested_network(peps)
        reduced = reduced_tensor(peps)
        for (twist_x, twist_y) in spin_structures
            @test nested_test_torus_scalar(
                nested.network; twist_x, twist_y
            ) ≈ nested_test_torus_scalar(
                reduced; twist_x, twist_y
            ) rtol=5e-13 atol=1e-13
        end
    end
end
```

- [ ] **Step 2: Run the focused test and verify RED**

```bash
cd /home/jkkong/work/2026/0731/nested-existing-ops-assembly-019fb340
./run_with_memory_guard.sh \
  --limit-gb 20 \
  --log log/red-assembly.log \
  -- bash -ic \
  'export PATH=/home/jkkong/bin:/usr/local/bin:/usr/bin:/bin;
   export JULIA_PKG_OFFLINE=true;
   julia_grassmann focused_runtests.jl'
```

Expected: missing `nested_network` method and missing CTMRG adapters. The
periodic helpers themselves must run far enough to demonstrate that the
assembly method is the absent behaviour.

- [ ] **Step 3: Implement the corrected checkerboard**

Add these exports to `src/GrassmannTensorNetworks.jl`:

```julia
export nested_network, initialize_nested_environment, run_nested_GCTMRG!
```

Add:

```julia
function nested_network(
    peps::Square_GPEPS{T},
    layout::NestedLayout=NestedLayout(peps),
) where {T}
    layout.source_size == size(peps) ||
        throw(ArgumentError("layout source size does not match PEPS unit cell"))

    rows, cols = size(peps)
    ket = [
        _nested_ket_for_network(peps.A[r, c])
        for r in 1:rows, c in 1:cols
    ]
    bra = [
        _nested_bra_for_network(peps.A[r, c])
        for r in 1:rows, c in 1:cols
    ]
    xraw = [
        _nested_x(
            size(bra[r, c])[1], even(bra[r, c])[1],
            size(ket[r, c])[4], even(ket[r, c])[4], T,
        )
        for r in 1:rows, c in 1:cols
    ]

    tensors = Matrix{Grassmann{T, 4}}(undef, size(layout)...)
    for r in 1:rows, c in 1:cols
        north_bra = bra[Nmod(r - 1, rows), c]
        east_ket = ket[r, Nmod(c + 1, cols)]
        yraw = _nested_y(
            _physical_identity(peps.A[r, c]),
            size(east_ket)[1], even(east_ket)[1],
            size(north_bra)[4], even(north_bra)[4],
        )

        tensors[layout.ket_sites[r, c]] = ket[r, c]
        tensors[layout.bra_sites[r, c]] = bra[r, c]
        tensors[layout.x_sites[r, c]] =
            _nested_x_for_network(xraw[r, c])
        tensors[layout.y_sites[r, c]] =
            _nested_y_for_network(yraw)
    end

    nested = NestedNetwork(tensors, layout, xraw)
    _check_nested_links(nested)
    return nested
end
```

Add:

```julia
function _check_nested_link(
    left::Grassmann,
    left_axis::Int,
    right::Grassmann,
    right_axis::Int,
    left_site::CartesianIndex{2},
    right_site::CartesianIndex{2},
)
    size(left)[left_axis] == size(right)[right_axis] ||
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
        right = CartesianIndex(
            site[1], Nmod(site[2] + 1, size(nested, 2))
        )
        below = CartesianIndex(
            Nmod(site[1] + 1, size(nested, 1)), site[2]
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

- [ ] **Step 4: Run assembly and periodic tests and verify GREEN**

```bash
cd /home/jkkong/work/2026/0731/nested-existing-ops-assembly-019fb340
./run_with_memory_guard.sh \
  --limit-gb 20 \
  --log log/green-periodic.log \
  -- bash -ic \
  'export PATH=/home/jkkong/bin:/usr/local/bin:/usr/bin:/bin;
   export JULIA_PKG_OFFLINE=true;
   julia_grassmann focused_runtests.jl'
```

Expected:

- `1 x 1`, `1 x 2`, `2 x 1`, and `2 x 2` pass all four spin structures;
- mixed `Float64` and `ComplexF64` pass;
- raw X remains stored in `x_crossings`;
- every periodic link passes the dimension/even/arrow audit.

- [ ] **Step 5: Write the CTMRG smoke test**

```julia
@testset "Corrected nested CTMRG smoke" begin
    peps = Square_GPEPS(2, 1, 2, 1, 1, Float64, false)
    nested = nested_network(peps)
    env = initialize_nested_environment(nested, 4)
    @test size(env) == size(nested)
    @test run_nested_GCTMRG!(
        nested, env, 4;
        ctmrg_iter=1, verbosity=0, save_iter=0,
    ) === env
end
```

- [ ] **Step 6: Run the CTMRG smoke test and verify RED**

```bash
cd /home/jkkong/work/2026/0731/nested-existing-ops-assembly-019fb340
./run_with_memory_guard.sh \
  --limit-gb 20 \
  --log log/red-ctmrg.log \
  -- bash -ic \
  'export PATH=/home/jkkong/bin:/usr/local/bin:/usr/bin:/bin;
   export JULIA_PKG_OFFLINE=true;
   julia_grassmann focused_runtests.jl'
```

Expected: missing `initialize_nested_environment` or
`run_nested_GCTMRG!`.

- [ ] **Step 7: Implement the thin GCTMRG adapters**

```julia
initialize_nested_environment(
    nested::NestedNetwork,
    chi::Int,
    chi_even::Int=div(chi, 2),
) = CTMRGEnv(nested.network, chi, chi_even)

function run_nested_GCTMRG!(
    nested::NestedNetwork,
    env::CTMRGEnv,
    chi::Int;
    kwargs...,
)
    size(env) == size(nested) ||
        throw(DimensionMismatch(
            "nested environment and network sizes differ"
        ))
    run_GCTMRG!(nested.network, nested.network, env, chi; kwargs...)
    return env
end
```

- [ ] **Step 8: Run focused and full server suites**

```bash
cd /home/jkkong/work/2026/0731/nested-existing-ops-assembly-019fb340
./run_with_memory_guard.sh \
  --limit-gb 20 \
  --log log/green-final.log \
  -- bash -ic \
  'export PATH=/home/jkkong/bin:/usr/local/bin:/usr/bin:/bin;
   export JULIA_PKG_OFFLINE=true;
   julia_grassmann focused_runtests.jl'

./run_with_memory_guard.sh \
  --limit-gb 20 \
  --log log/full.log \
  -- bash -ic \
  'export PATH=/home/jkkong/bin:/usr/local/bin:/usr/bin:/bin;
   export JULIA_PKG_OFFLINE=true;
   julia_grassmann server_runtests_offline.jl'
```

Expected:

- all corrected placement, local, periodic, link, and CTMRG smoke tests pass;
- full GrassmannTensorNetworks server suite passes;
- peak RSS remains below 20 GB.

- [ ] **Step 9: Commit**

```bash
git add algorithms/Nested_CTMRG/nested_network.jl \
        test/nested_network.jl src/GrassmannTensorNetworks.jl
git commit -m "feat: assemble corrected nested network"
```

## Downstream Interface Contract

After both tasks pass review:

- Task 4 must apply `_nested_y_for_network` to every identity- or
  operator-dressed raw Y before measurement.
- Task 6 must differentiate `_nested_ket_for_network`,
  `_nested_bra_for_network`, `_nested_x_for_network`, and
  `_nested_y_for_network`, rather than bypassing placement corrections.
- The corrections are static existing Grassmann operations; no correction
  parameter or tangent field is introduced.
- The parent plan resumes at Task 4 only after this supplement's two tasks
  both pass independent review.
