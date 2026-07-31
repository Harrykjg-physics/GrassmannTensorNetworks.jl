# Exact Nested Measurement Tests Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Validate one-site and nearest-neighbor nested measurement
factorizations by exact native contraction against reduced-layer references,
without constructing or iterating a CTMRG environment in measurement tests.

**Architecture:** Keep the environment-based production measurement methods
from parent-plan Tasks 4 and 5. Replace their CTMRG-based tests with exact
finite-periodic closures of the corrected nested checkerboard and independent
reduced tensors, comparing denominator, numerator, and normalized value under
all four spin structures.

**Tech Stack:** Julia, GrassmannTensorNetworks native `contract` and `trace`,
Test, required SSH Julia server.

## Global Constraints

- The approved design is
  `docs/superpowers/specs/2026-07-31-nested-exact-measurement-tests-design.md`.
- No test in `test/nested_measurements.jl` constructs `CTMRGEnv`, calls
  `initialize_nested_environment`, `run_GCTMRG!`, or
  `run_nested_GCTMRG!`.
- Retain the environment-based `compute_nested_exp_site`,
  `compute_nested_exp_hbond`, and `compute_nested_exp_vbond` production
  interfaces.
- Compare exact nested and reduced denominator, numerator, and normalized
  expectation using `rtol=5e-12` and `atol=1e-12`.
- Cover all four spin structures.
- Require at least one nonzero number or bond numerator; zero-equals-zero is
  insufficient evidence.
- Use only existing Grassmann tensors and native sign/contraction operations.
- Run Julia only on `jkkong@172.23.26.248` through `julia_grassmann`, with
  the 20 GB memory guard and local Markdown logs under
  `D:\C 盘备份\Back up\Work_Save\coding\codex_server_logs\2026\0731`.
- Do not change dependency versions.

---

### Task 1: Replace One-Site CTMRG Tests with Exact Closures

**Files:**

- Modify: `algorithms/Nested_CTMRG/measurements.jl`
- Modify: `test/nested_measurements.jl`
- Preserve the other parent Task 4 file changes.

**Interfaces:**

- Consumes: `nested_network`, `nested_y_operator`, `reduced_tensor`, and
  `nested_test_torus_scalar`.
- Produces: `_check_nested_operator_unit_cell(peps, operators)` for the
  environment-based matrix overload and its environment-free validation
  test.

- [ ] **Step 1: Replace CTMRG fixtures with exact one-site network helpers**

Keep `physical_identity` and add:

```julia
import GrassmannTensorNetworks: _check_nested_operator_unit_cell

const MEASUREMENT_SPIN_STRUCTURES =
    ((false, false), (true, false), (false, true), (true, true))

function exact_site_networks(peps, operator, source::CartesianIndex{2})
    nested = nested_network(peps)
    nested_impurity = copy(nested.network)
    nested_impurity[nested.layout.y_sites[source]] =
        nested_y_operator(nested, peps, source, operator)

    T = eltype(peps)
    reduced_bulk =
        Matrix{Grassmann{T, 4}}(reduced_tensor.(peps.A))
    reduced_impurity = copy(reduced_bulk)
    reduced_impurity[source] =
        reduced_tensor(peps.A[source], operator)

    return (
        nested=nested,
        nested_bulk=nested.network,
        nested_impurity=nested_impurity,
        reduced_bulk=reduced_bulk,
        reduced_impurity=reduced_impurity,
    )
end

function exact_site_scalars(
    networks;
    twist_x::Bool,
    twist_y::Bool,
)
    close(tensors) = nested_test_torus_scalar(
        tensors; twist_x, twist_y
    )
    return (
        nested_denominator=close(networks.nested_bulk),
        nested_numerator=close(networks.nested_impurity),
        reduced_denominator=close(networks.reduced_bulk),
        reduced_numerator=close(networks.reduced_impurity),
    )
end
```

- [ ] **Step 2: Write the exact identity, number, validation, and matrix tests**

Replace every Task 4 test that constructs or iterates an environment with:

```julia
@testset "Exact nested one-site measurements" begin
    Random.seed!(0x4e455354)
    peps = Square_GPEPS(2, 1, 2, 1, 2, Float64, false)
    source = CartesianIndex(1, 1)
    nested = nested_network(peps)
    identity = physical_identity()
    number = n_site(SpinlessFermionModel(1.0, 1.0, 3.0))

    @test nested_y_operator(nested, peps, source, identity) ≈
        nested[nested.layout.y_sites[source]]

    wrong_size = Grassmann(
        Matrix{Float64}(I, 4, 4),
        (4, 4), (2, 2), (:out, :in),
    )
    wrong_even = Grassmann(
        Matrix{Float64}(I, 2, 2),
        (2, 2), (2, 2), (:out, :in),
    )
    wrong_arrows = Grassmann(
        Matrix{Float64}(I, 2, 2),
        (2, 2), (1, 1), (:in, :out),
    )
    @test_throws ArgumentError nested_y_operator(
        nested, peps, (2, 1), number
    )
    @test_throws DimensionMismatch nested_y_operator(
        nested, peps, source, wrong_size
    )
    @test_throws DimensionMismatch nested_y_operator(
        nested, peps, source, wrong_even
    )
    @test_throws ArgumentError nested_y_operator(
        nested, peps, source, wrong_arrows
    )

    operators = fill(number, size(peps))
    @test _check_nested_operator_unit_cell(peps, operators) === nothing
    @test_throws DimensionMismatch _check_nested_operator_unit_cell(
        peps, fill(number, 1, 3)
    )

    identity_networks = exact_site_networks(peps, identity, source)
    number_networks = exact_site_networks(peps, number, source)
    number_numerators = Float64[]
    for (twist_x, twist_y) in MEASUREMENT_SPIN_STRUCTURES
        identity_data = exact_site_scalars(
            identity_networks; twist_x, twist_y
        )
        @test identity_data.nested_denominator ≈
            identity_data.reduced_denominator rtol=5e-12 atol=1e-12
        @test identity_data.nested_numerator ≈
            identity_data.reduced_numerator rtol=5e-12 atol=1e-12
        @test identity_data.nested_numerator /
            identity_data.nested_denominator ≈ 1 atol=1e-12

        number_data = exact_site_scalars(
            number_networks; twist_x, twist_y
        )
        @test number_data.nested_denominator ≈
            number_data.reduced_denominator rtol=5e-12 atol=1e-12
        @test number_data.nested_numerator ≈
            number_data.reduced_numerator rtol=5e-12 atol=1e-12
        @test number_data.nested_numerator /
            number_data.nested_denominator ≈
            number_data.reduced_numerator /
            number_data.reduced_denominator rtol=5e-12 atol=1e-12
        push!(number_numerators, abs(number_data.nested_numerator))
    end
    @test maximum(number_numerators) > 1e-12
end
```

- [ ] **Step 3: Run the focused test and verify RED**

Run in
`/home/jkkong/work/2026/0731/nested-exact-site-019fb340`:

```bash
./run_with_memory_guard.sh \
  --limit-gb 20 \
  --log log/red-exact-site.log \
  -- bash -ic \
  'export PATH=/home/jkkong/bin:/usr/local/bin:/usr/bin:/bin;
   export JULIA_PKG_OFFLINE=true;
   julia_grassmann focused_runtests.jl'
```

Expected: the import or validation test fails because
`_check_nested_operator_unit_cell` does not exist. No CTMRG iteration appears
in the command output.

- [ ] **Step 4: Implement the pure matrix-unit-cell validator**

Add:

```julia
function _check_nested_operator_unit_cell(
    peps::Square_GPEPS,
    operators::AbstractMatrix,
)
    size(operators) == size(peps) ||
        throw(DimensionMismatch("operator and PEPS unit cells differ"))
    return nothing
end
```

At the start of the environment-based matrix overload, replace its inline
size check with:

```julia
_check_nested_operator_unit_cell(peps, operators)
```

- [ ] **Step 5: Run focused and full tests and verify GREEN**

Use the same guarded focused command with `log/green-exact-site.log`, then:

```bash
./run_with_memory_guard.sh \
  --limit-gb 20 \
  --log log/full-exact-site.log \
  -- bash -ic \
  'export PATH=/home/jkkong/bin:/usr/local/bin:/usr/bin:/bin;
   export JULIA_PKG_OFFLINE=true;
   julia_grassmann server_runtests_offline.jl'
```

Expected: all exact identity/number/validation tests and the full package
suite pass with exit code `0`; peak RSS stays below 20 GB. Run:

```bash
rg -n \
  'CTMRGEnv|initialize_nested_environment|run_GCTMRG!|run_nested_GCTMRG!' \
  test/nested_measurements.jl
```

Expected: no matches.

- [ ] **Step 6: Commit the completed parent Task 4**

```bash
git add algorithms/Nested_CTMRG/measurements.jl \
        src/algorithms.jl src/GrassmannTensorNetworks.jl \
        test/nested_measurements.jl test/runtests.jl \
        test/server_runtests.jl
git commit -m "feat: measure nested one-site operators"
```

---

### Task 2: Replace Two-Site CTMRG Tests with Exact Bond Closures

**Files:**

- Modify: `algorithms/Nested_CTMRG/measurements.jl`
- Modify: `test/nested_measurements.jl`

**Interfaces:**

- Consumes: `_operator_schmidt`, `nested_y_operator`,
  `reduced_tensor_hbond`, `reduced_tensor_vbond`, and
  `nested_test_torus_scalar`.
- Produces and tests:
  `_check_nested_bond_operator(peps, operator, source, neighbor)`.
- Preserves: the parent Task 5 environment-based strip contractions and
  public horizontal/vertical measurement methods.

- [ ] **Step 1: Add exact reduced-bond closure helpers**

Append:

```julia
function close_reduced_hbond(
    tensor::Grassmann;
    twist_x::Bool,
    twist_y::Bool,
)
    closed = trace(
        tensor,
        ((1, 3, 4), (2, 5, 6));
        pbc=(!twist_x, !twist_y, !twist_y),
        sign_function=GrassmannTensorNetworks.global_sign,
    )
    return scalar(closed)
end

function close_reduced_vbond(
    tensor::Grassmann;
    twist_x::Bool,
    twist_y::Bool,
)
    closed = trace(
        tensor,
        ((1, 2, 5), (3, 4, 6));
        pbc=(!twist_x, !twist_x, !twist_y),
        sign_function=GrassmannTensorNetworks.global_sign,
    )
    return scalar(closed)
end

function exact_nested_bond_numerator(
    nested::NestedNetwork,
    peps::Square_GPEPS,
    operator::Grassmann,
    source::CartesianIndex{2},
    orientation::Symbol;
    twist_x::Bool,
    twist_y::Bool,
)
    neighbor = orientation === :horizontal ?
        CartesianIndex(
            source[1], Nmod(source[2] + 1, size(peps)[2])
        ) :
        CartesianIndex(
            Nmod(source[1] + 1, size(peps)[1]), source[2]
        )
    total = zero(nested_test_torus_scalar(
        nested.network; twist_x, twist_y
    ))
    for (left_operator, right_operator) in
        GrassmannTensorNetworks._operator_schmidt(operator)
        impurity = copy(nested.network)
        impurity[nested.layout.y_sites[source]] =
            nested_y_operator(
                nested, peps, source, left_operator
            )
        impurity[nested.layout.y_sites[neighbor]] =
            nested_y_operator(
                nested, peps, neighbor, right_operator
            )
        total += nested_test_torus_scalar(
            impurity; twist_x, twist_y
        )
    end
    return total
end
```

- [ ] **Step 2: Write exact operator reconstruction and bond comparisons**

Append:

```julia
@testset "Exact nested nearest-neighbor measurements" begin
    identity2 = two_site_identity()
    hamiltonian =
        nn_bond(SpinlessFermionModel(1.0, 1.0, 3.0))

    terms = GrassmannTensorNetworks._operator_schmidt(hamiltonian)
    reconstructed = sum(
        contract(
            left, right;
            sign_function=GrassmannTensorNetworks.global_sign,
        ) for (left, right) in terms
    )
    ordered = permutedims(
        reconstructed, (1, 3, 2, 4);
        sign_function=GrassmannTensorNetworks.global_sign,
    )
    @test ordered ≈ hamiltonian rtol=1e-12 atol=1e-12

    Random.seed!(0x48424f4e44)
    hpeps = Square_GPEPS(2, 1, 2, 1, 2, Float64, false)
    hnested = nested_network(hpeps)
    hsource = CartesianIndex(1, 1)
    hdenominator_reference = reduced_tensor_hbond(
        hpeps.A[1, 1], hpeps.A[1, 2], identity2
    )
    hnumerator_reference = reduced_tensor_hbond(
        hpeps.A[1, 1], hpeps.A[1, 2], hamiltonian
    )

    Random.seed!(0x56424f4e44)
    vpeps = Square_GPEPS(2, 1, 2, 2, 1, Float64, false)
    vnested = nested_network(vpeps)
    vsource = CartesianIndex(1, 1)
    vdenominator_reference = reduced_tensor_vbond(
        vpeps.A[1, 1], vpeps.A[2, 1], identity2
    )
    vnumerator_reference = reduced_tensor_vbond(
        vpeps.A[1, 1], vpeps.A[2, 1], hamiltonian
    )

    @test GrassmannTensorNetworks._check_nested_bond_operator(
        hpeps, hamiltonian,
        CartesianIndex(1, 1), CartesianIndex(1, 2),
    ) === nothing
    wrong_size = Grassmann(
        (3, 2, 3, 2), (1, 1, 1, 1),
        (:out, :out, :in, :in), Float64;
        init=:zeros,
    )
    wrong_even = Grassmann(
        (2, 2, 2, 2), (2, 1, 2, 1),
        (:out, :out, :in, :in), Float64;
        init=:zeros,
    )
    wrong_arrows = Grassmann(
        (2, 2, 2, 2), (1, 1, 1, 1),
        (:in, :out, :out, :in), Float64;
        init=:zeros,
    )
    odd_operator = Grassmann(
        (2, 2, 2, 2), (1, 1, 1, 1),
        (:out, :out, :in, :in), Float64;
        init=:zeros, parity=:odd,
    )
    @test_throws ArgumentError
        GrassmannTensorNetworks._check_nested_bond_operator(
            hpeps, hamiltonian,
            CartesianIndex(2, 1), CartesianIndex(1, 2),
        )
    @test_throws DimensionMismatch
        GrassmannTensorNetworks._check_nested_bond_operator(
            hpeps, wrong_size,
            CartesianIndex(1, 1), CartesianIndex(1, 2),
        )
    @test_throws DimensionMismatch
        GrassmannTensorNetworks._check_nested_bond_operator(
            hpeps, wrong_even,
            CartesianIndex(1, 1), CartesianIndex(1, 2),
        )
    @test_throws ArgumentError
        GrassmannTensorNetworks._check_nested_bond_operator(
            hpeps, wrong_arrows,
            CartesianIndex(1, 1), CartesianIndex(1, 2),
        )
    @test_throws ArgumentError
        GrassmannTensorNetworks._check_nested_bond_operator(
            hpeps, odd_operator,
            CartesianIndex(1, 1), CartesianIndex(1, 2),
        )

    bond_numerators = Float64[]
    for (twist_x, twist_y) in MEASUREMENT_SPIN_STRUCTURES
        hdenominator = nested_test_torus_scalar(
            hnested.network; twist_x, twist_y
        )
        hnumerator = exact_nested_bond_numerator(
            hnested, hpeps, hamiltonian, hsource, :horizontal;
            twist_x, twist_y
        )
        hdenominator_reduced = close_reduced_hbond(
            hdenominator_reference; twist_x, twist_y
        )
        hnumerator_reduced = close_reduced_hbond(
            hnumerator_reference; twist_x, twist_y
        )
        @test hdenominator ≈ hdenominator_reduced
            rtol=5e-12 atol=1e-12
        @test hnumerator ≈ hnumerator_reduced
            rtol=5e-12 atol=1e-12
        @test hnumerator / hdenominator ≈
            hnumerator_reduced / hdenominator_reduced
            rtol=5e-12 atol=1e-12

        vdenominator = nested_test_torus_scalar(
            vnested.network; twist_x, twist_y
        )
        vnumerator = exact_nested_bond_numerator(
            vnested, vpeps, hamiltonian, vsource, :vertical;
            twist_x, twist_y
        )
        vdenominator_reduced = close_reduced_vbond(
            vdenominator_reference; twist_x, twist_y
        )
        vnumerator_reduced = close_reduced_vbond(
            vnumerator_reference; twist_x, twist_y
        )
        @test vdenominator ≈ vdenominator_reduced
            rtol=5e-12 atol=1e-12
        @test vnumerator ≈ vnumerator_reduced
            rtol=5e-12 atol=1e-12
        @test vnumerator / vdenominator ≈
            vnumerator_reduced / vdenominator_reduced
            rtol=5e-12 atol=1e-12

        push!(bond_numerators, abs(hnumerator), abs(vnumerator))
    end
    @test maximum(bond_numerators) > 1e-12
end
```

- [ ] **Step 3: Run the focused test and verify RED**

Run in
`/home/jkkong/work/2026/0731/nested-exact-bond-019fb340`:

```bash
./run_with_memory_guard.sh \
  --limit-gb 20 \
  --log log/red-exact-bond.log \
  -- bash -ic \
  'export PATH=/home/jkkong/bin:/usr/local/bin:/usr/bin:/bin;
   export JULIA_PKG_OFFLINE=true;
   julia_grassmann focused_runtests.jl'
```

Expected: `_check_nested_bond_operator`, `_operator_schmidt`,
`compute_nested_exp_hbond`, or `compute_nested_exp_vbond` is missing. The
test output contains no CTMRG iteration.

- [ ] **Step 4: Implement parent Task 5 production methods**

Implement the exact `_operator_schmidt`, environment-based strip helpers,
three-node patch helpers, scalar public methods, and unit-cell overloads from
parent-plan Task 5. Do not add a second exact-contraction production API;
the exact closure helpers remain test-only.

- [ ] **Step 5: Run focused and full tests and verify GREEN**

Use the guarded focused command with `log/green-exact-bond.log`, followed by
the full offline command from Task 1 with `log/full-exact-bond.log`.

Expected: operator reconstruction, exact horizontal/vertical comparisons,
and the full suite pass with exit code `0`; peak RSS stays below 20 GB. Run
the static `rg` check from Task 1 and verify it still has no matches.

- [ ] **Step 6: Commit**

```bash
git add algorithms/Nested_CTMRG/measurements.jl \
        test/nested_measurements.jl
git commit -m "feat: measure nested nearest-neighbor operators"
```
