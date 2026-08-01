# X-Impurity Nested Network Simplification Design

## Goal

Adopt the simplified Grassmann nested-network construction in which the
physical identity or local operator belongs to the `X` tensor, while `Y` is a
pure virtual crossing. Update all dependent measurements, automatic
differentiation rules, examples, exact tests, and the external LaTeX monograph
without restoring the deleted placement-wrapper layer.

## User-owned baseline

The edited `algorithms/Nested_CTMRG/nested_network.jl` is the authoritative
design baseline:

- `_nested_ket` permutes `A[p,l,r,u,d]` to `A[l,r,p,u,d]`, applies the chosen
  parity factors, and fuses `(p,u)`.
- `_nested_bra` conjugates `A`, permutes to `Abar[l,p,r,u,d]`, applies the
  chosen parity factors, fuses `(p,r)`, and reverses the required arrows.
- `X` is to the right of the bra in the selected source-cell convention and
  carries the physical identity/operator.
- `Y` is to the left of the ket and contains only horizontal and vertical
  virtual identities plus the required vertical arrow/sign correction.

Deleted helpers such as `_graded_pair_sign`, `_placed_nested_x`,
`_nested_ket_for_network`, `_nested_bra_for_network`,
`_nested_x_for_network`, `_nested_y_for_network`, and the former reduced-basis
correction wrappers will not be restored.

## Canonical local tensors

All stored nested tensors retain the rank-four order `(left,right,up,down)` and
the arrow tuple `(:out,:in,:in,:out)`.

### Ket tensor

`_nested_ket(A)` implements the user-selected `(p,u)` fusion. Each nontrivial
line is preceded by an index equation comment, for example:

```julia
# A_perm[l, r, p, u, d] = A[p, l, r, u, d]
# Ao2[l, r, U, d] = Ao1[l, r, (p, u), d]
```

### Bra tensor

`_nested_bra(A)` implements conjugation, the `(p,r)` fusion, and the three
explicit arrow reversals. Comments expose both the pre-fusion and final arrow
orders.

### Operator-bearing X tensor

The canonical internal constructor is

```julia
_nested_x(operator, h_size, h_even, v_size, v_even)
```

where `operator` has order `(physical_out,physical_in)` and arrows
`(:out,:in)`. It contracts two true identity maps with `operator`, routes the
six indices, and fuses `(left,physical_out)` and `(down,physical_in)`.

The bulk overload

```julia
_nested_x(h_size, h_even, v_size, v_even,
          p_size, p_even, T)
```

constructs `Matrix{T}(I,p_size,p_size)` and delegates to the operator-bearing
constructor. `ones(T,n,n)` is not used because it is not an identity map when
an even or odd sector has dimension greater than one.

### Pure Y crossing

`_nested_y(h_size,h_even,v_size,v_even,T)` constructs only two true identity
maps, contracts them, reverses the vertical arrows, and applies the selected
vertical parity sign.

## Network assembly

`nested_network` constructs final `K`, `B`, bulk `X`, and `Y` tensors directly.
No raw-versus-placed wrapper is retained. The array `x_crossings` stores the
bulk identity-bearing `X` tensors so its meaning matches the actual network.
Nearest-neighbour size, even-sector, and opposite-arrow checks remain active.

## Measurement API and topology

The canonical public impurity API becomes:

```julia
nested_x_operator(nested, peps, site, operator)
```

It validates the source coordinate, physical dimensions, even split, and
arrows, then reconstructs the source `X` tensor with `operator` replacing the
physical identity.

For compatibility, `nested_y_operator(args...)` remains exported and forwards
to `nested_x_operator(args...)`. New implementation code, tests, ChainRules,
examples, and documentation use `nested_x_operator`.

The impurity routes are:

- one site: replace `layout.x_sites[source]`;
- horizontal bond: `X-B-X`;
- vertical bond: `X-K-X`.

The graded Schmidt decomposition remains parity resolved. The endpoint sign
is attached to the ket-mediated vertical route rather than assumed from the
old topology. Exact nested-versus-reduced tests determine and lock the final
sign convention; the expected topology transfer is no extra scalar sign for
`X-B-X` and `(-1)^tensor_parity(top_operator)` for `X-K-X`.

## Automatic differentiation and examples

`nested_chainrules.jl` follows the canonical `nested_x_operator` API and the
new patch topology. Rules for deleted helpers are removed. The compatibility
alias does not receive a separate handwritten rule unless dispatch requires
one; it delegates to the canonical differentiable function.

The spinless-fermion nested example keeps its external energy API but calls
the updated measurement implementation. No new MPO, swap-tensor container, or
boundary type is introduced.

## Comment style

Nontrivial contractions, permutations, fusions, traces, and impurity
replacements use equation-style comments. Examples:

```julia
# X0[l, r, u, d] = Ih[l, r] * Iv[u, d]
# X1[l, po, r, u, d, pi] = X0[l, r, u, d] * O[po, pi]
# X2[L, r, u, d, pi] = X1[(l, po), r, u, d, pi]
```

Comments name the result and both sides of the index transformation. Narrative
comments are retained only for API contracts or convention-level warnings.

## Strict algebraic tests

The acceptance tests do not run CTMRG iterations.

### Local tensor equality

For deterministic real and complex tensors, including anisotropic dimensions
and mixed even/odd splits:

1. contract one `K-Y-X-B` tile;
2. fuse external legs in the native Grassmann basis;
3. compare against `reduced_tensor(A)`;
4. require equal `size`, `even`, and `index_type`;
5. compare every dense element and every populated parity block;
6. report and bound `maximum(abs, candidate-target)`,
   `norm(candidate-target)`, and the relative norm error.

### Periodic network equality

For `1x1`, `1x2`, `2x1`, and `2x2` unit cells, compare exact finite-torus
closures of `nested.network` and `reduced_tensor(peps)` in all four periodic
spin structures. Include heterogeneous compatible bond dimensions.

### Exact observables

Without a CTMRG environment or iteration, compare nested and reduced finite
closures for:

- identity and number one-site operators;
- identity and spinless-fermion Hamiltonian horizontal bonds;
- identity and spinless-fermion Hamiltonian vertical bonds;
- denominator, numerator, and normalized ratio separately.

Tests fail with diagnostic values for maximum element error, absolute norm
error, relative norm error, and scalar differences.

## Documentation

The English academic monograph under
`D:/C 盘备份/Back up/Work_Save/latex file/2026/0801_Nested GTN` is revised so
all formulas, function tables, TikZ diagrams, sign audits, and source mappings
show an operator-bearing `X`, a pure `Y`, `X-B-X` horizontal measurements, and
`X-K-X` vertical measurements. The legacy `nested_y_operator` name is
documented only as a compatibility alias. The PDF is recompiled and every page
is visually inspected.

## Out of scope

- No new MPO representation.
- No new tensor or boundary container.
- No CTMRG convergence claim in the strict algebraic acceptance tests.
- No unrelated restructuring of the repository.
