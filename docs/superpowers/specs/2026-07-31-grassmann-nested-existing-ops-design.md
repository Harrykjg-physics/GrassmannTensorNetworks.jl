# Grassmann Nested Network Existing-Ops Design

## Status

Approved design for replacing the rejected correction-MPO proposal with a
native construction that uses only existing Grassmann tensor operations.

## Context

The TensorKit reference implementation represents fusion isomorphisms,
braids, twists, duals, and contraction coherence as categorical morphisms.
GrassmannTensorNetworks instead stores block arrays in a fixed linear index
order. Its native `fuse` is a sector-ordered reshape and concatenation, while
`contract(...; sign_function=auto_sign)` inserts Koszul phases from the
native contraction route.

The Task 2 K/Y/X/B translation is locally correct after an eight-leg
comparison map, but its first uncorrected periodic checkerboard double-counts
two route-exchange phases. A proposed correction MPO would encode the missing
phases, but exhaustive GF(2) and numerical oracles show that no new tensor
structure is necessary. Existing node-local sign operations span the exact
periodic correction.

## Design Goals

- Preserve the explicit `2m x 2n` K/Y/X/B checkerboard.
- Preserve raw categorical primitives, including raw X odd-odd value `-1`.
- Use only existing `add_parity_sign`, `add_perm_sign`, conjugation,
  Grassmann permutation, fusion, splitting, and contraction operations.
- Add no auxiliary legs, correction MPO, new tensor type, or new network
  contraction semantics.
- Work for arbitrary positive source unit-cell dimensions, including odd
  numbers of rows.
- Match the native reduced network under all four periodic spin structures.
- Keep the change local to `algorithms/Nested_CTMRG` and its tests.

## Rejected Alternatives

### X/Y-only correction

Correcting only X and Y is exact when the source unit cell has an even number
of rows, including the required `2 x 2` acceptance cell. It fails for odd-row
cells such as `1 x 1` and `1 x 2`, leaving

```text
sum_cells(kU + bU).
```

The public API must not silently depend on unit-cell parity, so this
specialization is rejected.

### Z2 correction MPO

A finite correction MPO with abstract bond dimension at most four can encode
the required quadratic phase, but it is unnecessary once the universal
node-local solution is used. It would add auxiliary dimensions and complicate
CTMRG and AD.

### Global fusion or contraction changes

Changing repository-wide `fuse` or `contract` semantics would affect
unrelated algorithms and violate the small-change constraint.

## Universal Native Placement Correction

Let `A` have native order `(P,L,R,U,D)`. Keep the raw constructors
`_nested_ket`, `_nested_bra`, `_nested_x`, and `_nested_y` unchanged as the
reference-level categorical primitives.

Only tensors placed in the native checkerboard receive the following
corrections:

```julia
A_up = add_parity_sign(A, 4; sign_function=global_sign)

K_network = _nested_ket(A_up)
B_network = _nested_bra(A_up)

X_network = add_perm_sign(
    _placed_nested_x(X_raw),
    (1, 3, 2, 4);
    sign_function=global_sign,
)

Y_network = add_perm_sign(
    Y_raw,
    (1, 3, 2, 4);
    sign_function=global_sign,
)
```

The K and B input twist contributes `kU` and `bU`, respectively. For B, the
twist must act on the original north parity before `(P,U)` fusion; twisting
the final B axis 3 would incorrectly apply `p+bU`.

`add_perm_sign` does not move axes. The permutation `(1,3,2,4)` has only the
adjacent inversion `(2,3)` and therefore contributes:

```text
X: q2*q3 = bL*kD
Y: q2*q3 = kR*bU.
```

The complete universal phase is

```text
kU + bU + bL*kD + kR*bU.
```

These factors correct a representation-level double count. Raw X remains a
fermionic braid with odd-odd coefficient `-1`, and raw Y continues to be
constructed from that braid. The native contraction route already contributes
the corresponding exchanges; applying the no-move signs at placement makes
the total graphical phase equal to the TensorKit reference.

## Private Interfaces

No new public type is introduced. Placement corrections should be centralized
in private helpers or one private node builder so that measurements and AD
cannot accidentally use a raw node where a network node is required.

The intended responsibilities are:

- raw constructors: reproduce the reference categorical pieces and support
  primitive tests;
- placement helpers: apply the four universal native corrections;
- `nested_network`: assemble only corrected placement tensors;
- `NestedNetwork.x_crossings`: retain the raw X crossings for measurement and
  differentiation logic, while `network[x_site]` stores corrected placed X;
- operator-dressed Y: apply the same Y placement correction after inserting
  the operator.

## Assembly Data Flow

For each source site `(r,c)`:

1. Apply the north parity twist to `A[r,c]`.
2. Construct corrected K and B from that twisted input.
3. Construct raw X from horizontal `B.W` and vertical `K.S`, in that order.
4. Apply the placed-X west twist, then the X no-move permutation sign.
5. Construct raw Y from the east-neighbour K route and north-neighbour B
   route.
6. Apply the Y no-move permutation sign.
7. Place K/Y/X/B at odd-odd, odd-even, even-odd, and even-even sites.
8. Validate dimensions, even-sector sizes, and opposite arrow directions on
   every periodic horizontal and vertical link.

The network remains an ordinary matrix of rank-four `Grassmann` tensors and
can be passed to the existing `CTMRGEnv` and `run_GCTMRG!`.

## Local Comparison Invariant

The Task 2 local factorization test remains useful, but its comparison gauge
must be updated after the universal placement corrections.

In bra-first order

```text
(bL,kL,bR,kR,bU,kU,bD,kD),
```

the corrected comparison phase reduces to

```text
bL + kL + kR + bU.
```

It is implemented by `add_parity_sign` on ordered axes `(1,2,4,5)` before
the existing reduced-style external fusion pipeline. This comparison map is
a diagnostic gauge map only; it is not inserted into each periodic bulk
tile.

## Spin Structures

For native geographic order `(L,R,U,D)`:

- horizontal anti-periodic twist acts on axis 2 of the last column;
- vertical anti-periodic twist acts on axis 4 of the last row.

The exact periodic regression covers:

```text
(twist_x,twist_y) =
    (false,false),
    (true,false),
    (false,true),
    (true,true).
```

The universal placement correction is independent of the selected spin
structure.

## Validation

### Algebraic validation

- Exhaust all 8192 compatible parity assignments on a `2 x 2` source torus.
- Check the universal residual is zero for source shapes `1 x 1` through
  `3 x 3` and all four spin structures.
- Retain an odd-row regression so the rejected X/Y-only specialization
  cannot return unnoticed.

### Numerical validation

- Minimal nonuniform tensors for `1 x 1`, `1 x 2`, `2 x 1`, `2 x 2`, and
  `3 x 2`.
- Nonuniform mixed-multiplicity `Float64` and `ComplexF64` cells.
- Exact local factorization for minimal and mixed multiplicities.
- `2 x 2` reduced versus `4 x 4` corrected nested periodic scalar for all
  four spin structures.
- Nested link validation with anisotropic horizontal and vertical dimensions.
- One-iteration nested CTMRG smoke test.
- Full server regression suite.

Existing oracle evidence gives:

```text
GF(2) universal residual:                 exactly zero
2 x 2 minimal maximum relative error:    8.57e-16
mixed Float64 maximum relative error:     5.37e-16
mixed ComplexF64 maximum relative error:  4.31e-16
```

Small `1 x 1`, `1 x 2`, and `2 x 1` scalar comparisons reach approximately
`1.5e-14` because of contraction-order roundoff; their parity residual is
exactly zero.

All Julia verification runs only on `jkkong@172.23.26.248` through
`julia_grassmann`, under the 20 GB memory guard, with local Markdown logs in
the required 2026 date directory.

## Downstream Measurements and AD

One-site and two-site operator insertions replace raw Y but must reuse the
same Y placement correction. Identity insertions must reproduce the corrected
bulk Y exactly.

The four correction operations are static and parameter-free. AD should
differentiate through their existing Grassmann operations. Task 6 must still
test the assembled `nested_network` pullback and operator-dressed paths by
finite differences, including the K/B input twist.

## Acceptance

Task 3 is accepted only when:

- arbitrary source unit-cell shapes use the universal four-term correction;
- raw X odd-odd remains `-1`;
- local minimal and mixed factorization tests pass;
- odd-row and even-row periodic tests pass;
- all four `2 x 2` spin-structure comparisons agree with the reduced network;
- link validation and nested CTMRG smoke tests pass;
- the full server suite passes without exceeding 20 GB.

The MPO design is removed from the implementation scope unless future
evidence invalidates the proven existing-operations construction.
