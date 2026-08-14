# X-Impurity Nested Network Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the simplified Grassmann nested network in which `X` carries the physical identity/operator and `Y` is a pure virtual crossing, then update exact tests, measurements, ChainRules, examples, and the academic LaTeX/PDF.

**Architecture:** Preserve the user's direct `_nested_ket` and `_nested_bra` routes, build final `K/Y/X/B` tensors without the deleted placement-wrapper layer, and make `nested_x_operator` the canonical impurity API. Exact finite contractions compare the nested and reduced representations without CTMRG iterations; `nested_y_operator` remains a compatibility forwarder.

**Tech Stack:** Julia, GrassmannTensorNetworks.jl, ChainRulesCore/Zygote extension, `Test`, remote `julia_grassmann`, XeLaTeX/TikZ/BibTeX, Poppler.

## Global Constraints

- Treat the user's edited `algorithms/Nested_CTMRG/nested_network.jl` as the design baseline; do not restore deleted wrapper helpers.
- Do not introduce an MPO, swap-container type, or new boundary structure.
- All Julia execution occurs on `jkkong@172.23.26.248` through `julia_grassmann`, with a 20 GB memory guard and a local Markdown log under `coding/codex_server_logs/2026/0801`.
- Strict nested-versus-reduced acceptance tests perform no CTMRG iteration.
- Nontrivial tensor operations receive equation-style comments such as `# C[i, k] = A[i, j] * B[j, k]`.
- Preserve the public compatibility name `nested_y_operator`, but use `nested_x_operator` canonically everywhere else.
- Keep repository restructuring local to `algorithms/Nested_CTMRG`, its tests, module exports, extension rules, example, and corresponding documentation.

---

### Task 1: Write strict failing network tests

**Files:**
- Modify: `test/nested_network.jl`

**Interfaces:**
- Consumes: `_nested_ket`, `_nested_bra`, `_nested_x`, `_nested_y`, `nested_network`, `reduced_tensor`.
- Produces: diagnostic helpers and exact local/periodic requirements that the simplified production code must satisfy.

- [ ] **Step 1: Replace deleted-helper imports with the simplified API**

Use only:

```julia
import GrassmannTensorNetworks:
    _nested_ket, _nested_bra, _nested_x, _nested_y
```

Remove tests whose only purpose was to validate deleted `_for_network`, raw-placement, and reduced-basis correction helpers.

- [ ] **Step 2: Add dense and blockwise tensor diagnostics**

```julia
function tensor_comparison(candidate, target)
    delta = convert(Array, candidate) - convert(Array, target)
    target_norm = norm(convert(Array, target))
    return (
        max_element_error=maximum(abs, delta; init=0.0),
        norm_error=norm(delta),
        relative_norm_error=norm(delta) / max(target_norm, eps(Float64)),
    )
end

function test_strict_tensor_equal(candidate, target; atol, rtol)
    @test size(candidate) == size(target)
    @test even(candidate) == even(target)
    @test index_type(candidate) == index_type(target)
    @test sort!(collect(nonzero_keys(candidate))) ==
          sort!(collect(nonzero_keys(target)))
    data = tensor_comparison(candidate, target)
    @test data.max_element_error <= atol + rtol * norm(convert(Array, target))
    @test data.norm_error <= atol + rtol * norm(convert(Array, target))
    @test data.relative_norm_error <= rtol
    @test convert(Array, candidate) ≈ convert(Array, target) atol=atol rtol=rtol
end
```

- [ ] **Step 3: Express the simplified local tile contraction**

The helper contracts the final `K/Y/X/B` tensors in their checkerboard order,
fuses the four external bra/ket leg pairs, and returns a rank-four tensor in
the same native basis as `reduced_tensor(A)`. Precede each transformation with
an index-equation comment.

- [ ] **Step 4: Add deterministic real, complex, and anisotropic cases**

Use `(2,2,2,2,2)/(1,1,1,1,1)` and
`(3,3,4,3,4)/(2,1,3,2,1)` source dimensions. Require strict dense, blockwise,
metadata, maximum-element, norm, and relative-norm agreement.

- [ ] **Step 5: Keep exact periodic comparisons without CTMRG**

For `1x1`, `1x2`, `2x1`, and `2x2` cells, including heterogeneous compatible
bonds, compare exact nested and reduced torus scalars in all four spin
structures. Do not initialize or update `CTMRGEnv`.

- [ ] **Step 6: Run the focused RED on the server**

Create remote workspace `/home/jkkong/work/2026/0801/x-impurity-nested`, upload
the worktree snapshot plus `run_with_memory_guard.sh`, and execute:

```bash
./run_with_memory_guard.sh --limit-gb 20 --log log/red-network.log -- \
  bash -ic 'export PATH=/home/jkkong/bin:/usr/local/bin:/usr/bin:/bin; \
  export JULIA_PKG_OFFLINE=true; julia_grassmann test/nested_network.jl'
```

Expected: nonzero exit caused by the incomplete `_nested_x`/`_nested_y`
signatures or a strict nested-versus-reduced mismatch, not by a test syntax
error. Record the command, failure headline, exit code, duration, and peak RSS
in the local task log.

---

### Task 2: Complete the simplified local construction

**Files:**
- Modify: `algorithms/Nested_CTMRG/nested_network.jl`
- Test: `test/nested_network.jl`

**Interfaces:**
- Consumes: the strict tests from Task 1.
- Produces: final `_nested_ket`, `_nested_bra`, operator-bearing/bulk `_nested_x`, pure `_nested_y`, and direct `nested_network` assembly.

- [ ] **Step 1: Preserve and document ket/bra routes**

Keep the user's operations and replace narrative transformation comments with
equations such as:

```julia
# A_perm[l, r, p, u, d] = A[p, l, r, u, d]
# K[l, r, U, d] = Ao1[l, r, (p, u), d]
# B[l, R, u, d] = Ao1[l, (p, r), u, d]
```

- [ ] **Step 2: Implement the operator-bearing X constructor**

```julia
function _nested_x(
    operator::Grassmann{T, 2},
    h_size::Int, h_even::Int,
    v_size::Int, v_even::Int,
) where {T}
    Ih = Grassmann(Matrix{T}(I, h_size, h_size),
        (h_size, h_size), (h_even, h_even), (:in, :out))
    Iv = Grassmann(Matrix{T}(I, v_size, v_size),
        (v_size, v_size), (v_even, v_even), (:in, :out))
    # X0[l, r, u, d] = Ih[l, r] * Iv[u, d]
    X0 = contract(Ih, Iv; sign_function=global_sign)
    # X1[l, po, r, u, d, pi] = X0[l, r, u, d] * operator[po, pi]
    X1 = contract(X0, operator;
        perm=(1, 5, 2, 3, 4, 6), sign_function=global_sign)
    # X2[L, r, u, d, pi] = X1[(l, po), r, u, d, pi]
    X2 = fuse(X1, (1, 2); index_type_fused=:out)
    # X3[L, r, u, D] = X2[L, r, u, (d, pi)]
    return fuse(X2, (4, 5); index_type_fused=:out)
end
```

The bulk overload creates a true `Matrix{T}(I,p_size,p_size)` physical map and
delegates. Validate all total/even dimensions before allocating.

- [ ] **Step 3: Implement pure Y**

Build true horizontal and vertical identity maps, contract them, reverse axes
`(3,4)`, and apply the axis-3 parity factor. Include equations for the input
and resulting arrow order.

- [ ] **Step 4: Assemble final tensors directly**

Pass the source physical dimensions to bulk `_nested_x`; pass `T` to
`_nested_y`; assign `xraw[r,c]` and `yraw` directly to the layout sites. Remove
the invalid `` `phy_even_size` `` binding and all one-argument `_nested_x` or
`_nested_y` calls.

- [ ] **Step 5: Run focused GREEN and refactor**

Repeat the guarded server command from Task 1 using `log/green-network.log`.
Expected: every strict local and periodic network test passes, with no CTMRG
iteration in output. Run `git diff --check` locally after GREEN.

---

### Task 3: Move observables and exact measurements to X

**Files:**
- Modify: `test/nested_measurements.jl`
- Modify: `algorithms/Nested_CTMRG/measurements.jl`
- Modify: `src/GrassmannTensorNetworks.jl`

**Interfaces:**
- Consumes: operator-bearing `_nested_x` and direct layout from Task 2.
- Produces: `nested_x_operator`, compatibility `nested_y_operator`, one-site X impurity, horizontal `X-B-X`, and vertical `X-K-X` exact/public measurements.

- [ ] **Step 1: Write failing X-impurity tests**

Change exact impurity replacement to:

```julia
nested_impurity[nested.layout.x_sites[source]] =
    nested_x_operator(nested, peps, source, operator)
```

Assert identity reconstruction equals the bulk X, the compatibility alias is
equal to `nested_x_operator`, and invalid site/size/even/arrows raise the same
errors through both names.

- [ ] **Step 2: Write failing exact one- and two-site comparisons**

Use finite torus closure only. Compare nested/reduced denominator, numerator,
and normalized ratio for every spin structure. Place Schmidt factors at X
sites. The RED audit ultimately fixes horizontal `X-B-X` to no extra scalar
sign and vertical `X-K-X` to `(-1)^tensor_parity(top_operator)`, together
with the local odd-X correction `(-1)^(q*(1+u))`.

- [ ] **Step 3: Run measurement RED on the server**

Use the same isolated remote workspace and guard:

```bash
./run_with_memory_guard.sh --limit-gb 20 --log log/red-measurements.log -- \
  bash -ic 'export PATH=/home/jkkong/bin:/usr/local/bin:/usr/bin:/bin; \
  export JULIA_PKG_OFFLINE=true; julia_grassmann test/nested_measurements.jl'
```

Expected: `nested_x_operator` missing or exact old-Y topology mismatch.

- [ ] **Step 4: Implement canonical and compatibility APIs**

Create `_nested_x_operator_raw` from the stored K/B dimensions and implement
validated `nested_x_operator`. Add:

```julia
nested_y_operator(args...) = nested_x_operator(args...)
```

with a doc comment explaining that the legacy name is retained only for source
compatibility. Export both names.

- [ ] **Step 5: Update patch selection and endpoint signs**

One-site measurement uses `layout.x_sites[source]`. Horizontal patch endpoints
use `x_sites` with the intervening B; vertical endpoints use `x_sites` with the
intervening K. Update variable names from `left_y/right_y` to
`left_x/right_x` and from `top_y/bottom_y` to `top_x/bottom_x`. Apply the
strict-test-confirmed endpoint sign convention.

- [ ] **Step 6: Run measurement GREEN**

Repeat the guarded command with `log/green-measurements.log`. Expected: all
validation, reconstruction, element/norm, and exact finite-closure comparisons
pass without CTMRG iteration.

---

### Task 4: Synchronize ChainRules and the nested example

**Files:**
- Modify: `test/nested_chainrules.jl`
- Modify: `algorithms/Nested_CTMRG/nested_chainrules.jl`
- Modify: `ext/GrassmannChainRulesCoreExt/GrassmannChainRulesCoreExt.jl`
- Modify: `examples/Spinless_Fermion_2D_Square_AD_nested/Spinless_Fermion_2D_Square_AD_nested.jl`

**Interfaces:**
- Consumes: canonical `nested_x_operator` and X patch functions.
- Produces: reverse rules and example code free of references to deleted helpers or Y impurities.

- [ ] **Step 1: Update ChainRules tests first**

Rename operator-dressed directional tests to X and call `nested_x_operator`.
Remove direct rules for deleted `_graded_pair_sign` and wrapper helpers. Keep
zero-cotangent, primal validation-order, and non-vacuous directional checks.

- [ ] **Step 2: Run ChainRules RED**

Execute the focused extension harness through the memory guard. Expected:
missing canonical rule or old Y/prepared-patch references.

- [ ] **Step 3: Update rules and prepared contractions**

Use `closed_left_x`, `left_x`, `top_x`, and the new endpoint signs. Implement
the canonical `nested_x_operator` rule. Let the compatibility function forward
through the canonical implementation; add a forwarding rule only if the
focused RED proves dispatch cannot reach the canonical rule.

- [ ] **Step 4: Update extension imports and example comments/calls**

Import `nested_x_operator` and `_nested_x_operator_raw`. Replace old impurity
names in the example or its diagnostic text. Preserve its public energy and
optimization interfaces.

- [ ] **Step 5: Run ChainRules GREEN**

Expected: focused directional derivatives remain within `1e-4`, zero tangents
retain their contract, and no deleted helper is referenced.

---

### Task 5: Update the monograph and rebuild the PDF

**Files:**
- Modify: `D:/C 盘备份/Back up/Work_Save/latex file/2026/0801_Nested GTN/main.tex`
- Modify: `D:/C 盘备份/Back up/Work_Save/latex file/2026/0801_Nested GTN/main.pdf`
- Modify: `D:/C 盘备份/Back up/Work_Save/latex file/2026/0801_Nested GTN/main.log`

**Interfaces:**
- Consumes: verified final source and exact test convention.
- Produces: an English 25--35 page academic monograph consistent with the simplified implementation.

- [ ] **Step 1: Rewrite theory-to-source descriptions**

Swap the physical roles of X and Y throughout formulas, function tables,
glossary, call graph, sign audit, and source correspondence. Document
`nested_x_operator` canonically and `nested_y_operator` as a compatibility
alias.

- [ ] **Step 2: Redraw affected TikZ diagrams**

Update the doubled cell, X construction, pure Y crossing, one-site patch,
horizontal `X-B-X`, vertical `X-K-X`, and public call graph. Preserve explicit
arrows on every tensor leg.

- [ ] **Step 3: Compile from an ASCII staging directory**

Copy `main.tex` and `references.bib` to
`C:/Users/Harry/.codex/tmp/nested-gtn-latex-build`, then execute:

```powershell
python compile_latex.py main.tex --compiler texlive --engine xelatex --json
```

Copy the successful `main.pdf` and `main.log` back to the requested directory.

- [ ] **Step 4: Render and inspect every page**

Render all pages with Poppler at 110 DPI, build contact sheets, and visually
check arrows, captions, tables, references, clipping, overlaps, and page
numbering. Require zero unresolved references, zero replacement characters,
and zero overfull boxes.

---

### Task 6: Final server verification and change audit

**Files:**
- Modify: local log `D:/C 盘备份/Back up/Work_Save/coding/codex_server_logs/2026/0801/x-impurity-nested-<id>.md`

**Interfaces:**
- Consumes: all implementation, test, ChainRules, example, and document changes.
- Produces: fresh evidence for completion.

- [ ] **Step 1: Run the combined nested server suite**

Use the guarded remote command to include `test/nested_network.jl`,
`test/nested_measurements.jl`, and `test/nested_chainrules.jl`. Record pass and
failure counts, peak RSS, duration, exit code, and memory-kill status.

- [ ] **Step 2: Run the repository server suite**

Execute `julia_grassmann test/server_runtests.jl` with the 20 GB guard. If the
known offline TestExtras version mismatch recurs, use a remote-only harness
that changes only the unavailable TestExtras version and document that the
repository file was not changed.

- [ ] **Step 3: Audit the final diff**

Run:

```powershell
git diff --check
rg -n "_graded_pair_sign|_nested_.*_for_network|_physical_identity" algorithms/Nested_CTMRG test ext examples
rg -n "nested_y_operator" algorithms/Nested_CTMRG test ext examples src
```

Expected: no whitespace errors; no production dependency on deleted helpers;
`nested_y_operator` appears only in its compatibility definition/export and
explicit compatibility tests/documentation.

- [ ] **Step 4: Verify delivered PDF and source artifacts**

Require 25--35 PDF pages, zero `??`, zero U+FFFD, zero LaTeX warning lines, and
a SHA-256 match between the staged and delivered PDF.
