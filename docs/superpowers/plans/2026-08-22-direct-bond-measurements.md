# Direct Bond Measurements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add low-memory CTMRG bond expectation APIs that compute two-site bond observables without materializing rank-6 bond impurity tensors.

**Architecture:** Keep the existing explicit `Timp::Matrix{Grassmann{Q,6}}` measurement APIs unchanged. Add overloads that accept `Square_GPEPS` and a rank-4 bond operator, decompose the operator into alpha-channel single-site factors, build only rank-5 single-site alpha impurities per measured bond, and contract the alpha channel directly in the measurement path.

**Tech Stack:** Julia, existing Grassmann tensor primitives, existing CTMRG measurement code, existing nested CTMRG alpha-channel helpers as reference.

## Global Constraints

- Do not construct `T_vbond_imp_mat` or `T_hbond_imp_mat` in the new path.
- Keep existing explicit bond impurity APIs backward-compatible.
- Use server-side Julia verification only.
- Compare the new API against the existing explicit rank-6 path on small tensors.

---

### Task 1: Add Direct Bond Measurement Tests

**Files:**
- Create: `test/bond_measurements.jl`
- Modify: `test/runtests.jl`
- Modify: `test/server_runtests.jl`

**Interfaces:**
- Consumes: existing `reduced_tensor(peps, H_bond)`, `compute_exp_hbond(Tbulk, Timp, env)`, and `compute_exp_vbond(Tbulk, Timp, env)`.
- Produces: failing tests for `compute_exp_hbond(Tbulk, peps, H_bond, env)` and `compute_exp_vbond(Tbulk, peps, H_bond, env)`.

- [ ] **Step 1: Write the failing test**

Create a deterministic small PEPS, compute old explicit bond measurements, and compare them against the new direct overloads.

- [ ] **Step 2: Run the targeted test on the server and verify it fails**

Run `runj.sh test/direct_bond_measurement_red.jl` in an isolated server task folder.
Expected: failure with `MethodError` for the new overload.

### Task 2: Implement Alpha-Channel Measurement Path

**Files:**
- Modify: `auxiliary/reduced_tensors.jl`
- Modify: `algorithms/CTMRG/measurements.jl`
- Modify: `src/GrassmannTensorNetworks.jl`

**Interfaces:**
- Produces: `reduced_tensor_alpha(Tdn::Grassmann{Q,5}, operator::Grassmann{Q,3})`.
- Produces: `compute_exp_hbond(Tbulk::Matrix{Grassmann{Q1,4}}, peps::Square_GPEPS{Q2}, H_bond::Grassmann{Q3,4}, env::CTMRGEnv)`.
- Produces: `compute_exp_vbond(Tbulk::Matrix{Grassmann{Q1,4}}, peps::Square_GPEPS{Q2}, H_bond::Grassmann{Q3,4}, env::CTMRGEnv)`.

- [ ] **Step 1: Implement the single-site alpha reduced tensor**

Use the existing single-site impurity construction pattern and preserve the alpha leg as an extra trailing index.

- [ ] **Step 2: Implement alpha-preserving horizontal and vertical contraction helpers**

Use `_left_move_keep_open` and `_up_move_keep_open` style contractions to carry the alpha channel through the environment and trace the two alpha legs only after both sites have been inserted.

- [ ] **Step 3: Add direct measurement overloads**

Each overload loops over bonds, constructs only the two alpha impurities for the current bond, contracts them, stores the scalar result, and releases the local references.

### Task 3: Verify Equivalence and Memory-Oriented Behavior

**Files:**
- Test: `test/bond_measurements.jl`

**Interfaces:**
- Consumes: all APIs from Task 2.
- Produces: passing small-D equivalence tests.

- [ ] **Step 1: Run targeted equivalence tests on the server**

Expected: direct horizontal and vertical bond measurements match explicit rank-6 measurements within tight numerical tolerances.

- [ ] **Step 2: Run the server test entrypoint**

Expected: package tests that include bond measurement coverage pass, or failures are reported with exact output.

## Self-Review

- Spec coverage: the plan covers the direct measurement API, low-memory alpha path, compatibility, and server verification.
- Placeholder scan: no placeholder tasks remain.
- Type consistency: method names and argument types match the existing codebase conventions.
