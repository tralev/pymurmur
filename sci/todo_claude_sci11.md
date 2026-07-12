# TODO — Ideas & Math from `sci/new11_sci.md` (Predecessor Codebase Reference) Not Implemented in the Codebase

Comparison of `sci/new11_sci.md` — the feature reference for the *predecessor*
implementation at `/Users/tralev/Developer/murmuration/` — against `pymurmur/`.
Because this document describes the codebase pymurmur was rebuilt from, most gaps
below are **features lost in the port** rather than never-built ideas; several are
outright regressions where the predecessor's correct detail was dropped.

Already implemented (the port's core is faithful): tetrahedron mesh + GPU LookAt
with gimbal guard, Blinn-Phong + speed tint, 6-float instanced rendering,
reference grid, orbit camera with matching defaults/clamps and auto-rotate,
deferred add/remove/reset mutation, three boundary modes + speed-floor re-seeding,
uniform-sphere noise vectors, k=7 local spacing, H₂ via symmetrized kNN Laplacian
with `J(m) = H₂ + 0.06·m`, day-length model, Knuth per-day predator hash,
FBO headless capture with LANCZOS GIF assembly.

---

## 1. Rendering regressions

- [ ] **§1.4 The VAO rebuild after buffer growth was dropped.** The predecessor
  grows the instance buffer and *rebuilds the VAO*; pymurmur reallocates the
  buffer but leaves the VAO bound to the old one
  ([renderer.py:104-109](pymurmur/viz/renderer.py#L104-L109)) — draws read a
  stale buffer after the first growth (defect independently found in
  `todo_claude2.md` §6; this doc confirms the original had it right).
- [ ] **§10.1 The headless FBO lost its depth attachment.** Predecessor:
  color renderbuffer **+ `depth_renderbuffer`**. pymurmur:
  `ctx.framebuffer(color_attachments=[texture])` with **no depth attachment**
  ([renderer.py:91-94](pymurmur/viz/renderer.py#L91-L94)) — with DEPTH_TEST
  enabled but no depth buffer, headless captures render birds in draw order,
  not depth order: every exported GIF has wrong occlusion.
- [ ] **§1.6 The mat4 upload workaround was dropped.** Predecessor uses an
  explicit `np.array(m.to_list(), dtype=np.float32).tobytes()` conversion
  because PyGLM `.to_bytes()` layout differs across platforms (found unreliable
  on macOS Metal — this project's platform); pymurmur calls `.to_bytes()`
  directly ([renderer.py:124-127](pymurmur/viz/renderer.py#L124-L127)).
- [ ] **§2.3 Cinematic capture sweep**: scripted camera during headless capture —
  `azimuth = 45° + t·180°`, `elevation = 25° + sin(t·2π)·0.15`,
  `distance = 650 + sin(t·1.5π)·100`. The Recorder uses a static camera; the
  sweep formulas are the canonical spec for the gap noted in `todo_claude3.md`
  §16.

## 2. Capture pipeline

- [ ] **§10.3 Pre-warm phase**: run ~60 frames un-captured so the flock settles
  before recording — pymurmur's Recorder captures from frame 0, so every GIF
  opens on the random initial soup.
- [ ] **§10.4 Environment-variable overrides** (`CAPTURE_W/H`, `CAPTURE_FRAMES`,
  `CAPTURE_OUT`) — needed by the docker-compose capture service under llvmpipe
  (the `ci/` setup has no way to shrink capture resolution today; CLI flags
  cover only output path and frame count).
- [ ] **§10.2 GIF flags**: `optimize=True`, `disposal=2` — two save kwargs the
  port dropped (smaller files, correct frame clearing).

## 3. Input & UI

- [ ] **§4.1 The φ-weight constraint is unenforced.** Predecessor: raising φp
  auto-reduces φa to preserve `φp + φa + φn = 1`. pymurmur clamps each key
  independently to [0,1] ([input_control.py:88-95](pymurmur/viz/input_control.py#L88-L95));
  no constraint, and φn doesn't exist (the missing Pearce noise term,
  `todo_claude3.md` §2).
- [ ] **§5 The a–h,w preset table.** Eight presets with exact
  (φp, φa, σ, mode) tuples and per-preset descriptions printed on activation.
  `analysis/presets.py` has seven presets with *different* names/values, no key
  bindings, and no consumer — reconcile its contents to this canonical table
  and wire the keys (gap chain: `todo_claude2.md` §4).
- [ ] **§3.5 Full title-bar metrics readout**: mode, N, φp/φa/σ,
  `α Θ Θ' L σr`, τρ, and FPS. pymurmur's title shows N/φp/φa/α/Θ only —
  no mode, σ, Θ', L, σr, τρ, or FPS (a `metrics.summary()` helper existed for
  this).

## 4. Physics & spatial index details lost

- [ ] **§6.2–6.3 Toroidal cell wrapping in the hash grid.** The predecessor's
  grid wraps cell indices with modulo on **both** rebuild and query — pymurmur's
  `SpatialHashGrid` uses unwrapped integer keys, so edge cells have truncated
  neighborhoods (one concrete half of the codebase-wide toroidal-distance gap;
  cKDTree `boxsize=` covers the other half — `todo_claude_sci5.md` §1).
- [ ] **§6.3 Radius-driven cell range**: the predecessor's query spans
  `⌊(pos±radius)/cell⌋` — arbitrary radii; pymurmur's `query_radius` hardcodes
  the ±1-cell (27-cell) neighborhood and silently ignores its `radius` argument
  beyond that.
- [ ] **§7.3 Velocity init with speed dispersion**: uniform direction ×
  `speed ~ U(1, V0)` — pymurmur fixes all speeds at `0.8·v0` (third source for
  this init-variant gap: `todo_claude_sci2.md` §12, `todo_claude_sci5.md` §10).
- [ ] **§16 `MAX_VISIBILITY_RANGE` occlusion candidate cutoff** (200 units):
  projection mode's neighbor query takes the σ nearest *at any distance* —
  the predecessor filtered occlusion candidates by a max range first.
- [ ] **§8 Fibonacci-sphere utility** (golden-angle spiral, 256 points) — absent;
  small, but it's also the primitive the crs48 stratified offsets need
  (`todo_claude_sci1.md` §2.2).

## 5. Metrics: formula deviations and missing estimators

- [ ] **§9.1 Gyration radius: wrong centroid, wrong trim.** Predecessor: distances
  from the **median** centroid (outlier-immune — the stated point of the
  estimator), trimming only the **far** 15% tail (`keep=0.85`). pymurmur
  ([metrics.py:340-358](pymurmur/analysis/metrics.py#L340-L358)) uses the
  **mean** centroid and trims **both** tails (15% inner + outer) — a different,
  less robust statistic.
- [ ] **§13.2 Thickness ratio formula mismatch.** Doc (and Young):
  `thickness = √(λ₃/λ₁) ∈ (0,1]`. pymurmur computes `√(λ₂/λ₃) ≥ 1`
  ([metrics.py:335-337](pymurmur/analysis/metrics.py#L335-L337)) — a different
  descriptor exported under the documented name.
- [ ] **§13.3 Shape-driven m\*** — the interpolation
  `m* = 9.78 + clamp((aspect−1)/2)·(6.05 − 9.78)` between transverse/
  longitudinal endpoints; pymurmur computes shape and m\* independently and
  never links them (canonical formulas for the gap in `todo_claude3.md` §4).
- [ ] **§14.3 Marginal efficiency** `η(m) = H₂(m−1) − H₂(m)` (+inf at the
  connectivity threshold) — not computed.
- [ ] **§14.1 Symmetrization variant**: predecessor `max(A, Aᵀ)` (binary edge);
  pymurmur `(A+Aᵀ)/2` (half-weight one-way edges) — different Laplacian spectra;
  pick and document one (Young et al. assume the undirected/max form).
- [ ] **§15 Correlation time τρ — the reference method is absent.** Predecessor:
  scalar density series from **ConvexHull volume** (`ρ = N/hull.volume`),
  sampled every 10 frames into a 500-slot ring buffer, then **trapezoidal
  integration** of the autocorrelation (half-weight lag 0, stop at first zero
  crossing or 0.25·buffer). pymurmur substitutes voxel-histogram Pearson
  correlations with an exponential fit — a different estimator, and
  `ConvexHull` (a declared scipy dependency for exactly this) is used nowhere.
- [ ] **§9.3 Number density** `ρ = kept/((4/3)π·R_g³)` — still missing
  (`todo_claude3.md` §4); depends on the §9.1 fix.

## 6. Ecology details (canonical formulas for known gaps)

- [ ] **§12.3 Logistic dusk ramp**: `1/(1+e^{−z})`, `z = (hour − sunset)/(width/4)`,
  overflow-clamped — pymurmur uses a linear last-hour ramp.
- [ ] **§12.5 Cold-weather roost boost**:
  `1 + 0.2·(T_mean − T(day))/T_amp` — `temperature()` exists and is never used.
- [ ] **§12.1 Critical-mass window**: smoothstep over `[0.4·N_crit, 1.2·N_crit]`
  (200→600 birds, inflection at 500); pymurmur ramps over [0, 500].
- [ ] **§12.2 `rng` parameter** for true-random predator presence alongside the
  deterministic hash — the hash is ported, the stochastic option is not.

## 7. Superseded (record only, no action)

- **§11 `features.py` import-time feature toggles** — superseded by pymurmur's
  config-driven mode dispatch; do not port.
- **§3.3 `ext` dict for behavioural state** — superseded by `ExtensionManager`;
  the one piece worth keeping from it is the explicit `hour`/`day`/`roost`
  state surface, which Ecology currently hides as private attributes.
