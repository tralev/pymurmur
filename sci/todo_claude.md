# todo_claude.md — math, ideas & tests to port from the reference implementation

**What this is.** A gap analysis produced by comparing this repository
(`pymurmur/`) against a second, independently-developed 3D murmuration codebase
(the *reference implementation*, at `/Users/tralev/Developer/murmuration`). It
lists the **scientific math and ideas that exist in the reference but are absent
or weaker here**, with concrete instructions for porting each, followed by the
**tests the reference has that this repo lacks**.

**Scope.** This only flags reference → `pymurmur` gaps, as requested. For
balance, a short "where `pymurmur` is already ahead" note is at the end so the
suggestions aren't read as a one-way verdict — several subsystems here (proper
modulo toroidal wrap, the `sphere` boundary, the panic/blackening predator, MSD,
R² goodness-of-fit) are genuinely more complete than the reference.

**Structure.** Part 1 = scientific math to port. Part 2 = tests to port.
Part 3 = engineering features the reference has and this repo lost or lacks
(renderer, capture, presets, input, grid — each verified against the reference
source with file:line citations). Part 4 = the **implementation roadmap**:
phased, dependency-ordered instructions with code sketches and acceptance
criteria, detailed enough to implement directly.

**Conventions.** File paths are repo-relative. Tests are written in the
`pytest` function style this repo already uses (`test/**/test_*.py`). Severity:
**[HIGH]** = a correctness/fidelity gap, **[MED]** = a missing feature or a
meaningfully different (less faithful) formula, **[LOW]** = polish.

---

## Part 1 — Math & ideas to add

### 1. Occlusion: implement true visibility culling  **[HIGH]**

`pymurmur/physics/occlusion.py::spherical_cap_occlusion` processes neighbours
closest-first but **never occludes** — the loop appends *every* non-blind
neighbour to `visible` and adds its cap to `theta`. The docstring claims
"occluded birds excluded", but no code excludes them. A bird directly behind a
nearer bird on the same view ray is still counted. (This is even baked into the
test suite — see Part 2, T1.)

This matters: occlusion *is* the Pearce model. Without it, δ̂ and Θ are computed
over the wrong neighbour set, and the density-regulation mechanism (interior
birds see a dark, cancelled view) is muted.

**Fix.** Process closest-first and drop a neighbour whose direction lies inside a
*nearer, already-visible* cap. In the reference this is a single batched dot
product per candidate:

```python
vis_dirs = np.empty((M, 3), dtype=np.float64)   # unit dirs of accepted birds
vis_cos  = np.empty(M, dtype=np.float64)         # cos(alpha) of accepted caps
n_vis = 0
for j in order:                                  # order = argsort(dists)
    d = dists[j]
    if d < 1e-6:            # self
        continue
    dir_j = diffs[j] / d
    if blind_cos is not None and float(dir_j @ (-obs_forward)) >= blind_cos:
        continue
    # ── OCCLUSION: hidden if inside any nearer accepted cap ──
    if n_vis and np.any(vis_dirs[:n_vis] @ dir_j >= vis_cos[:n_vis]):
        continue
    alpha = math.asin(min(b_eff / d, 1.0))       # exact cap half-angle
    vis_dirs[n_vis] = dir_j
    vis_cos[n_vis]  = math.cos(alpha)
    n_vis += 1
    # ... accumulate theta (item 2) and delta (item 3) here ...
```

### 2. Occlusion: probabilistic-union opacity Θ  **[HIGH]**

Here Θ is a **linear sum** of angular radii, clamped: `theta += cap_radius;
min(theta, 1.0)`. The reference uses Pearce's probabilistic union of the visible
caps' sky-fractions, which is what actually lives in [0, 1] and composes
correctly when caps overlap:

```
Ω_j = 2π (1 − cos α_j)          # solid angle of cap j
Θ   = 1 − Π_visible (1 − Ω_j / 4π)
```

Implement as a running product `one_minus *= (1 - omega_j / (4*math.pi))`, then
`theta = 1 - one_minus`. Two moderate caps then combine as
`1 − (1−Ω₁/4π)(1−Ω₂/4π)`, not their raw sum. Also switch the cap radius from the
small-angle `b_eff / d` to the exact `α = asin(min(b_eff/d, 1))` (item 1).

### 3. Occlusion: δ̂ boundary-length normalization  **[MED]**

The δ̂ here is `Σ dir_j · cap_radius_j` clamped to unit magnitude. The reference
weights each visible direction by `sin α_j` (the *boundary length* the cap
contributes) and normalizes by the **total** boundary length, not by its own
magnitude:

```
δ̂ = ( Σ_visible sin α_j · d̂_j ) / ( Σ_visible sin α_j )
```

This makes `|δ̂|` the density-regulation signal the model is built on: `→ 1` for
an edge bird (all boundaries resolve one way), `→ 0` for a surrounded bird
(boundaries cancel) — *without* an artificial clamp. Keep the vector unnormalized
otherwise; callers must read its magnitude.

### 4. Occlusion: bounded neighbour cap  **[LOW]**

The reference caps the O(V²) visibility test at the nearest
`MAX_OCCLUSION_NEIGHBOURS = 64` candidates (well above any realistic in-range
count, so ordinary flocks are unaffected). This repo currently bounds cost by
only passing σ neighbours through the SoA adapter, which is fine — but if
`spherical_cap_occlusion` is ever called with a large neighbour list, add
`order = order[:64]` after the argsort as a guard.

### 5. Ecology: seasonal flock-size curve + season window  **[MED]**

`pymurmur/physics/extensions/ecology.py` models day length, temperature and a
dusk pull, but has **no seasonal population model**. The reference has:

```python
PEAK_DAY, MIN_FACTOR = 15, 0.25          # peak ~mid-Jan, summer trough = 25% of peak
def seasonal_size_factor(day) -> float:  # cosine, 1.0 at PEAK_DAY, MIN_FACTOR at +182
def flock_size_for_day(day, peak_size, min_size=0) -> int
def is_murmuration_season(day) -> bool   # True in the Oct–Mar observation window
```

Add these as free functions (or `Ecology` methods). They let the sim scale flock
size and gate behaviour by time of year (Goodenough 2017).

### 6. Ecology: coherence gate as a weight multiplier  **[MED]**

Here the critical-mass idea appears only as a smoothstep on the *roost pull*
(`apply()`, N/500). The reference generalizes it into a reusable **coherence
factor** applied to the flocking weights themselves:

```python
CRITICAL_MASS = 500;  _LO, _HI = 0.4, 1.2      # ramp over [0.4·N, 1.2·N]
def coherence_factor(n, critical_mass=CRITICAL_MASS) -> float  # smoothstep 0→1
def has_critical_mass(n, critical_mass=CRITICAL_MASS) -> bool  # coherence ≥ 0.5
def gated_weight(weight, n, critical_mass=CRITICAL_MASS) -> float  # weight·coherence
```

Use `gated_weight` to scale φ_a / φ_p so a small flock is incoherent and a large
one "switches on" — the phenomenon the critical mass is supposed to model.

### 7. Ecology: logistic dusk + sunset + roosting predicates  **[MED]**

The dusk pull here is a **linear** ramp active only in the last hour before
sunset. The reference uses a smooth **logistic** ramp of configurable width with
overflow-safe saturation, plus helper predicates:

```python
def sunset_hour(day) -> float                 # 12 + day_length/2
def dusk_factor(hour, day=15) -> float         # logistic, 0 well before, 0.5 at, 1 after sunset
def is_roosting_time(hour, day=15, threshold=0.5) -> bool
```

`dusk_factor` clamps to exactly 0/1 far from sunset (guard `z<-60 / z>60` before
`exp`). This gives a continuous roost drive instead of a hard 1-hour window.

### 8. Ecology: temperature-coupled roost strength  **[MED]**

`temperature(day)` is computed but never used. The reference couples it to the
roost pull — colder days give a slightly stronger/longer descent (Goodenough:
duration correlates negatively with temperature):

```python
def roost_strength(hour, day=15, base=1.0) -> float:
    temp_boost = 1.0 + 0.2 * (TEMP_MEAN - temperature(day)) / TEMP_AMP
    return base * dusk_factor(hour, day) * max(0.0, temp_boost)

def roost_force(bird_pos, hour, roost, day=15, strength=1.0):  # unit(roost-pos)·roost_strength
```

Then the `apply()` pull becomes `roost_force(...)` — same shape, but temperature
now modulates it.

### 9. Flock shape → suggested m\*  **[MED]**

`pymurmur/analysis/metrics.py::compute_shape` returns `(aspect, thickness)` but
stops there. The reference maps aspect → Young's optimal neighbour count:

```python
M_STAR_TRANSVERSE, M_STAR_LONGITUDINAL = 9.78, 6.05   # round vs thin
_ASPECT_ROUND, _ASPECT_THIN = 1.0, 3.0
def suggested_m_star(aspect_ratio) -> float:           # clamped linear interp
    t = max(0.0, min(1.0, (aspect_ratio - 1.0) / (3.0 - 1.0)))
    return 9.78 + t * (6.05 - 9.78)
```

This is a *shape-driven* m\* prediction, complementary to the H₂ *cost-driven*
`find_optimal_m`. Return it alongside aspect/thickness (add a `suggested_m` field
to `FlockMetrics`).

### 10. H₂: marginal per-neighbour efficiency η(m)  **[MED]**

`find_optimal_m` minimizes `H₂ + 0.06·m` but exposes no **marginal** curve. The
reference adds η(m), the robustness gained from the m-th neighbour, with correct
handling of the connectivity transition (the neighbour that first connects a
disconnected graph is worth +∞):

```python
def eta_of_m(positions, m, m0=None) -> float:
    m0 = m - 1 if m0 is None else m0
    h0, h1 = h2_norm(positions, m0), h2_norm(positions, m)
    if math.isinf(h0) and math.isfinite(h1):   # m first connects the graph
        return math.inf
    if not (math.isfinite(h0) and math.isfinite(h1)):
        return 0.0                              # still disconnected
    return (h0 - h1) / (m - m0)
```

This requires `compute_h2` to return `inf` (not `0.0`) when the graph is
disconnected — right now it returns `0.0, 0.0` in that case, which conflates
"perfectly robust" with "disconnected". Fixing that is a prerequisite and a
fidelity improvement in its own right.

### 11. Correlation time: hull-volume density + integrated autocorrelation  **[MED]**

`compute_tau_rho` correlates coarse density **histograms** and fits an
exponential decay. The reference measures density as `ρ = N / convex_hull_volume`
(scipy `ConvexHull`, a coordinate-free physical density) and integrates the
normalized autocorrelation up to the **first zero crossing** (the standard
integrated-autocorrelation-time estimator) rather than fitting an exponential:

```python
def convex_hull_volume(flock) -> float          # scipy.spatial.ConvexHull(...).volume; 0 if degenerate
# τ = SAMPLE_INTERVAL · (0.5 + Σ_{lag≥1} r(lag)),  stop at first r(lag) ≤ 0
```

Consider adding this as an alternative estimator; it avoids the grid-resolution
dependence of the histogram method and returns a scale in real frame units.

### 12. External opacity Θ′: silhouette projection  **[MED]**

`compute_theta_prime` measures **3D voxel occupancy** (fraction of bounding-box
voxels containing a bird) — a volume-filling fraction. The reference computes the
Pearce **distant-observer silhouette**: project all birds onto the plane ⊥ a
chosen observer axis, rasterize each as a disk of radius `BOID_SIZE`, and return
covered-cells / total-cells (the union, so overlaps don't double-count). These
are *different quantities* — "how full is the volume" vs "how much sky does a
distant observer see blocked". The silhouette version is the one Pearce's Θ′
refers to; add it (e.g. `external_opacity(positions, observer_axis=0)`) rather
than replacing the voxel metric.

### 13. Density scaling: ideal-exponent comparison + gyration-sphere density  **[MED]**

`pymurmur/analysis/density_scaling.py` fits `spacing ~ N^β` for toroidal vs open
and reports β with R². The reference adds the scientific interpretation:
marginal opacity is N-independent only if `density ~ N^(−1/2)`, so it fits the
exponent against `IDEAL_DENSITY_EXPONENT = -0.5` (and size `R_g ~ N^{...}`), and
measures **number density in the gyration sphere** (`N_kept / (4/3 π R_g³)`) with
a straggler-robust R_g (median centre + top-quantile trim), not just the
7th-neighbour spacing. Port the ideal-target comparison and the
`number_density` / trimmed-`gyration_radius` observables so the sweep answers
"how far from marginal opacity is this model" rather than just reporting a slope.
(Keep this repo's nice toroidal-vs-open + R² framing — combine the two.)

### 14. Steric: clamp to MAX_FORCE  **[HIGH]** (robustness)

`pymurmur/physics/steric.py::steric_force` sums `strength · Σ r̂/d²` with no
upper bound. At small separations `1/d²` explodes (only a `1e-10` epsilon guards
the division), so a close pair can inject an arbitrarily large acceleration and
blow up the integrator. The reference clamps the result:

```python
force = np.sum(dirs / close_dists[:, None]**2, axis=0) * strength
mag = np.linalg.norm(force)
if mag > MAX_FORCE:
    force = force / mag * MAX_FORCE
```

Add a `max_force` parameter (default your global cap) and clamp. Cheap, prevents
a real instability.

### 15. Metrics: normalize angular momentum to O(1)  **[LOW]**

`m.angular_momentum` is the raw `⟨r × v⟩`, whose scale depends on the domain
size and speed. The reference normalizes it into a **rotational order parameter**
by dividing by `N · V0 · characteristic_radius`, so it sits ~O(1) and is
comparable across flock sizes/domains. Consider storing the normalized scalar
magnitude alongside (or instead of) the raw vector.

---

## Part 2 — Tests to add

This repo uses `pytest` function-style tests under `test/`. Each item below
names the target file and gives a sketch. Items T7–T11 depend on first
implementing the corresponding Part-1 math.

### T1. Occlusion — true culling (and fix the test that enshrines the bug)  **[HIGH]**

`test/physics/test_occlusion.py::test_occlusion_closest_first` currently asserts
`list(visible) == [2, 1, 0]` for three birds at (10,0,0),(30,0,0),(80,0,0) — all
on the **same +x ray**. Under real occlusion the two farther birds sit inside the
nearest bird's cap and are hidden, so the correct assertion is
`list(visible) == [2]` (closest only). After implementing Part-1 item 1, fix that
test and add:

```python
def test_occlusion_culls_birds_behind_a_nearer_one():
    obs_pos = np.array([0,0,0], np.float32); obs_vel = np.array([1,0,0], np.float32)
    nbr_pos = np.array([[30,0,0],[60,0,0],[90,0,0]], np.float32)   # collinear
    nbr_vel = np.ones((3,3), np.float32)
    _, visible, _ = spherical_cap_occlusion(obs_pos, obs_vel, nbr_pos, nbr_vel, boid_size=9.0)
    assert list(visible) == [0]          # only the nearest is visible

def test_occlusion_separated_caps_all_visible():
    # widely-separated directions: none occludes another
    obs_pos = np.zeros(3, np.float32); obs_vel = np.array([1,0,0], np.float32)
    nbr_pos = np.array([[60,0,0],[0,60,0],[0,0,60]], np.float32)
    nbr_vel = np.ones((3,3), np.float32)
    _, visible, _ = spherical_cap_occlusion(obs_pos, obs_vel, nbr_pos, nbr_vel)
    assert len(visible) == 3
```

### T2. Occlusion — probabilistic-union Θ  **[HIGH]**

```python
def test_theta_is_probabilistic_union_not_linear_sum():
    # two well-separated equal caps: Θ = 1-(1-Ω/4π)² < 2·(Ω/4π)
    obs_pos = np.zeros(3, np.float32); obs_vel = np.array([1,0,0], np.float32)
    nbr_pos = np.array([[60,0,0],[0,60,0]], np.float32); nbr_vel = np.ones((2,3), np.float32)
    _, _, th2 = spherical_cap_occlusion(obs_pos, obs_vel, nbr_pos, nbr_vel, boid_size=9.0)
    _, _, th1 = spherical_cap_occlusion(obs_pos, obs_vel, nbr_pos[:1], nbr_vel[:1], boid_size=9.0)
    assert th1 < th2 < 2*th1          # union grows sub-linearly, never > sum
    assert 0.0 <= th2 <= 1.0
```

### T3. Occlusion — δ̂ edge-vs-surrounded (exact)  **[HIGH]**

The reference pins the density-regulation signal. Port it:

```python
def test_delta_magnitude_edge_vs_surrounded():
    obs_pos = np.zeros(3, np.float32); obs_vel = np.array([1,0,0], np.float32)
    surround = np.array([[60,0,0],[-60,0,0],[0,60,0],[0,-60,0],[0,0,60],[0,0,-60]], np.float32)
    d_in, _, _ = spherical_cap_occlusion(obs_pos, obs_vel, surround, np.ones((6,3),np.float32))
    assert np.linalg.norm(d_in) < 1e-2               # octahedral surround cancels
    one, _, _ = spherical_cap_occlusion(obs_pos, obs_vel, surround[:1], np.ones((1,3),np.float32))
    assert abs(np.linalg.norm(one) - 1.0) < 1e-6     # single neighbour → |δ̂| = 1
```

### T4. Occlusion — anisotropy = 1 ≡ isotropic  **[MED]**

```python
def test_anisotropy_one_equals_isotropic():
    rng = np.random.default_rng(3); obs_pos = np.zeros(3, np.float32); obs_vel = np.array([1,0,0],np.float32)
    p = rng.normal(0,40,(12,3)).astype(np.float32); v = rng.normal(0,1,(12,3)).astype(np.float32)
    d1,vi1,t1 = spherical_cap_occlusion(obs_pos,obs_vel,p,v,anisotropy=1.0)
    d0,vi0,t0 = spherical_cap_occlusion(obs_pos,obs_vel,p,v)            # default
    assert np.allclose(d0,d1) and t0 == pytest.approx(t1) and list(vi0)==list(vi1)
```

### T5. Occlusion — invariant fuzzing  **[MED]**

Loop ~100 random configs asserting the contract holds for *all* of them:
`|δ̂| ≤ 1`, `Θ ∈ [0,1]`, `visible` closest-first and duplicate-free, no visible
bird inside the blind cone. Seed a `random.Random`/`default_rng` so it's
deterministic. (The reference's `TestOcclusionInvariants` is the template.)

### T6. Steric — MAX_FORCE clamp  **[HIGH]**

Depends on Part-1 item 14:

```python
def test_steric_clamped_to_max_force():
    f = steric_force(np.zeros(3,np.float32), np.array([[0.01,0,0]],np.float32),
                     strength=0.6, max_force=0.15)
    assert np.linalg.norm(f) == pytest.approx(0.15, abs=1e-6)   # 1/d² huge → clamped
```

### T7. Ecology — the new seasonal / coherence / dusk / roost math  **[MED]**

After Part-1 items 5–8, add to `test/physics/extensions/test_extensions.py`:

```python
def test_seasonal_peak_and_trough():
    assert seasonal_size_factor(PEAK_DAY) == pytest.approx(1.0)
    assert seasonal_size_factor(PEAK_DAY+182) == pytest.approx(MIN_FACTOR, abs=1e-2)
def test_is_murmuration_season():           # Jan in, Jul out
    assert is_murmuration_season(15) and not is_murmuration_season(196)
def test_coherence_gate():                  # gated_weight rises from ~0 to ~full with N
    assert gated_weight(0.8, 10) == pytest.approx(0.0, abs=1e-3)
    assert gated_weight(0.8, 600) > 0.7
def test_dusk_factor_saturates():           # logistic clamps far from sunset
    assert dusk_factor(0.0, 15) == 0.0 and dusk_factor(40.0, 15) == 1.0
def test_roost_strength_colder_is_stronger():
    assert roost_strength(22.0, day=20) > roost_strength(22.0, day=200)   # Jan > Jul, same hour
```

### T8. Flock shape — suggested m\*  **[MED]**

After Part-1 item 9, in `test/analysis/test_metrics.py`:

```python
def test_suggested_m_thin_vs_round():
    thin  = np.array([[x,0,0] for x in range(0,600,15)], np.float32)  # aspect ≫ 3
    round_ = np.random.default_rng(3).uniform(-50,50,(80,3)).astype(np.float32)
    assert suggested_m_star(compute_shape(thin)[0])  <= 7.0
    assert suggested_m_star(compute_shape(round_)[0]) >= 8.0
def test_suggested_m_monotone():
    assert suggested_m_star(1.0) > suggested_m_star(3.0)
```

### T9. H₂ — marginal efficiency & connectivity ∞  **[MED]**

After Part-1 item 10, in `test/analysis/test_h2.py`:

```python
def test_h2_infinite_when_disconnected():
    pts = np.array([[0,0,0],[1,0,0],[1000,0,0],[1001,0,0]], np.float32)  # two far pairs
    _, h2 = compute_h2(pts, m=1)
    assert math.isinf(h2)                    # requires the inf fix (item 10)
def test_eta_connectivity_transition():
    pts = np.array([[0,0,0],[10,0,0],[0,10,0],[1000,0,0],[1010,0,0],[1000,10,0]], np.float32)
    assert eta_of_m(pts, m=5, m0=1) == math.inf   # a neighbour first connects the graph
    assert eta_of_m(pts, m=2, m0=1) == 0.0        # still disconnected on both sides
```

### T10. Correlation time — hull volume + degenerate cases  **[MED]**

After Part-1 item 11, in `test/analysis/test_metrics.py`:

```python
def test_convex_hull_volume_cube():
    cube = np.array([[x,y,z] for x in (0,10) for y in (0,10) for z in (0,10)], np.float32)
    assert convex_hull_volume(cube) == pytest.approx(1000.0, abs=1e-3)
def test_convex_hull_volume_degenerate_zero():
    coplanar = np.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0]], np.float32)  # < 4 non-coplanar
    assert convex_hull_volume(coplanar) == 0.0
def test_tau_constant_density_is_zero():          # no variance → τ = 0, not 0/0
    ...  # feed identical density snapshots, assert τ == 0
```

### T11. Density scaling — ideal exponent + robust gyration  **[MED]**

After Part-1 items 13, in `test/analysis/test_density_scaling.py`:

```python
def test_gyration_trims_stragglers():
    pts = np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[10000,0,0]], np.float32)  # 1 outlier
    assert gyration_radius(pts, keep=0.8) < 5.0        # median centre + trim ignores it
def test_number_density_degenerate_flock_zero():
    assert number_density(np.tile([5,5,5],(4,1)).astype(np.float32)) == 0.0
def test_scaling_reports_ideal_target():
    r = measure_scaling(n_values=(40,80,150), frames=60, seeds=(0,))
    assert r["ideal_density_exponent"] == pytest.approx(-0.5)
    assert np.isfinite(r["density_exponent"])
```

### T12. Metamorphic invariance of the observables  **[MED]**

Symmetries the metrics must respect — fuzz over random flocks
(`test/analysis/test_metrics.py`):

```python
def test_order_parameter_rotation_invariant():
    rng = np.random.default_rng(0)
    for _ in range(30):
        v = rng.normal(0,4,(40,3)); a0 = order_parameter_of(v)
        q,_ = np.linalg.qr(rng.normal(size=(3,3)))          # random rotation
        assert order_parameter_of(v @ q.T) == pytest.approx(a0, abs=1e-6)
def test_dispersion_translation_invariant():
    rng = np.random.default_rng(1); p = rng.normal(0,80,(35,3))
    d0 = dispersion_of(p)
    assert dispersion_of(p + rng.normal(0,500,3)) == pytest.approx(d0, abs=1e-4)
```

### T13. Physics invariants (fuzz)  **[MED]**

In `test/physics/test_boid.py` — assert the integrator's guarantees over many
random states, not just one:

```python
def test_speed_ceiling_never_exceeds_v0():
    rng = np.random.default_rng(0)
    for _ in range(200):
        vel = rng.uniform(-2*V0, 2*V0, (50,3)).astype(np.float32); ...
        integrate(pos, vel, acc, active, W,H,D, V0, "toroidal", dt=1.0)
        assert np.linalg.norm(vel, axis=1).max() <= V0 + 1e-4
def test_toroidal_wrap_keeps_positions_in_bounds():
    ...  # after integrate(..., "toroidal"), assert 0 <= pos < [W,H,D] on every axis
```

### T14. Golden-trajectory regression  **[MED]**

This repo has within-run determinism tests but **no committed golden snapshot**,
so an *unintended physics change between versions* is undetected. Add one: run a
fixed seeded config (e.g. 15 birds × 30 frames, both modes) once, save
`test/data/golden_trajectory.npz`, and assert future runs match it within a small
tolerance (the reference uses `atol=1e-3` — libm ulp differences amplify through
the dynamics, so don't demand bit-exactness across platforms):

```python
def test_matches_committed_golden(tmp_path):
    pos, vel = run_reference_sim(seed=77, frames=30)
    g = np.load("test/data/golden_trajectory.npz")
    np.testing.assert_allclose(pos, g["pos"], rtol=0, atol=1e-3)
```

Ship a `regenerate_golden()` helper and document that a deliberate physics change
means re-pinning it in the same commit.

### T15. Discovery / collection-count guard  **[LOW]**

The reference pins each module's exact test count so a renamed or mis-indented
test can't silently vanish from collection. The `pytest` analogue: a small test
that asserts the collected count for a subpackage, or a CI step
`pytest --collect-only -q test/physics/ | wc -l` compared to a pinned number.
Lower value under pytest than unittest (pytest's collection is less fragile), but
cheap insurance for the science suites.

### T16. Doc-drift guard  **[LOW]**

The reference has a test that every module named in the README exists and every
`sci.md#anchor` markdown link resolves to a real heading (GitHub's slug
algorithm). Given this repo's many `.md` files (`arch.md`, `roadmap.md`,
`functional_*.md`, `sci/`), an analogous `test/test_docs.py` that validates
intra-repo doc links and referenced module paths would catch doc rot cheaply.

---

## Part 3 — Engineering features present in the reference but missing here

Verified against the reference source (paths are reference-repo files unless
prefixed `pymurmur/`).

### E1. Headless FBO has no depth attachment  **[HIGH]**

Reference (`renderer_3d.py:70-77`) attaches a **depth renderbuffer** to the
capture FBO; `pymurmur/viz/renderer.py:91-94` creates
`ctx.framebuffer(color_attachments=[ctx.texture((w,h),3)])` with **no depth
attachment**. DEPTH_TEST is enabled but has no buffer to test against, so every
headless capture renders birds in draw order — wrong occlusion in every GIF.

```python
self._fbo = self.ctx.framebuffer(
    color_attachments=[self.ctx.renderbuffer((width, height), components=3)],
    depth_attachment=self.ctx.depth_renderbuffer((width, height)),
)
```

### E2. Instance-buffer growth orphans the VAO  **[HIGH]**

Reference `_grow_instance_buffer` (`renderer_3d.py:144-159`) reallocates the
VBO **and rebuilds the VAO**. `pymurmur/viz/renderer.py:104-109` reallocates
the VBO only — `self._vao`, built once in `__init__`, keeps rendering from the
old (released) buffer after the first growth. Extract a `_build_bird_vao()`
helper, call it from `__init__` and after every growth.

### E3. PyGLM matrix upload workaround dropped  **[MED]**

Reference `_mat4_bytes` (`renderer_3d.py:32-41`) uploads
`np.array(m.to_list(), np.float32).tobytes()` because PyGLM builds differ in
memory layout — raw `bytes(mat4)` is row-major on some builds and silently
transposes every matrix ("throws all geometry off-screen", found on macOS
Metal, this project's platform). `pymurmur/viz/renderer.py:124-127` calls
`.to_bytes()` directly. Port `_mat4_bytes` and use it for `u_view`/`u_projection`.

### E4. Scenario presets: wrong values, no key bindings  **[MED]**

Reference `scenario_presets_3d.py` defines the canonical 8-preset table wired
to keys a–h,w in `input_handler_3d.py:124-129`, each printing its description:

| key | φp | φa | σ | mode | label |
|-----|----|----|---|------|-------|
| a | 0.04 | 0.80 | 6 | projection | 3D Pearce Default |
| b | 0.18 | 0.70 | 7 | projection | Ball of Birds |
| c | 0.06 | 0.45 | 3 | projection | Storm Cloud |
| d | 0.25 | 0.55 | 8 | spatial | 3D Stream |
| e | 0.10 | 0.75 | 6 | projection | Vertical Column |
| f | 0.02 | 0.85 | 3 | projection | 3D Acro |
| w | 0.08 | 0.82 | 10 | spatial | Spiral Vortex |
| h | 0.35 | 0.58 | 9 | spatial | 3D Void |

`pymurmur/analysis/presets.py` has seven presets with *different* names and
values, **no key bindings, and no importer anywhere**. Replace its contents
with the table above (fields: `label, phi_p, phi_a, sigma, mode, description`;
mode as the string `"projection"`/`"spatial"`), add
`apply_preset(config, key) -> str`, and bind keys in
`pymurmur/viz/input_control.py` (`input_control → presets` is already an
allowed dependency per the architecture docs). Skip `g` inside the a–h range —
it's the grid toggle (reference does the same).

### E5. φp + φa ≤ 1 constraint unenforced  **[MED]**

Reference (`input_handler_3d.py:84-95`): raising φa auto-reduces φp (and vice
versa on the decrease path) to preserve `φp + φa + φn = 1`.
`pymurmur/viz/input_control.py:88-95` clamps each independently to [0,1] — the
pair can sum to 2.0. Add after each φ mutation:
`if cfg.phi_p + cfg.phi_a > 1.0: <other> = 1.0 - <changed>`.

### E6. Window-title metrics readout incomplete  **[LOW]**

Reference title (`main_3d.py:125-129`):
`mode | N birds | φp φa σ | α Θ Θ' L σr | τρ | FPS`, via
`metrics.summary()` (`metrics_3d.py:224-228`). pymurmur's title
(`viz/visualizer.py:104-110`) shows only `N φp φa α Θ`. Add a
`FlockMetrics.summary()` returning
`f"α={alpha:.2f} Θ={theta:.2f} Θ'={theta_prime:.2f} L={|L|:.3f} σr={dispersion:.0f}"`
and extend the title with mode, σ, the summary, and `clock.get_fps()`
(pass the clock's fps into `Visualizer.run`'s title builder).

### E7. Cinematic capture sweep  **[MED]**

Reference `_auto_orbit` (`capture_3d.py:51-57`), applied every frame of a
headless capture:

```python
t = frame / total_frames
camera.azimuth   = math.radians(45) + t * math.radians(180)   # half-orbit
camera.elevation = math.radians(25) + math.sin(t * 2*math.pi) * 0.15
camera.distance  = 650 + math.sin(t * 1.5*math.pi) * 100      # breathing zoom
```

pymurmur's Recorder uses a static camera. Add the sweep to
`capture/recorder.py` (apply to `self._camera` before each captured frame; the
Recorder already counts frames — pass `capture_frames` as the total). Gate
behind a `capture_sweep: bool = True` config field.

### E8. Capture pre-warm phase  **[MED]**

Reference runs **60 un-captured frames** first so the flock settles
(`capture_3d.py:75-81`); pymurmur GIFs open on the random initial soup. In
`__main__.py`'s capture branch, before attaching the callback:
`sim.run_headless(steps=60)` (make it `capture_prewarm: int = 60` in config).

### E9. Env-var capture overrides  **[LOW]**

Reference reads `CAPTURE_W/CAPTURE_H/CAPTURE_FRAMES/CAPTURE_OUT` from the
environment (`capture_3d.py:41-48`) — the docker-compose `capture` service
depends on this to shrink llvmpipe renders (this repo's `ci/` has the same
service and no way to do it). In `__main__.py`, after loading config:
env vars override YAML, CLI flags override env. Also add `CAPTURE_W/H` →
`capture_width/height` — note the Recorder currently ignores those two fields
entirely (it uses `window_width/height`); fix that while there.

### E10. GIF save flags  **[LOW]**

Reference saves with `optimize=True, disposal=2` (`capture_3d.py:116-124`);
`pymurmur/capture/recorder.py:save_gif` passes neither (larger files; frames
composite instead of replacing). Add both kwargs.

### E11. Spatial grid: toroidal cell wrap + radius-driven range  **[MED]**

Reference grid (`spatial_grid_3d.py:47-82`) wraps cell indices with modulo on
**both** rebuild and query, and derives the queried cell range from the actual
radius (`(pos ± radius) // cell_size`). pymurmur's `SpatialHashGrid`
(`physics/flock.py:136-209`) uses unwrapped keys (edge cells lose their
across-the-seam neighbours under the default toroidal boundary) and hardcodes
the ±1-cell range, silently ignoring `query_radius`'s radius argument. Store
`cols/rows/slices = ceil(dim / cell_size)` from config at construction, apply
`% cols` etc. in `rebuild` and in the query loops, and compute the loop bounds
from the radius as the reference does.

### E12. Velocity init: speed dispersion  **[LOW]**

Reference (`boid_3d.py:72-73`): direction uniform on the sphere ×
`speed ~ U(1, V0)`. pymurmur fixes every bird at `0.8·v0`
(`physics/flock.py:34`). Change to
`random_unit_sphere(N, rng) * rng.uniform(min(1.0, 0.3*v0), v0, (N,1))` —
initial speed dispersion inside the legal band. (Deliberate physics change →
re-pin the golden trajectory, see roadmap.)

### E13. Thickness ratio: wrong formula under the documented name  **[MED]**

Reference (`flock_shape.py:104-105`): `thickness = √(λ₃/λ₁) ∈ (0,1]` (with
`aspect = √(λ₁/λ₃)`). `pymurmur/analysis/metrics.py:335-337` computes
`√(λ₂/λ₃) ≥ 1` and exports it as `thickness_ratio`. Fix the formula, keep the
degenerate-case guards, update the `FlockMetrics` docstring, and adjust any
test pinning the old value.

---

## Part 4 — Implementation roadmap

Dependency-ordered phases. Each phase is independently shippable; run
`pytest test/` green before moving on. Two phases intentionally change physics
output (P1, P5) — each ends by **re-pinning the golden trajectory in the same
commit** so the change is documented as deliberate.

```
P0 safety net ──► P1 occlusion/steric (physics, re-pin golden)
                  ├─► P2 renderer/capture parity   (no physics impact)
                  ├─► P3 ecology & analysis science (metrics only)
                  ├─► P4 presets & UI               (config mutation only)
                  └─► P5 grid & init details        (physics, re-pin golden)
P6 guard rails — any time after P0
```

Estimated total: **5–7 working days.**

### Phase 0 — Safety net  *(½ day; do first)*

1. **Golden trajectory (T14).** Add `test/data/` and a helper
   `test/regenerate_golden.py`: seeded `SimConfig(num_boids=15, seed=77)`,
   run 30 frames in each of projection and spatial mode via
   `SimulationEngine`, save positions/velocities to
   `test/data/golden_trajectory.npz`. Add
   `test/test_golden.py::test_matches_committed_golden` with
   `np.testing.assert_allclose(..., atol=1e-3)`. Commit the `.npz`.
2. **Physics invariant fuzz (T13)** in `test/physics/test_boid.py`: 200 random
   states → after `integrate(..., "toroidal", ...)`, assert speeds ≤ v0+1e-4
   and positions in-bounds.
3. **H₂ disconnected → `inf` fix** (prereq for item 10): in
   `pymurmur/analysis/metrics.py::compute_h2`, when the Laplacian has more
   than one ~zero eigenvalue (disconnected graph), return
   `(math.inf, math.inf)` instead of `(0.0, 0.0)`; in `find_optimal_m`, skip
   non-finite H₂ values when minimizing J.

**Accept:** suite green, golden committed, `compute_h2` on two far-apart pairs
with m=1 returns `inf`.

### Phase 1 — Scientific correctness  *(1–2 days; Part 1 items 1–4, 14)*

1. Rewrite `pymurmur/physics/occlusion.py::spherical_cap_occlusion` per the
   Part-1 item-1 sketch: closest-first sweep, **visibility cull**
   (`vis_dirs @ dir_j >= vis_cos` against nearer accepted caps), exact
   `α = asin(min(b_eff/d, 1))`, Θ as the running probabilistic union
   (item 2), δ̂ as `Σ sin α · d̂ / Σ sin α` (item 3, drop the magnitude
   clamp), `order = order[:64]` guard (item 4). Signature and return types
   unchanged → `forces/projection.py` and metrics need **no changes**.
2. `pymurmur/physics/steric.py::steric_force`: add
   `max_force: float = 0.15` parameter, clamp the summed force (item 14).
   Call site `forces/projection.py:83` → pass `config.max_force`.
3. Tests: fix `test_occlusion_closest_first` (`[2]`, not `[2,1,0]`), add
   T1–T6 as written in Part 2.
4. **Re-pin golden** (projection-mode dynamics legitimately change).

**Accept:** T1–T6 green; collinear birds → only nearest visible;
`Θ(two caps) < Θ₁+Θ₂`; surrounded bird `|δ̂| < 1e-2`, single neighbour
`|δ̂| = 1`; steric at d=0.01 returns exactly `max_force`.

### Phase 2 — Renderer & capture parity  *(1 day; E1–E3, E7–E10)*

All in `viz/renderer.py`, `capture/recorder.py`, `__main__.py`,
`core/config.py`:

1. E1 depth attachment (exact snippet above).
2. E2 `_build_bird_vao()` extracted; called in `__init__` and after growth.
3. E3 `_mat4_bytes(m)` module function; use for both mat4 uniforms.
4. E7 `_auto_orbit(camera, frame, total)` in `recorder.py`; call before each
   captured frame; config `capture_sweep: bool = True`.
5. E8 pre-warm: `capture_prewarm: int = 60` config; in `__main__` capture
   branch run `sim.run_headless(steps=cfg.capture_prewarm)` first (subtract
   nothing from `capture_frames`; prewarm is additional).
6. E9 env overrides in `__main__.load-config path`:
   `CAPTURE_W/H/FRAMES/OUT` → `capture_width/height/frames/output`
   (env > YAML, CLI > env). Make Recorder actually use
   `capture_width/height` for its headless renderer.
7. E10 `optimize=True, disposal=2` in `save_gif`.

Tests (`test/viz/`, skip-if-no-GL): FBO has a depth attachment
(`renderer._fbo.depth_attachment is not None`); after `add_boids` past the
chunk size, `update_instances` + `draw_birds` still renders (smoke);
GIF exists and opens after a 5-frame capture.

**Accept:** capture GIF shows correct near-over-far occlusion; growth smoke
test passes.

### Phase 3 — Ecology & analysis science  *(2 days; items 5–13, 15, E13)*

1. **Ecology** (`physics/extensions/ecology.py`): add module-level constants +
   free functions from items 5–8 (`seasonal_size_factor`,
   `flock_size_for_day`, `is_murmuration_season`, `coherence_factor`,
   `has_critical_mass`, `gated_weight`, `sunset_hour`, `dusk_factor`
   (logistic, `z = (hour − sunset)/(DUSK_WIDTH/4)`, clamp |z|>60),
   `is_roosting_time`, `roost_strength` (temperature boost), `roost_force`).
   Rewire `Ecology.apply()` to use `dusk_factor × roost_strength ×
   coherence_factor` instead of the linear last-hour ramp — and read
   `config.ecology_roost` / `config.ecology_critical_mass` instead of the
   hardcoded values (they are currently dead config fields).
2. **Shape** (`analysis/metrics.py`): fix thickness to `√(λ₃/λ₁)` (E13); add
   `suggested_m_star(aspect)` (item 9) and a `suggested_m` field on
   `FlockMetrics`, filled in `_compute_expensive_metrics`.
3. **H₂** : add `eta_of_m(positions, m, m0=None)` (item 10; uses the P0 inf
   fix).
4. **τρ**: add `convex_hull_volume(positions)` (scipy `ConvexHull().volume`,
   0 on degenerate) and the integrated-autocorrelation estimator (item 11:
   `τ = interval · (0.5 + Σ r(lag))`, stop at first `r ≤ 0`) as
   `compute_tau_rho_hull(density_series)`; collect a hull-density series in
   `MetricsCollector` at detail_level ≥ 2 alongside the histogram method.
5. **Θ′ silhouette**: `external_opacity(positions, observer_axis=0,
   boid_size=…)` — project ⊥ axis, rasterize disks, union coverage (item 12);
   report as a new field (`theta_prime_silhouette`), keep the voxel metric.
6. **Density scaling** (`analysis/density_scaling.py`): report
   `ideal_density_exponent = -0.5` next to the fitted β; add
   `number_density` on the trimmed gyration sphere; fix `compute_gyration`
   to **median** centroid + one-sided top-15% trim (item 13).
7. **Angular momentum**: add normalized scalar
   `|⟨r×v⟩| / (v0 · gyration_radius)` (item 15) as `angular_momentum_norm`.
8. Tests T7–T12 as written in Part 2.

**Accept:** T7–T12 green; thickness of a thin line flock < 0.2; τρ of constant
density = 0; silhouette Θ′ of one dense wall ≫ voxel Θ′ of the same wall.

### Phase 4 — Presets & UI parity  *(½–1 day; E4–E6)*

1. Replace `analysis/presets.py` with the E4 table +
   `apply_preset(config, key)` (sets `phi_p, phi_a, sigma, mode`, prints
   description, returns label).
2. `viz/input_control.py`: in `_handle_keydown`, add
   `elif (pygame.K_a <= key <= pygame.K_h and key != pygame.K_g) or key ==
   pygame.K_w: apply_preset(cfg, chr(key))`.
3. E5 constraint: after the φp/φa increments, renormalize the other weight.
4. E6 `FlockMetrics.summary()`; extend the visualizer title with
   `mode`, `σ`, summary, and `clock.get_fps()` (title update every ~20th
   frame is enough — the reference does the same to amortize cost).
5. Tests (`test/viz/test_input.py`, no window needed): construct
   `InputControl` with a stub camera, feed synthetic
   `pygame.event.Event(KEYDOWN, key=pygame.K_b)`, assert config matches the
   table row; assert `phi_p + phi_a <= 1.0` after hammering K_UP 100×.

**Accept:** pressing `b` in the running app switches to Ball-of-Birds and
prints the description; φ sum can no longer exceed 1.

### Phase 5 — Grid & init details  *(½ day; E11, E12; Part 1 item 4 note)*

1. E11: `SpatialHashGrid.__init__` stores
   `self._cols/_rows/_slices = max(1, ceil(dim/cell_size))` from config;
   `rebuild` keys become `(cx % cols, cy % rows, cz % slices)`;
   `query_radius` loops `range(int((p-r)//cell), int((p+r)//cell)+1)` per axis
   with modulo-wrapped lookups.
2. E12: velocity init magnitude `rng.uniform(min(1.0, 0.3*v0), v0)` per bird
   (both in `PhysicsFlock.__init__` and `add_boids`).
3. **Re-pin golden** (init distribution changed).
4. Tests: two birds on opposite faces of a toroidal domain appear in each
   other's `query_radius` result; init speeds all within `[0.3·v0, v0]` and
   non-constant.

**Accept:** cross-seam neighbour test passes; golden re-pinned.

### Phase 6 — Guard rails  *(½ day; T15–T16, any time after P0)*

1. T15 collection-count guard for `test/physics/` (pin the number, update on
   deliberate additions).
2. T16 doc-drift test: every `sci/*.md`-referenced module path exists; every
   intra-repo markdown link target resolves.

### Commit strategy

One commit per numbered step where practical; physics-visible steps (P1.1,
P1.2, P5.2) bundle their golden re-pin. Every commit leaves `pytest test/`
green. Suggested order if time is short: **P0 → P1 → P2** (correctness and
user-visible rendering fixes), then P3–P6 as capacity allows.

---

## Where `pymurmur` is already ahead (not gaps — noted for balance)

So the suggestions above aren't mistaken for a one-way verdict, this repo is
**more** complete than the reference in several places, and those should *not* be
"fixed" toward the reference:

- **Boundary modes.** `physics/boid.py` has proper modulo toroidal wrap
  (`pos %= size`, correct across the seam), an `open` mode, a graduated `margin`
  push, **and** a soft `sphere` boundary. The reference's wrap is a single
  set-to-0/L (not true modulo) and it has no sphere mode.
- **Zero-speed re-seed.** Here a frozen bird re-seeds to `0.5·v0` (inside the
  `[0.3·v0, v0]` band); the reference re-seeds to unit magnitude, which can land
  *below* its own floor. Yours is more correct.
- **Predator.** The approach/pass-through FSM with panic speed-boost and
  "blackening" cohesion is richer than the reference's radial-flee predator.
- **Extra metrics.** `speed_avg / force_avg / power_avg / MSD` and the density
  sweep's **R² goodness-of-fit** have no reference counterpart.
- **Architecture.** The plugin `forces/` and `extensions/` registries and the
  SoA hot path are a cleaner structure than the reference's flat modules.

Net: port the **occlusion fidelity** (items 1–3, T1–T3) and the **steric clamp**
(item 14, T6) first — those are correctness gaps. The ecology/shape/H₂/analysis
items are feature additions worth doing but not urgent.
