# todo_claude_git2.md — Porting `corentinpradier/collective-motion` to pymurmur (3D)

**Source:** https://github.com/corentinpradier/collective-motion — a 2D Vicsek
model with predator–prey dynamics (`collective_motion.py`, class
`CollectiveMotion`).
**Target:** `/Users/tralev/Developer/git_mur` (`pymurmur/`), whose `vicsek`
force mode ([pymurmur/physics/forces/vicsek.py](pymurmur/physics/forces/vicsek.py))
is a partial port of this repo.

**What this file is.** Every idea and piece of math in the source repo that is
*not* implemented (or implemented incorrectly) in pymurmur, re-derived for
**3D simulation and visualization**, with enough detail — formulas, array
shapes, file paths, config fields, acceptance tests — that each roadmap item
below can be implemented from this file alone.

**Already implemented in pymurmur (do not redo):** the Vicsek core skeleton
(mean-neighbour direction + noise blend, constant-speed position update,
radius neighbourhoods via cKDTree), toroidal position wrapping, and the
(η, D) phase-diagram sweep (`pymurmur/analysis/phase_diagram.py` — done more
thoroughly than the source's single-step `corr_diff`).

---

## Conventions used throughout

- **State (3D):** the source stores a heading *angle* θ per agent; in 3D the
  heading is a **unit vector** `û ∈ ℝ³`. pymurmur already stores velocities
  `(N,3) float32`; the vicsek mode treats `û = v/|v|` and rewrites `v = û·s`
  with per-species constant speed `s`.
- **Domain:** the source uses `[−W, +W]²` with wrap at ±W. pymurmur uses
  `[0, width) × [0, height) × [0, depth)` with `pos %= size`. All formulas
  below are written for pymurmur's convention; the half-width `W` of the
  source maps to `width/2`.
- **RNG:** every random draw below must come from a **seeded**
  `np.random.Generator` owned by the flock (`PhysicsFlock` should hold
  `self.rng = np.random.default_rng(config.seed)` and pass it down). Never
  `np.random.*` module calls — pymurmur's current vicsek code violates this.
- `active = flock.active` masks live birds; all per-agent arrays are indexed
  through `idx = np.where(active)[0]`.

---

## R0 — Prerequisite: make the vicsek parameters actually load

**Idea.** The source's constructor takes 13 parameters. pymurmur has config
fields for only 5 of them, and — critically — the YAML loader drops the whole
`vicsek:` section: `SimConfig.from_file`
([pymurmur/core/config.py:149-164](pymurmur/core/config.py#L149-L164))
flattens `vicsek: couplage: 0.8` to key `couplage`, which does not match the
dataclass field `vicsek_couplage`, so it is silently filtered out. Worse,
unprefixed collisions occur: `capture: width:` flattens to `width` and
**overwrites the domain width**. Nothing in this roadmap works until
parameters load.

**Implementation.**

1. In `SimConfig.from_file`, prefix flattened keys with their section name
   when the prefixed name is a valid field:

```python
valid = {f.name for f in fields(cls)}
flat: dict[str, Any] = {}
for section, data in raw.items():
    if isinstance(data, dict):
        for k, v in data.items():
            pk = f"{section}_{k}"          # vicsek.couplage -> vicsek_couplage
            if pk in valid:   flat[pk] = v
            elif k in valid:  flat[k] = v  # domain.width -> width (legacy)
            else: warnings.warn(f"config: unknown key {section}.{k}")
    else:
        flat[section] = data
```

   Special-case the documented legacy aliases (`capture.width` →
   `capture_width`, `metrics.detail_level` → `metrics_detail_level`,
   `visual.fps` → `fps`, `flock.*`, `domain.*`, `projection.*`, `spatial.*`
   already unprefixed-match).

2. Add the missing dataclass fields (defaults = source defaults):

```python
# ── Vicsek predator-prey (collective-motion) ─────────────────
vicsek_n_predators: int = 0            # source default n_predators=1; 0 keeps old behaviour
vicsek_velocity_predator: float = 1.0
vicsek_radius_predators: float = 5.0   # R_pred: prey fear radius
vicsek_weight_afraid: float = 3.0      # w_afraid
vicsek_predator_noise_ratio: float = 0.2
vicsek_detect_ratio: float = 1.5       # R_detect = ratio * R_pred
vicsek_time_step: float = 1.0          # Δt in the noise amplitude
```

   (`vicsek_couplage`, `vicsek_diffusion`, `vicsek_radius_influence`,
   `vicsek_radius_avoid`, `vicsek_velocity` already exist.)

3. Round-trip test: `SimConfig(...).to_file(p); SimConfig.from_file(p)` equal
   on every field; plus a test that loading `conf/murmuration_vicsek.yaml`
   yields `width == 40.0` (not 600) and `vicsek_couplage == 0.8` from YAML,
   not defaults.

**Accept:** editing `couplage:` in the YAML changes runtime behaviour; the
domain of every shipped preset survives loading.

---

## R1 — Correct 3D Vicsek update (memory term, noise amplitude, tangent-plane noise, constant speed)

**Idea (verbal).** The source's prey update is *not* "blend alignment with a
random direction". It blends the alignment target with the agent's **own
previous heading perturbed by noise** — persistence plus diffusion. pymurmur
blends against a fresh random unit vector, so 20% of every step (at η=0.8) is
memoryless, and lone birds random-walk with zero persistence. Additionally,
pymurmur normalises the noise vector, throwing away the diffusion amplitude
`√(2DΔt)` — the D axis of the phase diagram physically does nothing.

**Math (3D).**

Source (2D): `θ_new = η·θ_target + (1−η)·(θ_old + noise)`,
`noise ~ √(2D·Δt)·N(0,1)`.

3D form, per prey `i` with neighbour set `Nᵢ = {j : ‖pⱼ−pᵢ‖ < R_inf}`:

```
M_i      = Σ_{j∈N_i} w_j · û_j                       (w_j = 1 normally; see R3)
û_target = M_i / ‖M_i‖            (if ‖M_i‖ > 0, else û_target = û_noisy)
η̂_i      = √(2·D·Δt) · n_⊥                           noise IN THE TANGENT PLANE of û_old:
             n_⊥ = (g − (g·û_old)·û_old),  g ~ N(0, I₃)   (project out the parallel part)
û_noisy  = normalize(û_old + η̂_i)
û_new    = normalize(η·û_target + (1−η)·û_noisy)
v_new    = û_new · s                                  (s = vicsek_velocity, constant)
p_new    = p_old + v_new · Δt
```

Why tangent-plane: in 3D, isotropic additive noise has a component along
`û_old` that only rescales (then normalises away), biasing the effective
angular diffusion downward; projecting onto the plane ⊥ `û_old` makes
`√(2DΔt)` the true angular step scale, matching the 2D model's
`θ += noise` semantics.

**Implementation.** Rewrite `vicsek_forces` in
`pymurmur/physics/forces/vicsek.py`:

```python
def vicsek_forces(flock, config):
    active_idx = np.where(flock.active)[0]
    n = len(active_idx)
    if n == 0: return
    rng   = flock.rng
    dt    = config.vicsek_time_step
    eta   = config.vicsek_couplage
    amp   = np.sqrt(2.0 * config.vicsek_diffusion * dt)

    pos   = flock.positions[active_idx]
    vel   = flock.velocities[active_idx]
    speed = np.linalg.norm(vel, axis=1, keepdims=True)
    u_old = np.where(speed > 1e-9, vel / np.maximum(speed, 1e-9),
                     _random_units(rng, n))          # zero-speed fallback

    # tangent-plane noise
    g      = rng.normal(size=(n, 3)).astype(np.float32)
    g     -= (np.sum(g * u_old, axis=1, keepdims=True)) * u_old
    u_noisy = _normalize_rows(u_old + amp * g)

    u_target = _alignment_targets(flock, config, active_idx, u_old, u_noisy)  # R3/R4 hook
    u_new  = _normalize_rows(eta * u_target + (1.0 - eta) * u_noisy)

    flock.velocities[active_idx] = u_new * config.vicsek_velocity
```

Constant-speed contract: `integrate()`
([pymurmur/physics/boid.py:19-68](pymurmur/physics/boid.py#L19-L68)) applies a
`[0.3·v0, v0]` band with the *global* `v0` — with defaults it rescales every
vicsek bird from 1.0 to 1.2 per frame. Fix: give `integrate` a
`speed_mode: str = "band"` parameter; vicsek mode calls with
`speed_mode="fixed"` (skip both clamps — the mode already sets exact speeds).
Route via `PhysicsFlock.step`: `speed_mode = "fixed" if config.mode == "vicsek"
else "band"`.

**Accept:** (1) a single isolated bird's heading autocorrelation
`⟨û(t)·û(t+1)⟩` rises toward 1 as D→0 (persistence exists); (2) sweeping D at
fixed η now changes the steady-state order parameter (it currently doesn't);
(3) every vicsek bird's speed equals `vicsek_velocity` exactly after any
number of steps.

---

## R2 — Species layer: predators as flock members

**Idea.** The source's predators are *agents in the same arrays* as prey —
same state, different update rule and speed. pymurmur has no notion of agent
type. This is the substrate for R3–R5 and R9.

**Implementation.**

1. `PhysicsFlock.__init__` gains
   `self.is_predator = np.zeros(N, dtype=bool)`; the **last**
   `config.vicsek_n_predators` active slots are set True when
   `config.mode == "vicsek"` (source seats them at the end of the array too).
   Extend `_extend()` and `add_boids`/`remove_boids` to carry the column
   (new birds default to prey).
2. Convenience masks inside the vicsek mode:

```python
pred = flock.is_predator[active_idx]      # (n,) bool
prey = ~pred
```

3. Per-species speed at the end of R1's update:

```python
speeds = np.where(pred, config.vicsek_velocity_predator, config.vicsek_velocity)
flock.velocities[active_idx] = u_new * speeds[:, None]
```

4. Neighbour queries must be species-aware: build **two** cKDTrees per step,
   `tree_prey = cKDTree(pos[prey], boxsize=box)` and
   `tree_pred = cKDTree(pos[pred], boxsize=box)` (boxsize from R6).
   All alignment uses `tree_prey` (prey align with prey only — source
   behaviour); fear and hunting use cross-tree queries.

**Accept:** `N_active == n_prey + n_pred`; metrics (α, dispersion) computed
over **prey only** in vicsek mode (source's `param_ordre` uses prey only) —
gate with `flock.is_predator` in `MetricsCollector.collect`.

---

## R3 — Fear-weighted prey behaviour

**Idea (verbal).** Prey near a predator do two things at once: they steer
*away* from the predators, and they pay **more attention to each other**
(neighbour weight amplified ×`w_afraid`) — frightened birds watch the flock
harder. The blend between "align" and "flee" scales linearly with predator
proximity.

**Math (3D).** For prey `i`, let `P_i = {k predators : d_ik < R_pred}`
(minimum-image distances, R6):

```
d̄_pred      = mean_{k∈P_i} d_ik
fear_i      = clamp((R_pred − d̄_pred) / R_pred, 0, 1)        (0 if P_i empty)

û_flee,i    = normalize( Σ_{k∈P_i} (p_i − p_k)_mi / |P_i| )   (_mi = minimum-image difference)
              (if P_i empty: û_flee,i = random unit vector)

w_j         = w_afraid  if fear_i > 0 else 1.0                (per-neighbour weight in R1's M_i)
û_align,i   = normalize( Σ_{j∈N_i, prey} w_j · û_j )

û_combined,i = normalize( (1 − fear_i)·û_align,i + fear_i·û_flee,i )
```

`û_combined` replaces `û_target` in R1's blend for prey. Note the weight
amplification enters *before* normalisation — with mixed afraid/unafraid
neighbours it changes nothing (uniform scaling), but the source keeps the
formulation for the general per-neighbour case; implement it as a scalar
multiply of `M_i` gated on `fear_i > 0` for exact parity.

**Implementation.** Inside `_alignment_targets` (R1 hook):

```python
if n_pred_active:
    d_pred, _ = tree_pred.query(pos[prey], k=min(len_pred, K_PRED), ...)
    # or query_ball_point(pos[prey], R_pred) for the exact set
    near      = d_pred < config.vicsek_radius_predators
    fear      = np.clip((R_pred - mean_near_dist) / R_pred, 0, 1)   # (n_prey,)
    flee_vec  = mean of min-image (p_prey - p_pred) over near predators
    u_flee    = _normalize_rows(flee_vec); rows with no predator -> random units
    u_comb    = _normalize_rows((1-fear)[:,None]*u_align + fear[:,None]*u_flee)
```

**Accept:** with one stationary predator planted at the flock centre, mean
prey heading points radially outward within 5 steps (⟨û·r̂⟩ > 0.8 for prey
inside R_pred); with predators removed, behaviour is bit-identical to R1.

---

## R4 — Predator hunting (no couplage, reduced noise)

**Idea (verbal).** Predators are deliberate: they pursue the **nearest prey**
inside a detection radius *larger* than the prey's fear radius (they see prey
before prey panic), commit 100% to the target direction (no η blending, no
memory of previous heading), and carry 5× less noise. With no prey in range
they random-walk. Edge case: if there are no prey at all, every agent pure
random-walks and all interaction logic is skipped.

**Math (3D).**

```
R_detect = vicsek_detect_ratio · R_pred            (default 1.5 · 5 = 7.5)

for predator k:
    prey* = argmin_{prey j, d_kj < R_detect} d_kj          (minimum-image)
    û_target,k = normalize((p_{prey*} − p_k)_mi)            if prey* exists
    η̂_k = vicsek_predator_noise_ratio · √(2DΔt) · n_⊥      (tangent-plane, 0.2×)
    û_new,k = normalize(û_target,k + η̂_k)                  (prey found — NO couplage)
    û_new,k = normalize(û_old,k + η̂_k)                     (no prey — random walk)
```

All-predator edge case (`n_prey == 0`): every agent takes the random-walk
branch; skip trees, fear, and alignment entirely (early return).

**Implementation.** In `_alignment_targets`, the predator rows bypass the
η-blend — cleanest is to compute `û_new` for prey and predators separately
and merge:

```python
u_new = np.empty_like(u_old)
u_new[prey] = _normalize_rows(eta * u_comb[prey] + (1-eta) * u_noisy[prey])
u_new[pred] = _normalize_rows(u_target_pred + noise_pred)      # no eta
```

(so the R1 skeleton's final blend moves into the species branches).
`tree_prey.query(pos[pred], k=1)` gives nearest prey + distance in one call;
mask rows with `d > R_detect` to the random-walk branch.

**Accept:** a single predator seeded 10 units from a static prey cloud closes
distance monotonically (≥90% of steps decrease min distance); with
`n_preys=0`, α stays ≈ `1/√N` (disordered) for all η, D.

---

## R5 — Asymmetric collision avoidance (`dont_touch_predator`)

**Idea (verbal).** Collisions are resolved by **position correction**, not
force: overlapping same-type pairs split the correction 50/50; a prey
overlapping a predator absorbs **100 %** of the correction (the predator
plows through unaffected). This runs *after* the position update and *before*
boundary wrapping, using minimum-image pair vectors.

**Math (3D).** For a pair at minimum-image separation `d = ‖Δ_mi‖`,
`n̂ = Δ_mi / d` pointing i→j:

```
prey–prey,   d < R_avoid:   p_i −= n̂·(R_avoid − d)/2;   p_j += n̂·(R_avoid − d)/2
pred–pred,   d < R_avoid:   same, symmetric
prey–pred,   d < R_pred:    p_prey −= n̂·(R_pred − d)    (n̂ points prey→predator)
                            p_pred unchanged
```

**Implementation.** New function in `vicsek.py`, called from the mode after
setting velocities... — position corrections need the *post-move* positions,
and in pymurmur the move happens in `integrate()`. Two options; use (a):

(a) give the vicsek mode its own position update (it already owns velocity):
    set `v`, then in `PhysicsFlock.step` for vicsek mode call
    `_vicsek_collisions(flock, config)` **after** `integrate` and **before**
    metrics — `integrate` wraps positions, so unwrap is unnecessary if
    collision uses minimum-image differences (correct on the torus by
    construction).

Vectorised pair resolution with the species trees:

```python
def _vicsek_collisions(flock, config):
    idx  = np.where(flock.active)[0]
    pos  = flock.positions[idx]
    pred = flock.is_predator[idx]
    box  = np.array([config.width, config.height, config.depth], np.float32)

    def _pairs(tree_a, tree_b, r):            # unique overlapping pairs
        return tree_a.query_ball_tree(tree_b, r)

    def _min_image(delta):
        return delta - box * np.round(delta / box)

    # same-type, symmetric half corrections
    for mask, r in ((~pred, config.vicsek_radius_avoid),
                    (pred,  config.vicsek_radius_avoid)):
        sub = np.where(mask)[0]
        if len(sub) < 2: continue
        tree = cKDTree(pos[sub], boxsize=box)
        for i, nbrs in enumerate(tree.query_ball_point(pos[sub], r)):
            for j in nbrs:
                if j <= i: continue
                delta = _min_image(pos[sub[j]] - pos[sub[i]])
                d = np.linalg.norm(delta)
                if d < 1e-9 or d >= r: continue
                corr = (r - d) / 2.0 * (delta / d)
                pos[sub[i]] -= corr; pos[sub[j]] += corr

    # prey–predator, fully asymmetric (prey takes all of it)
    ... same loop over (prey_tree, pred_tree, R_pred):
        pos[prey_i] -= (R_pred - d) * (delta / d)     # delta = pred - prey, min-image

    flock.positions[idx] = pos % box                  # re-wrap
```

For large N, replace the inner Python loops with flattened pair index arrays
and `np.add.at(pos, i_idx, -corr); np.add.at(pos, j_idx, +corr)` — the
source uses exactly this `np.add.at` pattern.

**Accept:** after 100 steps with `R_avoid=1`, no same-type pair sits closer
than `0.5·R_avoid` (corrections may not fully resolve chains in one pass —
the source accepts this too); planting a predator inside a prey clump ejects
every prey to ≥ R_pred within a few steps while the predator's position
trace is unaffected by the contacts.

---

## R6 — Minimum-image (toroidal) distances everywhere

**Idea.** Everything above assumes distances are measured **on the torus**.
pymurmur currently measures raw Euclidean distances under its default
toroidal boundary — birds near opposite faces are mutually invisible.

**Math.** Per axis of box length `L`:

```
Δ_mi = Δ − L·round(Δ / L)            (component-wise; |Δ_mi| = min(|Δ|, L−|Δ|))
d²   = ‖Δ_mi‖²
```

**Implementation.** Two mechanisms cover all uses:

1. **Neighbour queries:** `cKDTree(pos, boxsize=(W, H, D))` — scipy then
   returns torus-correct `query`, `query_ball_point`, `query_ball_tree`.
   Requirement: positions must be pre-wrapped into `[0, L)` (pymurmur's
   toroidal integrate already guarantees this). Gate on
   `config.boundary_mode == "toroidal" and config.use_toroidal_distance`
   (the latter field exists and is currently dead — this makes it live).
2. **Pair vectors:** the `_min_image(delta)` helper above, used in R3's flee
   vector, R4's pursuit vector, R5's corrections, and R8's MSD.

**Accept:** two birds at `x = 0.5` and `x = W − 0.5` (same y,z) report
distance ≈ 1.0 via the tree and appear in each other's `R_inf`
neighbourhoods; disabling `use_toroidal_distance` restores raw behaviour.

---

## R7 — Nematic order parameter (2D pairwise + 3D Q-tensor)

**Idea (verbal).** The source's `param_ordre` measures **nematic** order —
alignment modulo 180° — so anti-parallel lanes count as ordered. pymurmur has
only the polar α. Both are needed: polar to detect flocking, nematic to
distinguish a coherent flock (both ≈ 1) from a two-lane configuration
(polar ≈ 0, nematic ≈ 1).

**Math.**

Source (2D pairwise): `S = (2 / N(N−1)) · Σ_{i<j} cos(2(θᵢ − θⱼ))`.

3D (liquid-crystal Q-tensor — the correct generalisation, O(N) not O(N²)):

```
Q_αβ = (1/N) Σ_j ( (3/2)·û_j^α û_j^β − (1/2)·δ_αβ )     (3×3 symmetric, traceless)
S    = λ_max(Q)                                          ∈ [−1/2, 1]
```

`S = 1` all parallel *or* anti-parallel; `S ≈ 0` isotropic.

**Implementation.** In `pymurmur/analysis/metrics.py`:

```python
def compute_nematic_order(velocities: np.ndarray) -> float:
    norms = np.linalg.norm(velocities, axis=1, keepdims=True)
    u = velocities / np.maximum(norms, 1e-9)               # (N,3) unit
    Q = 1.5 * (u.T @ u) / len(u) - 0.5 * np.eye(3)
    return float(np.linalg.eigvalsh(Q)[-1])
```

Add `nematic: float = 0.0` to `FlockMetrics`, fill in `collect()` (fast,
every frame; prey-only in vicsek mode per R2). Add an
`order: "polar" | "nematic"` parameter to
`analysis/phase_diagram.py::sweep_vicsek_phase` selecting which observable
fills `alpha_grid` (the shipped YAML's `phase_diagram_use_nematic: true`
finally means something).

**Accept:** two half-flocks moving exactly opposite (+x̂ / −x̂):
polar α ≈ 0, nematic S ≈ 1; isotropic random headings: both < 0.15 at
N = 500.

---

## R8 — MSD(τ) per-lag curve with minimum-image, and the ballistic→diffusive crossover

**Idea (verbal).** The model's signature dynamical observable: mean squared
displacement as a **function of lag τ**, whose log-log slope crosses over
from 2 (ballistic, coherent flight) at short lags to 1 (diffusive, noise
dominated) at long lags. pymurmur computes only one first-vs-last scalar with
no wrap correction, which spikes every time a bird crosses the toroidal seam.

**Math.**

```
MSD(τ) = (1/N) Σ_i  (1/(N_t − τ)) Σ_{t=0}^{N_t−τ−1} ‖ p_i(t+τ) − p_i(t) ‖²_unwrapped
```

Unwrapping: accumulate true displacements before wrapping —

```
step_t   = ( p(t+1) − p(t) )_mi          (minimum-image per axis, R6)
p_unwrap(t) = p(0) + Σ_{s<t} step_s
```

Slope diagnostic: `slope(τ) = d log MSD / d log τ`; report
`τ_cross = argmin_τ |slope(τ) − 1.5|` as the crossover lag.

**Implementation.** In `MetricsCollector`: keep the existing snapshot list
but store **unwrapped** positions — maintain `self._p_unwrap` updated each
`collect()` from the min-image step (needs previous wrapped positions:
keep `self._prev_pos`). Replace `compute_msd(snapshots)` with:

```python
def compute_msd_curve(snaps: list[np.ndarray], lags: list[int]) -> dict[int, float]:
    out = {}
    for L in lags:
        if L >= len(snaps): break
        disp = [snaps[t+L] - snaps[t] for t in range(len(snaps)-L)]
        out[L] = float(np.mean([np.mean(np.sum(d*d, axis=1)) for d in disp]))
    return out
```

Default `lags = [1, 2, 4, 8, 16, 32, 64]` (log-spaced). Keep the scalar
`msd` field (last-lag value) for compatibility; add `msd_curve: dict | None`
and `msd_crossover: float | None` to `FlockMetrics` (detail_level ≥ 2).

**Accept:** noiseless straight-line flight (D=0, η=1, aligned init):
slope ≈ 2.0 at every lag; pure random walk (η=0, large D): slope ≈ 1.0 for
τ ≥ 4; a bird crossing the seam produces no spike (MSD(1) ≈ `s²·Δt²`
exactly).

---

## R9 — 3D visualization of species and heading

**Idea (verbal).** The source renders agents as coloured arrows — prey blue,
predators red and longer (3.0 vs 4.5). pymurmur's instanced tetrahedra
already encode heading by orientation; what's missing is any per-instance
colour or size, so predators are **invisible distinctions**. 3D adjustment:
per-instance colour + scale in the instance buffer; optional heading-hue
colouring as a debug view.

**Implementation.**

1. Extend the instance layout in `pymurmur/viz/renderer.py` from 6 to 7
   floats per bird: `[pos.xyz, vel.xyz, flag]` where `flag = 1.0` for
   predator, else 0.0. VAO format string becomes `'3f 3f 1f/i'` with a new
   `in_bird_flag` attribute; **rebuild the VAO wherever the buffer is
   (re)created** (there is a known bug where growth skips the VAO rebuild —
   fix together).
2. Vertex shader (`pymurmur/viz/shaders.py`): pass the flag through;
   scale the mesh `* mix(1.0, 1.5, in_bird_flag)` (predators 1.5× — the
   source's 4.5/3.0 ratio).
3. Fragment shader: `base_color = mix(themeColor, vec3(0.85, 0.1, 0.1),
   v_flag)` — predators red in every theme (source behaviour: red overrides
   scheme).
4. `Renderer3D.update_instances` packs the flag column from
   `flock.is_predator` (R2).
5. Optional heading-hue mode (`theme: "heading"`): colour =
   HSV(azimuth(û)/2π, 0.7, 0.9) computed in the vertex shader from
   `normalize(in_bird_vel)`; azimuth = `atan(v.y, v.x)`.

**Accept:** running `--config murmuration_vicsek` with
`vicsek_n_predators: 1` shows one red, larger tetrahedron; screenshots in
each theme keep predators red.

---

## R10 — Preset, invariants, and regression coverage

**Idea.** Tie R0–R9 together so the shipped Vicsek experience matches the
source repo, and lock behaviour with tests.

**Implementation.**

1. **Preset:** update `conf/murmuration_vicsek.yaml` keys to the R0 names
   (`vicsek: n_predators: 1`, etc. — the file already *documents* these
   values; after R0 they load). Source defaults for parity:
   `n_preys=100, n_predators=1, R_inf=5, R_avoid=1, R_pred=5, v=1, v_pred=1,
   Δt=1, W=20 (→ domain 40³), D=0.8, η=0.8, w_afraid=3.0`.
2. **Invariant tests** (`test/physics/test_vicsek.py`):
   - constant speed: all prey speeds == `vicsek_velocity` after 50 steps;
   - determinism: same seed → identical positions after 100 steps (twice);
   - order transition: α(η=0.95, D=0.05) > 0.8 and α(η=0.05, D=2.0) < 0.3
     at N=200 after 300 settled steps;
   - fear response, pursuit, collision asymmetry, min-image visibility, and
     nematic/two-lane assertions as given in R3–R7 above.
3. **Golden trajectory:** re-pin the repo's golden snapshot for vicsek mode
   after R1 (dynamics deliberately change), in the same commit.

---

## Roadmap summary (dependency order)

| # | Item | Depends on | Size | Files touched |
|---|------|-----------|------|---------------|
| R0 | Config loading + fields | — | ½ day | `core/config.py`, tests |
| R6 | Minimum-image distances | R0 | ½ day | `forces/vicsek.py`, `physics/flock.py` |
| R1 | Correct 3D Vicsek update | R0, R6 | 1 day | `forces/vicsek.py`, `physics/boid.py`, `physics/flock.py` |
| R2 | Species layer | R1 | ½ day | `physics/flock.py`, `analysis/metrics.py` |
| R3 | Fear-weighted prey | R2 | ½ day | `forces/vicsek.py` |
| R4 | Predator hunting | R2 | ½ day | `forces/vicsek.py` |
| R5 | Asymmetric collisions | R2, R6 | 1 day | `forces/vicsek.py`, `physics/flock.py` |
| R7 | Nematic order | R0 | ¼ day | `analysis/metrics.py`, `analysis/phase_diagram.py` |
| R8 | MSD(τ) curve | R6 | ½ day | `analysis/metrics.py` |
| R9 | Species visualization | R2 | ½ day | `viz/renderer.py`, `viz/shaders.py` |
| R10 | Preset + tests + golden | all | ½ day | `conf/`, `test/` |

Total ≈ **6 working days**. R7 and R8 are independent of the species chain
and can be done any time after their prerequisites; R9 can proceed in
parallel with R3–R5 once R2 lands.
