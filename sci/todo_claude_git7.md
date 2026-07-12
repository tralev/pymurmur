# todo_claude_git7.md — Porting `TheAmirHK/BirdMurmuration` to pymurmur (3D)

**Source:** https://github.com/TheAmirHK/BirdMurmuration — a Multi-Agent
Reinforcement Learning approach to murmuration: a gymnasium environment with
built-in Reynolds physics, a centralized PPO agent that learns velocity
adjustments on top of them, a reward encoding flock quality, and dual-view
3D matplotlib animation. Source verified: `Codes/BirdMurmuration_v1.ipynb`
(the whole implementation), README.
**Target:** `/Users/tralev/Developer/git_mur` (`pymurmur/`).

**What this file is.** Every idea and piece of math in the source that
pymurmur lacks, adapted to pymurmur's 3D pipeline (the source is already 3D;
adaptation = embedding its `[−1,1]³` world into pymurmur's domain and
translating matplotlib rendering to moderngl), with formulas, shapes, file
paths, config fields, and acceptance tests — each roadmap item implementable
from this file alone.

**Scope boundary, stated up front.** `functional_decomposition.md` excludes
ML training from the core ("MARL is an external research mode"). This
roadmap therefore puts everything *except* the training loop in-repo
(control hook, environment, reward, rendering — R1–R4, R6), and specifies
training/inference as **dependency-gated scripts** (R5): fully implementable
from this file, but importing `stable_baselines3` lazily and living in
`scripts/`, so the core package never depends on ML.

**Already implemented in pymurmur (do not redo):** a headless step loop
(`SimulationEngine.run_headless`), Reynolds primitives (used differently —
R2 needs a variant, not a rewrite), GIF capture via PIL, dispersion
(= the cohesion-reward magnitude), and seeded-config infrastructure.

**Source constants (verified):**

```
num_birds = 200
obs   = concat(positions, velocities)  → shape (6N,), bounded [−1, 1]
action = per-bird Δv                    → shape (3N,), bounded [−1, 1]
action scaling      : v += action·0.1 ;  v clipped to [−0.1, 0.1]³
flocking-rule weight: 0.01 (all three rules)
separation radius   : 0.2
init                : p ~ U(−1,1)³ ;  v ~ U(−0.1, 0.1)³
training            : PPO("MlpPolicy"), total_timesteps = 5000
rollout             : 500 steps, render each frame
animation           : two 3D views (elev/azim 15°/15° and 45°/45°),
                      axes limits [−3,3]³, PNG frames → GIF at 10 fps
```

---

## Conventions

- **Unit mapping.** The source world is `[−1,1]³`-ish (drifting to ±3).
  Define `U = min(W,H,D)/6` (so the source's ±3 render box spans the
  domain) and `C = (W/2, H/2, D/2)`. Source length `ℓ` → `ℓ·U` world units;
  source speed 0.1/step → `0.1·U` per step.
- `idx = np.where(flock.active)[0]`; arrays `(N,3) float32`.
- **RNG:** the flock-owned seeded generator (`flock.rng`).
- Normalisation helpers used by R3:

```
p_norm = (p − C) / (3·U)          ∈ ≈[−1, 1]   (positions)
v_norm = v / v_cap                 ∈ [−1, 1]    (velocities; v_cap = 0.1·U per-step)
```

---

## R0 — Prerequisite: config fields

New `SimConfig` fields (apply the YAML section-prefix loader fix first if
still pending — flatten `marl: x:` → `marl_x`, warn on unknown keys, stop
`capture: width:` overwriting the domain):

```python
# ── MARL environment (BirdMurmuration) ───────────────────────
marl_action_scale: float = 0.1        # Δv per unit action, in v_cap units
marl_velocity_cap: float = 0.1       # component cap, source units (×U → world)
marl_rule_weight: float = 0.01        # environment flocking-rule gain
marl_separation_radius: float = 0.2   # source units (×U → world)
marl_episode_steps: int = 500
# ── Reward weights (R4) ──────────────────────────────────────
reward_alignment_w: float = 1.0
reward_cohesion_w: float = 1.0
reward_angular_w: float = 0.0         # optional extension terms
reward_boundary_w: float = 0.0
reward_altitude_w: float = 0.0
reward_altitude_target: float = 0.0   # world-z; 0 = domain centre z
reward_faithful_signs: bool = True    # True = source's sign convention (see R4)
```

---

## R1 — Per-bird external control hook (the core enabler)

**Idea (verbal).** Nothing in pymurmur lets caller code inject per-bird
control into a step — extensions mutate accelerations internally, but there
is no API for "here is a `(N,3)` adjustment, apply it this frame". The
source's whole paradigm is exactly that: an external agent supplies velocity
adjustments each step, *on top of* built-in physics. One hook serves RL,
scripted choreography, and any future pilot mode.

**Math (source step semantics).**

```
v ← clip(v + a_ext·(marl_action_scale·v_cap·U), −v_cap·U, +v_cap·U)   (component-wise)
```

**Implementation.** `pymurmur/simulation/engine.py`:

```python
def step(self, dt: float = 1/60, control: np.ndarray | None = None) -> None:
    self.extensions.pre_step(self.flock)
    if control is not None:
        idx = np.where(self.flock.active)[0]
        cap = self.config.marl_velocity_cap * self._unit_scale()   # v_cap·U
        v = self.flock.velocities[idx]
        v += control.reshape(-1, 3)[: len(idx)] * (self.config.marl_action_scale * cap)
        np.clip(v, -cap, cap, out=v)
        self.flock.velocities[idx] = v
    self.flock.step(self.config, dt)
    self.metrics.collect(self.flock, self.frame)
    self.frame += 1
```

`run_headless` gains a `controller: Callable[[SimulationEngine], np.ndarray]
| None` parameter that supplies `control` per step. Note the clip: with a
control active, the velocity cap **replaces** the usual speed band for those
birds — the MARL mode (R2) runs `integrate` with `speed_mode="none"`
(move + boundary only), matching the source's dynamics, which have no speed
floor/ceiling beyond the component clip.

**Accept:** feeding `control = 0` reproduces the uncontrolled trajectory
bit-for-bit; feeding a constant +x control drives the flock in +x with
per-component speeds capped at `v_cap·U`; the parameter is `None`-safe for
every existing caller.

---

## R2 — Environment-embedded flocking rules (the source's physics variant)

**Idea (verbal).** The source's built-in physics is a deliberately weak,
*global-neighbourhood* Reynolds set: alignment and cohesion are computed
against **all** birds (no radius — effectively the global mean velocity and
CoM), separation against a small hard radius, all scaled by a tiny 0.01
gain. Crucially, the rules run **after** the position update — they prepare
the velocity for the *next* step, while the agent's action (10× stronger)
dominates the current one. This two-layer, deferred-rule control is a
distinct dynamics mode pymurmur doesn't have.

**Math (per bird i, applied to v after positions moved).**

```
F_sep,i   = Σ_{j: ‖p_i−p_j‖ < 0.2·U} (p_i − p_j)
F_align,i = (1/N)·Σ_j v_j − v_i                         (global mean; N = all birds)
F_coh,i   = (1/N)·Σ_j p_j − p_i                         (global CoM attraction)
v_i ← v_i + 0.01·(F_sep,i + F_align,i + F_coh,i)
```

**Implementation.** New force mode `"marl"` in
`pymurmur/physics/forces/marl.py`, registered in `_DISPATCH`:

```python
def marl_forces(flock, config):
    idx = np.where(flock.active)[0]
    p, v = flock.positions[idx], flock.velocities[idx]
    U = _unit_scale(config)
    com, vbar = p.mean(0), v.mean(0)
    sep = np.zeros_like(p)
    tree = cKDTree(p)
    for i, nbrs in enumerate(tree.query_ball_point(p, config.marl_separation_radius * U)):
        for j in nbrs:
            if j != i:
                sep[i] += p[i] - p[j]
    w = config.marl_rule_weight
    flock.velocities[idx] = v + w * (sep + (vbar - v) + (com - p))
```

Ordering: `PhysicsFlock.step` for `mode == "marl"` calls **integrate first**
(move with current velocity, `speed_mode="none"`), **then** `marl_forces`
(deferred-rule semantics). Vectorise the separation loop later via
`np.add.at` over the pair lists if N grows.

**Accept:** with zero control, a scattered flock slowly coheres toward its
CoM and aligns (global rules); the rule effect on a step's *positions* is
delayed by exactly one step (assert: positions after step k depend only on
velocities from step k−1's rules + step k's control).

---

## R3 — Gymnasium environment wrapper

**Idea (verbal).** The bridge that makes pymurmur usable for MARL research:
a standard `gymnasium.Env` exposing the centralized observation
(all positions + velocities, normalized) and action (per-bird Δv) spaces,
delegating dynamics to `SimulationEngine` + R1/R2. Optional import — the
core package must not require gymnasium.

**Math / spaces.**

```
observation_space = Box(−1, 1, shape=(6N,), float32)
    obs = concat( flatten((p − C)/(3U)),  flatten(v / (v_cap·U)) )
action_space      = Box(−1, 1, shape=(3N,), float32)
reset:  p = C + U(−1,1)³·U   ;   v = U(−1,1)³·(v_cap·U)         (seeded)
step:   control ← action  → engine.step(control=...)  → obs, reward(R4),
        terminated=False, truncated=(frame ≥ marl_episode_steps), info={}
```

**Implementation.** `pymurmur/analysis/gym_env.py`:

```python
class MurmurationEnv(gymnasium.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}
    def __init__(self, config: SimConfig | None = None, render_mode=None):
        import gymnasium  # lazy; raise ImportError with install hint
        cfg = config or SimConfig(mode="marl", num_boids=200)
        self.sim = SimulationEngine(cfg); ...
    def reset(self, *, seed=None, options=None):
        if seed is not None: self.sim.config.seed = seed
        self.sim.reset(); self._init_marl_state(); return self._obs(), {}
    def step(self, action): ...
    def render(self):  # rgb_array via the R6 matplotlib fallback or FBO
```

Register nothing globally; document
`from pymurmur.analysis.gym_env import MurmurationEnv`.

**Accept:** `gymnasium.utils.env_checker.check_env(MurmurationEnv())`
passes; two same-seed episodes with the same action sequence produce
identical observations; obs values stay within [−1, 1] for 500 random-action
steps.

---

## R4 — The reward module (and two metrics pymurmur is missing)

**Idea (verbal).** The source's reward is two terms: a velocity-deviation
term and a compactness term. Faithful note: the source *adds* the mean
velocity deviation (positive sign — atypical) and *subtracts* the mean
distance from CoM; the agent then trades the two off, settling at
intermediate order. Implement the faithful signs behind
`reward_faithful_signs=True`, and the "corrected" variant (both terms
negative, maximum 0 at perfect order) as the default-off alternative.
Independent of RL, both quantities are flock observables pymurmur lacks.

**Math.**

```
alignment_dev = (1/N)·Σ_i ‖v̄ − v_i‖          v̄ = mean velocity
cohesion_dev  = (1/N)·Σ_i ‖p_i − CoM‖        (≡ existing `dispersion`)

faithful:   R = +w_a·alignment_dev − w_c·cohesion_dev        (source)
corrected:  R = −w_a·alignment_dev − w_c·cohesion_dev        (max 0)

Optional extension terms (source doc §2.5), all ×their weights:
  R += −w_L·‖Σ_i (p_i−CoM)×v_i‖/N              (excess rotation penalty)
  R += −w_b·Σ_i max(0, ‖p_i − C‖ − R_dom)      (out-of-domain overshoot; R_dom = 3U)
  R += −w_z·(1/N)·Σ_i |p_z,i − z_target|       (altitude keeping)
```

**Implementation.**

1. `pymurmur/analysis/rewards.py`: `compute_reward(flock, config) -> float`
   assembling the weighted terms; pure numpy, no gym dependency (also
   reusable as an EvoFlock scalarization alternative).
2. `pymurmur/analysis/metrics.py`: add `velocity_deviation: float`
   (alignment_dev above — unlike the order parameter α it captures *speed*
   dispersion, not just direction) and `boundary_overshoot: float` to
   `FlockMetrics`, computed every frame (both O(N)).

**Accept:** a perfectly aligned, co-located flock scores
`alignment_dev = 0` and maximal reward under the corrected variant; unit
tests pin both sign conventions; `velocity_deviation` distinguishes a flock
with equal headings but mixed speeds from a truly uniform one (α cannot).

---

## R5 — Training and inference scripts (dependency-gated tier)

**Idea.** The source's entire ML lifecycle is ~15 lines; ship it as
`scripts/` so the external-research mode is demonstrated and reproducible
without making `stable_baselines3` a core dependency.

**Implementation.** `scripts/train_marl.py`:

```python
"""Requires: pip install 'stable-baselines3>=2.0' gymnasium"""
from pymurmur.analysis.gym_env import MurmurationEnv
from stable_baselines3 import PPO

env = MurmurationEnv()                       # 200 birds, mode="marl"
model = PPO("MlpPolicy", env, verbose=1, seed=42)
model.learn(total_timesteps=5000)            # source hyperparameters
model.save("output/marl_ppo")
```

`scripts/rollout_marl.py`: load the model, `reset()`, loop 500 steps of
`model.predict(obs, deterministic=True)` → `env.step`, collecting frames via
R6, save GIF. Docstring guidance (from the source's own scaling analysis):
the centralized MLP is 6N→3N (1200→600 at N=200) and grows quadratically —
for N ≫ 200 use per-agent policies with local observations (IPPO); that is
a research direction, not in scope.

**Accept:** on a machine with sbl3 installed, `train_marl.py` completes
5 000 timesteps and the rollout GIF shows visibly more cohesion than a
random-action rollout (compare mean `cohesion_dev` over the episode:
trained < random by ≥ 20 %).

---

## R6 — Dual-view rendering + GPU-free capture

**Idea (verbal).** Two rendering ideas port independently of RL:
(1) **dual simultaneous viewpoints** — every frame rendered from two camera
angles (low 15°/15° and elevated 45°/45°) side by side, which exposes
whether a flock has true 3D structure or is merely planar (one view can
lie; two can't); (2) the source's whole pipeline is **matplotlib scatter →
PNG → GIF**, i.e. it needs no GL — the blueprint for a capture fallback on
GPU-less machines (pymurmur's Recorder currently produces zero frames
silently without a GL context).

**Implementation.**

1. **Dual-viewport mode** (`viz/renderer.py` + `viz/visualizer.py`):
   `config.dual_view: bool = False`. When on, per frame render the same
   instance buffer twice with `ctx.viewport = (0, 0, w//2, h)` then
   `(w//2, 0, w//2, h)`, using two `OrbitCamera`s fixed at
   (elev 15°, azim 15°) and (elev 45°, azim 45°) around `C` (distance =
   `3.2·U`). Camera-uniform upload already happens per draw — only the
   viewport split and second camera are new. Recorder captures the full
   split FBO (both views in every GIF frame, as the source does).
2. **Matplotlib capture fallback** (`capture/mpl_recorder.py`):
   `MplRecorder(sim, config)` with the same `on_frame`/`save_gif` interface
   as `Recorder`; each sampled frame:

```python
fig = plt.figure(figsize=(8, 4))
for k, (elev, azim) in enumerate(((15, 15), (45, 45))):
    ax = fig.add_subplot(1, 2, k+1, projection="3d")
    ax.scatter(p[:,0], p[:,1], p[:,2], c="black", s=2)
    ax.set_xlim(C[0]-3*U, C[0]+3*U); ...   # fixed limits — stable framing
    ax.view_init(elev=elev, azim=azim); ax.set_axis_off()
frames.append(figure_to_rgb_array(fig)); plt.close(fig)
```

   `save_gif` via imageio (or PIL) at 10 fps. `__main__.py` capture branch:
   try the GL Recorder, on `ImportError`/context failure **fall back to
   MplRecorder with a warning** — replacing the current silent
   `except Exception: pass`.

**Accept:** `dual_view: true` renders two synchronized views; a deliberately
planar flock looks flat in one view and line-like in the other; on a machine
without GL (`moderngl.create_context` raising), `--no-viz --capture` still
produces a GIF via matplotlib and prints which backend it used.

---

## R7 — Preset, tests

**Implementation.**

1. **Preset** `conf/murmuration_marl.yaml`: `mode: marl`,
   `num_boids: 200`, `boundary: open`,
   `marl: {action_scale: 0.1, velocity_cap: 0.1, rule_weight: 0.01,
   separation_radius: 0.2, episode_steps: 500}`,
   `visual: {dual_view: true}`, `seed: 42`.
2. **Tests** (`test/analysis/test_marl_port.py`; gym tests
   `pytest.importorskip("gymnasium")`):
   - control-hook identity and cap tests (R1 acceptance);
   - deferred-rule ordering (R2);
   - `check_env` conformance + obs bounds + determinism (R3);
   - reward sign conventions and `velocity_deviation` discrimination (R4);
   - fallback recorder produces ≥ 1 frame with matplotlib only (R6).
3. No golden re-pin needed: `"marl"` is a new mode; pin its own golden
   trajectory (zero-control, seeded) alongside.

---

## Roadmap summary (dependency order)

| # | Item | Depends on | Size | Files touched |
|---|------|-----------|------|---------------|
| R0 | Config fields (+ loader fix if pending) | — | ¼ day | `core/config.py` |
| R1 | Per-bird control hook | R0 | ½ day | `simulation/engine.py`, `physics/boid.py` |
| R2 | Deferred global-rule "marl" mode | R1 | ½ day | `forces/marl.py` (new), `forces/__init__.py`, `physics/flock.py` |
| R3 | Gymnasium wrapper | R1, R2 | ½ day | `analysis/gym_env.py` (new) |
| R4 | Reward module + 2 new metrics | R0 | ½ day | `analysis/rewards.py` (new), `analysis/metrics.py` |
| R5 | Train/rollout scripts (gated) | R3, R4 | ¼ day | `scripts/` |
| R6 | Dual-view + matplotlib fallback | — | 1 day | `viz/renderer.py`, `viz/visualizer.py`, `capture/mpl_recorder.py` (new), `__main__.py` |
| R7 | Preset + tests + mode golden | all | ½ day | `conf/`, `test/` |

Total ≈ **3½–4 working days**. R6 is fully independent (and useful beyond
MARL — it fixes the silent-capture-failure defect); R4's metrics land value
even if the RL tier is never exercised.
