# MARL Mode

This document defines the MARL (multi-agent reinforcement learning)
mode: an external per-bird control channel layered over a small
deferred flocking-rule composite, the gymnasium observation/action/
reward contract, and the two-step control lag this mode is built
around.

---

## 1. External Control + Deferred Rules

Unlike every other steering mode, MARL mode does not compute a
self-contained force law from neighbour geometry each step. Instead,
each step applies two things in sequence:

```
1. Apply external control:
   v_i += clip(a_ext_i · action_scale · v_cap, −v_cap, v_cap)

2. Move: p_i += v_i · dt   (handled by the integrator, not this mode)

3. Rules prep the *next* step:
   v_i += rule_weight · (F_sep,i + F_align,i + F_coh,i)
```

`a_ext` is the externally supplied action (one 3-vector per bird,
already clipped to [−1, 1] by the caller), read from a one-shot stash
on the config object. If no action is present for the current frame
(e.g. a bare `compute()` call outside the gym wrapper), step 1 is
skipped and only the deferred rules apply.

**Key property — the two-step lag.** Step 3's rules use `positions`
and `velocities` as they stand *after* step 1's control application
but *before* this frame's integration — meaning the rule contribution
computed here doesn't influence position until the *following* step.
This lag is deliberate: it keeps a clean causal separation between
"what the external policy chose this step" and "what the background
rules nudge next," which the observation model (§3) depends on to stay
consistent across the control boundary.

### 1.1 Deferred Rules — Formulas

All three rules use the **global** neighbourhood (every active bird,
no spatial index, no per-mode radius cap on alignment/cohesion) —
appropriate for MARL mode's typical small flock sizes (tens of birds,
not thousands):

```
Separation (within sep_radius only, toroidal-wrapped distance):
  F_sep,i = Σ_{j: 0 < d_ij < sep_radius} clip(r̂_ji / d_ij², −1, 1)

Alignment (global mean velocity):
  F_align,i = v̄ − v_i,   where v̄ = (1/n) Σ_j v_j

Cohesion (global centre of mass):
  F_coh,i = CoM − p_i,   where CoM = (1/n) Σ_j p_j
```

Each per-neighbour separation contribution is clipped to unit
magnitude before summing, preventing an exploding force at very close
range. Separation is the only one of the three that's actually
distance-gated — alignment and cohesion pull every active bird toward
the whole-flock mean/centre regardless of distance, by construction
(there is no spatial index in this mode at all).

### 1.2 Speed Enforcement

After both steps, speed is clamped directly by the mode itself (not
via the generic post-integrate speed-model dispatch other modes use),
because the two are on different unit scales — the generic dispatch's
reference speed is `v0`, while this mode's is `v_cap` (unit-scale
relative, see §4):

```
if ‖v_i‖ > v_cap:        v_i *= v_cap / ‖v_i‖
if ‖v_i‖ < 0.3·v_cap:     v_i *= (0.3·v_cap) / ‖v_i‖
```

---

## 2. Unit Scale

Both the control action and the deferred rules are expressed relative
to a shared unit scale `U`, derived from the domain size:

```
U = min(width, height, depth) / 6
v_cap = marl_velocity_cap · U
sep_radius = marl_separation_radius · U
```

This keeps the mode's dynamics domain-size-invariant — the same
`marl_velocity_cap`/`marl_separation_radius` values produce
proportionally similar behaviour whether the domain is small or large.

---

## 3. Observation Space

The environment wrapper builds a flat `(6N,)` observation per step,
concatenating normalized position and velocity for every bird:

```
U = min(W, H, D) / 6
obs_scale = 3 · U
center = (W/2, H/2, D/2)

pos_norm_i = (p_i − center) / obs_scale
vel_norm_i = v_i / v_cap

obs = concat(pos_norm.flatten(), vel_norm.flatten())     // (6N,)
obs = clip(obs, −1, 1)
```

The declared gymnasium space is `Box(−1, 1, (6N,))`. Neither
normalized term is a hard physical limit by construction — a bird can
sit near a domain edge (pushing `pos_norm` toward ±1 and potentially
beyond) and speed can transiently exceed the soft `v_cap` — so the
final clip is load-bearing: without it, values outside the declared
box would reach the policy and violate the contract that gymnasium's
own environment checker (and anything trusting the declared bounds)
relies on.

---

## 4. Action Space

The action is a flat `(3N,)` array, one 3D adjustment vector per bird,
declared as `Box(−1, 1, (3N,))`. The environment clips any out-of-range
action to `[−1, 1]` before passing it into `compute()` (§1), where it
is further scaled by `action_scale · v_cap` and re-clipped to
`±v_cap` component-wise. This double-clipping (once at the space
boundary, once at the physical-scale boundary) means an action of
exactly `±1` on every axis always maps to the same physical delta
regardless of domain size.

---

## 5. Reward — Five-Term Penalty Composite

The reward is a linear combination of five existing flock-metric
observables — no new physics, only a re-weighted composite of
signals the metrics pipeline already computes every step:

```
R = ±w_a·velocity_deviation
    − w_c·dispersion
    − w_L·‖Σᵢ(pᵢ−CoM)×vᵢ‖ / N
    − w_b·boundary_overshoot
    − w_z·altitude_deviation
```

The `velocity_deviation` term's sign depends on a `faithful_signs`
flag: under `faithful_signs=True` it is `+w_a·velocity_deviation` (the
agent is allowed to trade deviation against compactness, deliberately
left un-"corrected" relative to its source formulation); every other
term is always negative. Under `faithful_signs=False` (the default the
gym wrapper actually uses) the alignment term also flips negative, so
the reward becomes a pure penalty composite — maximum value 0 at
perfect order: full alignment (`velocity_deviation = 0`), zero
dispersion, zero net rotation, and no boundary or altitude excursion.

Three of the five weights (`w_L`, `w_b`, `w_z`) default to 0 —
opt-in extension terms not active unless explicitly configured. Only
`w_a` and `w_c` (velocity deviation and dispersion) are live by
default, making the out-of-the-box reward a simple
order-versus-compactness trade-off.

The reward is computed from the most recent metrics-history entry
each step; if no metrics history exists yet (e.g. the very first
step before any collection), the reward is 0.

---

## 6. Episode Structure

```
terminated = False                          // no terminal-failure condition
truncated  = step_count ≥ episode_steps     // fixed-horizon truncation only
```

Episodes never terminate early — a MARL episode always runs to its
configured horizon (`marl_episode_steps`, default 500) and then
truncates. There is no failure condition (e.g. flock collapse) that
ends an episode prematurely.

`reset()` rebuilds a fresh simulation engine from a copy of the base
config (re-seeded if a new seed was passed), so successive episodes
don't share simulation state.

---

## 7. Contrast with the Other Six Modes

Every other force mode computes velocities/accelerations purely from
internal state — neighbour geometry, per-mode config, and (at most)
the flock's own history. MARL mode instead defers primary control to
an external actor: the force law itself only ever contributes a small
background nudge (`rule_weight`, default 0.01 — two orders of
magnitude smaller than a typical steering weight in the other modes),
while the dominant behaviour comes from whatever policy is driving the
action vector each step. This makes MARL mode structurally a *control
surface* rather than a self-contained steering law — its "physics" is
deliberately minimal so that a learned policy has the largest possible
influence over outcomes.

---

## 8. Taxonomy

MARL mode is one of pymurmur's 7 interchangeable force-computation
strategies — a per-strategy dispatch registry: an ABC (or
shared-signature callable) plus a decorator populating a lookup table
that a call site looks up at runtime instead of branching on a
hardcoded if/elif chain. The six sibling strategies: projection
(occlusion-geometry-driven steering), spatial (Reynolds force
summation), field (named-term blob/anchor compositing), vicsek
(constant-speed angle-coupling alignment), influencer (scripted
tick-driven Lissajous targets), and angle (direct heading rotation,
no force accumulation).

Within that family, MARL mode is architecturally the odd one out: the
other six each implement a complete, self-contained physics law —
given flock state, they deterministically (up to configured noise)
produce the next velocity or acceleration. MARL mode instead is a
*thin adapter*: its own contribution to the dynamics (§1.1's deferred
rule composite) is deliberately minor, and the dominant driver of
behaviour is an external policy supplied through the same
observation/action/reward contract described in §2–§6. Registering it
alongside the other six lets the simulation engine, CLI, and rendering
layer treat "hand the flock over to a learned policy" as just another
selectable mode rather than a special-cased code path — the same
`M`-key mode-cycling and config-driven mode selection that switches
between projection and spatial also reaches MARL mode.

## 9. Beyond pymurmur: Unimplemented Extensions

A few directions from the wider multi-agent reinforcement-learning
literature that this mode's current design does not attempt:

- **Centralized critic, decentralized actors (CTDE).** The reward
  here is a single scalar shared across all birds' identical policy
  input/output contract — there's no mechanism for a training loop to
  use a centralized value function that sees the whole flock's state
  while keeping each bird's actor limited to its own local
  observation. Adding this would be a training-loop concern (outside
  `compute()` itself) rather than a change to the per-step contract.
- **Inter-agent communication channels.** Some multi-agent flocking
  research augments each agent's observation with an explicit learned
  message vector broadcast by nearby agents (rather than only raw
  position/velocity), letting agents coordinate on intent rather than
  just react to state. This mode's `(6N,)` observation (§3) carries no
  such channel.
- **Curriculum scheduling over flock size or reward weights.** The
  reward composite's five weights (§5) are static for an episode;
  there's no built-in schedule for e.g. starting training with only
  the compactness term active and gradually introducing the
  angular-momentum or boundary penalties as the policy matures.
- **Per-bird heterogeneous policies.** The action space assumes one
  policy type is applied uniformly (or a shared policy queried once
  per bird); there's no contract here for mixing e.g. a majority of
  Vicsek-following birds with a minority of independently-learned
  "scout" agents within the same episode.
- **Terminal failure conditions.** §6 notes episodes never terminate
  early — no reward-based or state-based early-stopping (e.g. ending
  the episode immediately if the flock fragments) is implemented,
  unlike some RL-flocking setups that terminate on a collapse
  condition to shape learning signal more sharply near failure.

## 10. Summary of Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `marl_velocity_cap` | 0.5 | `v_cap` multiplier on unit scale U |
| `marl_rule_weight` | 0.01 | Deferred sep/align/coh rule weight |
| `marl_separation_radius` | 2.0 | Separation radius, in units of U |
| `marl_action_scale` | 0.05 | External action scaling factor |
| `marl_episode_steps` | 500 | Truncation horizon (steps per episode) |
| `marl_reward_w_a` | 1.0 | Velocity-deviation reward weight |
| `marl_reward_w_c` | 1.0 | Dispersion reward weight |
| `marl_reward_w_L` | 0.0 | Angular-momentum penalty weight (opt-in) |
| `marl_reward_w_b` | 0.0 | Boundary-overshoot penalty weight (opt-in) |
| `marl_reward_w_z` | 0.0 | Altitude-deviation penalty weight (opt-in) |
