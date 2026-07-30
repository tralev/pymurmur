# Speed Models

This document defines the six speed control strategies found across 22
surveyed flocking implementations — how each controls a bird's velocity
magnitude independently of its heading.

---

## 1. Fixed Speed

Velocity magnitude is forced to a constant every frame.  Steering only
changes direction, never speed.

```
v_new = speed_constant · normalize(v_desired)
```

**Characteristics:** Predictable, uniform movement.  Sacrifices speed
variation for simplicity.  Common in angle-based and GPU compute
implementations where speed variation would complicate the math.

**Used by:** pymurmur angle mode, pymurmur vicsek mode, pymurmur
influencer mode, §04, §06, §09, §10a, §10b from the survey.

---

## 2. Clamped Variable

Speed varies naturally from steering forces, then is capped at a
maximum (or clamped to a min+max range) each frame.  This is the most
common model — 14 of 22 implementations use some form of it.

### 2.1 Normalize-Then-Scale

Direction is preserved; magnitude is reset to exactly `maxSpeed` when
over the limit:

```
if |v| > maxSpeed:
    v = normalize(v) × maxSpeed
```

**Used by:** §01, §03, §08, §13 from the survey.

### 2.2 Direct Truncation

Magnitude is capped without re-normalizing first — the direction is
preserved but the magnitude is simply limited:

```
if |v| > maxSpeed:
    |v| = maxSpeed
```

**Used by:** §02 from the survey.

### 2.3 Min + Max Clamp

Both lower and upper bounds are enforced.  Birds cannot stop, ensuring
the flock never stagnates:

```
|v| = clamp(|v|, minSpeed, maxSpeed)
```

**Used by:** §14 from the survey.

### 2.4 Band Clamp

Speed is clamped to a band `[0.3·v0, v0]` — a floor prevents stalling,
a ceiling caps overspeed.  This is pymurmur's default speed policy:

```
if |v| < 0.3·v0:   |v| = 0.3·v0    // floor — prevent stall
if |v| > v0:        |v| = v0        // ceiling — cap overspeed
```

**Used by:** pymurmur (default), murmuration (§21).

### 2.5 With Damping

A slight friction term reduces speed every frame, requiring continuous
steering to maintain velocity:

```
v = v × (1 − damping × dt)
if |v| > maxSpeed:  |v| = maxSpeed
```

**Used by:** §15 from the survey.

### 2.6 State-Dependent Speed

Speed varies by behavioural mode — e.g. halved when turning at the edge,
full speed when fleeing:

```
if state == "turning":       speed = base_speed × 0.5
elif state == "turning_hard": speed = base_speed
else:                        speed = random(base_speed_low, base_speed_high)
```

**Used by:** §16, §21, §22 from the survey.

---

## 3. Noise-Modulated

Speed is a continuous function of the bird's position in a procedural
noise field, creating organic slow and fast zones:

```
q = position × frequency                          // frequency = 0.01 cycles/unit
noise = sin(q.x·1.3 + cos(q.y·1.7 + 0.5)) × cos(q.z·1.1 + sin(q.x·0.9 + 1.7))
noise_01 = (clip(noise, −1, 1) + 1) / 2
mult = 0.5 + 1.5 · noise_01³
speed_cap = base_speed_cap × mult
```

**Characteristics:** Creates natural-looking speed variation without
any behavioural logic. In pymurmur this is not literal simplex noise —
there is no simplex-noise dependency anywhere in the codebase — but
the same "value noise" family: a deterministic sinusoidal field built
from the same construction the physics core's `curl_flow` primitive
uses for its own pseudo-noise, sampled at each bird's position. Only
the speed *cap* is modulated; the minimum-speed floor is left
untouched, so slow zones still respect the mode's normal minimum
rather than letting birds stall.

**Used by:** §05 (3D Simplex), §20 (3D value noise) from the survey;
pymurmur implements the same taxonomic strategy via the value-noise
variant.

---

## 4. Neighbor-Count Adaptive

Isolated birds fly faster to rejoin the flock; surrounded birds slow
down.  Speed is a function of local density:

```
deficit = target_neighbors − actual_neighbors
if deficit > 0:
    speed = base_speed + bonus(deficit)
else:
    speed = base_speed
```

Three bonus laws are available in pymurmur's angle mode:

| Law | Formula | Character |
|-----|---------|-----------|
| Linear | `deficit × 5.0` | Proportional, uncapped |
| Quadratic | `min(cap, deficit²)` | Gentle for small deficits, saturating |
| Softened | `min(cap, deficit² / 2)` | Half the quadratic gain |

**Used by:** pymurmur angle mode, pymurmur neighbour-adaptive speed
extension, §17 from the survey.

---

## 5. Velocity-Adaptive

Speed smoothly approaches a randomized-bonus target via exponential
lerp, at a single fixed rate:

```
bonus ~ Uniform(0.85, 1.15)                   // re-rolled every frame, per bird
goal = speed_cap × bonus
rate = clamp(LERP_RATE × dt, 0, 1)            // LERP_RATE = 3.0 s⁻¹
v_new = lerp(v, direction(v) × goal, rate)
```

pymurmur implements the randomized-bonus + smooth-approach mechanism
from §11's taxonomy, but simplified: §11's full form keys the target
speed on an external per-bird behavioural state (normal vs. emergency
acceleration rate) — this speed-enforcement strategy has no channel
for that (it would need a per-bird state array threaded in from
outside, tracked separately as a steering-decoupling concern). The
bonus also re-rolls every frame here rather than on §11's periodic
timer, since there is no per-bird persistent state to remember "next
reroll time."

**Used by:** §11 from the survey; pymurmur implements the
randomized-bonus/smooth-approach core of it, without the
behavioural-state-dependent acceleration rate.

---

## 6. None

No speed enforcement at all — the renderer skeleton (§19) has no
flocking math implemented.

---

## 7. Taxonomy

Speed enforcement is a plugin family in its own right — the
SpeedModel registry (6 distinct strategies, 7 registry entries
counting the `clamp` alias of `band`) — dispatched from a fixed point
in the per-frame pipeline: after acceleration integration, before the
frame is considered complete. This is a per-strategy dispatch
registry, structurally identical to the mode-computation family
above it: an ABC (or shared-signature callable) plus a decorator
populating a lookup table, chosen at runtime instead of a hardcoded
if/elif chain.

The key architectural property is *orthogonality*: which speed model
runs is independent of which of the 7 force-computation strategies
produced the acceleration that frame. Any force mode can, in
principle, pair with any speed model — the same way any of the 5
boundary-handling strategies can pair with any force mode. In
practice most modes have one speed model they're tuned around
(angle/vicsek/influencer default to `fixed`; spatial/projection
default to `band`), but the registry doesn't enforce that pairing.

---

## 8. Beyond pymurmur

Candidate speed-enforcement strategies from the broader
flocking-simulation literature not implemented here — framed as
candidates, not verified-correct recommendations:

**Priority-ordered speed override.** Several implementations use a
priority-ordered rule stack where a triggered high-priority rule (an
imminent obstacle, a predator) fully overrides the normal speed
policy rather than blending with it — e.g. an obstacle-avoidance rule
forcing a hard deceleration regardless of what the base speed model
would otherwise compute. pymurmur's closest analogue is the panic
speed-*ceiling* raise (a multiplicative cap increase, not an
override) — a genuine override would need the speed-enforcement step
itself to accept a per-bird "bypass normal policy" flag from upstream
extensions.

**Two-rate acceleration (normal vs. emergency).** `velocity_adaptive`
(§5) already implements randomized-bonus smooth-approach at one fixed
lerp rate; some implementations use two distinct rates — a slow
"normal" approach rate and a fast "emergency" rate triggered by an
external behavioural-state signal (e.g. a bird that just detected a
predator snaps to its target speed almost instantly, while routine
speed adjustments stay smooth). Would need a per-bird state channel
this speed-enforcement layer doesn't currently have.

**Distance-to-wall speed damping.** A speed policy that slows birds
as they approach a domain boundary (independent of the edge-steering
force itself) — reasoning that a bird about to turn sharply near a
wall benefits from being slower going into the turn, similar to how
some boundary-avoidance implementations pair a steering force with an
explicit speed reduction rather than relying on the turn itself to
bleed speed. Would need the speed-enforcement step to read distance-
to-boundary, which it currently doesn't (that information lives in
the boundary-handling layer instead).

---

## 9. Summary

| Strategy | Count | Speed varies? | In pymurmur? |
|----------|:-----:|:------------:|:------------:|
| Fixed speed | 5 | No | ✅ (angle, vicsek, influencer) |
| Clamped variable | 14 | Yes, then capped | ✅ (default: band clamp) |
| Noise-modulated | 2 | Yes, by noise field | ✅ (`noise_modulated` strategy) |
| Neighbor-count adaptive | 2 | Yes, by local density | ✅ (angle mode, extension) |
| Velocity-adaptive | 1 | Yes, by state + bonus | ✅ (`velocity_adaptive`, without behavioural state) |
| None | 1 | — | ❌ |
