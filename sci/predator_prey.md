# Predator-Prey Species Dynamics

This document defines the predator-prey species system: an autonomous
threat agent with a full approach/egress finite state machine, a
four-term force bundle on nearby birds, panic speed-ceiling raises,
blackening cohesion/separation modulation, and predator hunting with
fear-weighted prey fleeing.

---

## 1. Threat Agent FSM

The predator follows an autonomous approach/egress state machine with
configurable targeting modes.

### 1.1 Phase Transitions

```
dist_to_center = ‖p_threat − C_flock‖

capture_dist = max(0.18, threat_radius · 0.72) · U
pass_dist    = (0.92 + threat_radius · 2.6 + momentum · 1.32) · U
clear_dist   = pass_dist · (0.72 + momentum · 0.16)

Approach → Egress:  dist_to_center ≤ capture_dist
Egress  → Approach: dist_to_center > clear_dist  AND  dot(dir, to_center̂) < −0.12
```

`U` is the unit scale — either `field_unit_scale` from config or
`0.4 · min(W, H, D)`.

### 1.2 Targeting Modes

| Mode | Target | FSM |
|------|--------|-----|
| `"autonomous"` (default) | Flock centre (approach) / beyond centre with arc (egress) | Full FSM |
| `"orbit"` | Always beyond centre with arc offset | Always egress |
| `"cursor"` | User-controlled cursor position (if live) | Frozen phase |
| `"off"` | No movement, no force | None |

### 1.3 Egress Arc

In egress or orbit mode, the target is placed beyond the flock centre
with a sinusoidal arc offset for lift and drift:

```
base_target = C + dir · pass_dist
lift = turn_axis · sin(t · 0.18 + 0.7) · pass_dist · 0.24
drift = normalize(cross(turn_axis, dir)) · cos(t · 0.13 + 1.4) · pass_dist · 0.24 · 0.72
target = base_target + lift + drift
```

`drift` uses the **normalized** cross product — `turn_axis` and `dir`
are both unit vectors, so `cross(turn_axis, dir)` alone has magnitude
`sin(angle between them)` rather than a constant 1; normalizing first
keeps the drift amplitude at a fixed `pass_dist · 0.24 · 0.72`
regardless of how close the two vectors are to parallel.

The turn axis is an EMA-blended version of `dir × to_center̂`, with
sign-aligned blending to prevent discontinuous flips when the threat
crosses the centre line.

### 1.4 Steering

The predator steers toward its target via Rodrigues rotation capped
at a mode-dependent turn rate:

```
Approach turn:  turn_rate = (0.54 + accel · 0.025) · (1 − momentum · 0.24)
                response  = 1.86 + (1 − momentum) · 0.48
Egress turn:    turn_rate = 0.42 · (1 − momentum · 0.24)
                response  = 0.34 + (1 − momentum) · 0.44

max_turn = turn_rate · response · dt
dir_new  = rotate_toward(dir, normalize(target − pos), max_turn)
pos_new  = pos + dir_new · speed · dt
```

Approach turns are 4–5× more responsive than egress turns — the
predator commits hard to closing distance during approach and cruises
lazily during egress.

### 1.5 Speed

```
speed = 2.0 · v0 · (1 + 0.5 · momentum)
```

Roughly twice the prey cruise speed, scaled by momentum.

---

## 2. Force Bundle on Nearby Birds

Birds within `threat_dist = threat_radius · U · 2.0` receive a
four-term force:

```
prox = clamp(1 − d/threat_dist, 0, 1)        // proximity in [0, 1]
broad = √(prox + ε)                           // softened proximity

F_push  = away_dir · threat_strength · (2.5 + vacuole · 1.7) · broad
F_wake  = (away_dir − dir_threat · 0.35) · min(1.8, |v_threat|/v0) · threat_strength · broad · 0.42
F_split = cross(z_axis, away_dir) · 1.45 · split_gain · broad   // xy-plane
F_split_z = away_dir_z · 0.28 · split_gain · broad               // z-component
F_wave  = v̂_prey · wave_gain · broad · 0.22              // velocity-aligned
F_total = F_push + F_wake + F_split + F_wave
```

**Push:** Radial repulsion away from the threat.
**Wake:** Drags birds along the threat's path — they follow slightly
behind the predator.
**Split:** Horizontal tear perpendicular to the radial direction,
splitting the flock left and right.
**Wave:** Velocity-aligned perturbation — birds already moving away
from the threat get an extra boost.

When the priority stack is enabled, the threat force is isolated into
its own tier (`predator_priority_accel`) rather than fused into the
main acceleration buffer.

---

## 3. Panic Speed Ceiling (P3.8)

Birds near a threat get a raised speed ceiling — they can fly faster
to escape:

```
panic     = prox · threat_strength
boost     = panic · (0.72 + wave_gain · 0.18 + vacuole_strength · 0.12)
speed_mult = 1 + min(1.35, boost)

max_speed_i = max(existing_max_speed_i, v0 · speed_mult)
```

The ceiling raise is capped at 2.35× `v0` (1 + 1.35).  This is a
ceiling raise, not a compound multiply — the bird's actual speed
still comes from its normal steering, but the cap is lifted so it
*can* go faster if it chooses to flee.

The boost formula weights threat_strength at 0.72, wave propagation
at 0.18, and vacuole formation at 0.12 — the direct threat proximity
dominates.

---

## 4. Blackening: Cohesion/Separation Modulation (P3.8)

Threat proximity modulates the effectiveness of cohesion and
separation forces in field mode:

```
prox  = clamp(1 − d/threat_dist, 0, 1)
black = 1 + blackening_gain · prox · 0.85

sep_eff = separation · (2 − black)         // weaker near threat
coh_eff = cohesion · black                 // stronger near threat
```

When `black ≈ 1` (far from threat): normal behaviour.
When `black ≈ 1 + blackening_gain · 0.85` (near threat): stronger
cohesion (birds cluster), weaker separation (personal space relaxes).
The `(2 − black)` term ensures separation weakens as blackening
increases — at `black ≈ 1.85`, `sep_eff ≈ 0.15 · separation`.

This implements the "safer together" anti-predator response: near a
threat, birds tighten their formation and relax personal space to
maximize the dilution and confusion effects.

---

## 5. Fear-Weighted Alignment (Vicsek P6.1)

In Vicsek mode, prey birds near predators blend their alignment with
a flee direction. Fear is driven by the **mean** distance to all
nearby predators, not just the closest one:

```
R_pred = vicsek_radius_predators                 // detection radius
near_dists = { ‖p_prey − p_pred_k‖ : predator k within R_pred }   // min-image
fear = clamp((R_pred − mean(near_dists)) / R_pred, 0, 1)
flee_dir = normalize( mean(p_prey − p_pred_k over near predators) )

u_combined = normalize( (1 − fear)·η·u_align + w_afraid·fear·flee_dir + (1 − η)·u_noisy )
```

where `η` is the base Vicsek coupling (`vicsek_couplage`),
`w_afraid = vicsek_weight_afraid` (default 3.0 — the flee term is
weighted independently of `fear`, not just scaled by it), `u_align` is
the bird's normal topological-neighbour alignment direction, and
`u_noisy` is its normal tangent-plane-noised heading — this is a
genuine **three-term** blend, not a simple interpolation between
alignment and fleeing: the noise term's `(1 − η)` weight is untouched
by fear.

A bird with no predators within `R_pred` skips fear blending entirely
(its direction is whatever the base Vicsek step already computed).
Solo prey (birds with no topological neighbours, so no `u_align` to
blend with) use a simpler path: direction is set to a 70%
`flee_dir` / 30% existing-noisy-direction mix instead of the full
three-term blend. If the mean predator direction is degenerate (a
predator exactly on top of the prey), `flee_dir` falls back to a
random unit vector.

---

## 6. Predator Hunting (Vicsek P6.2)

Predators in Vicsek mode pursue the nearest prey within a detection
radius scaled off the same `vicsek_radius_predators`, and the
resulting heading is perturbed by hunting noise rather than pointed
exactly at the target:

```
detect_r = vicsek_detect_ratio · vicsek_radius_predators   // default 1.5× R_pred

for each predator:
    nearest_prey = argmin ‖p_pred − p_prey‖ over all prey     // min-image
    if nearest_prey within detect_r:
        target = normalize(p_nearest_prey − p_pred)
        u_hunt = normalize(target + vicsek_predator_noise_ratio · η̂)   // η̂ ~ N(0,I₃), normalized
    else:
        u_hunt = random unit vector on S²
```

`vicsek_predator_noise_ratio` (default 0.1) keeps the pursuit slightly
imprecise rather than a perfect beeline. When no prey are in range,
predators random-walk. When all birds are predators (no prey), the
flock skips all interaction and performs a pure random walk at
predator speed.

---

## 7. Summary of Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `predator_enabled` | false | Enable the predator extension |
| `predator_mode` | "autonomous" | Targeting mode |
| `predator_threat_radius` | 0.3 | Threat influence radius fraction |
| `predator_strength` | 1.0 | Threat force magnitude |
| `predator_momentum` | 0.5 | Speed and turn-rate modifier |
| `predator_acceleration` | 1.0 | Approach turn-rate modifier |
| `predator_split_gain` | 0.3 | Horizontal split force |
| `predator_vacuole_strength` | 0.0 | Vacuole formation strength |
| `predator_blackening_gain` | 0.6 | Cohesion/separation modulation gain |

Vicsek-mode fear/hunting fields (§5, §6) are separate from the
`predator_*` threat-agent fields above — they live in the vicsek
sub-config and only apply when `mode = "vicsek"` with mixed
predator/prey species:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `vicsek_radius_predators` | 80.0 | Detection radius `R_pred` for fear (§5) and hunting base radius (§6) |
| `vicsek_weight_afraid` | 3.0 | Flee-term weight `w_afraid` in the fear-blend (§5) |
| `vicsek_detect_ratio` | 1.5 | Multiplier on `R_pred` giving the predator's hunting detection radius (§6) |
| `vicsek_predator_noise_ratio` | 0.1 | Directional noise mixed into predator hunting headings (§6) |
| `vicsek_radius_avoid` | 1.0 | Same-species collision-avoidance radius (§1's asymmetric collision resolution, not covered above) |

---

## 8. Taxonomy

Predator-prey behaviour in pymurmur spans **two distinct architectural
mechanisms**, not one — a fact easy to miss since both are described in
this single document.

The first (§1–§4: the threat agent's FSM, its four-term force bundle,
panic speed ceiling, blackening) is the **Threat extension** — one of
pymurmur's roughly eight opt-in, per-step behavioural extensions. This
is a plugin family distinct from force-computation strategies: rather
than being the one active steering law for the whole flock, an
extension is a hook that runs every frame (if enabled) and either
mutates simulation state directly or injects an additional force
alongside whatever the active steering strategy already computed.
Multiple extensions can be active simultaneously and compose with each
other and with whichever steering strategy is running — the Threat
extension works identically whether the flock is in Reynolds-style,
angle-based, or any other steering mode, because it operates on the
force/state layer underneath the mode-specific logic.

The second (§5–§6: fear-weighted alignment, predator hunting) is
**not** a registry-dispatched plugin at all — it's inline logic
specific to one particular steering strategy (the alignment-coupling
family), conditionally executed only when that strategy detects mixed
predator/prey species in the flock. Unlike the Threat extension, this
logic cannot be reused by other steering strategies without being
reimplemented for each one; it is tightly coupled to that strategy's
own internal alignment/noise-blending machinery.

The practical consequence: a flock can have the Threat extension's
autonomous chasing agent active under *any* steering strategy, but
gets the fear-blend/hunting *behaviour* described in §5–§6 only under
the one strategy that implements it inline. A future "port fear/hunting
to every steering strategy" effort would mean re-deriving §5–§6's logic
per strategy, not flipping one shared flag.

## 9. Beyond pymurmur: Unimplemented Extensions

A few predator-prey mechanisms from the broader collective-motion
literature and comparable simulations are not implemented here:

- **Multi-predator inverse-lerp fear weighting with a full/zero fear
  band.** This document's fear formula (§5) is a simple linear ramp
  from the detection radius down to zero. Some implementations instead
  use two radii — a `fullFearRadius` beyond which fear is exactly 0
  and a `fearRadius` inside which fear is exactly 1, with an
  inverse-lerp interpolation on *squared* distance in between — giving
  a genuine dead zone of total calm at range, rather than this
  document's approach where fear only reaches exactly 0 at the single
  boundary distance. Both approaches already weight multiple nearby
  predators by summing weighted contributions rather than picking one
  nearest predator, so the structural difference is specifically the
  two-radius dead-zone shape, not the multi-predator averaging itself.
- **Selfish-herd / confusion-effect modelling.** Real predator evasion
  research models two distinct benefits of grouping under attack: the
  geometric "selfish herd" effect (each individual minimizes its own
  domain of danger by moving toward denser neighbours, distinct from
  ordinary cohesion) and the "confusion effect" (a predator's attack
  success rate drops as target density/motion complexity increases,
  usually modelled as a per-predator success-probability penalty
  proportional to local prey density). Neither is implemented — this
  document's blackening mechanism (§4) increases cohesion near a
  threat but doesn't model confusion as a probability of the predator
  actually landing a strike, and doesn't give individuals an incentive
  to specifically minimize their own exposed "domain of danger."
- **Visual-looming-based threat detection**, where a predator is
  detected (and its urgency judged) by the rate of angular expansion
  of its silhouette on the prey's retina, rather than by raw Euclidean
  distance as this document's `nearest_pred_dist`/mean-distance fear
  formulas do. Looming-based response is closer to how real animals
  actually detect approach threats (a fast-approaching predator looms
  even at a large absolute distance) and would couple naturally with
  this document's occlusion/opacity machinery if extended to a
  predator-specific silhouette, but is a materially different signal
  than a plain distance gate.
- **Post-attack recovery/cooldown state.** Currently the threat FSM
  (§1.1) alternates purely between Approach and Egress based on
  distance and closing angle — there is no explicit "startled" or
  "scattering" state for prey that persists briefly after a close call
  independent of the predator's current distance, the way some
  behavioural models give prey a decaying elevated-alertness period
  after an attack rather than alertness being a pure function of
  instantaneous predator proximity.
