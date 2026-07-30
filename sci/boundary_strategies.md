# Boundary Strategies

How a flocking simulation keeps boids within (or lets them leave) its
simulation volume. Strategies fall into three categories: **soft**
(a force nudges velocity, no position override), **hard** (position is
directly teleported or clamped), and **hybrid** (both mechanisms
combined).

---

## 1. Soft Strategies (Force Nudges Velocity)

No position override — the boid's velocity is steered away from the
edge, and it is still possible (briefly) to overshoot the boundary.

| Strategy | Mechanism |
|----------|-----------|
| Steer-back | A constant turn factor is added to velocity whenever the boid is within a margin distance of an edge. |
| Center attraction | A constant pull toward the origin, amplified on one axis to discourage motion along it (e.g. suppressing altitude drift). |
| Obstacle-based only | No world boundary at all — an open, effectively infinite volume, with containment implemented entirely via obstacle avoidance rather than a domain edge. |
| Linear wall ramp | The steering force ramps linearly from 0 at a configurable "avoid distance" up to full strength exactly at the wall. |
| 1/d wall repulsion | A six-face force that grows as `weight × wallDistance / |dist|` — asymptotic, approaching infinity as distance to the wall approaches zero. |
| Spherical spring | `normalize(toCenter) × (sphereRadius − distance)` — a linear spring pulling inward once outside a spherical boundary. |
| Constant cage push | A fixed-magnitude push (e.g. ±0.5) on whichever axis brings the boid within a fixed distance of a cage wall. |
| Home force with dead zone | Attraction toward the origin that is completely inactive inside an inner-radius dead zone, only engaging once the boid strays far enough out. |

## 2. Hard Strategies (Position Directly Modified)

| Strategy | Mechanism |
|----------|-----------|
| Wrap-around (conditional) | `if pos > width: pos = 0` — a toroidal teleport applied independently on each edge. |
| Wrap-around (modulo) | `pos = MODULO(pos + v·dt, size)` — true mathematical modulo, computed in one step rather than a conditional. |
| Zone clamp | Position is clamped inside a slab, box, or sphere defined in a *local* coordinate space, requiring a world↔local matrix transform. |
| Wrap via if/else chain | A chain of `if`/`else if` tests per edge; wrapping and normal movement are mutually exclusive for that frame (the boid's position jumps rather than continuing to integrate). |

## 3. Hybrid Strategies (Soft Force + Hard Override)

| Strategy | Mechanism |
|----------|-----------|
| Spherical confinement + behavioral override | Soft: an inward push proportional to distance once outside a radius. Hard: flocking acceleration is *skipped entirely* while outside the radius — boundary avoidance completely overrides normal behavior until the boid re-enters. |
| Wrap + six-wall force | Hard: wrap on all 6 box faces. Soft: an inverse-square repulsion from each wall is added to acceleration before the wrap is applied, so the wrap is rarely actually triggered in practice. |
| Wrap + asymmetric wall force | Hard: wrap on one axis only. Soft: one wall (e.g. the ground) is strongly avoided, the others weakly, and one face may be left fully open. |
| Reflective + wall force | Hard: position is clamped to the boundary and the offending velocity component is mirrored (`v ← −v`). Soft: an additional 1/d repulsion is applied after force clamping (GPU variant only). |
| Wrap + edge state machine | Hard: screen/volume wrap with a buffer zone. Soft: a three-level state machine (idle → turning → turning harder) applies exponential attenuation of flocking forces, half-speed turns, and a hard turn-angle saturation cap as the boid approaches the edge. |
| Edge-steer + optional wrap | Soft: within a margin, flocking is overridden and heading is set to face directly away from the nearest edge, with turn rate ramping up the closer the boid gets. Hard: an optional wrap mode teleports to the opposite edge instead of steering away. |
| Multi-mode (toroidal default) | Default: toroidal wrap on all axes (`pos %= size`). Also offers a margin mode (soft nudge + hard clamp combined) and a fully open mode (no enforcement at all) as alternatives. |

## 4. No Boundary Handling

Some implementations have no boundary logic at all — either because
they're a minimal steering demo with too few boids to need one, because
target-seeking behavior alone keeps boids loosely centered without any
explicit containment, or because no flocking math exists yet (a
renderer-only skeleton).

## 5. Patterns Across Implementations

- **Wrap-around is the single most common hard strategy** — roughly
  40% of surveyed implementations wrap at least one axis, usually
  because it's the cheapest way to guarantee boids never truly leave
  the visible volume.
- **Soft-only approaches dominate in 3D** — most pure-soft
  implementations are 3D; hard position clamping is comparatively rarer
  in 3D because it requires either a spherical-distance check or a full
  local-space matrix transform, both more work than a simple 2D
  rectangle clamp.
- **Hybrid strategies appear in feature-rich implementations** — they
  combine the reliability of a hard teleport/clamp (boids provably
  cannot escape) with the natural look of a soft steering force (no
  visible "pop" at the boundary under normal conditions).
- **A full local-space containment system** (slab, box, or sphere
  clamped via a world↔local matrix transform) is the most sophisticated
  hard-boundary approach surveyed.
- **A multi-level edge state machine** with exponential force
  attenuation and speed modulation is the most elaborate soft-boundary
  approach surveyed — it produces visibly organic turning behavior
  rather than an abrupt correction.
- **Very few implementations have genuinely no boundary at all** while
  still doing real flocking — those that do rely entirely on target
  attraction to keep the flock loosely centered.

---

## 6. Boundary Modes

Five boundary strategies exist as interchangeable, hot-swappable
options — a per-run choice, not a fixed design decision. All five share
one calling contract: given positions, velocities, an active mask, the
domain dimensions, a sphere radius and avoidance-factor parameter, and
an optional domain centre, each strategy mutates positions/velocities
in place for whichever birds it applies to.

### 6.1 Toroidal (Default)

Positions wrap modulo the domain size on all three axes independently:

```
p_x ← p_x mod width
p_y ← p_y mod height
p_z ← p_z mod depth
```

A pure hard strategy — no velocity change, no soft nudge. A bird
exiting one face reappears at the opposite face travelling in the same
direction. This is the biologically-neutral default: it removes edge
effects entirely so the flock's own dynamics (not the boundary) shape
its behavior.

### 6.2 Open

No enforcement at all — birds may leave the nominal domain freely, and
neither position nor velocity is touched. Useful for open-space studies
where the "domain" is only a reference frame for initialization and
camera framing, not a real constraint.

### 6.3 Margin (Soft)

Per axis, a bird within `margin` units of either wall receives a
velocity nudge proportional to how deep it is into the margin band,
and its position is clamped so it cannot cross the wall entirely:

```
for each axis, size in {width, height, depth}:
  if p < margin:            v += avoidance_factor · (margin − p) / margin
  if p > size − margin:     v −= avoidance_factor · (p − (size − margin)) / margin
  p ← clamp(p, 0, size)
```

A hybrid in spirit — the velocity push is soft (proportional, not a
hard reflect), but the trailing position clamp is a hard backstop
against ever numerically exceeding the domain.

### 6.4 Sphere (Hard)

A hard spherical boundary centred on the domain centre `C` (not the
origin). Birds beyond `sphere_radius` are projected back exactly onto
the sphere's surface and given an inward velocity correction
proportional to how far they overshot:

```
r = ‖p − C‖
if r > R:
  r̂ = (p − C) / r
  p ← C + r̂ · R                                    // hard projection to surface
  v ← v − r̂ · avoidance_factor · (r − R)            // inward correction, proportional to overshoot
```

### 6.5 Sphere-Soft (Asymptotic)

The soft counterpart to §6.4 — never hard-projects position, only
applies an increasingly strong inward velocity push as a bird
approaches or crosses the boundary, growing asymptotically as the gap
to the surface shrinks:

```
gap = max(R − r, 0.05·R)                            // clamped to avoid divide-by-zero
push = avoidance_factor · R / gap
v ← v − r̂ · push                                    // for birds with r > 0.9·R
```

Because `push` grows as `1/gap`, a bird deep past the boundary
experiences an increasingly forceful correction without ever having its
position hard-clamped — it can briefly overshoot and is smoothly pulled
back rather than snapped to the surface. The activation threshold
(`r > 0.9·R`) means the soft push engages slightly before the nominal
boundary is reached, giving birds a chance to turn away before crossing
it at all.

---

## 7. Summary of Parameters

| Parameter | Default | Description |
|-----------|---------|--------------|
| `boundary_mode` | "toroidal" | Which of the 5 strategies is active |
| `width`, `height`, `depth` | domain-dependent | Box dimensions for toroidal/open/margin |
| `margin` | 50.0 | Distance from a wall at which the margin strategy engages |
| `sphere_radius` | domain-dependent | Radius for sphere / sphere_soft |
| `avoidance_factor` | domain-dependent | Strength of the velocity correction for margin / sphere / sphere_soft |
| `center` | domain centre | Sphere centre for sphere / sphere_soft (not the origin) |

---

## 8. Taxonomy

The 5 strategies in §6 form the BoundaryMode plugin family — a
per-strategy dispatch registry structurally identical to the
force-computation family, but serving a narrower, orthogonal concern:
domain-edge handling rather than a full steering law. It is one of
roughly seven "other computational plugin" families that sit alongside
force-mode selection (neighbour-selection filtering, obstacle
avoidance, speed enforcement, kernel dispatch, noise injection, and
spatial-index selection are the others).

The key architectural property is orthogonality: boundary handling is
dispatched from the position/velocity integration step, completely
independent of which of the 7 force modes produced the acceleration
that frame. Any force mode can combine with any boundary mode — a
projection-mode flock can run inside a sphere, a field-mode flock can
wrap toroidally, and so on — because the boundary strategies operate
only on positions and velocities after the force computation is
already done, with no knowledge of which force law produced them.

## 9. Beyond pymurmur: Unimplemented Extensions

- **Reflective boundary.** A hard box boundary with velocity
  mirroring on contact — when a bird crosses a face, its position is
  clamped to the wall and the offending velocity component is negated:

  ```
  if p.x > +bound.x:  p.x = +bound.x;  v.x = −v.x
  if p.x < −bound.x:  p.x = −bound.x;  v.x = −v.x
  // same for y, z
  ```

  Confirmed not implemented — none of pymurmur's 5 boundary modes
  perform velocity mirroring; the closest analogues (sphere, margin)
  redirect velocity toward the interior via a proportional correction
  rather than an exact reflection. Adding it would mean a 6th
  `BoundaryMode` registered under a name like `"reflect"`.
- **Local-space zone clamp (slab/box/sphere via matrix transform).** A
  containment system where the boundary shape is defined in a
  *local* coordinate frame (potentially rotated or offset from the
  simulation's world axes) and enforced via a world↔local matrix
  transform each frame, rather than pymurmur's world-axis-aligned
  toroidal/margin box or origin-relative sphere. Would allow, for
  example, an oriented capsule or an arbitrarily-rotated bounding box
  as the containment volume.
- **Multi-level edge state machine.** Rather than pymurmur's single
  linear turn-rate ramp (§6.3's margin mode), a three-level state
  machine (idle → turning → turning-harder) with exponential force
  attenuation, half-speed turns at the middle level, and a hard
  turn-angle saturation cap at the outermost level — producing a more
  visibly graduated, organic approach-and-turn behaviour than a single
  linear ramp between two rates.
