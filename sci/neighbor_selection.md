# Neighbor Selection

How a flocking simulation chooses which other boids to consider when
computing separation, alignment, and cohesion. This choice shapes
emergent behavior as much as the force formulas themselves — the same
separation/alignment/cohesion math produces qualitatively different
flocks depending on whether "neighbor" means "everyone within a fixed
distance," "my 7 closest boids regardless of distance," or "everyone in
front of me."

---

## 1. Strategies Surveyed

| Strategy | Description |
|----------|-------------|
| Single radius | One Euclidean distance threshold shared by all three rules (separation, alignment, cohesion). The most common approach by far. |
| Multi-radius | Separate thresholds per rule — typically a tight separation radius and wider alignment/cohesion radii, reflecting that personal space is smaller than social awareness. |
| Radius + cap | A distance threshold combined with a hard cap on the *number* of neighbors considered, even if more fall within range — usually the closest N are kept. |
| k-NN topological | A fixed neighbor *count* (e.g. exactly 6 or 7 closest boids) with no distance threshold at all — a boid at the edge of a sparse flock still has exactly that many neighbors, just farther away. |
| FOV cone | A directional constraint layered on top of distance — only boids within a forward-facing cone count as neighbors, modeling the fact that animals can't see behind themselves. |
| Global | No filtering at all — every boid considers every other boid, producing a single world-coherent flock rather than local sub-flocks. |
| Occlusion-based | Neighbors are filtered by line-of-sight visibility (closer boids can block the view of farther ones), not just distance or count. |

## 2. Radius-Based Selection (the Dominant Approach)

The overwhelming majority of surveyed implementations use simple
Euclidean distance thresholds — cheapest to implement and cheapest to
evaluate per pair. Multi-radius variants (different thresholds for
separation vs. alignment/cohesion) are notably more common than
single-radius ones, consistently reflecting the same biological
intuition: personal space (separation) should be tighter than social
awareness (alignment/cohesion). Ratios between the tightest and widest
radius vary enormously across implementations — from roughly 1.5× up
to 7-8× in the most extreme cases.

## 3. FOV Cone

A minority of implementations add a genuine directional constraint.
Two distinct patterns appear:

**Fixed cone with dynamic vision distance.** The cone half-angle is
fixed, but the *distance* a boid can see adapts every frame based on
how many neighbors it currently has in view — isolated boids
progressively widen their vision distance to find flockmates, while
crowded boids narrow it. This produces a feedback loop that keeps
local neighbor counts roughly stable regardless of overall density.

**Forward hemisphere.** A simpler fixed 90°-half-angle cone (the
entire forward-facing hemisphere) combined with an ordinary distance
threshold — a boid simply cannot see anything behind itself, full stop,
with no adaptive component.

**Per-interaction cones.** The most granular variant applies an
independent FOV cone to each of separation, alignment, and cohesion
separately, rather than one shared cone for all three — allowing, for
example, a narrow forward-only separation cone (only react to threats
ahead) combined with wide-open cohesion (stay aware of the whole flock
including boids behind).

## 4. k-NN Topological Selection

Rather than a distance radius, a fixed *number* of nearest boids is
always selected regardless of how far away they are:

```
neighbours = the k closest boids by distance, always exactly k
             (excluding self)
```

A boid in a dense cluster sees only its k nearest, tightly-packed
neighbors; a boid isolated at the flock's edge still sees exactly k
neighbors, just much farther away. The effective perception distance
adapts implicitly to local density with zero explicit logic for it —
this is the behavior real starling flocks are understood to exhibit
(each bird tracking a roughly constant number of neighbors, historically
cited as 6–7, independent of flock density).

A pure distance-radius fallback sometimes exists alongside k-NN
selection purely as a performance safety net (bounding the search when
the flock is extremely sparse) but is rarely, if ever, actually
triggered in practice.

## 5. Capped Radius

A middle ground between pure radius and pure k-NN: all boids within a
distance threshold are candidates, but only the closest N of them
actually count, discarding the rest even though they're technically in
range. Unlike pure k-NN, a capped-radius boid with fewer than N boids
in range simply gets fewer neighbors rather than reaching further out
to find more.

## 6. Global Selection

The simplest possible rule: no filtering at all. Every boid contributes
to every other boid's alignment and cohesion computation, typically via
a single precomputed flock-wide average velocity and centroid rather
than a genuine per-pair loop. This produces a single world-coherent
swarm — every boid steers toward the *same* global average — rather
than the local sub-flock structure that distance- or count-limited
selection naturally produces. Separation is usually kept local even in
otherwise-global implementations, since unlimited-range separation
would make no physical sense.

## 7. Patterns

- **Radius-based selection is the overwhelming default** across
  surveyed implementations — simplest to implement, cheapest per-pair
  to evaluate.
- **Multi-radius is more common than single-radius**, consistent with
  the biological intuition that personal space and social awareness
  operate at different scales.
- **Directional (FOV) constraints are comparatively rare** — they're
  biologically well-motivated but add a per-pair dot-product cost that
  pure-distance approaches skip.
- **Pure topological (k-NN, no distance term at all) selection is the
  rarest pattern of all**, despite being the one most directly
  supported by real starling-flock field research — most
  implementations favor the simplicity of a distance threshold even
  when citing the same biological research as motivation for a
  specific neighbor *count*.
- **Global averaging is a fundamentally different regime** from every
  other strategy — it produces one coherent whole-flock swarm rather
  than emergent local structure, behaving more like a single coupled
  particle system than a flock of individually-reacting agents.

---

## 8. pymurmur's Approach: Per-Mode Selector Registry

pymurmur does not use one universal neighbor-selection strategy —
different force modes have different needs, so three distinct
strategies are registered under one shared dispatch pattern (an ABC
plus a name-keyed lookup table), each wrapping the specific query logic
its owning mode already needs:

### 8.1 Hybrid Selector

Used by the Reynolds-style spatial mode. Combines a spatial-index k-NN
query with a metric distance filter and an optional topological cap —
genuinely hybrid in the taxonomic sense above (§1): a batched k-nearest
query up to a configured neighbor count, further filtered to only those
within a visual-range distance, with the closest survivors kept if more
than the target count remain after filtering. Supports switching
between purely metric, purely topological, hybrid, or unfiltered
behavior at runtime via a configurable filter mode, and is aware of a
predator-perception-boost multiplier (threatened prey can temporarily
see farther).

### 8.2 Topological-Visibility Selector

Used by the projection mode. A pure k-NN topological query — a fixed
count of nearest neighbors per bird via repeated spatial-index queries,
with unfilled slots (when fewer than the target count exist) marked
with a sentinel value rather than left undefined. This is the closest
match to §4's pure topological pattern in the survey — no distance
threshold at all, always exactly the configured count wherever
possible.

### 8.3 Ball-Tree Radius Selector

Used by the Vicsek-alignment mode. A pure metric-radius query via a
ball-tree, producing a sparse adjacency structure (which pairs of birds
are within radius of each other) plus a per-bird neighbor count,
restricted to birds with a well-defined heading (near-zero-velocity
birds are excluded as neighbors, since their direction is undefined).
This is the closest match to §2's single-radius pattern in the survey.

### 8.4 Modes With No Registered Selector

Two force modes intentionally have nothing registered here. The
angle-steering mode has its own fused, per-bird neighbor-banding logic
built directly into its steering loop rather than a separably-callable
selector — pulling it out cleanly would require restructuring that
loop, deliberately not done as part of formalizing this registry.
The influencer, MARL, and field modes don't perform index-based
neighbor queries at all — they use global (all-active-birds) formulas
or bespoke per-bird targeting instead, so neighbor *selection* in the
sense described here simply doesn't apply to them.

---

## 9. Taxonomy

pymurmur's neighbour-selection system is the NeighborSelector plugin
family — a per-strategy dispatch registry with 3 entries (§8.1-§8.3),
the same architectural pattern (an ABC plus a name-keyed lookup table,
chosen at runtime instead of a hardcoded chain) used for every other
swappable computation: force-mode selection, domain-edge handling,
obstacle avoidance, post-integrate speed enforcement, spatial-index
selection, kernel dispatch for separation/alignment/cohesion, and
noise injection are each their own such registry.

This family sits one layer above the SpatialIndexStrategy family (a
different, lower-level registry): a NeighborSelector strategy decides
*which birds count as neighbours and under what rule* (radius,
topological count, hybrid, FOV-filtered), while the spatial-index
layer underneath it decides *how the underlying position query is
executed efficiently* (grid cells, k-d tree, or an automatic choice
between them). The hybrid selector (§8.1), for instance, issues its
k-NN query through whichever spatial index is currently active without
needing to know which one that is — the two registries compose rather
than overlap.

## 10. Beyond pymurmur: Unimplemented Extensions

Selection mechanisms from §1's survey not present in pymurmur's
3-strategy registry:

- **Dynamic vision-distance feedback** (§3's fixed-cone-with-adaptive-
  distance pattern) — a fixed FOV half-angle combined with a
  perception *distance* that expands when a bird has too few
  neighbours in view and contracts when it has too many, creating a
  feedback loop that keeps local neighbour counts roughly stable
  regardless of overall flock density. pymurmur's closest analogue is
  the neighbour-count-adaptive *speed* law (isolated birds fly faster
  to rejoin the flock) rather than an adaptive *perception radius* —
  the two solve a similar problem (isolation) through different
  mechanisms (speed vs. sight). Adding true adaptive vision distance
  would mean a new NeighborSelector strategy carrying per-bird
  persistent state (the current vision distance) across frames, which
  none of the 3 existing selectors need.
- **Forward-hemisphere-only selection** (§3's simple 90°-cone
  pattern) — a selector with no metric or topological component at
  all, purely "everything in front of me, regardless of distance or
  count." pymurmur's hybrid selector (§8.1) supports per-interaction
  FOV cones as a *filter* layered on top of a metric+topological base,
  but has no mode where FOV is the *only* selection criterion.
- **Capped-radius selection as a first-class strategy** (§5) —
  pymurmur's hybrid selector already behaves this way internally (a
  metric filter followed by a topological cap on the survivors), but
  it isn't exposed as an independently selectable strategy distinct
  from the full hybrid behaviour — a bird-mode wanting *only*
  capped-radius selection (no separate metric-only or
  topological-only variants) currently has no way to request that
  narrower behaviour directly.

## 11. Summary of Parameters

| Parameter | Default | Description |
|-----------|---------|--------------|
| `neighbor_filter` (spatial mode) | "hybrid" | Which sub-behavior the hybrid selector uses: metric, topological, hybrid, global, or none |
| `influence_count` | 7 | Target/cap neighbor count for the hybrid and topological-visibility selectors |
| `visual_range` | 70.0 | Metric distance limit for the hybrid selector's filter step |
| `sigma` (projection mode) | 4 | Topological neighbor count for the topological-visibility selector |
| `vicsek_radius_influence` | 5.0 | Metric radius for the ball-tree radius selector |
