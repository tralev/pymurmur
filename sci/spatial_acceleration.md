# Spatial Acceleration

This document defines the six spatial acceleration techniques used across
22 surveyed flocking implementations — how each avoids the O(N²) all-pairs
neighbour query by organizing birds in space.

---

## 1. O(N²) All-Pairs

Every bird checks every other bird.  No spatial organization, no
acceleration structure.  Simplest to implement; expensive at scale.

```
for each bird i:
    for each bird j ≠ i:
        if distance(i, j) < radius:
            process neighbour
```

**Used by:** 16 of 22 implementations (§01, §02, §04, §05, §06, §07a,
§08, §09, §10a, §11, §12, §13, §14, §15, §16, §18).  Adequate for
N < 500; dominates runtime above N ≈ 1,000.

---

## 2. Grid Cells (Spatial Hash)

The simulation volume is divided into a regular grid of cells.  Each
bird is filed into its cell; queries check only the 27 neighbouring
cells (3×3×3 in 3D, 3×3 in 2D):

```
cell_size = max(perception_radius, boid_size × 2)
cell = floor(position / cell_size)
file bird into hash_map[cell]

for each bird i:
    for cell_offset in 27_neighbour_offsets:
        for bird j in hash_map[cell_i + cell_offset]:
            if distance(i, j) < radius:
                process neighbour
```

**Characteristics:** O(N) per frame with constant per-query cost when
cell density is bounded.  Best for N < 5,000.  Degrades when birds
cluster densely (many birds per cell → per-query cost rises).

**pymurmur implementation:** `SpatialHashGrid` — modulo-wrapped cell
keys for toroidal domains, incremental rebuild (only re-files birds
that cross cell boundaries), dictionary-based storage.

**Used by:** pymurmur (default for small N), murmuration (§21), §15,
§17 from the survey.

---

## 3. cKDTree (K-D Tree)

A balanced binary tree partitioning space along alternating axes.
Queries are O(log N) per bird:

```
tree = cKDTree(positions)
neighbours = tree.query_ball_point(position, radius)
```

**Characteristics:** O(N log N) build, O(log N) per query.  Handles
non-uniform density well — the tree depth adapts to local clustering.
More expensive to build than a hash grid but more robust to clustering.

**pymurmur implementation:** `KDTreeIndex` — used when N exceeds a
threshold or when the configuration requests it explicitly.  Supports
parallel queries via `workers=-1`.

**Used by:** pymurmur (large N, projection mode), §07b from the survey.

---

## 4. Grid + KDTree (Adaptive)

Automatically selects between grid cells and k-d tree based on flock
size and density.  The grid is faster to build and cheaper per-query
for small N; the tree scales better for large N:

```
if N < threshold:     use SpatialHashGrid
else:                 use KDTreeIndex
```

**pymurmur implementation:** The spatial index strategy plugin
(`SpatialIndexStrategy`) auto-selects based on `spatial_index` config
field.  Options: `"hash_grid"`, `"kdtree"`, or `"auto"` (default —
auto-switches at N ≈ 5,000).

**Used by:** pymurmur (§22).

---

## 5. Hashed Grid + Bitonic Sort (§20)

A GPU-friendly variant: birds are sorted by cell index using a bitonic
sort on the GPU, then neighbour lookups traverse contiguous cell
segments in the sorted array.  The sort is O(log² N) parallel steps.

```
// GPU dispatch:
sort birds by cell_index (bitonic sort, parallel)
for each cell:
    for each bird in cell_and_neighbours:
        process neighbour
```

**Characteristics:** Fully GPU-resident — no CPU↔GPU transfer per
frame.  The bitonic sort is the only O(log² N) parallel sorting
network available on all GPU generations.

**Used by:** §20 from the survey.  Not implemented in pymurmur.

---

## 6. Parallel Prefix-Sum Reduction (§10b)

All birds contribute to a single global average via GPU parallel
reduction.  No per-bird neighbour queries at all — a compute shader
accumulates the sum of all positions and velocities in O(log N) steps:

```
// Parallel prefix-sum (log N steps):
for offset = 1; offset < N; offset <<= 1:
    if thread_id >= offset:
        sum[thread_id] += sum[thread_id − offset]
```

**Characteristics:** Produces global (not local) flocking — every bird
steers toward the same world centre of mass.  Extremely fast on GPU but
produces qualitatively different behaviour than local interactions.

**Used by:** §10b from the survey.  Not implemented in pymurmur.

---

## 7. Summary

| Technique | Complexity | Best for | In pymurmur? |
|-----------|:---------:|----------|:------------:|
| O(N²) all-pairs | O(N²) | N < 500 | ❌ (not used) |
| Grid cells (spatial hash) | O(N) | N < 5,000 | ✅ (default for small N) |
| cKDTree (k-d tree) | O(N log N) | N > 5,000 | ✅ (large N, projection) |
| Grid + KDTree (adaptive) | O(N) or O(N log N) | All N | ✅ (auto-select) |
| Hashed grid + bitonic sort | O(log² N) GPU | Very large N | ❌ (GPU-only) |
| Parallel prefix-sum | O(log N) GPU | Global flocking | ❌ (GPU-only) |

---

## 8. Taxonomy

This document covers the spatial-index-selection plugin family — a
per-strategy dispatch registry distinct from, and smaller than, the
7-entry force-computation family: 4 entries (auto, hash_grid, kdtree,
none). It is one of roughly seven "other computational plugin"
families that sit alongside force-mode selection, each a registry
serving one specific computation rather than a full force law:
boundary handling (5 strategies), neighbour-selection filtering (3
strategies), obstacle avoidance (1 strategy currently), speed
enforcement (6 strategies), kernel dispatch for separation/alignment/
cohesion (11/4/3 entries), noise injection (5 strategies), and this
spatial-index family.

Unlike the force-mode choice, which every run must pick exactly one
of, spatial-index selection is an implementation detail invisible to
the force math itself — it applies transparently underneath whichever
force mode needs neighbour queries (the modes described in §2 and §3
above), swappable without changing any steering behaviour, only its
computational cost profile.

## 9. Beyond pymurmur: Unimplemented Extensions

- **Hashed grid + bitonic sort (GPU).** §5 above already surveys this
  — a GPU-resident cell-sort acceleration structure requiring no
  per-frame CPU↔GPU transfer. Not implemented; the current grid and
  tree implementations are both CPU-resident.
- **Parallel prefix-sum reduction (GPU).** §6 above — relevant only to
  modes using a global (whole-flock) neighbour filter, where it would
  replace an O(N) CPU reduction with an O(log N) GPU one without
  changing the resulting behaviour.
- **Verlet/skin-radius neighbour lists.** A common technique from
  molecular-dynamics simulation: build the neighbour list using a
  slightly larger "skin" radius than the actual interaction radius, so
  the list stays valid for several frames without a full rebuild —
  only refreshing when a bird has moved more than half the skin
  distance since the last build. Distinct from the current incremental
  hash-grid rebuild (which re-files individual birds as they cross
  cell boundaries every frame, not the neighbour list itself); would
  need tracking cumulative per-bird displacement since the last full
  build.
