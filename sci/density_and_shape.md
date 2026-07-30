# Density, Shape, and Motion Metrics

This document defines the spatial-structure observables computed from
flock position snapshots: density scaling analysis (power-law fit,
toroidal vs open boundaries), convex-hull density, PCA shape metrics
(aspect ratio, thickness ratio), gyration radius, robust number
density, pairwise diameter R_max, suggested neighbour count from
shape, jamming index, and normalized angular momentum.

---

## 1. Density Scaling Analysis

Density scaling measures how local flock spacing changes with
population size `N`.  The key question: does a flock of 800 birds
have the same nearest-neighbour spacing as a flock of 50, or does
density change with scale?

### 1.1 Sweep Protocol

For each population size `N` in a sweep (default: [50, 100, 200, 400,
800]), a headless spatial-mode simulation is run to steady state under
both toroidal and open boundary conditions.  The domain size scales
with `N` to maintain approximately constant volume per bird:

```
domain_size = max(80, N^(1/3) · 15)
```

This keeps the nominal density roughly constant across the sweep,
so any observed change in spacing is a genuine scaling effect rather
than a domain-size artifact.

After a settling period (default: 50% of total steps), the median
`local_spacing` (the 7th-neighbour distance, a robust density proxy)
is recorded from the steady-state frames.

### 1.2 Power-Law Fit

The spacing-vs-N relationship is modelled as a power law:

```
spacing(N) = C · N^β
```

Taking logs:

```
log(spacing) = β · log(N) + log(C)
```

A linear regression on the log-log data estimates the exponent `β` and
the goodness-of-fit `R²`:

```
β = Σ_i (x_i − x̄)(y_i − ȳ) / Σ_i (x_i − x̄)²
```

where `x_i = log(N_i)` and `y_i = log(spacing_i)`.

```
SS_res = Σ_i (y_i − ŷ_i)²         // residual sum of squares
SS_tot = Σ_i (y_i − ȳ)²           // total sum of squares
R² = 1 − SS_res / SS_tot          // coefficient of determination
```

Requires at least 3 valid data points for the fit.

### 1.3 Theoretical Ideal

For a uniform ideal gas in `d` dimensions, density scales as:

```
ρ(N) ∝ N^(−1/(d−1))
```

In 3D (`d = 3`), this gives `ρ ∝ N^(−1/2)`, meaning spacing (which is
`ρ^(−1/3)`) scales as:

```
spacing(N) ∝ N^(1/6) ≈ N^0.167
```

The measured `β` for toroidal boundaries is typically close to 0
(constant spacing, birds maintain a fixed personal distance regardless
of `N`), while open boundaries produce a non-zero `β` (birds spread
out in open space, reducing local density at larger `N`).  The
toroidal result is the biologically relevant one — starlings in a
murmuration maintain constant local density independent of the total
flock size.

### 1.4 Size Scaling and External Opacity

Alongside the spacing power law, the same sweep independently fits a
**size** power law — how the flock's overall spatial extent (its
gyration radius `Rg`, §5) grows with `N` — using the same log-log
linear-regression machinery as §1.2:

```
Rg(N) = C · N^β_size
```

For an ideal 3D gas at constant density, volume scales as `N`, so
radius scales as `N^(1/3)`; but a flock that also compacts as it
grows (as the spacing power law above can show) instead follows
`Rg ~ N^(1/2)` — this implementation's `ideal_size_exponent = 0.5` is
the theoretical reference line the measured `β_size` is compared
against, not a strict physical requirement.

The sweep also tracks the median **external 2D silhouette opacity**
(`theta_ext` — the fraction of the flock's projected disk covered by
birds, via disk rasterisation) at each `N`, as a raw per-population
median rather than a separate power-law fit — a companion signal for
whether the flock visually thins out or stays opaque as it scales.

---

## 2. Convex-Hull Density

Density is computed as birds per unit volume using the convex hull of
the flock:

```
ρ = N / V_hull
```

where `V_hull = ConvexHull(positions).volume` is the volume of the
smallest convex polyhedron containing all N birds.

Requires at least 4 non-coplanar points.  Returns 0 for degenerate
configurations (coplanar, colinear, or fewer than 4 birds).

The convex hull is computed via scipy's `ConvexHull`, which uses the
Qhull algorithm (O(N log N) expected, O(N²) worst-case).  Hull volume
is a more robust density estimator than e.g. bounding-box volume
because it conforms to the actual flock shape rather than the
axis-aligned extents.

### 2.1 Density Autocorrelation Time τ_ρ

The density autocorrelation time measures how long the flock's spatial
density pattern persists before decorrelating — the characteristic
timescale over which density "forgets" its previous configuration.

Two computation methods exist:

**Hull-density method** (used in the metrics collector at detail_level ≥ 2):

Density samples `ρ(t)` from the convex-hull density at each metrics
interval (default every 20 frames) are accumulated in a ring buffer.
The autocorrelation function is:

```
r(lag) = ⟨(ρ_t − ρ̄)(ρ_{t+lag} − ρ̄)⟩ / Var(ρ)
```

The autocorrelation time is the integrated correlation:

```
τ_ρ = interval · (0.5 + Σ_{lag ≥ 1} r(lag))
```

The sum terminates at the first lag where `r(lag) ≤ 0`, or at
`lag = 0.25 · buffer_size` (a cap that keeps τ_ρ finite on
slowly-varying series that never cross zero).  `interval` is the
number of frames between consecutive density samples, converting the
sample-step units to frame units.

Constant-density series (zero variance) produce τ_ρ = 0.

**Histogram method** (alternative, for histogram-based density):

Pearson correlation `r(τ)` is computed between density histograms at
lag τ.  An exponential decay model `r(τ) ≈ exp(−τ / τ_ρ)` is fit to
extract the characteristic timescale via weighted median of per-lag
estimates:

```
τ_ρ = median_{lag} ( −lag / log(r(lag)) )
```

Requires at least 4 histogram snapshots and at least 2 lags with
positive correlation.  Returns 0 if insufficient data or no positive
correlations exist.

τ_ρ is reported in frame units — divide by the frame rate to convert
to seconds.

---

## 3. PCA Shape Metrics

Flock shape is characterised by principal component analysis of the
`N×3` position matrix.

### 3.1 Covariance and Eigenvalues

```
C = (1/N) · (P − P̄)ᵀ · (P − P̄)       // 3×3 covariance matrix
```

The eigenvalues `λ₁ ≥ λ₂ ≥ λ₃` of `C` describe the variance along the
three principal axes.  For a sphere, λ₁ ≈ λ₂ ≈ λ₃.  For an elongated
cigar, λ₁ ≫ λ₂ ≈ λ₃.  For a flat pancake, λ₁ ≈ λ₂ ≫ λ₃.

### 3.2 Aspect Ratio (Elongation)

```
aspect = sqrt(λ₁ / λ₃)     ≥ 1
```

- aspect ≈ 1: spherical (equal extent in all directions).
- aspect ≈ 3: moderately elongated.
- aspect ≫ 10: highly elongated (line-like).

Degenerate case: if λ₃ ≈ 0 but λ₁ > 0 (flat or linear flock),
aspect → +∞.

### 3.3 Thickness Ratio (Flatness)

```
thickness = sqrt(λ₃ / λ₁)     ∈ (0, 1]
```

- thickness ≈ 1: spherical.
- thickness ≈ 0.4: the Young et al. (2013) "fully 3D" transition
  threshold — below this, flocks behave like quasi-2D sheets for
  robustness purposes.
- thickness → 0: flat plane or line.

### 3.4 Fully-3D Regime Classifier (A16)

```
is_fully_3d = (thickness ≥ 0.4)
```

Young et al. (2013) report that robustness (H₂) stops responding to
increased thickness once the flock crosses ~0.4 — beyond this
threshold, adding depth does not meaningfully improve consensus.
Below it, the flock behaves more like a 2D sheet and can gain
robustness by becoming thicker.

---

## 4. Suggested Neighbour Count from Shape (P9.5)

The PCA aspect ratio predicts an optimal neighbour count `m*` via a
linear interpolation between spherical and elongated extremes:

```
t = clamp((aspect − 1) / 2, 0, 1)
m* = 9.78 + t · (6.05 − 9.78)
```

- aspect = 1 (sphere) → m* = 9.78
- aspect ≥ 3 (elongated) → m* = 6.05

The intuition: rounder flocks need more neighbours to achieve the
same robustness because information can flow in more directions;
elongated flocks can achieve adequate consensus with fewer neighbours
because the dominant axis constrains information flow.

This is labelled `suggested_m` in the metrics output — distinct from
the cost-optimal `m★` computed from the H₂ objective.

---

## 5. Gyration Radius

### 5.1 Robust Gyration (P9.7)

```
R_g = sqrt( (1/N_kept) · Σ_{i=1}^{N_kept} d_i² )
```

where `d_i` are the distances from each bird to the **median**
centroid (not the mean — resistant to outliers), sorted ascending,
and only the innermost 85% of birds are kept (top 15% trimmed).

```
com = median(positions, axis=0)       // outlier-resistant centroid
dists = sort(||positions − com||)
keep = int(N · 0.85)
R_g = sqrt(mean(dists[:keep]²))
```

The median centroid ensures that a single bird 10,000 units away
changes `R_g` by less than 5%.  The top-15% trim further reduces
sensitivity to extreme outliers on the flock periphery.

### 5.2 Robust Number Density

```
ρ_robust = N_kept / ((4/3) · π · R_g³)
```

The number density implied by treating the trimmed flock as a uniform
sphere of radius `R_g`.

---

## 6. Maximum Pairwise Distance R_max (B3)

```
R_max(t) = max_{i,j} ||p_i(t) − p_j(t)||
```

The flock's 3D diameter — the largest Euclidean distance between any
two birds.  Computed via scipy's `pdist` (O(N²) condensed distance
matrix).

Pearce et al. (2014) use R_max to test for fragmentation: the swarm
does NOT fragment as long as φp > 0.  Even a tiny projection coupling
maintains 3D cohesion stronger than local Reynolds models achieve.
This metric makes fragmentation (or its absence) directly observable:
a monotonic increase in R_max over time indicates the flock is
spreading without bound; a bounded R_max indicates stable cohesion.

---

## 7. Jamming Index (B14)

The jamming index measures steering saturation — whether birds are
at their maximum turning rate or have converged to a steady heading:

```
jamming = 1 − (⟨|a|⟩ / max_force)
```

clamped to [0, 1].

- jamming ≈ 0: steering is fully saturated at `max_force` —
  birds are constantly maneuvering (turbulent, unconstrained).
- jamming ≈ 1: steering has converged near zero — `v_desired ≈ v`,
  the flock is in a rigid, locked configuration.

In the shipped defaults (φp = 0.03, φa = 0.80), steering typically
saturates at `max_force` every frame (jamming ≈ 0), corresponding
to the dynamic, fluid-like murmuration behaviour.  Higher φp and φa
values produce partial desaturation (jamming ≈ 0.35–0.55), as
predicted by the Pearce et al. (2014) jamming analysis.

The jamming index is an engineered proxy — the source paper describes
a phenomenon, not a specific formula.

---

## 8. Normalized Angular Momentum (P9.8)

Angular momentum about the centre of mass, normalized to remove
system-size and speed dependence:

```
r_i = p_i − ⟨p⟩                    // position relative to centre of mass
L = (1/N) · Σ_i r_i × v_i         // mean angular momentum vector
L_norm = ||L|| / (v0 · R_g)
```

where `v0` is the characteristic cruise speed and `R_g` is the
gyration radius.

- L_norm ≈ 0: purely radial or linear motion (no coherent rotation).
- L_norm ≈ 1: coherent rotation (the flock is spinning as a whole).

Normalizing by `v0 · R_g` makes this an O(1) quantity invariant under
domain scaling — two geometrically similar flocks of different sizes
produce the same L_norm.

---

## 9. Summary of Quantities

| Symbol | Name | Formula | Range |
|--------|------|---------|-------|
| β | Density scaling exponent | log(spacing) = β·log(N) + C | (−∞, +∞) |
| ρ | Convex-hull density | N / V_hull | [0, +∞) |
| aspect | PCA aspect ratio | sqrt(λ₁/λ₃) | [1, +∞) |
| thickness | PCA thickness ratio | sqrt(λ₃/λ₁) | (0, 1] |
| m*(shape) | Suggested m* from shape | 9.78 + t·(6.05−9.78), t = clamp((aspect−1)/2,0,1) | [6.05, 9.78] |
| R_g | Robust gyration radius | RMS distance to median centroid, top-15% trim | [0, +∞) |
| ρ_robust | Robust number density | N_kept / ((4/3)·π·R_g³) | [0, +∞) |
| R_max | Maximum pairwise distance | max ||p_i − p_j|| | [0, +∞) |
| jamming | Jamming index | 1 − ⟨|a|⟩/max_force | [0, 1] |
| L_norm | Normalized angular momentum | ||⟨r×v⟩|| / (v0·R_g) | [0, +∞) |

---

## 10. Taxonomy

None of the quantities in this document are force-computation plugins —
they belong to a separate scientific-metrics layer: pure functions that
read a snapshot of flock state (positions, sometimes velocities) and
compute an observable, without ever writing back into the simulation.
This is architecturally distinct from the plugin-registry system that
drives the physics itself (an ABC plus a decorator populating a lookup
table per computation, selected by config and dispatched every frame) —
metrics have no registry and no interchangeable strategies; each one is
a single fixed calculation reported at a collection interval.

Within that metrics layer, this document covers **spatial-structure**
observables — how the flock is arranged in space (density, shape, size,
diameter) rather than how aligned or mobile it is. Sibling observables
elsewhere in the same layer report on order and motion (polar/nematic
order parameters, mean squared displacement), consensus robustness (a
graph-Laplacian pipeline over the neighbour network), visual opacity
(the fraction of the sky blocked by flock-mates, from either an
internal or external vantage point), and a cross-correlation between
opacity and acceleration. This document's quantities are the ones
sensitive to *where* birds sit relative to each other and to the flock
centroid — not to which direction they're pointing or how correlated
their neighbours' headings are.

## 11. Beyond pymurmur: Unimplemented Extensions

A few spatial-structure techniques from the broader collective-motion
literature are not implemented here, offered as candidates rather than
claims of correctness:

**Voronoi-cell local density.** Instead of a single global density
figure (convex hull or gyration-sphere volume), tessellate the flock
into a 3D Voronoi diagram and use each bird's own cell volume as its
*local* density: `ρ_i = 1 / V_Voronoi(i)`. This is the density
estimator most commonly used in fish-schooling studies, since it
naturally handles anisotropic and non-uniform arrangements that a
single flock-wide number averages away. Would need a 3D Voronoi
tessellation library (e.g. a Qhull-based one, alongside the convex-hull
computation already used here) and a per-bird rather than per-flock
output field.

**Radial distribution function g(r).** A pair-correlation function
measuring the probability of finding a neighbour at distance `r`
relative to a uniform random baseline, `g(r) = ρ(r) / ρ_avg`. This
reveals short-range structural order (e.g. a preferred inter-bird
spacing showing up as a peak in g(r)) that a single spacing-power-law
exponent cannot — it's a full distribution, not a summary statistic.
Would need binned pairwise-distance histograms per frame, normalized
by the expected count under a uniform density.

**Fractal / box-counting dimension.** Covering the occupied volume
with cubes of side `ε` and measuring how the occupied-cube count scales
with `ε` gives a fractal dimension `D_f`, sometimes used to
characterise whether a flock's spatial arrangement is closer to a
solid 3D blob (`D_f ≈ 3`) or a sparser, more filamentary structure
(`D_f < 3`) — a complementary lens to the PCA aspect/thickness ratios
already computed here, which only capture the *gross* ellipsoidal
shape, not fine-scale sparseness.
