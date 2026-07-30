# Order, Motion, and MSD Metrics

This document defines the order parameters (polar and nematic),
motion observables (speed, force, power, angular momentum, velocity
deviation, dispersion), boundary metrics (overshoot, altitude
deviation), steering saturation (jamming index), and the mean
squared displacement curve with its ballistic→diffusive crossover.

---

## 1. Polar Order α

The polar order parameter measures the degree of alignment — how
closely all birds fly in the same direction:

```
v̂_i = v_i / ||v_i||                  // unit velocity, or zero if ||v_i|| < 10⁻⁶
α = (1/N) · || Σ_{i=1}^{N} v̂_i ||
```

where `N` is the number of birds (typically restricted to non-predator
birds when species filtering is active).

α ∈ [0, 1]:
- α = 0: velocities are uniformly distributed on the sphere (complete
  disorder — no coherent flight direction).
- α = 1: all birds fly in exactly the same direction (perfect
  alignment — the flock moves as a rigid body).

α is the standard order parameter used throughout the collective
behaviour literature.  It is computed every frame (O(N) cost) and
reported as `alpha`.

---

## 2. Nematic Order S (P9.1)

The nematic order parameter measures alignment **without regard to
direction** — birds flying in opposite directions still contribute
to nematic order if their axes are parallel.  This is the appropriate
order parameter for systems where ±v̂ are equivalent (e.g. rods,
magnetic spins, or birds that can fly either way along a corridor).

S is computed from the 3×3 traceless Q-tensor:

```
Q_αβ = (1/N) · Σ_{i=1}^{N} [ (3/2) · û_i^α · û_i^β − (1/2) · δ_αβ ]
S    = λ_max(Q)
```

where `û_i` are unit velocity directions, `δ_αβ` is the Kronecker
delta, and `λ_max(Q)` is the maximum eigenvalue of the 3×3 symmetric
matrix Q.

S ∈ [0, 1]:
- S ≈ 1: perfect nematic order — all birds fly along the same axis,
  regardless of whether they fly +v̂ or −v̂ along it.
- S ≈ 0: isotropic — unit vectors are uniformly distributed on the
  sphere.

The key difference from polar α: if half the flock flies north and
half flies south (anti-aligned), α ≈ 0 (vectors cancel) but S ≈ 1
(axes are parallel).  Nematic order is invariant under `û → −û`,
making it the correct metric for systems where direction reversal
is a symmetry of the dynamics.

---

## 3. Speed, Force, and Power

### 3.1 Mean Speed

```
⟨|v|⟩ = (1/N) · Σ_i ||v_i||
```

The arithmetic mean of the bird speed magnitudes.  Reported as
`speed_avg`.

### 3.2 Mean Force

Accelerations are stashed before being zeroed by the integrator.
The mean force magnitude is:

```
⟨|a|⟩ = (1/N) · Σ_i ||a_i||
```

Reported as `force_avg`.

### 3.3 Mean Power

The mean absolute mechanical power (force dotted into velocity):

```
⟨|a · v|⟩ = (1/N) · Σ_i |a_i · v_i|
```

The absolute value prevents positive and negative power from
cancelling — a bird accelerating forward and a bird decelerating
both expend energy.  Reported as `power_avg`.

### 3.4 Jamming Index (B14)

Steering saturation — whether the flock is at its maximum turning
rate or has converged to a steady heading:

```
jamming = 1 − ⟨|a|⟩ / max_force
         clamped to [0, 1]
```

- jamming ≈ 0: steering fully saturated at `max_force` — birds are
  constantly maneuvering (turbulent, fluid murmuration).
- jamming ≈ 1: steering converged near zero — `v_desired ≈ v`,
  flock is locked into a rigid configuration.

The jamming index is an engineered proxy — the source paper
describes a phenomenon, not a formula.  Reported as `jamming_index`.

---

## 4. Velocity Deviation

How much individual bird velocities differ from the flock mean:

```
⟨v⟩ = (1/N) · Σ_i v_i
velocity_deviation = (1/N) · Σ_i ||⟨v⟩ − v_i||
```

This is the mean Euclidean distance from each velocity to the flock
average — a measure of velocity spread.  Zero when all birds fly at
exactly the same velocity; large when birds fly in different
directions or at different speeds.

Reported as `velocity_deviation`.

---

## 5. Dispersion

The mean distance from each bird to the centre of mass:

```
p_com = (1/N) · Σ_i p_i                  // centre of mass
dispersion = (1/N) · Σ_i ||p_i − p_com||
```

A measure of flock compactness — how spread out the birds are.
Zero for a single point; large for a widely dispersed flock.

Reported as `dispersion`.

---

## 6. Angular Momentum

### 6.1 Raw Angular Momentum

The mean angular momentum vector about the centre of mass:

```
r_i = p_i − p_com                        // position relative to CoM
L = (1/N) · Σ_i r_i × v_i               // mean angular momentum (3D vector)
```

Reported as `angular_momentum`.

### 6.2 Normalized Angular Momentum (P9.8)

Normalized by the characteristic speed and size to remove system
dependence:

```
L_norm = ||L|| / (v0 · R_g)
```

where `v0` is the characteristic cruise speed and `R_g` is the
gyration radius:

```
com_med = median(positions, axis=0)               // outlier-resistant centroid
d_i = sort(||p_i − com_med||)                     // distances, sorted ascending
N_kept = int(N · 0.85)                            // top-15% trim
R_g = sqrt( (1/N_kept) · Σ_{i=1}^{N_kept} d_i² )
```

This robust estimator uses a median centroid and trims the outermost
15% of birds, making it resistant to single outliers.

L_norm ≥ 0:
- L_norm ≈ 0: purely radial or linear motion (no coherent rotation).
- L_norm ≈ 1: coherent rotation — the flock is spinning as a whole.

Normalizing by `v0 · R_g` makes this an O(1) quantity invariant under
domain scaling — two geometrically similar flocks of different sizes
produce the same L_norm.

Reported as `normalized_angular_momentum`.

---

## 7. Boundary Metrics

### 7.1 Boundary Overshoot (P9.8)

Total overshoot distance beyond the domain boundary:

```
C = (W/2, H/2, D/2)                     // domain centre
R_dom = min(W, H, D) / 2                 // inscribed sphere radius
boundary_overshoot = Σ_i max(0, ||p_i − C|| − R_dom)
```

Measures how far birds have strayed outside the domain.  Zero when
all birds are inside the inscribed sphere; positive when birds
cross the boundary (open or margin modes).

Reported as `boundary_overshoot`.

### 7.2 Altitude Deviation (P9.8)

Mean absolute deviation from a target altitude (default 500.0):

```
altitude_deviation = (1/N) · Σ_i |z_i − z_target|
```

Measures how closely the flock maintains a preferred altitude.
Useful for roosting behaviour analysis where birds settle at a
specific height.  Reported as `altitude_deviation`.

---

## 8. Mean Squared Displacement (MSD)

MSD measures how far birds travel from their starting positions over
time — distinguishing ballistic motion (persistent velocity) from
diffusive motion (random walk).

### 8.1 Position Unwrapping

For toroidal domains, raw position differences are corrected via
minimum-image displacement to avoid spurious jumps:

```
Δ_unwrap(t) = Δp(t) − box · round(Δp(t) / box)    // per-axis
p_unwrap(t) = p_unwrap(t−1) + Δ_unwrap(t)
```

This reconstructs a continuous trajectory from wrapped snapshots.

### 8.2 MSD(τ) Curve

MSD is computed at log-spaced lags τ = 1, 2, 4, 8, …, max_lag
(default max_lag = 64 snapshot steps):

```
MSD(τ) = (1/N) · (1/(T−τ)) · Σ_{i=1}^{N} Σ_{t=0}^{T−τ−1} ||p_i(t+τ) − p_i(t)||²
```

where `N` is the number of birds, `T` is the number of snapshots,
and `p_i(t)` is the unwrapped position of bird `i` at snapshot `t`.
The average is over both time origins and birds.

The MSD at the longest available lag is reported as `msd`.  The full
curve of MSD values at each lag is reported as `msd_curve`.

### 8.3 Log-Log Slope

The slope of log(MSD) vs log(τ) over the first 3 lags quantifies
the motion regime:

```
log(MSD_k) ≈ b · log(τ_k) + c     for k = 0, 1, 2   (linear regression)
```

- b ≈ 2: ballistic — birds fly in straight lines (velocities are
  persistent).
- b ≈ 1: diffusive — birds random-walk (velocities decorrelate
  quickly).
- 1 < b < 2: super-diffusive — persistent but with turning.

Reported as `msd_slope`.

### 8.4 Ballistic→Diffusive Crossover

The crossover lag is the first τ where the **local** log-log slope
between consecutive lags drops below 1.5:

```
local_slope(i) = log(MSD_i / MSD_{i-1}) / log(τ_i / τ_{i-1})
crossover = first τ_i where local_slope(i) < 1.5
```

If no lag crosses below 1.5, the motion remains ballistic at all
measured timescales — common for small flocks in toroidal domains
where birds have long velocity persistence.  Reported as
`msd_crossover` (None if never crosses).

---

## 9. Summary of Quantities

| Symbol | Name | Formula | Range |
|--------|------|---------|-------|
| α | Polar order | (1/N)·‖Σ v̂_i‖ | [0, 1] |
| S | Nematic order | λ_max(Q), Q_αβ = (1/N)·Σ((3/2)·ûα·ûβ − δ_αβ/2) | [0, 1] |
| ⟨‖v‖⟩ | Mean speed | (1/N)·Σ‖v_i‖ | [0, +∞) |
| ⟨‖a‖⟩ | Mean force | (1/N)·Σ‖a_i‖ | [0, +∞) |
| ⟨\|a·v\|⟩ | Mean power | (1/N)·Σ\|a_i·v_i\| | [0, +∞) |
| jamming | Jamming index | 1 − ⟨‖a‖⟩/max_force, clamped [0,1] | [0, 1] |
| — | Velocity deviation | (1/N)·Σ‖⟨v⟩ − v_i‖ | [0, +∞) |
| — | Dispersion | (1/N)·Σ‖p_i − p_com‖ | [0, +∞) |
| L | Angular momentum | (1/N)·Σ r_i × v_i | 3D vector |
| L_norm | Normalized ang. mom. | ‖L‖ / (v0·R_g) | [0, +∞) |
| — | Boundary overshoot | Σ max(0, ‖p_i−C‖ − R_dom) | [0, +∞) |
| — | Altitude deviation | (1/N)·Σ\|z_i − z_target\| | [0, +∞) |
| MSD(τ) | Mean sq. displacement | ⟨‖p(t+τ) − p(t)‖²⟩ | [0, +∞) |
| msd_slope | MSD log-log slope | d(log MSD)/d(log τ), first 3 lags | [0, 2] |
| crossover | Ballistic→diffusive | first τ with local slope < 1.5 | integer or None |

---

## 10. Taxonomy

None of the quantities in this document are force-computation plugins —
they belong to a separate scientific-metrics layer: pure functions that
read a snapshot of flock state (positions and velocities) and compute
an observable, without ever writing back into the simulation. This is
architecturally distinct from the plugin-registry system that drives
the physics itself (an ABC plus a decorator populating a lookup table
per computation, selected by config and dispatched every frame) —
metrics have no registry and no interchangeable strategies; each one is
a single fixed calculation reported at a collection interval.

Within that metrics layer, this document covers **order-parameter and
motion** observables — how aligned and how mobile the flock is, as
opposed to its spatial arrangement (density, shape, gyration radius —
a sibling document) or its visual opacity and consensus robustness
(also sibling documents, covering the graph-Laplacian pipeline over the
neighbour network and the internal/external opacity fractions). This
document's quantities are the ones sensitive to *which direction* birds
point and *how far* they travel — not to where they sit relative to
each other spatially.

## 11. Beyond pymurmur: Unimplemented Extensions

A few order/motion techniques from the broader collective-motion
literature are not implemented here, offered as candidates rather than
claims of correctness:

**Spatial correlation function C(r) and correlation length.** Rather
than a single flock-wide polar order value, compute how strongly two
birds' velocity fluctuations are correlated as a function of the
distance between them: `C(r) = ⟨δv̂_i · δv̂_j⟩` for pairs at separation
`r` (where `δv̂_i = v̂_i − ⟨v̂⟩` is the deviation from the mean
heading), then fit a correlation length `ξ` from its decay. This is
the technique behind a well-known empirical finding in real starling
flocks: correlations extend across the *whole* flock regardless of
size ("scale-free correlation"), a qualitatively different claim than
"the flock has high polar order," and one this document's single
scalar `α` cannot distinguish from a flock with only short-range
correlation. Would need pairwise velocity-fluctuation products binned
by distance, then a fit (power-law or exponential) to extract `ξ`.

**Van Hove self-correlation function.** MSD (§8) captures only the
*second moment* of how far birds travel over a lag `τ`. The van Hove
function `G_s(Δr, τ)` is the full probability distribution of
displacements at that lag — its shape reveals whether motion is
Gaussian-diffusive (as classical Brownian motion predicts) or has
heavy tails from occasional large excursions (common in active-matter
systems, including real flocking data). Would need a histogram of
per-bird displacement magnitudes at each sampled lag, in addition to
the mean-squared summary already computed.

**Susceptibility near the ordering transition.** In Vicsek-style
models, the variance of the order parameter across the flock (or
across repeated runs at the same noise level) peaks near the
order-disorder transition — a diagnostic used to locate that
transition precisely, analogous to susceptibility in statistical
mechanics: `χ = N · (⟨α²⟩ − ⟨α⟩²)`. Not currently reported; would need
either multiple concurrent runs at the same parameters or a
sufficiently long single-run time series to estimate the variance.
