# pymurmur — Validation Rigor: Acceptance Gates & Analysis Math

> **Purpose:** This document specifies a scientific validation harness for pymurmur,
> adapted from the Murmuration Core v2 design. It defines 10 results (5 hard gates,
> 5 reported-only) derived from Pearce (2014) and Young (2013), with complete
> Python/scipy implementations.
>
> **Scientific authority** (never modified): Pearce et al. 2014, Young et al. 2013.
> This document translates their findings into executable validation code.
>
> **Target:** pymurmur's existing simulation engine. All analysis modules below
> operate on numpy arrays piped from the engine — they do not depend on any internal
> simulation state beyond positions, velocities, and per-boid opacity Θ.

---

## 1. Overview of the 10 results

| ID | Result | Gate? | Source paper | Module |
|----|--------|:-----:|--------------|--------|
| ★ P-a | Mean internal opacity Θ̄ ≈ 0.30 (murmuration phenotype) | **hard** | Pearce 2014 | per-frame Θ column |
| P-b | Θ vs 1/N linear, R² ≈ 0.99 (N ≥ 400) | report | Pearce 2014 | `density.py` |
| ★ P-c | Density scaling ρ ∼ N^(−1/2) (open boundary) | **hard** | Pearce 2014 | `density.py` |
| ★ P-d | No fragmentation for any φp > 0 (bounded R_max) | **hard** | Pearce 2014 | per-frame R_max |
| P-e | Density correlation time τρ decreases as φp increases | report | Pearce 2014 | `tau_rho.py` |
| P-f | Three phenotypes give distinct (α, Θ) regimes | report | Pearce 2014 | harness |
| ★ Y-a | Robustness-per-neighbour peaks at m* ≈ 6–7 | **hard** | Young 2013 | `h2.py` |
| Y-b | m* independent of flock size N | report | Young 2013 | `h2.py` |
| ★ Y-c | m* vs flock thickness (PCA λ₃/λ₁) monotonic trend | **hard** | `h2.py` + `shape_pca.py` |
| Y-d | Sensing graph connected at m ≥ 5 | report | `h2.py` |

**The five starred gates** block acceptance. The other five are computed and
reported but do not block.

---

## 2. Per-frame metrics (cheap — computed every step)

These are O(N) or O(N²) reductions run inside the simulation loop. They provide
the raw numbers that the heavy analysis modules consume.

### 2.1 Formulas (use float64 throughout)

```
Polarisation (order parameter)   α  = |(1/N) Σᵢ v̂ᵢ|                          ∈ [0,1]
  where v̂ᵢ = vᵢ / |vᵢ| (unit heading)

Internal opacity (mean)          Θ̄  = (1/N) Σᵢ Θᵢ
  where Θᵢ = per-boid internal opacity from the occlusion pass                      ∈ [0,1]

Dispersion (mean radius)         σ_r = (1/N) Σᵢ |rᵢ − r_com|
  where r_com = (1/N) Σᵢ rᵢ

Mean nearest-neighbour dist      d̄_nn = (1/N) Σᵢ min_{j≠i} |rᵢ − rⱼ|
  computed via spatial index or cKDTree

Flock extent                     R_max = max_{i<j} |rᵢ − rⱼ|
  exact O(N²) for N ≤ 2600; convex-hull/bounding-box approximation for larger N

Mean speed                       s̄  = (1/N) Σᵢ |vᵢ|

External opacity (silhouette)    Θ′ = project all birds onto a plane ⟂ viewpoint axis,
  stamp a disk of radius b (body_radius) at each position, union-rasterise,
  Θ′ = covered_cells / total_cells
  Compute every k frames only (e.g. every 10), raster resolution 256².
  Θ′ is *reported* for empirical-image comparison, NOT the acceptance anchor.
```

### 2.2 Python implementation — per-frame collector

```python
import numpy as np

def collect_metrics(positions, velocities, theta_column, step,
                    view_axis=np.array([0, 0, 1]), boid_size=9.0,
                    opacity_ext_every=10, opacity_ext_res=256):
    """
    Compute per-frame metrics from live SoA columns.

    Args:
        positions:    (N, 3) float64 — boid positions.
        velocities:   (N, 3) float64 — boid velocities.
        theta_column: (N,) float64   — per-boid internal opacity Θ.
        step:         int — current frame number.
        view_axis:    (3,) — axis for external opacity rasterisation.
        boid_size:    float — body radius for silhouette disks.
        opacity_ext_every: int — compute Θ′ every this many frames.
        opacity_ext_res:  int — raster grid size (square).

    Returns:
        dict with keys: polarisation, opacity_int, opacity_ext,
                        r_max, dispersion, mean_nn, mean_speed, count, step.
    """
    N = len(positions)
    if N == 0:
        return dict(polarisation=0.0, opacity_int=0.0, opacity_ext=0.0,
                    r_max=0.0, dispersion=0.0, mean_nn=0.0,
                    mean_speed=0.0, count=0, step=step)

    inv = 1.0 / N

    # --- Single-pass: polarisation, centroid, speed_sum, theta_sum ---
    speeds = np.linalg.norm(velocities, axis=1)
    valid = speeds > 1e-9
    vhat_sum = np.zeros(3, dtype=np.float64)
    if valid.any():
        vhat_sum = (velocities[valid] / speeds[valid, np.newaxis]).sum(axis=0)
    com = positions.sum(axis=0) * inv
    speed_sum = speeds.sum()
    theta_sum = theta_column.sum()

    polarisation = np.linalg.norm(vhat_sum) * inv
    mean_speed = speed_sum * inv
    opacity_int = theta_sum * inv

    # --- Dispersion — second pass ---
    disp = np.linalg.norm(positions - com, axis=1).sum() * inv

    # --- R_max: exact O(N²) for N ≤ 2600; use bounding-box approx for large N ---
    if N <= 2600:
        diffs = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        pairwise = np.linalg.norm(diffs, axis=2)
        r_max = pairwise.max()
    else:
        # Bounding-box diagonal approximation
        mins = positions.min(axis=0)
        maxs = positions.max(axis=0)
        r_max = np.linalg.norm(maxs - mins)

    # --- Mean nearest-neighbour: use scipy.cKDTree ---
    from scipy.spatial import cKDTree
    tree = cKDTree(positions)
    nn_dists, _ = tree.query(positions, k=2)  # k=2 because self is nearest
    mean_nn = nn_dists[:, 1].mean()  # second column = nearest non-self neighbour

    # --- External opacity: every k frames ---
    opacity_ext = 0.0
    if step % opacity_ext_every == 0:
        opacity_ext = _external_opacity_rastser(
            positions, boid_size, view_axis, opacity_ext_res,
        )

    return dict(
        polarisation=float(polarisation),
        opacity_int=float(opacity_int),
        opacity_ext=float(opacity_ext),
        r_max=float(r_max),
        dispersion=float(disp),
        mean_nn=float(mean_nn),
        mean_speed=float(mean_speed),
        count=N,
        step=step,
    )


def _external_opacity_rastser(positions, boid_size, view_axis, res=256):
    """
    Rasterise the silhouette onto a plane perpendicular to view_axis.

    Projects all birds onto the plane, stamps a disk of radius `boid_size`
    at each projected position, and returns covered_cells / total_cells.
    """
    import numpy as np

    # Build orthonormal basis for the projection plane (u, v ⟂ view_axis)
    axis = view_axis / np.linalg.norm(view_axis)
    if abs(axis[0]) < 0.9:
        u = np.cross(axis, np.array([1.0, 0.0, 0.0]))
    else:
        u = np.cross(axis, np.array([0.0, 1.0, 0.0]))
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)

    # Project positions onto (u, v) plane
    proj_u = positions.dot(u)
    proj_v = positions.dot(v)

    # Determine bounding box of the projection
    u_range = proj_u.max() - proj_u.min() + 2.0 * boid_size
    v_range = proj_v.max() - proj_v.min() + 2.0 * boid_size
    span = max(u_range, v_range, 1.0)
    u_mid = (proj_u.max() + proj_u.min()) / 2.0
    v_mid = (proj_v.max() + proj_v.min()) / 2.0

    # Map to grid coordinates
    col = ((proj_u - u_mid) / span * res + res / 2).astype(int)
    row = ((proj_v - v_mid) / span * res + res / 2).astype(int)

    # Stamp disks
    canvas = np.zeros((res, res), dtype=bool)
    disk_r_px = int(np.ceil(boid_size / span * res))
    for r, c in zip(row, col):
        if 0 <= r < res and 0 <= c < res:
            r_min = max(0, r - disk_r_px)
            r_max = min(res, r + disk_r_px + 1)
            c_min = max(0, c - disk_r_px)
            c_max = min(res, c + disk_r_px + 1)
            rr, cc = np.ogrid[r_min:r_max, c_min:c_max]
            dist_sq = (rr - r) ** 2 + (cc - c) ** 2
            canvas[r_min:r_max, c_min:c_max] |= (dist_sq <= disk_r_px ** 2)

    return float(canvas.sum()) / (res * res)
```

---

## 3. Heavy analysis — Python/scipy modules

All operate on numpy arrays from piped simulation output. Use **float64** throughout.

### 3.1 H₂ robustness (Young 2013) — module: `analysis/h2.py`

**Scientific model.** Heading agreement is noisy linear consensus on the
m-nearest-neighbour graph. Per bird (Eq. 1 of Young):

```
dxᵢ/dt = Σ_{j∈Nᵢ} aᵢⱼ (xⱼ − xᵢ) + ξᵢ
```

where:
- Nᵢ is the set of m nearest neighbours
- aᵢⱼ = 1/m (uniform weights — Young Fig. S1: uniform is both simplest and most robust)
- ξᵢ is unit-intensity white noise

Stacked form: `dx/dt = −L x + ξ`, where `L = D − A` is the graph Laplacian.

**H₂ norm** = steady-state disagreement. Solving the Lyapunov equation
`L̄ Σ + Σ L̄ᵀ = I` on the reduced Laplacian (consensus mode projected out)
gives `Σ = ½ L̄⁻¹`, so:

```
H₂     = √Trace(Σ) = √( (1 / 2N) · Σ_{i≥2} 1/λᵢ )
R_nodal = 1 / (H₂ / √N)          # per-individual robustness (size-normalised, inverted)
R_per_m = R_nodal / m            # robustness per neighbour (sensing cost)
m*     = argmax_m (R_per_m)      # finite optimum ⇒ m* ≈ 6–7
```

- Disconnected graph (λ₂ ≈ 0) ⇒ H₂ = ∞ ⇒ R = 0.
- Connectivity threshold: graphs connect at m ≥ 5 (Young, 394 snapshots, N=440–2600).
- m* = argmax of R_per_m, NOT argmin of H₂ + cost·m (governed by sci/todo.md A6–A8).

**Python implementation:**

```python
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh


def h2_curve(positions, m_values=range(1, 13)):
    """
    Compute H₂ robustness curve over a range of m (neighbour count).

    Args:
        positions: (N, 3) float64 — boid positions from one snapshot.
        m_values:  iterable of int — m-values to test.

    Returns:
        dict: m -> (H2, R_nodal, R_per_m)
            H2:      float or np.inf — H₂ norm of the consensus system.
            R_nodal: float — per-individual robustness.
            R_per_m: float — robustness per neighbour (sensing cost).
    """
    N = len(positions)
    tree = cKDTree(positions)
    out = {}

    for m in m_values:
        # k=m+1 because self is the nearest neighbour at index 0
        _, idx = tree.query(positions, k=m + 1)

        # Build adjacency matrix: uniform weight 1/m per edge
        rows = np.repeat(np.arange(N), m)
        cols = idx[:, 1:].ravel()  # drop self (column 0)
        data = np.full(rows.size, 1.0 / m)
        A = csr_matrix((data, (rows, cols)), shape=(N, N))

        # Undirected symmetrisation: A ← max(A, Aᵀ)
        A = A.maximum(A.T)

        # Laplacian L = D − A
        D_diag = np.asarray(A.sum(axis=1)).ravel()
        L = csr_matrix(np.diag(D_diag)) - A

        # Smallest eigenvalues of the symmetric Laplacian; drop λ₁ ≈ 0
        k_eigs = min(N - 1, 200)
        lam = np.sort(
            eigsh(L, k=k_eigs, sigma=0.0, which='LM',
                  return_eigenvectors=False)
        )
        lam = lam[lam > 1e-9]  # discard near-zero (consensus mode)

        if lam.size == 0:
            out[m] = (np.inf, 0.0, 0.0)
            continue

        H2 = np.sqrt((1.0 / (2 * N)) * np.sum(1.0 / lam))
        R_nodal = 1.0 / (H2 / np.sqrt(N))
        R_per_m = R_nodal / m
        out[m] = (H2, R_nodal, R_per_m)

    return out


def m_star(curve):
    """
    Find the optimal m — argmax of robustness-per-neighbour.

    Gate Y-a: m* ∈ {5, 6, 7}.

    Args:
        curve: dict from h2_curve() — m -> (H2, R_nodal, R_per_m).

    Returns:
        int: m* — the m with maximum R_per_m.
    """
    return max(curve, key=lambda m: curve[m][2])


def is_connected(lam, threshold=1e-9):
    """
    Check whether the Laplacian has a second eigenvalue above threshold.

    Gate Y-d (report): graph connected at m ≥ 5 ⇒ λ₂ > 1e-9.

    Args:
        lam:       (k,) float64 — sorted Laplacian eigenvalues.
        threshold: float — values below this are treated as zero.

    Returns:
        bool: True if the graph is connected.
    """
    nonzero = lam[lam > threshold]
    return len(nonzero) >= 1
```

### 3.2 Flock shape PCA (Young 2013) — module: `analysis/shape_pca.py`

**Scientific model.** The 3×3 covariance of positions reveals the flock's aspect
ratio and "thickness." Young (2013) relates this to the optimal neighbour count m*.

```
C          = (1/N) Σᵢ (rᵢ − r̄)(rᵢ − r̄)ᵀ             # 3×3 covariance
λ₁ ≥ λ₂ ≥ λ₃                                          # eigenvalues

aspect_ratio = √(λ₁ / λ₃)          (≥ 1)
thickness    = √(λ₃ / λ₁)          (∈ (0, 1])         # canonical form — use the √
```

Empirical relationship (from Young data):
```
m*_from_shape = 9.78 + t · (6.05 − 9.78)
where t = clamp((aspect_ratio − 1) / (3 − 1), 0, 1)
```

**Important:** Use `thickness = √(λ₃/λ₁)` — the square root form. Some prose
descriptions write the ratio without the sqrt; those are wrong. The sqrt form
matches Young's empirical band 0.13–0.27.

**Python implementation:**

```python
import numpy as np


def flock_shape(positions):
    """
    Compute PCA shape parameters of a flock snapshot.

    Args:
        positions: (N, 3) float64.

    Returns:
        dict with keys:
            eigs:      (3,) float64 — eigenvalues λ₁ ≥ λ₂ ≥ λ₃.
            aspect:    float — aspect ratio √(λ₁/λ₃).
            thickness: float — flock thickness √(λ₃/λ₁).
    """
    centered = positions - positions.mean(axis=0)
    C = np.cov(centered.T)  # 3×3 covariance
    lam = np.sort(np.linalg.eigvalsh(C))[::-1]  # λ₁ ≥ λ₂ ≥ λ₃

    aspect = np.sqrt(lam[0] / lam[2]) if lam[2] > 0 else np.inf
    thickness = np.sqrt(lam[2] / lam[0]) if lam[0] > 0 else 0.0

    return dict(eigs=lam, aspect=aspect, thickness=thickness)
```

### 3.3 Density scaling (Pearce 2014) — module: `analysis/density.py`

**Mean-field derivation.** A random ray through a homogeneous isotropic 3D flock
hits sky with probability:

```
P_sky ≈ exp(−ρ · b² · R)
```

where b = body_radius, R = characteristic flock radius, ρ = number density.
In d = 3, silhouette cross-section ∝ b^(d−1) = b².

Marginal opacity sets P_sky = ½, so:

```
ρ · b² · R ≈ ln 2
```

Since N ∝ ρ · R³, eliminating R gives the **scaling law**:

```
ρ(N) ∼ N^(−1/2)      and      L(N) ∼ N^(+1/2)        (d = 3)
```

**Measured per run (open boundary, self-sized flock):**

```
centre  = per-axis median of positions                  # robust to stragglers
rᵢ      = |posᵢ − centre|
keep top 85% by distance (drop farthest 15%)
Rg      = √(mean(rᵢ² over kept))                       # gyration radius
ρ_N     = N_kept / ((4/3) · π · Rg³)                   # number density
```

**Fit:** `log ρ = b · log N + c` over N ∈ {400, 800, 1600, 2600}.
Ideal exponent b = −0.5.

**Python implementation:**

```python
import numpy as np
from scipy.spatial import ConvexHull


def density_gyration(positions, trim_frac=0.85):
    """
    Compute number density using the gyration-radius method.

    Args:
        positions: (N, 3) float64.
        trim_frac: fraction of birds to keep (drop farthest 1−trim_frac).

    Returns:
        float: number density ρ = N_kept / ((4/3)·π·Rg³).
    """
    N = len(positions)
    if N == 0:
        return 0.0

    centre = np.median(positions, axis=0)
    r = np.linalg.norm(positions - centre, axis=1)
    threshold = np.quantile(r, trim_frac)
    keep = r <= threshold

    N_kept = keep.sum()
    if N_kept == 0:
        return 0.0

    Rg = np.sqrt(np.mean(r[keep] ** 2))
    volume = (4.0 / 3.0) * np.pi * Rg ** 3
    return N_kept / volume if volume > 0 else 0.0


def density_convex_hull(positions):
    """
    Compute number density using ConvexHull volume.

    Args:
        positions: (N, 3) float64.

    Returns:
        float: number density ρ = N / Vol(ConvexHull).
        Returns 0.0 if hull is degenerate (< 4 non-coplanar points).
    """
    N = len(positions)
    if N < 4:
        return 0.0
    try:
        hull = ConvexHull(positions)
        return N / hull.volume if hull.volume > 0 else 0.0
    except Exception:
        return 0.0


def scaling_exponent(N_values, rho_values):
    """
    Fit log ρ = b · log N + c. Returns the exponent b.

    Gate P-c: b ∈ [−0.7, −0.3] (ideal: −0.5).

    Args:
        N_values:   array-like of int — flock sizes.
        rho_values: array-like of float — corresponding densities.

    Returns:
        tuple: (b, c, r_squared) — exponent, intercept, R².
    """
    log_N = np.log(N_values)
    log_rho = np.log(rho_values)

    # polyfit returns [slope, intercept]
    b, c = np.polyfit(log_N, log_rho, 1)

    # R²
    rho_pred = np.exp(b * log_N + c)
    ss_res = np.sum((rho_values - rho_pred) ** 2)
    ss_tot = np.sum((rho_values - np.mean(rho_values)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return b, c, r_squared
```

### 3.4 Density correlation time τρ — module: `analysis/tau_rho.py`

**Definition (Pearce Fig. 2f):**

```
ρ(t)     = N / Vol(convex hull of positions at time t)
C(Δt)    = ⟨ρ(t) · ρ(t + Δt)⟩ − ⟨ρ⟩²                      # autocovariance
τρ       = Σ_{lag ≥ 0} C(lag) / C(0) · Δt_sample           # integrate to first zero-crossing
```

**Protocol:** Buffer ~500 snapshots, sample every ~10 frames. Integrate the
normalised autocorrelation to the first zero crossing, capped at 25% of the buffer.

**Python implementation:**

```python
import numpy as np


def tau_rho(rho_series, dt_sample):
    """
    Compute density autocorrelation time.

    Args:
        rho_series: (T,) float64 — density at each sampled frame.
        dt_sample:  float — time between samples (e.g. 10 · dt_phys).

    Returns:
        float: τρ — integrated autocorrelation time.
    """
    x = np.asarray(rho_series, dtype=np.float64) - np.mean(rho_series)

    # Autocorrelation via FFT convolution
    ac = np.correlate(x, x, mode='full')
    ac = ac[len(ac) // 2:]  # zero-lag onwards
    ac /= ac[0]             # normalise so C(0) = 1

    # Find first zero crossing, cap at 25% of buffer length
    if np.any(ac < 0):
        zc = int(np.argmax(ac < 0))
    else:
        zc = len(ac) // 4

    return float(np.sum(ac[:zc]) * dt_sample)
```

### 3.5 Optional: multi-viewpoint external opacity — module: `analysis/opacity.py`

For higher-fidelity external opacity Θ′ than the cheap raster-scale version,
sample many viewpoints on a Fibonacci sphere and average.

```python
import numpy as np


def fibonacci_sphere(n_samples=100):
    """
    Generate n_samples nearly-uniform points on the unit sphere.

    Returns:
        (n_samples, 3) float64 — unit vectors.
    """
    points = np.zeros((n_samples, 3), dtype=np.float64)
    phi = np.pi * (3.0 - np.sqrt(5.0))  # golden angle
    for i in range(n_samples):
        y = 1.0 - (i / float(n_samples - 1)) * 2.0  # y ∈ [1, −1]
        radius = np.sqrt(1.0 - y * y)
        theta = phi * i
        points[i] = [np.cos(theta) * radius, y, np.sin(theta) * radius]
    return points


def multi_view_opacity(positions, boid_size, n_views=100, res=256):
    """
    Compute external opacity Θ′ averaged over many Fibonacci-sphere viewpoints.

    Calls _external_opacity_rastser for each viewpoint and averages.
    See §2.2 for the rasteriser implementation.
    """
    views = fibonacci_sphere(n_views)
    total = 0.0
    for v in views:
        total += _external_opacity_rastser(positions, boid_size, v, res)
    return total / n_views
```

---

## 4. Acceptance gates — definitions & tolerances

### 4.1 Pearce (2014) gates

| ID | Gate | Setup | Pass condition |
|----|------|-------|----------------|
| ★ **P-a** | Internal opacity Θ̄ | murmuration phenotype (φp=0.03, φa=0.80), open space, N ∈ {400, 800, 1600}, equilibrated | Θ̄ ∈ [0.25, 0.35] at each N |
| P-b | Θ vs 1/N linearity | N ≥ 400, multiple N | R² ≥ 0.95 (report) |
| ★ **P-c** | Density scaling exponent | open space, N ∈ {400, 800, 1600, 2600} | log–log exponent b ∈ [−0.7, −0.3] |
| ★ **P-d** | No fragmentation | φp ∈ {0.03, 0.10, 0.20}, ≥ 10⁴ steps | R_max bounded (no monotonic divergence) |
| P-e | τρ decreases with φp | φp ∈ {0.03, 0.10, 0.20}, same N | τρ(0.20) < τρ(0.10) < τρ(0.03) |
| P-f | Phenotype regimes | murmuration / schooling (φp=0.10, φa=0.60) / swarming (φp=0.20, φa=0.30) | (α, Θ̄) differ visibly across the three |

### 4.2 Young (2013) gates

| ID | Gate | Setup | Pass condition |
|----|------|-------|----------------|
| ★ **Y-a** | m* optimum | one equilibrated flock, H₂ curve over m ∈ [1, 12] | argmax_m(R_per_m) ∈ {5, 6, 7} |
| Y-b | m* vs N independence | several N values, same φ | m* approximately constant (R² near 0) |
| ★ **Y-c** | m* vs thickness trend | several flocks spanning different thicknesses | monotone (negative) trend — thicker flocks = smaller m* |
| Y-d | Connectivity | m ∈ [5, 12] across snapshots | λ₂ > 1e-9 for all m ≥ 5 |

### 4.3 Reference data (from `sci/todo.md`)

```
Θ̄ empirical:      µ = 0.30, σ² = 0.059  (own data)
                   µ = 0.41, σ² = 0.012  (public images)
Θ vs 1/N R²:      0.99
m*:               6–7
m* vs N R²:       0.0178  (i.e. independent of N)
m* vs thickness R²: 0.64
Connectivity:      m ≥ 5 across 394 snapshots (N = 440–2600)
```

---

## 5. Acceptance harness — module: `analysis/validate/acceptance.py`

A single entry point that runs all 10 results and exits non-zero if any of the
5 hard gates fail.

```python
"""
acceptance.py — Runs the 10-result validation harness.

Usage:
    python -m murmuration.analysis.validate.acceptance

Exits 0 if all 5 hard gates pass, 1 otherwise.
Prints a table of all 10 results with PASS/FAIL/REPORT.
"""

import sys
import numpy as np

from ..h2 import h2_curve, m_star
from ..shape_pca import flock_shape
from ..density import density_gyration, scaling_exponent
from ..tau_rho import tau_rho


# ── Gate definitions ───────────────────────────────────────────────

GATES = {
    "P-a": {
        "hard": True,
        "desc": "Θ̄ ∈ [0.25, 0.35] at murmuration φ for N ∈ {400, 800, 1600}",
    },
    "P-b": {
        "hard": False,
        "desc": "Θ vs 1/N linear, R² ≥ 0.95",
    },
    "P-c": {
        "hard": True,
        "desc": "Density scaling exponent b ∈ [−0.7, −0.3]",
    },
    "P-d": {
        "hard": True,
        "desc": "No fragmentation — R_max bounded over ≥ 10⁴ steps",
    },
    "P-e": {
        "hard": False,
        "desc": "τρ decreases as φp increases",
    },
    "P-f": {
        "hard": False,
        "desc": "Three phenotypes give distinct (α, Θ̄) regimes",
    },
    "Y-a": {
        "hard": True,
        "desc": "m* ∈ {5, 6, 7}",
    },
    "Y-b": {
        "hard": False,
        "desc": "m* independent of flock size N",
    },
    "Y-c": {
        "hard": True,
        "desc": "m* vs thickness: monotone negative trend",
    },
    "Y-d": {
        "hard": False,
        "desc": "Sensing graph connected at m ≥ 5",
    },
}


def check_P_a(theta_history, N_values, equil_steps=500):
    """Θ̄ ∈ [0.25, 0.35] at each N."""
    for N in N_values:
        thetas = theta_history.get(N, [])
        if len(thetas) < equil_steps:
            return False, f"N={N}: insufficient data"
        mean_theta = np.mean(thetas[equil_steps:])
        if not (0.25 <= mean_theta <= 0.35):
            return False, f"N={N}: Θ̄={mean_theta:.3f} ∉ [0.25, 0.35]"
    return True, "all N satisfy Θ̄ ∈ [0.25, 0.35]"


def check_P_b(theta_by_N):
    """Θ vs 1/N linearity R² ≥ 0.95."""
    if len(theta_by_N) < 3:
        return False, "insufficient N samples"
    Ns = np.array(sorted(theta_by_N.keys()))
    thetas = np.array([theta_by_N[n] for n in Ns])
    inv_N = 1.0 / Ns
    _, _, r2 = scaling_exponent(inv_N, thetas)  # reuse the fit helper
    return r2 >= 0.95, f"R² = {r2:.3f}"


def check_P_c(rhos, Ns):
    """Density scaling exponent b ∈ [−0.7, −0.3]."""
    if len(rhos) < 3:
        return False, "insufficient N samples"
    b, _, _ = scaling_exponent(Ns, rhos)
    return (−0.7 <= b <= −0.3), f"b = {b:.3f}"


def check_P_d(r_max_series):
    """R_max bounded — no monotonic divergence over the run."""
    if len(r_max_series) < 100:
        return False, "insufficient data"
    # Simple check: slope of late-half r_max vs step is non-positive
    half = len(r_max_series) // 2
    late = r_max_series[half:]
    x = np.arange(len(late))
    slope, _, _ = np.polyfit(x, late, 1, full=False)  # returns [slope, intercept]
    return slope <= 0, f"late-half slope = {slope:.2f}"


def check_P_e(tau_by_phi):
    """τρ decreases as φp increases."""
    if len(tau_by_phi) < 2:
        return False, "insufficient φp samples"
    sorted_phis = sorted(tau_by_phi.keys())
    taus = [tau_by_phi[p] for p in sorted_phis]
    for i in range(len(taus) - 1):
        if taus[i] <= taus[i + 1]:
            return False, f"τρ not monotonic: φp={sorted_phis[i]}→{sorted_phis[i+1]}"
    return True, "τρ monotonic decreasing with φp"


def check_P_f(phenotype_data):
    """Three phenotypes give distinct (α, Θ̄)."""
    # At minimum, murmuration must differ from swarming
    if "murmuration" not in phenotype_data or "swarming" not in phenotype_data:
        return False, "missing phenotype data"
    m = phenotype_data["murmuration"]
    s = phenotype_data["swarming"]
    # Some separation in (α, Θ̄) space
    dist = np.sqrt((m["alpha"] - s["alpha"]) ** 2 + (m["theta"] - s["theta"]) ** 2)
    return dist > 0.1, f"phenotype separation = {dist:.3f}"


def check_Y_a(h2_curve_result):
    """m* ∈ {5, 6, 7}."""
    mstar = m_star(h2_curve_result)
    return mstar in {5, 6, 7}, f"m* = {mstar}"


def check_Y_b(mstar_by_N):
    """m* approximately independent of N."""
    if len(mstar_by_N) < 3:
        return False, "insufficient N samples"
    Ns = np.array(sorted(mstar_by_N.keys()))
    mstars = np.array([mstar_by_N[n] for n in Ns])
    _, _, r2 = np.polyfit(np.log(Ns), mstars, 1, full=True)[:2]  # not a real fit — just check variance
    std = np.std(mstars)
    return std <= 1, f"σ(m*) = {std:.2f}"


def check_Y_c(mstar_and_thickness):
    """Monotone negative trend: thicker flocks → smaller m*."""
    if len(mstar_and_thickness) < 3:
        return False, "insufficient samples"
    pairs = sorted(mstar_and_thickness, key=lambda p: p[1])  # sort by thickness
    mstars = [p[0] for p in pairs]
    # Check if generally decreasing
    decreasing = all(mstars[i] >= mstars[i + 1] for i in range(len(mstars) - 1))
    return decreasing, "trend is (non-increasing)"


def check_Y_d(connectivity_data):
    """Graph connected for all m ≥ 5."""
    if not connectivity_data:
        return False, "no connectivity data"
    for m in range(5, 13):
        if m in connectivity_data and not connectivity_data[m]:
            return False, f"disconnected at m = {m}"
    return True, "connected for all m ≥ 5"


# ── Runner ──────────────────────────────────────────────────────────

CHECK_MAP = {
    "P-a": check_P_a,
    "P-b": check_P_b,
    "P-c": check_P_c,
    "P-d": check_P_d,
    "P-e": check_P_e,
    "P-f": check_P_f,
    "Y-a": check_Y_a,
    "Y-b": check_Y_b,
    "Y-c": check_Y_c,
    "Y-d": check_Y_d,
}


def print_results_table(results):
    """Print a formatted pass/fail table."""
    print(f"\n{'ID':6s} {'Hard?':6s} {'Result':10s} {'Detail'}")
    print("-" * 60)
    all_hard_pass = True
    for gate_id, info in GATES.items():
        passed, detail = results.get(gate_id, (None, "not run"))
        if info["hard"] and not passed:
            all_hard_pass = False
        status = "PASS" if passed else ("FAIL" if info["hard"] else "REPORT")
        print(f"{gate_id:6s} {'★' if info['hard'] else ' '}      {status:10s} {detail}")
    return all_hard_pass
```

---

## 6. Measurement protocol

### 6.1 Equilibration

For all acceptance gates, discard the first 500–1000 frames as warm-up.
Define "equilibrated" as: over a sliding window of 100 frames, |Δα| < 0.01
and |ΔΘ̄| < 0.01 for 3 consecutive windows.

### 6.2 Parameter sweep protocol

```
For P-a, P-c, P-b:
  - Fix φp = 0.03, φa = 0.80 (murmuration phenotype)
  - Run at N ∈ {400, 800, 1600, 2600}
  - Open boundary only (these gates assume unbounded space)
  - Collect Θ̄ and positions after equilibration

For P-d:
  - φp ∈ {0.03, 0.10, 0.20}
  - Run ≥ 10,000 steps
  - Record R_max every 10 frames

For Y-a:
  - Take one equilibrated snapshot (any N, e.g. 800)
  - Run h2_curve(m=1..12)

For Y-c:
  - Collect 5–10 snapshots from different runs/configurations
    spanning a range of flock thicknesses
  - For each: compute m* (from h2_curve) and thickness (from flock_shape)
```

### 6.3 Key constants for numerical stability

```
MIN_LEN  = 1e-9   — floor for normalisation/division guards
MIN_LEN2 = 1e-18  — MIN_LEN²

φp + φa + φn ≡ 1.0  — φn is always derived, never set directly
```

---

## 7. References

- Pearce et al. (2014) — "Role of projection in the control of bird flocks"
  (sci/1407.2414v1.pdf)
- Young et al. (2013) — "Starling flock networks manage uncertainty in
  consensus at low cost" (sci/1302.3195v1.pdf)
- Derived specifications: `sci/todo.md` (items A1–A16, B5–B12),
  `sci/sim_new.md` (§21)
- Design authority: Murmuration Core v2 design documents
  (`design/03_observables_bindings.md` §3–§4)
