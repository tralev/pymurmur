# H₂ Consensus Robustness (Young et al. 2013)

This document defines the H₂ consensus-robustness pipeline from Young
et al. (2013): the k-nearest-neighbour graph Laplacian, the H₂ norm
(via eigenvalue shortcut and Lyapunov cross-validation), nodal
robustness R_nodal, robustness-per-neighbour R_per_m, sensing-cost-
optimal neighbour count m★, cost-optimal neighbour count m★
(linear-penalty), algebraic connectivity λ₂, connectivity threshold,
and marginal efficiency η(m). An implementation note (§4.3) also
documents a known H₂-normalisation discrepancy against the paper.

---

## 1. The Consensus Model

Young et al. (2013) model a flock as a network of `N` agents (birds),
each connected to its `m` nearest neighbours.  Each agent `i` holds a
scalar state `x_i(t)` (which could represent heading, speed, or any
consensus variable).  The dynamics follow linear consensus on the
m-nearest-neighbour graph with additive noise:

```
dx_i/dt = Σ_{j ∈ N_i} a_ij · (x_j − x_i) + ξ_i(t)
```

where `N_i` is the set of agent `i`'s m nearest neighbours, `a_ij` is
the edge weight, and `ξ_i(t)` is zero-mean white noise with covariance
`⟨ξ_i(t)·ξ_j(τ)⟩ = δ_ij · δ(t−τ)` (independent across agents, unit
intensity).

In vector form, with `x ∈ ℝᴺ`:

```
dx/dt = −L·x + ξ
```

where `L` is the graph Laplacian matrix and `ξ` is the noise vector.

---

## 2. Graph Laplacian Construction

### 2.1 Adjacency Matrix

For a flock snapshot with positions `{p_i}`, the m-nearest-neighbour
graph is built as follows:

1. For each bird `i`, find the `m+1` nearest neighbours by Euclidean
   distance (the `+1` accounts for `i` itself, which is excluded).
2. For each neighbour `j ≠ i`, add a directed edge `i → j` with
   weight `a_ij = 1/m`.

The uniform `1/m` weighting matches Young et al.'s finding — they
show (Fig. S1) that uniform weights give strictly better robustness
than distance-proportional or order-based weights, because they
equalise each bird's total incoming influence regardless of how close
or far its neighbours are.

### 2.2 Symmetrization

The directed adjacency matrix is symmetrized via the element-wise
maximum:

```
A = max(A_dir, A_dirᵀ)
```

An edge exists at full weight `1/m` if either endpoint's k-NN set
includes the other.  This is the `max`-form symmetrization rather than
the average (`(A + Aᵀ)/2`), ensuring that two birds who
are each other's neighbours do not double-count the edge weight.

### 2.3 Laplacian

The graph Laplacian `L` is computed from the symmetrized adjacency:

```
D = diag( Σ_j A_ij )            // degree matrix
L = D − A                        // Laplacian
```

`L` is an `N×N` real symmetric positive-semidefinite matrix.
Its eigenvalues are: `0 = λ₀ ≤ λ₁ ≤ λ₂ ≤ … ≤ λ_{N-1}`.

- λ₀ = 0 always (the all-ones vector is the consensus direction).
- λ₁ is the **algebraic connectivity** (Fiedler value).  λ₁ > 0 iff
  the graph is connected; λ₁ = 0 means at least two disjoint
  components exist.

---

## 3. H₂ Norm — Eigenvalue Shortcut

The H₂ norm of the consensus network measures the steady-state
disagreement under unit-intensity white noise — i.e. how far agents
wander from consensus due to noise.  It is the `H₂` system norm of
the transfer function from noise input to disagreement output.

When the graph is connected, the H₂ norm can be computed from the
Laplacian eigenvalues without solving a Lyapunov equation (§4):

```
H₂² = (1 / 2N) · Σ_{k=1}^{N-1} 1/λ_k     (summing over all λ_k > 10⁻¹⁰)
H₂  = √(H₂²)
```

where `λ_k` are the ascending eigenvalues of `L` (λ₀ = 0 is
skipped).  Eigenvalues ≤ 10⁻¹⁰ are treated as numerically zero and
also excluded from the sum — in practice only λ₁ can fall in this
regime (disconnected or near-disconnected graphs).

**This `1/2N` normalisation is this implementation's own convention,
not the paper's** — see §4.3 below for the actual discrepancy and its
consequence for `R_nodal`.

When the graph is **disconnected** (λ₁ < 10⁻¹⁰), H₂ is defined as
`+∞` — disagreement never decays because information cannot flow
between components.  This is the mathematically correct divergence:
the integral of the transfer function's squared magnitude diverges
when the network has zero algebraic connectivity.

Key properties of H₂:

- **Smaller H₂ = more robust**.  Flocks that stay more tightly
  aligned under noise have lower H₂.
- H₂ decreases monotonically as `m` increases (more neighbours →
  more consensus paths → less disagreement).
- H₂ scales with `√N` for fixed `m` (larger flocks have more
  disagreement simply because there are more agents).

---

## 4. H₂ Norm — Lyapunov Cross-Validation

The eigenvalue shortcut (§3) is computationally efficient but depends
on the specific form of the consensus dynamics.  An independent
derivation via the Lyapunov equation provides cross-validation.

### 4.1 Disagreement Subspace

The consensus direction (all agents equal) is factored out.  Let
`Q ∈ ℝ^(N-1)×N` be an orthonormal basis orthogonal to the all-ones
vector `1 = (1,1,…,1)/√N`:

```
Q · 1 = 0          (orthogonal to consensus)
Q · Qᵀ = I_(N-1)    (orthonormal rows)
```

The disagreement state is `y = Q·x ∈ ℝ^(N-1)`.  Substituting into the
consensus dynamics:

```
dy/dt = Q · dx/dt = Q · (−L·x + ξ) = −Q·L·x + Q·ξ
```

Since `x` can be decomposed into the consensus component (projected
onto `1`) and the disagreement component (projected onto `Qᵀ`), and
`L·1 = 0` (the Laplacian annihilates the consensus direction), we have
`x ≈ Qᵀ·y` (ignoring the zero-consensus-mode part), giving:

```
dy/dt = −L̄·y + η
```

where `L̄ = Q·L·Qᵀ` is the reduced `(N-1)×(N-1)` Laplacian and
`η = Q·ξ` is the projected noise with covariance `⟨η·ηᵀ⟩ = I_(N-1)`.

### 4.2 Lyapunov Equation

The steady-state covariance `Σ = ⟨y·yᵀ⟩` of the disagreement subspace
satisfies the continuous-time Lyapunov equation:

```
L̄·Σ + Σ·L̄ᵀ = I
```

This is solved numerically via `solve_continuous_lyapunov(L̄, I)`.
The total steady-state disagreement is the trace:

```
H₂² = Trace(Σ) / N
H₂  = √(Trace(Σ) / N)
```

The Lyapunov and eigenvalue routes produce identical values (verified
to float precision across 39 seed/N/m combinations) — but this
cross-validates the two routes *against each other*, not against the
paper: both bake in the same extra `/N` (§4.3), so agreement between
them confirms internal consistency, not agreement with Young et al.'s
own normalisation.

### 4.3 Implementation Note: Normalisation vs. the Paper

Young et al. (2013)'s own Eq. 4–5 define `H₂ = √(Trace(Σ))` directly
from the same Lyapunov equation above — **with no division by `N`
anywhere in that definition**. The paper introduces `N`-independence
as a separate, later step when building nodal robustness (§5): divide
the paper's own `H₂` by `√N` to get a per-individual disagreement
figure, then invert.

This implementation instead divides by `N` *inside* `H₂` itself
(`H₂² = Trace(Σ)/N` above, equivalently `(1/2N)·Σ 1/λ_k` in §3) —
one extra factor of `√N` baked in before any further normalisation is
applied. Concretely: `H₂_here = H₂_paper / √N`.

The consequence carries through to §5: `R_nodal` here computes
`√N / H₂_here`, which equals `N / H₂_paper` — a full extra factor of
`√N` beyond the paper's own `R_nodal = √N / H₂_paper`. For a fixed
flock size this is just a constant rescale of the reported numbers
(it does not, for instance, explain §6's separately-flagged
`R_per_m` peak-location discrepancy), but it does mean this
codebase's H₂/R_nodal/R_per_m values are not on Young et al.'s own
numeric scale. Flagged here as a known discrepancy for a future
code-level decision — not resolved in this document, and
`consensus_robustness.py` is unchanged.

---

## 5. Nodal Robustness R_nodal

The raw H₂ norm grows with system size `N` — a larger flock
necessarily has more total disagreement.  To remove system-size
dependence, H₂ is normalised by `√N`:

```
R_nodal = √N / H₂
```

The inversion (`1/H₂` rather than `H₂`) makes **larger = more
robust**, which this implementation treats as more intuitive for a
"robustness" figure (a design choice, not itself drawn from a specific
numbered section of the paper).

`R_nodal = 0` when H₂ is non-finite (disconnected graph — the
paper's stated convention) or when H₂ ≤ 0 (degenerate case:
N < 2 or m < 1, where no meaningful Laplacian can be built).

Key property: `R_nodal` is approximately independent of `N` for
connected graphs — the `√N` factor cancels H₂'s own `√N` dependence,
so two flocks of different sizes with the same `m` have similar
`R_nodal` values.

---

## 6. Robustness Per Neighbour R_per_m

Each additional neighbour requires neurological and sensory effort —
maintaining a connection has a cost.  Raw `R_nodal` overstates the
benefit of large `m` because it ignores this cost.

```
R_per_m = R_nodal / m
```

This accounts for sensing cost by dividing robustness by the number
of neighbours.  The paper reports that `R_per_m` peaks at an interior
`m★ ≈ 6–7` for field-observed starling flocks.

In practice, the H₂ eigenvalue route produces a monotonically
decreasing `R_per_m(m)` curve from the connectivity threshold onward
for both random and simulated flocks — the interior peak at m★≈6–7
is not reproduced.  This discrepancy is a known open question.

---

## 7. Optimal Neighbour Count m★

Two distinct objectives produce two different optimal neighbour counts.

### 7.1 Sensing-Cost m★

```
m★ = argmax_{m ∈ [2, min(20, N)]} R_per_m(m)
```

Maximises robustness-per-neighbour — the neighbour count that gives
the best return on investment per connection.  Returns `(2, 0.0)` if
no `m` in `[2, 20]` produces a connected graph.

### 7.2 Linear-Penalty m★ (Cost-Optimal)

```
J(m) = H₂(m) + 0.06 · m
m★ = argmin_{m ∈ [2, 20]} J(m)
```

Minimises a linear combination of disagreement and neighbour cost.
The coefficient `0.06` converts neighbours into H₂-equivalent units.
This is a convex objective — H₂ decreases with `m` while the linear
penalty increases, guaranteeing a unique interior minimum for
connected graphs.

The two m★ values may differ: the sensing-cost criterion (§7.1)
favours the `m` that maximises efficiency, while the linear-penalty
criterion (§7.2) favours the `m` where an additional neighbour stops
reducing H₂ enough to justify its cost.

The linear-penalty m★ is the one reported in the FlockMetrics output
(`optimal_m` field).  The sensing-cost m★ is available as a separate
analysis function.

---

## 8. Algebraic Connectivity λ₂ — Convergence Speed

The second-smallest eigenvalue of the Laplacian, λ₂(L), is the
**algebraic connectivity** (Fiedler value).  It measures how fast
information propagates through the network:

```
dx/dt ≈ −λ₂ · x_perp     (dominant slowest mode)
```

The consensus error decays as `exp(−λ₂·t)` — larger λ₂ means faster
consensus.

```
λ₂ = 0 iff the graph is disconnected
```

The paper's headline result is that **speed and robustness trade
off oppositely as m grows**:

- λ₂ increases monotonically with `m` (faster consensus, no interior
  optimum — more neighbours always spread information faster).
- H₂ decreases with `m` (less steady-state disagreement, but with
  diminishing returns).

Real flocks are shaped by **robustness** (low H₂), not raw convergence
speed.  This is why the linear-penalty m★ (§7.2) uses H₂ in its
objective rather than λ₂ — the tradeoff curve between H₂ and λ₂ as
functions of `m` is the observable signature of this design principle.

`λ₂` is reported in the FlockMetrics output as `convergence_speed`.

---

## 9. Connectivity Threshold

The connectivity threshold is the smallest `m` for which the
m-nearest-neighbour graph becomes connected (H₂ finite):

```
m_conn = min{ m ∈ [1, m_max] : H₂(m) < ∞ }
```

Young et al. report `m_conn ≤ 5` across 394 field-observed
starling-flock snapshots (N = 440–2600): almost always connected
at m = 5, and m = 1, 2 almost always disconnected.

The function scans `m ∈ [1, m_max]` (default `m_max = 20`) and
returns the first `m` where `compute_h2(positions, m)` reports a
finite H₂.  Returns `None` if no `m` in the range connects the graph
(extremely unlikely for N ≥ 10 in practice).

---

## 10. Marginal Efficiency η(m) (P9.6)

Marginal efficiency measures how much robustness improves when adding
two extra neighbours, relative to a baseline:

```
m₀ = max(2, m★ − 2)                          // baseline neighbour count
η(m★) = (H₂(m₀) − H₂(m★)) / (m★ − m₀)
```

The `m₀` offset of 2 provides a meaningful difference — evaluating at
m★ − 1 would capture too little change, while larger offsets risk
crossing the connectivity threshold.

Special cases:

- Both disconnected: η = 0 (no improvement from disconnected →
  disconnected).
- Baseline disconnected, m★ connected: η = +∞ (the graph just became
  connected — an infinite improvement from disconnected to connected).
- Connected → disconnected (pathological, should never occur for
  increasing m): η = 0 (guard against degenerate input).
- N < 4: η = None (not enough birds for a meaningful efficiency
  measurement).

A high η(m★) means the flock's robustness improves rapidly near the
optimal neighbour count — the network is in a sensitive regime where
each additional neighbour provides substantial benefit.  A low η(m★)
means robustness changes slowly near m★ — the flock is in a broad,
flat optimum where the exact neighbour count matters little.

---

## 11. Computational Notes

### 11.1 Cost

The H₂ pipeline is O(N³) per `m` value: building the k-NN graph is
O(N·m·log N), and the dense eigenvalue decomposition is O(N³).  For
large flocks (N > 200), the Batched observers path in occlusion
culling is orders of magnitude cheaper than H₂.

The Lyapunov route (§4) is approximately 10× more expensive than the
eigenvalue shortcut (§3) because it requires an additional
`solve_continuous_lyapunov` call on an (N−1)×(N−1) matrix, though
both are O(N³).  The Lyapunov route exists solely for cross-validation
and is not wired into the metrics collector.

### 11.2 Gating

H₂ is computed only at `detail_level ≥ 2` and only every
`metrics_interval` frames (default 20), because the O(N³) cost is
prohibitive at every-frame cadence.  The `history_cap` ring-buffer
(default 10,000 entries) limits memory growth.

### 11.3 Determinism

The k-NN graph is built from positions only — there is no random
component.  For a given flock snapshot, H₂ is fully deterministic.

---

## 12. Summary of Quantities

| Symbol | Name | Formula | Interpretation |
|--------|------|---------|----------------|
| L | Graph Laplacian | D − A | Encodes the consensus network |
| A | Adjacency | max(A_dir, A_dirᵀ), a_ij = 1/m | Symmetrized m-NN edges |
| λ₂ | Algebraic connectivity | 2nd eigenvalue of L | Convergence speed; 0 = disconnected |
| H₂² | H₂ norm squared | (1/2N)·Σ_{k≥1} 1/λ_k | Steady-state disagreement |
| H₂ | H₂ norm | √(H₂²) | Robustness metric; smaller = better |
| R_nodal | Nodal robustness | √N / H₂ | Size-normalised robustness; larger = better |
| R_per_m | Robustness per neighbour | R_nodal / m | Robustness per unit sensing cost |
| m★ (sensing-cost) | Sensing-cost optimum | argmax R_per_m(m) | Best return per neighbour |
| m★ | Linear-penalty optimum | argmin (H₂(m) + 0.06·m) | Cost-optimal neighbour count |
| m_conn | Connectivity threshold | min{ m : H₂(m) < ∞ } | Smallest m connecting the graph |
| η(m★) | Marginal efficiency | (H₂(m₀) − H₂(m★)) / (m★ − m₀) | Robustness gain per extra neighbour |

---

## 13. Taxonomy

H₂ consensus robustness is **not** a force-computation plugin — none
of the quantities in this document ever feed back into how a bird
steers. It belongs to a different part of pymurmur's architecture
entirely: the scientific-metrics layer, a set of pure, post-hoc
observables computed from a flock snapshot (positions, velocities,
per-bird opacity) purely for reporting and analysis. Force-mode
plugins are selected once per run and dictate motion every frame;
metrics are the opposite — they never influence motion, and can in
principle be computed for *any* mode's output, though in practice a
given metric (like this one) is usually only meaningful for certain
modes' dynamics.

Sibling observables in this same metrics layer include: density
scaling (how local spacing/flock size change with population count),
PCA-based flock shape analysis (aspect ratio, thickness), opacity
(the same Θ/Θ′ quantities the projection-mode document defines,
computed independently as a standalone observable regardless of
which force mode produced the positions), mean-squared displacement
and polar/nematic order parameters, and a cross-correlation between
opacity and acceleration magnitude. Unlike the force-mode registry
(one active strategy per run, switched via configuration), these
metrics are not mutually exclusive — a single run can report all of
them simultaneously, gated by how expensive each is to compute per
frame.

## 14. Beyond pymurmur: Unimplemented Extensions

A few consensus/network-robustness techniques from the broader
collective-motion and networked-control literature are not
implemented here:

- **Time-varying (dynamic) network robustness.** This document's H₂
  pipeline treats the m-nearest-neighbour graph as static within a
  single snapshot — it never accounts for how quickly the network
  topology itself changes as birds move and re-sort their neighbour
  sets frame to frame. A genuinely dynamic-network extension would
  track the *switching* consensus dynamics (a jump-linear system
  whose Laplacian changes discretely each frame) rather than treating
  each frame as an independent static-graph problem — a substantially
  harder stability analysis (average dwell-time / joint connectivity
  arguments) than the single-snapshot H₂ norm computed here.
- **Directed (non-symmetrized) consensus.** §2.2 explicitly
  symmetrizes the adjacency matrix via an element-wise maximum before
  building the Laplacian. Real starling neighbour relationships are
  not necessarily reciprocal (bird A having B in its m-nearest set
  doesn't guarantee the reverse), and a directed-graph consensus
  analysis (using the actual asymmetric Laplacian, whose eigenvalues
  can be complex) would test whether robustness conclusions still
  hold without the symmetrization simplification — at the cost of a
  much more involved eigenvalue analysis (no longer guaranteed real,
  no longer diagonalizable by an orthonormal basis).
- **Robustness under adversarial/targeted node removal**, rather than
  this document's uniform-weight, random-topology robustness measure.
  A predator selectively "removing" (through capture or scatter) the
  most-connected individuals would stress-test the network
  differently than the m-nearest-neighbour graph's baseline
  robustness — closer to percolation-theory questions about targeted
  vs. random attack on scale-free-like networks, which m-NN graphs
  are not (they're closer to a spatial random geometric graph).
- **Higher-order (hypergraph) consensus**, where three or more birds
  that mutually influence each other are modelled as a single
  hyperedge rather than as three separate pairwise edges. This
  document's Laplacian is built entirely from pairwise m-nearest-
  neighbour edges; a hypergraph formulation could in principle
  capture genuinely group-level (not just pairwise) alignment
  effects, though the resulting spectral theory (hypergraph
  Laplacians) is far less standardized than the classical graph case
  used throughout this document.
