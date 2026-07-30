# Acceleration–Opacity Cross-Correlation (B9)

This document defines the B9 cross-correlation between horizontal
centre-of-mass acceleration and internal opacity Θ — the observable
from Pearce et al. (2014) that tests whether opacity changes mediate
long-range information transfer in starling murmurations.

---

## 1. Scientific Motivation (B9)

Real starling murmurations show opacity Θ changing significantly
within seconds of rapid horizontal acceleration — suggesting that
opacity (the fraction of the visual field blocked by flock-mates)
mediates 3D information transfer faster than nearest-neighbour
propagation.

The Pearce et al. (2014) projection model predicts this correlation:
when the flock accelerates horizontally (e.g. in response to a
predator or a turn), birds on the leading edge see more sky (lower Θ),
while birds behind see more flock (higher Θ).  If opacity is a causal
signal — birds steer based on how much sky they see — then Θ should
change **after** acceleration, with a measurable time lag.

The B9 observable measures the cross-correlation function C(δt)
between horizontal COM acceleration `a(t)` and opacity `Θ(t+δt)`,
searching for the characteristic lag at which correlation is
strongest.

---

## 2. Data Collection

Two ring buffers are maintained at the metrics-collection cadence
(configurable `interval`; production default is every 60 frames):

- **COM velocity ring**: Centre-of-mass velocity vectors (3D), sampled
  at each metrics interval.
- **Θ ring**: Mean internal opacity across the flock, sampled at the
  same instants.

Both rings accumulate samples with the oldest-first convention: index
0 is the oldest sample, index −1 is the newest.  The ring buffer has
configurable capacity (default `buffer_size = 500` entries).

---

## 3. Acceleration Derivation

Horizontal COM acceleration is derived as the per-sample-step
difference of horizontal COM velocity:

```
v_COM(t) = (1/N) · Σ_i v_i(t)            // centre-of-mass velocity
a_horiz(t) = Δv_COM,xy / Δt_sample       // horizontal acceleration
```

Only the `x` and `y` components are used — `z` (altitude in this
codebase's convention) is excluded because horizontal acceleration is
the ecologically relevant signal for predator evasion and turning.

Since Pearson correlation is invariant under positive linear rescaling
of one series, the acceleration magnitude is computed as the
Euclidean norm of the velocity difference without dividing by the
physical timestep:

```
Δv_xy[k] = v_COM,xy[k+1] − v_COM,xy[k]     // (2,) per sample step
a_mag[k] = ||Δv_xy[k]||                      // scalar magnitude
```

The dividing factor `Δt_sample = interval · dt_phys` is a positive
constant, so omitting it does not change the correlation coefficients
or the peak lag location — only the raw magnitude of the acceleration
series, which is irrelevant for correlation.

The acceleration and opacity series are aligned so that `a_mag[k]`
corresponds to the same sample step as `Θ[k+1]` (the opacity at the
end of the interval over which acceleration was measured), producing
`m = n_samples − 1` valid pairs.

---

## 4. Cross-Correlation C(δt)

The cross-correlation function measures the Pearson correlation
between horizontal acceleration at time `t` and opacity at a later
time `t + δt`:

```
ā = (1/m) · Σ_k a_mag[k]                          // acceleration mean
Θ̄ = (1/m) · Σ_k Θ[k]                              // opacity mean
σ_a = sqrt( (1/m) · Σ_k (a_mag[k] − ā)² )        // acceleration std
σ_Θ = sqrt( (1/m) · Σ_k (Θ[k] − Θ̄)² )            // opacity std

C(δt) = (1/(m−δt)) · Σ_{k=0}^{m−1−δt} (a_mag[k] − ā) · (Θ[k+δt] − Θ̄)
         / (σ_a · σ_Θ)
```

where δt is in sample-step units (each step = `interval` frames).
C(δt) ∈ [−1, 1]:

- C(δt) > 0: positive correlation — horizontal acceleration tends to
  be followed by increased opacity (more birds blocking the sky) at
  lag δt.
- C(δt) < 0: negative correlation — horizontal acceleration tends to
  be followed by decreased opacity (more sky visible) at lag δt.
- C(δt) ≈ 0: no linear relationship at that lag.

The correlation is computed for δt = 0, 1, 2, …, max_lag, where:

```
max_lag = min(m − 1, max(1, int(0.25 · buffer_size)))
```

The `0.25 · buffer_size` cap keeps the correlation scan bounded even
on long runs — for the default buffer of 500 entries, max_lag ≤ 125
sample steps.

If either series has zero variance (perfectly steady flock), the
correlation is undefined and `(None, None)` is returned.

---

## 5. Peak Lag Detection

The lag δt at which the **absolute** correlation is largest identifies
the characteristic timescale:

```
peak_sample_lag = argmax_{δt ∈ [0, max_lag]} |C(δt)|
peak_lag_frames = peak_sample_lag · interval
```

The peak lag is converted from sample-step units to **frame** units
by multiplying by `interval` — this matches the convention used by
the density autocorrelation time τ_ρ, which is also reported in frame
units.

A peak lag of zero means the strongest correlation is instantaneous:
acceleration and opacity change simultaneously.  A positive peak lag
means opacity changes **follow** acceleration — the causal signal
that the B9 hypothesis predicts.

---

## 6. Interpretation

The B9 observable addresses a specific scientific question:

> *Does horizontal acceleration cause opacity to change (birds
> reconfigure their relative positions), and if so, at what delay?*

If the peak lag is:

- **Zero or near-zero**: acceleration and opacity are simultaneously
  correlated — changes in flock shape and changes in flight direction
  happen at the same time.  This suggests a fast mechanical response
  rather than a signalling cascade.

- **Positive (several sample steps)**: opacity changes lag behind
  acceleration — birds first change direction, then gradually
  reconfigure the visual field.  This is consistent with the Pearce
  model where δ̂ (boundary projection) responds to changes in the
  visible neighbour set, and Θ integrates over the resulting new
  configuration.

- **Negative**: opacity changes **precede** acceleration — birds
  reconfigure before changing direction.  This would suggest opacity
  is a predictive signal, not a response.

The sign of C(δt) at the peak lag indicates whether acceleration is
associated with increased or decreased opacity:

- C > 0 at peak: acceleration → more birds visible (Θ increases) —
  the flock compresses or turns toward the observer's visual field.
- C < 0 at peak: acceleration → fewer birds visible (Θ decreases) —
  the flock expands or the observer moves to the periphery.

---

## 7. Computational Notes

### 7.1 Gating

The B9 cross-correlation is computed only at `detail_level ≥ 2` and
is a gated expensive metric — like H₂, τ_ρ, and MSD, it requires a
history buffer and is not computed every frame.

### 7.2 Mode Restriction

Θ is only computed in projection mode (Pearce et al. 2014) — in all
other modes, `theta` is `NaN` and the correlation is not computed.
The B9 observable is therefore exclusively a projection-mode metric.

### 7.3 Buffer Warm-Up

The ring buffer requires at least 6 samples before the cross-
correlation can be computed (≤ 4 degrees of freedom after the
acceleration differencing loses one sample).  The `max_lag` cap at
`0.25 · buffer_size` further requires the buffer to have at least
4 entries of usable history.

### 7.4 Output

The cross-correlation curve `C(δt)` for δt = 0..max_lag is reported
as `theta_accel_correlation` (a list of floats).  The peak lag in
frame units is reported as `theta_accel_peak_lag` (an integer, or
None if no correlation could be computed).

A full-3D sibling metric is also computed and reported alongside it:
`theta_accel_correlation_3d` and `theta_accel_peak_lag_3d`, using the
same ring-buffer/cross-correlation/peak-lag machinery but against the
**full 3D** COM acceleration magnitude `‖Δv_COM‖` (not just the
horizontal `‖Δv_COM,xy‖` from §3–§5 above). Pearce et al.'s own result
(§1) is horizontal-only because real flock footage can't measure
vertical motion; this simulation has genuine 3D velocity data, so the
3D variant asks a related but genuinely different question — whether
vertical manoeuvring also correlates with opacity changes — not a
replacement for the horizontal result.

---

## 8. Summary

| Symbol | Name | Formula | Units |
|--------|------|---------|-------|
| a_mag[k] | Acceleration magnitude (horizontal) | ‖Δv_COM,xy[k]‖ | arbitrary (unscaled) |
| a_mag_3d[k] | Acceleration magnitude (full 3D) | ‖Δv_COM[k]‖ | arbitrary (unscaled) |
| Θ[k] | Mean opacity | (1/N)·Σ Θ_i | [0, 1] |
| C(δt) | Cross-correlation | Pearson r(a_mag(t), Θ(t+δt)) | [−1, 1] |
| peak_lag | Peak correlation lag | argmax_δt \|C(δt)\| · interval | frames |
| max_lag | Maximum lag scanned | min(m−1, max(1, int(0.25·buffer_size))) | sample steps |

Both the horizontal (`theta_accel_correlation`/`theta_accel_peak_lag`)
and full-3D (`theta_accel_correlation_3d`/`theta_accel_peak_lag_3d`)
variants share every formula above — only `a_mag` vs. `a_mag_3d`
differs (§7.4).

---

## 9. Taxonomy

This observable is not a force-computation plugin — it belongs to
pymurmur's scientific-metrics layer, pure functions that read flock
state and compute a reported quantity without ever influencing the
simulation. But within that layer it sits one level *above* the other
metrics: it isn't computed directly from a raw position/velocity
snapshot, it's a **derived** observable computed from two other
metrics' own time series (opacity Θ and centre-of-mass acceleration,
itself derived from consecutive velocity samples). Sibling metrics
elsewhere report on spatial structure (density, shape, gyration
radius), order and motion (polar/nematic order, mean squared
displacement) computed directly from a snapshot, and consensus
robustness (a graph-Laplacian pipeline over the neighbour network) —
this is the only one of the group whose input is another metric's
output rather than the raw simulation state, and correspondingly the
only one gated to a single force mode (§7.2), since its opacity input
only exists there.

## 10. Beyond pymurmur: Unimplemented Extensions

Two related statistical techniques from time-series analysis are not
implemented here, offered as candidates rather than claims of
correctness:

**Granger causality test.** The cross-correlation in this document
establishes that acceleration and opacity are related at some lag, but
correlation at a lag is not a formal causality claim. A Granger test
asks a sharper question: does including past acceleration values
improve a statistical model's ability to predict *future* opacity,
beyond what past opacity values alone already predict? This requires
fitting two autoregressive models (opacity-on-its-own-past vs.
opacity-on-its-own-past-plus-acceleration's-past) and comparing their
residual variance via an F-test — a heavier statistical machinery than
the single Pearson-correlation-per-lag scan used here, but a stronger
claim if it passes.

**Transfer entropy.** An information-theoretic generalisation of
Granger causality that doesn't assume a linear relationship: it
measures how much uncertainty about future opacity is reduced by
knowing past acceleration, in bits, using estimated probability
distributions rather than a linear model. This would catch a
nonlinear acceleration→opacity coupling that a Pearson correlation
(which only detects linear relationships) could miss entirely, at the
cost of needing enough samples to estimate joint probability
distributions reliably — a real concern given this metric's own
buffer-warm-up requirement of just 6 samples (§7.3).
