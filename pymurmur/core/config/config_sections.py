"""Sub-config dataclasses composed by SimConfig (I7.1).

Extracted from config.py (file-size split) — DomainConfig through
CaptureConfig, ~20 small per-subsystem parameter dataclasses.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# ── Sub-config dataclasses (I7.1) ─────────────────────────────────

@dataclass
class DomainConfig:
    """Domain dimensions (static — requires restart)."""
    width: float = 1000.0
    height: float = 700.0
    depth: float = 400.0


@dataclass
class FlockConfig:
    """Flock parameters (static — requires restart)."""
    num_boids: int = 150
    boid_size: float = 9.0       # body radius for 3D spherical-cap occlusion
    v0: float = 4.0              # cruise speed (units/frame)
    max_force: float = 0.15      # max steering force per frame
    visual_range: float = 70.0   # max distance for neighbor candidate filtering
    dt_phys: float = 1.0 / 60.0  # P8.10: fixed physics timestep (seconds)
    speed_min_factor: float = 0.3  # P11.5: min speed = v0 · factor (band clamp)
    n_predators: int = 0          # number of predator boids (0 = off)


@dataclass
class BoundaryConfig:
    """Boundary conditions."""
    boundary_mode: str = "toroidal"     # toroidal | open | margin | sphere | sphere_soft
    boundary_sphere_radius: float = 300.0
    boundary_avoidance_factor: float = 0.05
    boundary_radius_factor: float = 1.0
    boundary_margin: float = 50.0       # margin distance for margin mode


@dataclass
class ProjectionConfig:
    """Projection mode weights (Pearce 2014)."""
    phi_p: float = 0.03          # projection weight (δ̂ coherence)
    phi_a: float = 0.80          # alignment weight (neighbor heading)
    sigma: int = 4               # topological neighbor count
    max_visibility: int = 64      # max visible neighbors for occlusion culling
    max_occlusion_neighbors: int = 64  # max occlusion neighbor candidates


@dataclass
class SpatialConfig:
    """Spatial mode weights (Reynolds 1987)."""
    separation_weight: float = 4.5
    alignment_weight: float = 0.65
    cohesion_weight: float = 0.75
    noise_scale: float = 0.0
    noise_mode: str = "additive"       # additive | maxwellian | none | seed_sinusoidal | velocity
    acceleration_scale: float = 0.3
    influence_count: int = 7           # P4.1: max topological neighbours (hybrid filter)
    speed_mode: str = "clamp"          # clamp | band | none — how speed is enforced
    flow_weight: float = 0.0           # P11.5: global flow contribution weight
    neighbor_filter: str = "hybrid"    # hybrid | metric | topological | global | none
    # sum | mean | unit | exp | linear_ramp | asymptotic | velocity_weighted | cosine_zone
    separation_kernel: str = "sum"     # how sep forces are combined (kernels.py registry)
    separation_kernel_radius: float = 20.0  # only used by exp/linear_ramp/asymptotic
    cohesion_kernel: str = "unweighted"      # unweighted | inverse_distance
    # S2.B1: dual-radii — alignment sees a tighter subset than sep/coh.
    # 1.0 = no extra restriction beyond visual_range (back-compat default);
    # the starlings preset sets 0.75 per the source-parity table.
    alignment_radius_ratio: float = 1.0
    # S2.B1: absolute metric gate for separation neighbours (0 = off,
    # uses the shared hybrid-filtered set unrestricted; starlings/boids
    # presets set 20).
    separation_distance: float = 0.0
    # P11.5: Per-interaction perception cones
    max_dist_sep: float = 0.0          # max sep distance (0 = disabled)
    max_dist_align: float = 0.0        # max align distance (0 = disabled)
    max_dist_coh: float = 0.0          # max coh distance (0 = disabled)
    angle_sep: float = -1.0            # cos(θ) sep cone (−1 = full sphere)
    angle_align: float = -1.0          # cos(θ) align cone (−1 = full sphere)
    angle_coh: float = -1.0            # cos(θ) coh cone (−1 = full sphere)
    # P4.3: Predator boids (species)
    predator_escape_factor: float = 10_000_000.0
    predator_speed_boost: float = 1.8
    predator_perception_boost: float = 1.5
    predator_accel_boost: float = 1.4
    # P4.5: Per-frame parameter jitter (adds organic variation)
    jitter_separation: float = 0.0   # 0 = off, 0.5 = ±50% variation
    jitter_cohesion: float = 0.0
    jitter_alignment: float = 0.0
    # P4.8: Coherence gate
    coherence_factor: float = 1.0       # reduce align/coh for small flocks (<1)
    # P11.5: Evolvable forward-thrust weight
    w_fwd: float = 0.0                  # forward force toward cruise speed
    # P11.5: Obstacle avoidance (EvoFlock genes, consumed by _ObjectiveCollector)
    static_avoid_weight: float = 0.0     # static SDF-gradient avoidance weight
    predictive_avoid_weight: float = 0.0 # predictive avoidance look-ahead weight
    fly_away_max_dist: float = 0.0       # fly-away trigger distance for obstacles
    min_time_to_collide: float = 0.0     # min time-to-collide for predictive avoidance


@dataclass
class FieldConfig:
    """Field mode parameters (crs48 blob-anchor)."""
    field_separation: float = 0.92
    field_alignment: float = 0.90
    field_cohesion: float = 1.80
    field_flow: float = 0.30
    field_chase_strength: float = 0.82
    # P3 presets — extended field parameters
    field_noise: float = 0.035
    field_target_pull: float = 0.22
    field_drift_pull: float = 0.10
    field_drift_direction: tuple[float, float, float] = (0.0, 0.0, 0.0)
    field_shell_influence: float = 1.0
    field_tangent_pull: float = 0.04
    field_wave_gain: float = 0.04
    field_ripple_trains: int = 3
    field_inertia: float = 0.82
    field_shell_radius_base: float = 0.22
    field_inner_radius_factor: float = 0.35
    field_num_groups: int = 7
    field_leader_fraction: float = 0.16
    # P3.6: Additional field term parameters
    field_flow_pull: float = 1.0        # curl flow / fold noise pull strength
    field_unit_scale: float | None = None  # explicit unit scale (None→auto from domain)
    # C3: Field term toggles
    disabled_terms: list[str] = field(default_factory=list)  # names of field sub-terms to skip


@dataclass
class WanderConfig:
    """Wander extension parameters."""
    wander_attractor_speed: float = 0.10
    wander_attractor_radius: float = 300.0


@dataclass
class SpeedNoiseConfig:
    """Noise-modulated speed extension parameters.

    Drives per-boid cruising speed continuously from a 3D value-noise
    field sampled at position (pymurmur.core.noise.value_noise3),
    producing organic slow/fast zones. Distinct from FieldConfig's
    noise_scale, which feeds the unrelated sinusoidal "noise" field
    term (seed_noise3) — kept name-prefixed speed_noise_* to avoid
    confusion between the two.
    """
    speed_noise_frequency: float = 0.003    # spatial frequency (cycles/unit)
    speed_noise_min_mult: float = 0.5       # multiplier at noise sample 0.0
    speed_noise_max_mult: float = 2.0       # multiplier at noise sample 1.0
    speed_noise_power: float = 1.0          # shaping exponent before lerp
    speed_noise_time_scale: float = 0.0     # 0 = static field; >0 = drifts over time


@dataclass
class NeighborAdaptiveSpeedConfig:
    """Neighbor-count adaptive speed extension parameters.

    Generalizes angle.py's deficit-based speed law (isolated boids fly
    faster) across all 7 modes. Reuses angle.py's speed_mode vocabulary
    (linear/quadratic/softened) but is otherwise independent — angle.py
    is not modified, it keeps its own separate config/implementation.
    """
    neighbor_adaptive_speed_target: int = 4        # desired neighbor count
    neighbor_adaptive_speed_radius: float = 70.0    # query radius (~visual_range default)
    neighbor_adaptive_speed_mode: str = "linear"    # linear | quadratic | softened
    neighbor_adaptive_speed_linear_scale: float = 5.0  # only used by "linear"


@dataclass
class VicsekConfig:
    """Vicsek mode parameters."""
    vicsek_couplage: float = 0.8       # alignment coupling η ∈ [0,1]
    vicsek_diffusion: float = 0.8      # angular noise D
    vicsek_radius_influence: float = 5.0
    vicsek_radius_avoid: float = 1.0
    vicsek_velocity: float = 1.0       # constant speed
    vicsek_time_step: float = 0.1      # Δt for memory-term noise (P1.8)
    # P6: Predator-prey species dynamics
    vicsek_radius_predators: float = 80.0      # prey detection radius for predators
    vicsek_velocity_predator: float = 2.0      # predator speed (faster than prey)
    vicsek_detect_ratio: float = 1.5          # predator hunting range multiplier
    vicsek_weight_afraid: float = 3.0          # neighbour weight boost when afraid
    vicsek_predator_noise_ratio: float = 0.1   # hunting directional noise


@dataclass
class InfluencerConfig:
    """Influencer mode parameters (P7.1–P7.6)."""
    influencer_rank_exponent: float = 1.8
    influencer_substeps: int = 5
    influencer_scale: float = 1.0         # P7.1: Lissajous spatial scale (≤1 stays in-domain)
    influencer_influence_mode: str = "rank"  # P7.3: "rank" | "distance"
    influencer_near_dist_sq: float = 100.0  # P7.3: near-distance for distance-based influence
    influencer_init_separation: float = 0.5  # P7.4: density-scaled init spacing factor
    influencer_tick_rate: float = 1.0       # P7.1: tick increment per substep
    # S2.E1: optional trajectory-shaping coefficients — default to the
    # verbatim spec formula (T_raw(t) below); overriding any of these
    # scales/reshapes the Lissajous path without touching the mode code.
    influencer_target_freq_primary: tuple[float, float, float] = (97.0, 29.0, 41.0)
    influencer_target_freq_secondary: tuple[float, float, float] = (217.0, 13.0, 7.0)
    influencer_target_amp_primary: tuple[float, float, float] = (200.0, 200.0, 100.0)
    influencer_target_amp_secondary: tuple[float, float, float] = (30.0, 30.0, 27.0)
    influencer_target_vert_offset: float = 40.0
    influencer_target_phase_offsets: tuple[float, float, float] = (0.0, 53.0, 61.0)
    # S2.E3: configurable clip bounds for influence_mode="distance"
    # (was hardcoded 0.2/0.8).
    influencer_influence_min: float = 0.2
    influencer_influence_max: float = 0.8
    # S2.E3: force rank-based influence regardless of influencer_influence_mode.
    influencer_use_rank_override: bool = False
    # S2.E2: move-then-steer is the only implemented update order; the field
    # exists so conf/*.yaml can document the contract explicitly — setting
    # it False raises at validate() time rather than being silently ignored.
    influencer_move_then_steer: bool = True
    # S2.E4: auto-apply the density-scaled Gaussian init for this mode
    # without requiring position_init="influencer_density" explicitly.
    influencer_density_scaled_init: bool = False
    # S2.E6: pilotable-flock mode — a command-queue-driven pilot point
    # replaces the Lissajous target when enabled.
    influencer_pilot_enabled: bool = False
    influencer_pilot_speed: float = 40.0    # unit-scale U per second


@dataclass
class AngleConfig:
    """Angle mode parameters (P5.1–P5.7)."""
    turn_rate: float = 120.0           # base steering rate (deg/s)
    max_turn_rate: float = 200.0        # max steering rate near boundaries (deg/s)
    turn_threshold: float = 0.5         # dead-zone half-angle (deg)
    jitter_deg: float = 4.0             # heading jitter amplitude (deg)
    base_speed: float = 150.0           # base cruise speed (units/frame)
    angle_neighbors: int = 7            # nearest neighbour count
    sep_radius_bodies: float = 1.0      # separation radius in body units
    align_radius_bodies: float = 5.0    # alignment radius in body units
    range_radius_bodies: float = 12.0   # cohesion radius in body units
    # S2.C3: adaptive speed law — linear | quadratic | softened
    # (named angle_speed_mode: "speed_mode" is already SpatialConfig's flat name)
    angle_speed_mode: str = "linear"


@dataclass
class MarlConfig:
    """MARL mode parameters (P12.1)."""
    marl_velocity_cap: float = 0.5       # v_cap multiplier for unit scale U
    marl_rule_weight: float = 0.01       # deferred-rule weight
    marl_separation_radius: float = 2.0  # sep radius in unit scale U
    marl_action_scale: float = 0.05      # external action scaling factor
    marl_episode_steps: int = 500        # truncation horizon
    # S3.9: five-term penalty-composite reward weights (analysis/rewards.py)
    marl_reward_w_a: float = 1.0         # velocity_deviation weight
    marl_reward_w_c: float = 1.0         # dispersion weight
    marl_reward_w_L: float = 0.0         # angular-momentum penalty weight
    marl_reward_w_b: float = 0.0         # boundary_overshoot weight
    marl_reward_w_z: float = 0.0         # altitude_deviation weight


@dataclass
class IndexConfig:
    """Spatial index parameters."""
    spatial_index: str = "auto"        # auto | hash_grid | kdtree | none
    topological_cap: int = 50          # cap on k-NN neighbor count
    use_toroidal_distance: bool = True


@dataclass
class RefinementConfig:
    """SI Refinements (live-mutable)."""
    refinements: bool = True
    steric: float = 0.6               # φ_s: 1/d² repulsion strength (0 = off)
    blind_deg: float = 60.0           # rear blind cone full angle (degrees)
    anisotropy: float = 2.0           # body axis ratio a/b (1.0 = isotropic)
    # steric_radius default (10.0) matches steric_force()'s prior hardcoded
    # threshold — zero behavior change out of the box. boid_size * 4.0 is
    # the Pearce-SI-faithful value (sim_new.md STERIC_RADIUS); opt in
    # explicitly rather than changing the default and altering existing runs.
    steric_radius: float = 10.0
    steric_visible_only: bool = False  # restrict steric to occlusion-visible neighbors only


@dataclass
class ExtensionConfig:
    """Extension toggles (live-mutable)."""
    predator_enabled: bool = False
    roosting_enabled: bool = False
    wander_enabled: bool = False
    ripple_enabled: bool = False
    speed_noise_enabled: bool = False
    priority_stack_enabled: bool = False
    neighbor_adaptive_speed_enabled: bool = False


@dataclass
class PredatorConfig:
    """Predator parameters."""
    predator_threat_radius: float = 12.0
    predator_strength: float = 1.0
    predator_momentum: float = 0.5
    predator_split_gain: float = 0.8
    predator_acceleration: float = 0.8      # threat steering aggressiveness
    predator_vacuole_strength: float = 0.0  # vacuole formation strength
    predator_blackening_gain: float = 0.6   # panic blackening gain
    # C1: Threat mode selector — off | cursor | orbit | autonomous
    predator_mode: str = "autonomous"


@dataclass
class RoostConfig:
    """Roost target altitude for metrics and ecology."""
    z_target: float = 500.0     # target Z altitude for altitude_deviation metric


@dataclass
class EcologyConfig:
    """Ecology parameters."""
    ecology_roost: tuple[float, float, float] = (500.0, 350.0, 40.0)
    ecology_critical_mass: int = 500
    # P4.8: Logistic dusk, seasonal amplitude, temperature boost
    ecology_dusk_width: float = 6.0          # logistic transition width (minutes)
    ecology_seasonal_amplitude: float = 0.5  # 0=no season, 1=full seasonal modulation
    ecology_temperature_boost: float = 0.3   # temperature influence on roost pull
    # S2.B8: predator-presence draw mode
    ecology_predator_presence: str = "deterministic"  # deterministic | stochastic


@dataclass
class PerfConfig:
    """Performance parameters."""
    metrics_detail_level: int = 1     # 0=off, 1=fast, 2=full
    metrics_interval: int = 60        # frames between expensive metric computations
    instance_buffer_chunk: int = 50000
    parallel_workers: int = 1         # n_jobs for occlusion culling (1 = sequential)
    # P4.4: Physical metrics — real-world unit conversions
    bird_mass_kg: float = 0.075       # typical starling mass
    cruise_speed_ms: float = 8.94     # ~32 km/h cruising speed
    acc_peak_ms2: float = 40.0        # peak acceleration in m/s²
    target_fps: int = 60              # P8.6: adaptive quality target frame rate
    history_cap: int = 10000          # D19: max metrics history entries (ring buffer)
    # C6: Numba / threading / fastmath toggles
    use_numba: bool = True            # enable numba JIT acceleration
    fastmath: bool = False            # enable fastmath (may reduce determinism)
    num_threads: int = 0              # numba threading (0 = auto)
    adaptive_quality: bool = True     # P8.6: adaptive quality governor
    readout_smooth: float = 0.04      # S3.11: EMA factor for display-only smoothed readout (0=raw)


@dataclass
class VizConfig:
    """Visualization parameters."""
    fps: int = 60
    window_width: int = 1200
    window_height: int = 800
    show_grid: bool = False
    auto_rotate: bool = False
    theme: str = "ink"                # ink | inverse | paper | graphite
    point_sprites: bool = False       # P8.1: sphere impostors vs tetrahedra
    winged_mesh: bool = True           # P8.4: winged bird mesh vs tetrahedron
    gradient_sky: bool = True          # P8.4: gradient sky background
    dual_view: bool = False             # P8.8: split-screen dual camera view
    trails: str = "off"               # P8.3: off | velocity | ring | accumulation | lines
    trail_length: int = 30            # P8.3: trail history length / fade duration
    density_mode: bool = False         # P8.11: alpha-accumulation density (murmuratR aesthetic)
    density_alpha: float = 0.2         # P8.11: sprite alpha in density mode
    # C5: Extended viz parameters
    per_bird_color: bool = False       # per-bird colour variation
    background_top: tuple[float, float, float] = (0.05, 0.05, 0.15)  # sky top colour (RGB)
    background_bottom: tuple[float, float, float] = (0.02, 0.02, 0.05)  # sky bottom colour
    bird_mesh: str = "auto"           # auto | tetrahedron | winged | sphere
    flap_period: float = 0.35         # wing flap period (seconds)
    hud: bool = True                  # show SliderHUD overlay


@dataclass
class CaptureConfig:
    """Capture parameters."""
    capture_width: int = 800
    capture_height: int = 600
    capture_frames: int = 240
    capture_every: int = 3
    capture_fps: int = 20
    capture_output: str = "output/murmuration.gif"
    capture_metrics_csv: str = "output/metrics.csv"
    capture_metrics_json: str = "output/metrics.json"
    capture_with_viz: bool = True
    capture_prewarm: int = 60          # P8.7: frames to skip before capturing
    capture_sweep: bool = True         # P8.7: cinematic camera sweep during capture
    capture_scale: float = 1.0         # P8.7: distance scale factor for sweep
    capture_mpl_fallback: bool = True   # P8.9: fall back to matplotlib when GPU unavailable
    capture_mpl_dpi: int = 72           # P8.9: DPI for matplotlib fallback renders
    capture_frame_cap: int = 10000      # D19: ring-buffer cap for frames + metrics
