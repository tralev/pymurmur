"""Simulation configuration — the shared parameter contract.

Level 2 — depends on PyYAML (stdlib only otherwise). Every component
reads from SimConfig; only InputControl and __main__ write to it.

I7.1: SimConfig composes sub-dataclasses (DomainConfig, FlockConfig, etc.)
with flat attribute access via __getattr__/__setattr__ for backward compat.

YAML nesting convention (I7.4 — collision-free):
    Sections nest by sub-config; leaf keys are flat _FIELD_MAP names.
    No two sections share the same leaf key — round-trip is exact.
    domain.width     → width        capture.width → capture_width
    flock.num_boids  → num_boids    capture.fps   → capture_fps
    projection.phi_p → (nested only — flat shim retired)
    extensions       → predator_enabled
    spatial.sep_wt   → separation_weight
    visual.fps       → fps          performance   → metrics_interval
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
    noise_mode: str = "additive"       # additive | maxwellian | none | seed_sinusoidal
    acceleration_scale: float = 0.3
    influence_count: int = 7           # P4.1: max topological neighbours (hybrid filter)
    speed_mode: str = "clamp"          # clamp | band | none — how speed is enforced
    flow_weight: float = 0.0           # P11.5: global flow contribution weight
    neighbor_filter: str = "hybrid"    # hybrid | metric | topological | global | none
    separation_kernel: str = "sum"     # sum | mean | unit — how sep forces are combined
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


@dataclass
class ExtensionConfig:
    """Extension toggles (live-mutable)."""
    predator_enabled: bool = False
    roosting_enabled: bool = False
    wander_enabled: bool = False
    ripple_enabled: bool = False


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


# ── Flat-field-name → (sub_config_attr, field_name) mapping (I7.1) ─

_FIELD_MAP: dict[str, tuple[str, str]] = {
    # DomainConfig
    "width": ("_domain", "width"),
    "height": ("_domain", "height"),
    "depth": ("_domain", "depth"),
    # FlockConfig
    "num_boids": ("_flock", "num_boids"),
    "boid_size": ("_flock", "boid_size"),
    "v0": ("_flock", "v0"),
    "max_force": ("_flock", "max_force"),
    "visual_range": ("_flock", "visual_range"),
    "dt_phys": ("_flock", "dt_phys"),  # P8.10
    "speed_min_factor": ("_flock", "speed_min_factor"),  # P11.5
    "n_predators": ("_flock", "n_predators"),  # C4
    # BoundaryConfig
    "boundary_mode": ("_boundary", "boundary_mode"),
    "boundary_sphere_radius": ("_boundary", "boundary_sphere_radius"),
    "boundary_avoidance_factor": ("_boundary", "boundary_avoidance_factor"),
    "boundary_radius_factor": ("_boundary", "boundary_radius_factor"),
    "boundary_margin": ("_boundary", "boundary_margin"),  # C1
    # ProjectionConfig
    "phi_a": ("_projection", "phi_a"),
    "sigma": ("_projection", "sigma"),
    "max_visibility": ("_projection", "max_visibility"),  # C4
    "max_occlusion_neighbors": ("_projection", "max_occlusion_neighbors"),  # C4
    # SpatialConfig
    "separation_weight": ("_spatial", "separation_weight"),
    "alignment_weight": ("_spatial", "alignment_weight"),
    "cohesion_weight": ("_spatial", "cohesion_weight"),
    "noise_scale": ("_spatial", "noise_scale"),
    "acceleration_scale": ("_spatial", "acceleration_scale"),
    "influence_count": ("_spatial", "influence_count"),
    "predator_escape_factor": ("_spatial", "predator_escape_factor"),
    "predator_speed_boost": ("_spatial", "predator_speed_boost"),
    "predator_perception_boost": ("_spatial", "predator_perception_boost"),
    "predator_accel_boost": ("_spatial", "predator_accel_boost"),
    "jitter_separation": ("_spatial", "jitter_separation"),
    "jitter_cohesion": ("_spatial", "jitter_cohesion"),
    "jitter_alignment": ("_spatial", "jitter_alignment"),
    # SpatialConfig — new leaves
    "noise_mode": ("_spatial", "noise_mode"),
    "speed_mode": ("_spatial", "speed_mode"),
    "flow_weight": ("_spatial", "flow_weight"),
    "neighbor_filter": ("_spatial", "neighbor_filter"),
    "separation_kernel": ("_spatial", "separation_kernel"),
    "max_dist_sep": ("_spatial", "max_dist_sep"),
    "max_dist_align": ("_spatial", "max_dist_align"),
    "max_dist_coh": ("_spatial", "max_dist_coh"),
    "angle_sep": ("_spatial", "angle_sep"),
    "angle_align": ("_spatial", "angle_align"),
    "angle_coh": ("_spatial", "angle_coh"),
    "coherence_factor": ("_spatial", "coherence_factor"),
    "w_fwd": ("_spatial", "w_fwd"),
    # SpatialConfig — obstacle avoidance (EvoFlock genes)
    "static_avoid_weight": ("_spatial", "static_avoid_weight"),
    "predictive_avoid_weight": ("_spatial", "predictive_avoid_weight"),
    "fly_away_max_dist": ("_spatial", "fly_away_max_dist"),
    "min_time_to_collide": ("_spatial", "min_time_to_collide"),
    # FieldConfig
    "field_separation": ("_field", "field_separation"),
    "field_alignment": ("_field", "field_alignment"),
    "field_cohesion": ("_field", "field_cohesion"),
    "field_flow": ("_field", "field_flow"),
    "field_chase_strength": ("_field", "field_chase_strength"),
    "field_noise": ("_field", "field_noise"),
    "field_target_pull": ("_field", "field_target_pull"),
    "field_drift_pull": ("_field", "field_drift_pull"),
    "field_drift_direction": ("_field", "field_drift_direction"),
    "field_shell_influence": ("_field", "field_shell_influence"),
    "field_tangent_pull": ("_field", "field_tangent_pull"),
    "field_wave_gain": ("_field", "field_wave_gain"),
    "field_ripple_trains": ("_field", "field_ripple_trains"),
    "field_inertia": ("_field", "field_inertia"),
    "field_shell_radius_base": ("_field", "field_shell_radius_base"),
    "field_inner_radius_factor": ("_field", "field_inner_radius_factor"),
    "field_num_groups": ("_field", "field_num_groups"),
    "field_leader_fraction": ("_field", "field_leader_fraction"),
    # FieldConfig — new leaves
    "field_flow_pull": ("_field", "field_flow_pull"),
    "field_unit_scale": ("_field", "field_unit_scale"),
    "disabled_terms": ("_field", "disabled_terms"),
    # WanderConfig
    "wander_attractor_speed": ("_wander", "wander_attractor_speed"),
    "wander_attractor_radius": ("_wander", "wander_attractor_radius"),
    # VicsekConfig
    "vicsek_couplage": ("_vicsek", "vicsek_couplage"),
    "vicsek_diffusion": ("_vicsek", "vicsek_diffusion"),
    "vicsek_radius_influence": ("_vicsek", "vicsek_radius_influence"),
    "vicsek_radius_avoid": ("_vicsek", "vicsek_radius_avoid"),
    "vicsek_velocity": ("_vicsek", "vicsek_velocity"),
    "vicsek_time_step": ("_vicsek", "vicsek_time_step"),
    "vicsek_radius_predators": ("_vicsek", "vicsek_radius_predators"),
    "vicsek_velocity_predator": ("_vicsek", "vicsek_velocity_predator"),
    "vicsek_detect_ratio": ("_vicsek", "vicsek_detect_ratio"),
    "vicsek_weight_afraid": ("_vicsek", "vicsek_weight_afraid"),
    "vicsek_predator_noise_ratio": ("_vicsek", "vicsek_predator_noise_ratio"),
    # InfluencerConfig
    "influencer_rank_exponent": ("_influencer", "influencer_rank_exponent"),
    "influencer_substeps": ("_influencer", "influencer_substeps"),
    "influencer_scale": ("_influencer", "influencer_scale"),
    "influencer_influence_mode": ("_influencer", "influencer_influence_mode"),
    "influencer_near_dist_sq": ("_influencer", "influencer_near_dist_sq"),
    "influencer_init_separation": ("_influencer", "influencer_init_separation"),
    "influencer_tick_rate": ("_influencer", "influencer_tick_rate"),
    # AngleConfig
    "turn_rate": ("_angle", "turn_rate"),
    "max_turn_rate": ("_angle", "max_turn_rate"),
    "turn_threshold": ("_angle", "turn_threshold"),
    "jitter_deg": ("_angle", "jitter_deg"),
    "base_speed": ("_angle", "base_speed"),
    "angle_neighbors": ("_angle", "angle_neighbors"),
    "sep_radius_bodies": ("_angle", "sep_radius_bodies"),
    "align_radius_bodies": ("_angle", "align_radius_bodies"),
    "range_radius_bodies": ("_angle", "range_radius_bodies"),
    "angle_speed_mode": ("_angle", "angle_speed_mode"),
    # MarlConfig
    "marl_velocity_cap": ("_marl", "marl_velocity_cap"),
    "marl_rule_weight": ("_marl", "marl_rule_weight"),
    "marl_separation_radius": ("_marl", "marl_separation_radius"),
    "marl_action_scale": ("_marl", "marl_action_scale"),
    "marl_episode_steps": ("_marl", "marl_episode_steps"),
    # IndexConfig
    "spatial_index": ("_index", "spatial_index"),
    "topological_cap": ("_index", "topological_cap"),
    "use_toroidal_distance": ("_index", "use_toroidal_distance"),
    # RefinementConfig
    "refinements": ("_refinement", "refinements"),
    "steric": ("_refinement", "steric"),
    "blind_deg": ("_refinement", "blind_deg"),
    "anisotropy": ("_refinement", "anisotropy"),
    # ExtensionConfig
    "predator_enabled": ("_extension", "predator_enabled"),
    "roosting_enabled": ("_extension", "roosting_enabled"),
    "wander_enabled": ("_extension", "wander_enabled"),
    "ripple_enabled": ("_extension", "ripple_enabled"),
    # PredatorConfig
    "predator_threat_radius": ("_predator", "predator_threat_radius"),
    "predator_strength": ("_predator", "predator_strength"),
    "predator_momentum": ("_predator", "predator_momentum"),
    "predator_split_gain": ("_predator", "predator_split_gain"),
    "predator_acceleration": ("_predator", "predator_acceleration"),
    "predator_vacuole_strength": ("_predator", "predator_vacuole_strength"),
    "predator_blackening_gain": ("_predator", "predator_blackening_gain"),
    "predator_mode": ("_predator", "predator_mode"),
    # EcologyConfig
    "ecology_roost": ("_ecology", "ecology_roost"),
    "ecology_critical_mass": ("_ecology", "ecology_critical_mass"),
    "ecology_dusk_width": ("_ecology", "ecology_dusk_width"),
    "ecology_seasonal_amplitude": ("_ecology", "ecology_seasonal_amplitude"),
    "ecology_temperature_boost": ("_ecology", "ecology_temperature_boost"),
    "ecology_predator_presence": ("_ecology", "ecology_predator_presence"),
    # RoostConfig
    "roost_z_target": ("_roost", "z_target"),
    # PerfConfig
    "metrics_detail_level": ("_perf", "metrics_detail_level"),
    "metrics_interval": ("_perf", "metrics_interval"),
    "instance_buffer_chunk": ("_perf", "instance_buffer_chunk"),
    "parallel_workers": ("_perf", "parallel_workers"),
    "bird_mass_kg": ("_perf", "bird_mass_kg"),
    "cruise_speed_ms": ("_perf", "cruise_speed_ms"),
    "acc_peak_ms2": ("_perf", "acc_peak_ms2"),
    "target_fps": ("_perf", "target_fps"),           # P8.6
    "history_cap": ("_perf", "history_cap"),         # D19
    # PerfConfig — new leaves
    "use_numba": ("_perf", "use_numba"),
    "fastmath": ("_perf", "fastmath"),
    "num_threads": ("_perf", "num_threads"),
    "adaptive_quality": ("_perf", "adaptive_quality"),
    "readout_smooth": ("_perf", "readout_smooth"),
    # VizConfig
    "fps": ("_viz", "fps"),
    "window_width": ("_viz", "window_width"),
    "window_height": ("_viz", "window_height"),
    "show_grid": ("_viz", "show_grid"),
    "auto_rotate": ("_viz", "auto_rotate"),
    "theme": ("_viz", "theme"),
    "point_sprites": ("_viz", "point_sprites"),     # P8.1
    "winged_mesh": ("_viz", "winged_mesh"),         # P8.4
    "gradient_sky": ("_viz", "gradient_sky"),       # P8.4
    "dual_view": ("_viz", "dual_view"),               # P8.8
    "trails": ("_viz", "trails"),                   # P8.3
    "trail_length": ("_viz", "trail_length"),       # P8.3
    "density_mode": ("_viz", "density_mode"),       # P8.11
    "density_alpha": ("_viz", "density_alpha"),     # P8.11
    # VizConfig — new leaves
    "per_bird_color": ("_viz", "per_bird_color"),
    "background_top": ("_viz", "background_top"),
    "background_bottom": ("_viz", "background_bottom"),
    "bird_mesh": ("_viz", "bird_mesh"),
    "flap_period": ("_viz", "flap_period"),
    "hud": ("_viz", "hud"),
    # CaptureConfig
    "capture_width": ("_capture", "capture_width"),
    "capture_height": ("_capture", "capture_height"),
    "capture_frames": ("_capture", "capture_frames"),
    "capture_every": ("_capture", "capture_every"),
    "capture_fps": ("_capture", "capture_fps"),
    "capture_output": ("_capture", "capture_output"),
    "capture_metrics_csv": ("_capture", "capture_metrics_csv"),
    "capture_metrics_json": ("_capture", "capture_metrics_json"),
    "capture_with_viz": ("_capture", "capture_with_viz"),
    "capture_prewarm": ("_capture", "capture_prewarm"),     # P8.7
    "capture_sweep": ("_capture", "capture_sweep"),         # P8.7
    "capture_scale": ("_capture", "capture_scale"),         # P8.7
    "capture_mpl_fallback": ("_capture", "capture_mpl_fallback"),  # P8.9
    "capture_mpl_dpi": ("_capture", "capture_mpl_dpi"),            # P8.9
    "capture_frame_cap": ("_capture", "capture_frame_cap"),          # D19
}

# Fields not in _FIELD_MAP are stored directly on SimConfig
_DIRECT_FIELDS: set[str] = {"mode", "seed", "position_init", "velocity_init"}

# Nested-only fields: no flat alias at all (shim fully retired).
# Access via sub-config only, e.g. config.projection.phi_p.
# from_file() routes their YAML keys to the sub-config explicitly.
_NESTED_ONLY: dict[str, tuple[str, str]] = {
    "phi_p": ("_projection", "phi_p"),
}

# All known field names (for YAML from_file filtering)
_ALL_FIELD_NAMES: set[str] = set(_FIELD_MAP.keys()) | _DIRECT_FIELDS | set(_NESTED_ONLY.keys())

# G5: Top-level YAML keys that are deliberately NOT SimConfig fields —
# consumed by a separate loader instead (e.g. analysis/evoflock.py's
# load_obstacle_scene reads `obstacles:` directly from the raw YAML).
# Exempted from from_file's unknown-key check by *name*, not by value
# shape, so a typo'd list key is still caught.
_NON_FIELD_TOP_LEVEL_LISTS: set[str] = {"obstacles"}


# Known tuple-typed fields that need YAML round-trip coercion.
# YAML loads sequences as Python lists; these fields expect tuples.
_TUPLE_FIELDS: set[str] = {
    "background_top", "background_bottom",
    "field_drift_direction", "ecology_roost",
}


def _coerce_tuples(cfg: SimConfig) -> None:
    """Post-load: cast list values to tuples for tuple-typed dataclass fields.

    YAML parses sequences as Python lists, but dataclass type hints expect
    tuples.  Walk each known tuple field and coerce list→tuple.
    Done after the config is fully constructed, so setattr delegation works.
    """
    for flat_name in _TUPLE_FIELDS:
        sub_attr, field_name = _FIELD_MAP[flat_name]
        sub_cfg = object.__getattribute__(cfg, sub_attr)
        val = getattr(sub_cfg, field_name)
        if isinstance(val, list):
            object.__setattr__(sub_cfg, field_name, tuple(val))


# ── SimConfig (I7.1: composed, not a dataclass) ───────────────────

class SimConfig:
    """Shared parameter contract between every subsystem.

    I7.1: Composes sub-dataclasses (DomainConfig, FlockConfig, etc.).
    Flat attribute access (config.width, config.v0) is preserved via
    __getattr__/__setattr__ delegation for backward compatibility.
    Sub-configs are accessible directly: config.domain, config.flock, etc.
    """

    def __init__(self, **kwargs: Any) -> None:
        # Sub-config instances
        object.__setattr__(self, "_domain", DomainConfig())
        object.__setattr__(self, "_flock", FlockConfig())
        object.__setattr__(self, "_boundary", BoundaryConfig())
        object.__setattr__(self, "_projection", ProjectionConfig())
        object.__setattr__(self, "_spatial", SpatialConfig())
        object.__setattr__(self, "_field", FieldConfig())
        object.__setattr__(self, "_wander", WanderConfig())
        object.__setattr__(self, "_vicsek", VicsekConfig())
        object.__setattr__(self, "_influencer", InfluencerConfig())
        object.__setattr__(self, "_angle", AngleConfig())
        object.__setattr__(self, "_marl", MarlConfig())
        object.__setattr__(self, "_index", IndexConfig())
        object.__setattr__(self, "_refinement", RefinementConfig())
        object.__setattr__(self, "_extension", ExtensionConfig())
        object.__setattr__(self, "_predator", PredatorConfig())
        object.__setattr__(self, "_ecology", EcologyConfig())
        object.__setattr__(self, "_roost", RoostConfig())
        object.__setattr__(self, "_perf", PerfConfig())
        object.__setattr__(self, "_viz", VizConfig())
        object.__setattr__(self, "_capture", CaptureConfig())

        # Direct fields
        object.__setattr__(self, "mode", kwargs.get("mode", "projection"))
        object.__setattr__(self, "seed", kwargs.get("seed", None))
        object.__setattr__(self, "position_init", kwargs.get("position_init", "box"))
        object.__setattr__(self, "velocity_init", kwargs.get("velocity_init", "sphere"))

        # P3.2: Per-config field mode time — set by engine.step() before
        # force computation.  Private field, NOT in _FIELD_MAP (not YAML-serialised).
        object.__setattr__(self, "_field_time", 0.0)

        # Apply kwargs to sub-configs and direct fields
        for key, value in kwargs.items():
            if key in _DIRECT_FIELDS or key.startswith("_"):
                continue  # already set above
            if key in _FIELD_MAP:
                sub_attr, field_name = _FIELD_MAP[key]
                sub_cfg = getattr(self, sub_attr)
                setattr(sub_cfg, field_name, value)

    # ── Sub-config accessors ─────────────────────────────────

    @property
    def domain(self) -> DomainConfig:
        return self._domain

    @property
    def flock(self) -> FlockConfig:
        return self._flock

    @property
    def boundary(self) -> BoundaryConfig:
        return self._boundary

    @property
    def projection(self) -> ProjectionConfig:
        return self._projection

    @property
    def spatial(self) -> SpatialConfig:
        return self._spatial

    @property
    def field(self) -> FieldConfig:
        return self._field

    @property
    def wander(self) -> WanderConfig:
        return self._wander

    @property
    def vicsek(self) -> VicsekConfig:
        return self._vicsek

    @property
    def influencer(self) -> InfluencerConfig:
        return self._influencer

    @property
    def angle(self) -> AngleConfig:
        return self._angle

    @property
    def marl(self) -> MarlConfig:
        return self._marl

    @property
    def index(self) -> IndexConfig:
        return self._index

    @property
    def refinement(self) -> RefinementConfig:
        return self._refinement

    @property
    def extension(self) -> ExtensionConfig:
        return self._extension

    @property
    def predator(self) -> PredatorConfig:
        return self._predator

    @property
    def ecology(self) -> EcologyConfig:
        return self._ecology

    @property
    def roost(self) -> RoostConfig:
        return self._roost

    @property
    def perf(self) -> PerfConfig:
        return self._perf

    @property
    def viz(self) -> VizConfig:
        return self._viz

    @property
    def capture(self) -> CaptureConfig:
        return self._capture

    # ── Flat access delegation (backward compat) ──────────────

    def __getattr__(self, name: str) -> Any:
        """Delegate flat attribute access to the correct sub-config."""
        if name in _FIELD_MAP:
            sub_attr, field_name = _FIELD_MAP[name]
            sub_cfg = object.__getattribute__(self, sub_attr)
            return getattr(sub_cfg, field_name)
        raise AttributeError(
            f"'SimConfig' has no attribute '{name}'"
        )

    def __setattr__(self, name: str, value: Any) -> None:
        """Delegate flat attribute mutation to the correct sub-config."""
        if name in _FIELD_MAP:
            sub_attr, field_name = _FIELD_MAP[name]
            sub_cfg = object.__getattribute__(self, sub_attr)
            object.__setattr__(sub_cfg, field_name, value)
        else:
            object.__setattr__(self, name, value)

    # ── YAML I/O ──────────────────────────────────────────────

    @classmethod
    def from_file(cls, path: str | Path, strict: bool = True) -> "SimConfig":
        """Load config from a YAML file. Nested keys are flattened.

        Args:
            path: YAML file path.
            strict: if True (default), unknown section keys raise ValueError
                    naming the section and key (G5). Set False for configs
                    that carry extra sections (e.g. evoflock GA parameters).

        Raises FileNotFoundError if path doesn't exist.
        Raises ValueError if strict=True and unknown keys are found.
        """
        import yaml  # type: ignore[import-untyped]

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        raw = yaml.safe_load(path.read_text()) or {}
        flat: dict[str, Any] = {}

        # Flatten nested sections with section-aware key normalisation
        unknown_keys: list[tuple[str, str]] = []  # G5: track unknown for actionable errors
        for section_name, section_data in raw.items():
            if isinstance(section_data, dict):
                for key, value in section_data.items():
                    # Normalise short keys: field/noise → field_noise,
                    # wander/attractor_speed → wander_attractor_speed, etc.
                    if not key.startswith(f"{section_name}_"):
                        prefixed = f"{section_name}_{key}"
                        if prefixed in _ALL_FIELD_NAMES:
                            key = prefixed
                    # Special case: 'performance'/'metrics' → metrics_ prefix
                    if section_name in ("performance", "metrics"):
                        if not key.startswith("metrics_"):
                            metrics_key = f"metrics_{key}"
                            if metrics_key in _ALL_FIELD_NAMES:
                                key = metrics_key
                    # G5: Track unknown keys for actionable error messages
                    if key in _ALL_FIELD_NAMES:
                        flat[key] = value
                    else:
                        unknown_keys.append((section_name, key))
            elif isinstance(section_data, list):
                # Non-field top-level list, e.g. `obstacles:` (a scene spec
                # consumed separately by analysis/evoflock.py's
                # load_obstacle_scene, not a SimConfig field). Only the
                # known name is exempt — a typo'd list key (e.g.
                # `obstalces:`) is still a mistake worth surfacing.
                if section_name not in _NON_FIELD_TOP_LEVEL_LISTS:
                    unknown_keys.append(("<top-level>", section_name))
            else:
                # G5: top-level (non-nested) scalar key, e.g. `mode: spatial`.
                # Must still be validated — an unrecognized one (typo'd
                # section/field name) was previously swallowed silently.
                if section_name in _ALL_FIELD_NAMES:
                    flat[section_name] = section_data
                else:
                    unknown_keys.append(("<top-level>", section_name))

        # G5: Actionable YAML errors — name the offending key AND section
        if strict and unknown_keys:
            lines = [f"  [{sec}] {key}" for sec, key in unknown_keys]
            raise ValueError(
                f"Unknown config keys in {path.name}:\n"
                + "\n".join(lines)
                + f"\n\nKnown fields (non-exhaustive): {sorted(_ALL_FIELD_NAMES)[:20]}..."
            )

        # Nested-only fields (flat shim retired) — route explicitly
        nested_vals = {
            key: flat.pop(key) for key in list(_NESTED_ONLY) if key in flat
        }

        # Filter to known fields only
        filtered = {k: v for k, v in flat.items() if k in _ALL_FIELD_NAMES}

        cfg = cls(**filtered)
        for key, value in nested_vals.items():
            sub_attr, field_name = _NESTED_ONLY[key]
            setattr(getattr(cfg, sub_attr), field_name, value)

        # YAML round-trip: coerce lists back to tuples for tuple-typed
        # dataclass fields (background_top/bottom, ecology_roost, etc.)
        _coerce_tuples(cfg)
        return cfg

    def to_file(self, path: str | Path) -> None:
        """Write config to a YAML file. Round-trip preserves all fields."""
        import yaml

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data: dict[str, Any] = {
            "domain": {"width": self.width, "height": self.height,
                       "depth": self.depth},
            "flock": {"num_boids": self.num_boids, "boid_size": self.boid_size,
                      "v0": self.v0, "max_force": self.max_force,
                      "dt_phys": self.dt_phys,
                      "speed_min_factor": self.speed_min_factor,
                      "n_predators": self.n_predators,
                      "visual_range": self.visual_range},
            "mode": self.mode,
            "projection": {"phi_p": self.projection.phi_p, "phi_a": self.phi_a,
                           "sigma": self.sigma,
                           "max_visibility": self.max_visibility,
                           "max_occlusion_neighbors": self.max_occlusion_neighbors},
            "spatial": {"separation_weight": self.separation_weight,
                        "alignment_weight": self.alignment_weight,
                        "cohesion_weight": self.cohesion_weight,
                        "noise_scale": self.noise_scale,
                        "noise_mode": self.noise_mode,
                        "acceleration_scale": self.acceleration_scale,
                        "influence_count": self.influence_count,
                        "speed_mode": self.speed_mode,
                        "flow_weight": self.flow_weight,
                        "neighbor_filter": self.neighbor_filter,
                        "separation_kernel": self.separation_kernel,
                        "max_dist_sep": self.max_dist_sep,
                        "max_dist_align": self.max_dist_align,
                        "max_dist_coh": self.max_dist_coh,
                        "angle_sep": self.angle_sep,
                        "angle_align": self.angle_align,
                        "angle_coh": self.angle_coh,
                        "coherence_factor": self.coherence_factor,
                        "w_fwd": self.w_fwd,
                        "predator_escape_factor": self.predator_escape_factor,
                        "predator_speed_boost": self.predator_speed_boost,
                        "predator_perception_boost": self.predator_perception_boost,
                        "predator_accel_boost": self.predator_accel_boost,
                        "jitter_separation": self.jitter_separation,
                        "jitter_cohesion": self.jitter_cohesion,
                        "jitter_alignment": self.jitter_alignment,
                        "static_avoid_weight": self.static_avoid_weight,
                        "predictive_avoid_weight": self.predictive_avoid_weight,
                        "fly_away_max_dist": self.fly_away_max_dist,
                        "min_time_to_collide": self.min_time_to_collide},
            "boundary": {"boundary_mode": self.boundary_mode,
                         "boundary_sphere_radius": self.boundary_sphere_radius,
                         "boundary_avoidance_factor": self.boundary_avoidance_factor,
                         "boundary_radius_factor": self.boundary_radius_factor,
                         "boundary_margin": self.boundary_margin},
            "refinements": {"refinements": self.refinements,
                            "steric": self.steric,
                            "blind_deg": self.blind_deg,
                            "anisotropy": self.anisotropy},
            "extensions": {"predator_enabled": self.predator_enabled,
                           "roosting_enabled": self.roosting_enabled,
                           "wander_enabled": self.wander_enabled,
                           "ripple_enabled": self.ripple_enabled},
            "predator": {"predator_threat_radius": self.predator_threat_radius,
                         "predator_strength": self.predator_strength,
                         "predator_momentum": self.predator_momentum,
                         "predator_split_gain": self.predator_split_gain,
                         "predator_acceleration": self.predator_acceleration,
                         "predator_vacuole_strength": self.predator_vacuole_strength,
                         "predator_blackening_gain": self.predator_blackening_gain,
                         "predator_mode": self.predator_mode},
            "ecology": {"ecology_roost": list(self.ecology_roost),
                        "ecology_critical_mass": self.ecology_critical_mass,
                        "ecology_dusk_width": self.ecology_dusk_width,
                        "ecology_seasonal_amplitude": self.ecology_seasonal_amplitude,
                        "ecology_temperature_boost": self.ecology_temperature_boost,
                        "ecology_predator_presence": self.ecology_predator_presence},
            "roost": {"roost_z_target": self.roost.z_target},
            "vicsek": {"vicsek_couplage": self.vicsek_couplage,
                       "vicsek_diffusion": self.vicsek_diffusion,
                       "vicsek_radius_influence": self.vicsek_radius_influence,
                       "vicsek_radius_avoid": self.vicsek_radius_avoid,
                       "vicsek_velocity": self.vicsek_velocity,
                       "vicsek_time_step": self.vicsek_time_step,
                       "vicsek_radius_predators": self.vicsek_radius_predators,
                       "vicsek_velocity_predator": self.vicsek_velocity_predator,
                       "vicsek_detect_ratio": self.vicsek_detect_ratio,
                       "vicsek_weight_afraid": self.vicsek_weight_afraid,
                       "vicsek_predator_noise_ratio": self.vicsek_predator_noise_ratio},
            "influencer": {"influencer_rank_exponent": self.influencer_rank_exponent,
                           "influencer_substeps": self.influencer_substeps,
                           "influencer_scale": self.influencer_scale,
                           "influencer_influence_mode": self.influencer_influence_mode,
                           "influencer_near_dist_sq": self.influencer_near_dist_sq,
                           "influencer_init_separation": self.influencer_init_separation,
                           "influencer_tick_rate": self.influencer_tick_rate},
            "field": {"field_separation": self.field_separation,
                      "field_alignment": self.field_alignment,
                      "field_cohesion": self.field_cohesion,
                      "field_flow": self.field_flow,
                      "field_chase_strength": self.field_chase_strength,
                      "field_noise": self.field_noise,
                      "field_target_pull": self.field_target_pull,
                      "field_drift_pull": self.field_drift_pull,
                      "field_drift_direction": list(self.field_drift_direction),
                      "field_shell_influence": self.field_shell_influence,
                      "field_tangent_pull": self.field_tangent_pull,
                      "field_wave_gain": self.field_wave_gain,
                      "field_ripple_trains": self.field_ripple_trains,
                      "field_inertia": self.field_inertia,
                      "field_shell_radius_base": self.field_shell_radius_base,
                      "field_inner_radius_factor": self.field_inner_radius_factor,
                      "field_num_groups": self.field_num_groups,
                      "field_leader_fraction": self.field_leader_fraction,
                      "field_flow_pull": self.field_flow_pull,
                      "field_unit_scale": self.field_unit_scale,
                      "disabled_terms": list(self.disabled_terms) if self.disabled_terms else []},
            "index": {"spatial_index": self.spatial_index,
                      "topological_cap": self.topological_cap,
                      "use_toroidal_distance": self.use_toroidal_distance},
            "performance": {"target_fps": self.target_fps,
                            "metrics_detail_level": self.metrics_detail_level,
                            "metrics_interval": self.metrics_interval,
                            "instance_buffer_chunk": self.instance_buffer_chunk,
                            "parallel_workers": self.parallel_workers,
                            "bird_mass_kg": self.bird_mass_kg,
                            "cruise_speed_ms": self.cruise_speed_ms,
                            "acc_peak_ms2": self.acc_peak_ms2,
                            "history_cap": self.history_cap,
                            "use_numba": self.use_numba,
                            "fastmath": self.fastmath,
                            "num_threads": self.num_threads,
                            "adaptive_quality": self.adaptive_quality,
                            "readout_smooth": self.readout_smooth},
            "visual": {"fps": self.fps,
                       "window_width": self.window_width,
                       "window_height": self.window_height,
                       "show_grid": self.show_grid,
                       "auto_rotate": self.auto_rotate,
                       "theme": self.theme,
                       "point_sprites": self.point_sprites,
                       "winged_mesh": self.winged_mesh,
                       "gradient_sky": self.gradient_sky,
                       "dual_view": self.dual_view,
                       "trails": self.trails,
                       "trail_length": self.trail_length,
                       "density_mode": self.density_mode,
                       "density_alpha": self.density_alpha,
                       "per_bird_color": self.per_bird_color,
                       "background_top": list(self.background_top),
                       "background_bottom": list(self.background_bottom),
                       "bird_mesh": self.bird_mesh,
                       "flap_period": self.flap_period,
                       "hud": self.hud},
            "capture": {"capture_width": self.capture_width,
                        "capture_prewarm": self.capture_prewarm,
                        "capture_sweep": self.capture_sweep,
                        "capture_scale": self.capture_scale,
                        "capture_height": self.capture_height,
                        "capture_frames": self.capture_frames,
                        "capture_every": self.capture_every,
                        "capture_fps": self.capture_fps,
                        "capture_output": self.capture_output,
                        "capture_metrics_csv": self.capture_metrics_csv,
                        "capture_metrics_json": self.capture_metrics_json,
                        "capture_with_viz": self.capture_with_viz,
                        "capture_mpl_fallback": self.capture_mpl_fallback,
                        "capture_mpl_dpi": self.capture_mpl_dpi},
            "wander": {"wander_attractor_speed": self.wander_attractor_speed,
                      "wander_attractor_radius": self.wander_attractor_radius},
            "angle": {"turn_rate": self.turn_rate,
                      "max_turn_rate": self.max_turn_rate,
                      "turn_threshold": self.turn_threshold,
                      "jitter_deg": self.jitter_deg,
                      "base_speed": self.base_speed,
                      "angle_neighbors": self.angle_neighbors,
                      "sep_radius_bodies": self.sep_radius_bodies,
                      "align_radius_bodies": self.align_radius_bodies,
                      "range_radius_bodies": self.range_radius_bodies,
                      "angle_speed_mode": self.angle_speed_mode},
            "marl": {"marl_velocity_cap": self.marl_velocity_cap,
                     "marl_rule_weight": self.marl_rule_weight,
                     "marl_separation_radius": self.marl_separation_radius,
                     "marl_action_scale": self.marl_action_scale,
                     "marl_episode_steps": self.marl_episode_steps},
            "seed": self.seed,
            "velocity_init": self.velocity_init,
        }

        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    # ── Validation ───────────────────────────────────────────

    _VALID_MODES = {"projection", "spatial", "field", "vicsek", "influencer", "angle", "marl"}
    _VALID_BOUNDARY_MODES = {"toroidal", "open", "margin", "sphere", "sphere_soft"}
    _VALID_INDEX_TYPES = {"auto", "hash_grid", "kdtree", "none"}
    # S4.4a: Valid themes and mesh names — mirror mesh_registry.py values.
    # Defined here statically to avoid core→viz import (forbidden per arch.md).
    _VALID_THEMES = frozenset({"ink", "inverse", "paper", "graphite"})
    _VALID_MESH_NAMES = frozenset({
        "auto", "sphere", "tetra", "winged", "impostor",
        "ellipsoid", "cone", "arrow", "points",
    })
    _VALID_METRICS_LEVELS = {0, 1, 2}
    _VALID_POSITION_INITS = {
        "box", "random", "sphere", "gaussian", "grid", "sphere_shell", "blob",
        "influencer_density",  # C4: composer for influencer_density_init
    }
    _VALID_VELOCITY_INITS = {"sphere", "blob", "drift", "cube", "speed_uniform", "tangential", "fixed"}
    _VALID_PREDATOR_MODES = {"off", "cursor", "orbit", "autonomous"}
    _VALID_ANGLE_SPEED_MODES = {"linear", "quadratic", "softened"}

    def validate(self) -> None:
        """Check cross-field consistency.

        Raises ValueError with all issues aggregated if any rule fails.
        Call at engine creation time to catch misconfiguration early.
        """
        issues: list[str] = []
        cfg = self  # shorthand for flat access

        # ── Type guards: catch non-numeric values early ───────
        _numeric_fields = (
            "width", "height", "depth",
            "num_boids", "boid_size", "v0", "max_force", "visual_range",
            "phi_a", "sigma",
            "separation_weight", "alignment_weight", "cohesion_weight",
            "noise_scale", "acceleration_scale",
            "influence_count",
            "predator_escape_factor", "predator_speed_boost",
            "predator_perception_boost",            "predator_accel_boost",
            "jitter_separation", "jitter_cohesion", "jitter_alignment",
            "steric", "blind_deg", "anisotropy",
            "parallel_workers", "metrics_interval", "metrics_detail_level",
            "bird_mass_kg", "cruise_speed_ms", "acc_peak_ms2",
            "topological_cap",
            "boundary_sphere_radius",
            "fps", "window_width", "window_height",
            "capture_width", "capture_height", "capture_frames",
            "capture_every", "capture_fps",
            "vicsek_couplage", "vicsek_diffusion",
            "vicsek_radius_influence", "vicsek_radius_avoid",
            "vicsek_velocity", "vicsek_time_step",
            "vicsek_radius_predators", "vicsek_velocity_predator",
            "vicsek_detect_ratio", "vicsek_weight_afraid",
            "vicsek_predator_noise_ratio",
            "vicsek_radius_predators", "vicsek_velocity_predator",
            "vicsek_detect_ratio", "vicsek_weight_afraid",
            "vicsek_predator_noise_ratio",
            "influencer_rank_exponent", "influencer_substeps",
            "influencer_scale",
            "influencer_near_dist_sq", "influencer_init_separation",
            "influencer_tick_rate",
            "predator_threat_radius", "predator_strength",
            "predator_momentum", "predator_split_gain",
            "field_separation", "field_alignment", "field_cohesion",
            "field_flow", "field_chase_strength",
            "field_noise", "field_target_pull", "field_drift_pull",
            "field_shell_influence", "field_tangent_pull",
            "field_wave_gain", "field_inertia",
            "field_shell_radius_base", "field_ripple_trains",
            "field_inner_radius_factor", "field_leader_fraction",
            "field_num_groups",
            "wander_attractor_speed", "wander_attractor_radius",
            "boundary_avoidance_factor", "boundary_radius_factor",
            "acceleration_scale",
            "ecology_dusk_width", "ecology_seasonal_amplitude",
            "ecology_temperature_boost",
            "trail_length",
            # AngleConfig
            "turn_rate", "max_turn_rate", "turn_threshold",
            "jitter_deg", "base_speed", "angle_neighbors",
            "sep_radius_bodies", "align_radius_bodies",
            "range_radius_bodies",
            # MarlConfig
            "marl_velocity_cap", "marl_rule_weight",
            "marl_separation_radius", "marl_action_scale",
            "marl_episode_steps",
            # PredatorConfig extras
            "predator_acceleration", "predator_vacuole_strength",
            "predator_blackening_gain",
            # New spatial leaves
            "flow_weight", "w_fwd",
            "readout_smooth",
            "max_dist_sep", "max_dist_align", "max_dist_coh",
            "angle_sep", "angle_align", "angle_coh", "coherence_factor",
            # New boundary leaves
            "boundary_margin",
            # New projection leaves
            "max_visibility", "max_occlusion_neighbors",
            # New flock leaves
            "n_predators",
            # New field leaves
            "field_flow_pull",
            # New perf leaves
            "num_threads",
            # New viz leaves
            "flap_period",
            # New spatial obstacle avoidance leaves
            "static_avoid_weight", "predictive_avoid_weight",
            "fly_away_max_dist", "min_time_to_collide",
            # New roost config leaves
            "roost_z_target",
        )
        _type_bad: set[str] = set()
        for fname in _numeric_fields:
            val = getattr(cfg, fname)
            if not isinstance(val, (int, float)):
                issues.append(
                    f"{fname} must be numeric, got {type(val).__name__} {val!r}"
                )
                _type_bad.add(fname)

        def _ok(fname: str) -> bool:
            """True if this field passed the type guard (safe for comparisons)."""
            return fname not in _type_bad

        # ── Domain dimensions ──────────────────────────────────
        if _ok("width") and cfg.width <= 0:
            issues.append(f"domain.width must be > 0, got {cfg.width}")
        if _ok("height") and cfg.height <= 0:
            issues.append(f"domain.height must be > 0, got {cfg.height}")
        if _ok("depth") and cfg.depth <= 0:
            issues.append(f"domain.depth must be > 0, got {cfg.depth}")

        # ── Flock ─────────────────────────────────────────────
        if _ok("num_boids") and cfg.num_boids < 0:
            issues.append(f"num_boids must be >= 0, got {cfg.num_boids}")
        if _ok("boid_size") and cfg.boid_size <= 0:
            issues.append(f"boid_size must be > 0, got {cfg.boid_size}")
        if _ok("v0") and cfg.v0 < 0:
            issues.append(f"v0 must be >= 0, got {cfg.v0}")
        if _ok("max_force") and cfg.max_force < 0:
            issues.append(f"max_force must be >= 0, got {cfg.max_force}")
        if _ok("visual_range") and cfg.visual_range <= 0:
            issues.append(f"visual_range must be > 0, got {cfg.visual_range}")

        # ── Boundary ──────────────────────────────────────────
        if cfg.boundary_mode not in self._VALID_BOUNDARY_MODES:
            issues.append(
                f"boundary_mode must be one of {self._VALID_BOUNDARY_MODES}, "
                f"got {cfg.boundary_mode!r}"
            )
        if _ok("boundary_sphere_radius") and cfg.boundary_sphere_radius <= 0:
            issues.append(
                f"boundary_sphere_radius must be > 0, got {cfg.boundary_sphere_radius}"
            )

        # ── Direct fields ─────────────────────────────────────
        if cfg.position_init not in self._VALID_POSITION_INITS:
            issues.append(
                f"position_init must be one of {self._VALID_POSITION_INITS}, "
                f"got {cfg.position_init!r}"
            )
        if cfg.velocity_init not in self._VALID_VELOCITY_INITS:
            issues.append(
                f"velocity_init must be one of {self._VALID_VELOCITY_INITS}, "
                f"got {cfg.velocity_init!r}"
            )

        # ── Mode ──────────────────────────────────────────────
        if cfg.mode not in self._VALID_MODES:
            issues.append(
                f"mode must be one of {self._VALID_MODES}, got {cfg.mode!r}"
            )

        # ── Mode-specific constraints ─────────────────────────
        # Explicit phi_p validation (shim retired — read from sub-config)
        if not isinstance(cfg.projection.phi_p, (int, float)):
            issues.append(
                f"projection.phi_p must be numeric, "
                f"got {type(cfg.projection.phi_p).__name__} {cfg.projection.phi_p!r}"
            )

        if cfg.mode == "projection":
            if _ok("sigma") and cfg.sigma <= 0:
                issues.append(f"projection.sigma must be > 0, got {cfg.sigma}")
            if isinstance(cfg.projection.phi_p, (int, float)) and cfg.projection.phi_p < 0:
                issues.append(f"projection.phi_p must be >= 0, got {cfg.projection.phi_p}")
            if _ok("phi_a") and cfg.phi_a < 0:
                issues.append(f"projection.phi_a must be >= 0, got {cfg.phi_a}")

        if cfg.mode == "spatial":
            if _ok("separation_weight") and cfg.separation_weight < 0:
                issues.append(f"spatial.separation_weight >= 0, got {cfg.separation_weight}")
            if _ok("alignment_weight") and cfg.alignment_weight < 0:
                issues.append(f"spatial.alignment_weight >= 0, got {cfg.alignment_weight}")
            if _ok("cohesion_weight") and cfg.cohesion_weight < 0:
                issues.append(f"spatial.cohesion_weight >= 0, got {cfg.cohesion_weight}")
            if _ok("influence_count") and cfg.influence_count < 1:
                issues.append(f"spatial.influence_count must be >= 1, got {cfg.influence_count}")
            if _ok("noise_scale") and cfg.noise_scale < 0:
                issues.append(f"spatial.noise_scale >= 0, got {cfg.noise_scale}")

        if cfg.mode == "vicsek":
            if _ok("vicsek_couplage") and not (
                0.0 <= cfg.vicsek_couplage <= 1.0
            ):
                issues.append(
                    f"vicsek_couplage must be in [0,1], got {cfg.vicsek_couplage}"
                )
            if _ok("vicsek_diffusion") and cfg.vicsek_diffusion < 0:
                issues.append(
                    f"vicsek_diffusion must be >= 0, got {cfg.vicsek_diffusion}"
                )
            if (
                _ok("vicsek_radius_influence")
                and _ok("vicsek_radius_avoid")
                and cfg.vicsek_radius_influence <= cfg.vicsek_radius_avoid
            ):
                issues.append(
                    f"vicsek_radius_influence ({cfg.vicsek_radius_influence}) "
                    f"must be > vicsek_radius_avoid ({cfg.vicsek_radius_avoid})"
                )
            if _ok("vicsek_velocity") and cfg.vicsek_velocity <= 0:
                issues.append(
                    f"vicsek_velocity must be > 0, got {cfg.vicsek_velocity}"
                )
            if _ok("vicsek_time_step") and cfg.vicsek_time_step <= 0:
                issues.append(
                    f"vicsek_time_step must be > 0, got {cfg.vicsek_time_step}"
                )

        if cfg.mode == "influencer":
            if _ok("influencer_substeps") and cfg.influencer_substeps < 1:
                issues.append(
                    f"influencer_substeps must be >= 1, got {cfg.influencer_substeps}"
                )
            if _ok("influencer_rank_exponent") and cfg.influencer_rank_exponent <= 0:
                issues.append(
                    f"influencer_rank_exponent must be > 0, got {cfg.influencer_rank_exponent}"
                )
            if _ok("influencer_scale") and cfg.influencer_scale <= 0:
                issues.append(
                    f"influencer_scale must be > 0, got {cfg.influencer_scale}"
                )
            if cfg.influencer_influence_mode not in {"rank", "distance"}:
                issues.append(
                    f"influencer_influence_mode must be 'rank' or 'distance', "
                    f"got {cfg.influencer_influence_mode!r}"
                )
            if _ok("influencer_near_dist_sq") and cfg.influencer_near_dist_sq <= 0:
                issues.append(
                    f"influencer_near_dist_sq must be > 0, got {cfg.influencer_near_dist_sq}"
                )
            if _ok("influencer_init_separation") and cfg.influencer_init_separation <= 0:
                issues.append(
                    f"influencer_init_separation must be > 0, got {cfg.influencer_init_separation}"
                )
            if _ok("influencer_tick_rate") and cfg.influencer_tick_rate <= 0:
                issues.append(
                    f"influencer_tick_rate must be > 0, got {cfg.influencer_tick_rate}"
                )

        # ── Refinements ───────────────────────────────────────
        if _ok("blind_deg") and (cfg.blind_deg < 0 or cfg.blind_deg >= 360):
            issues.append(
                f"blind_deg must be in [0, 360), got {cfg.blind_deg}"
            )
        if _ok("anisotropy") and cfg.anisotropy < 1.0:
            issues.append(
                f"anisotropy must be >= 1.0 (body axis ratio a/b), got {cfg.anisotropy}"
            )
        if _ok("steric") and cfg.steric < 0:
            issues.append(f"steric must be >= 0, got {cfg.steric}")

        # ── Angle mode ──────────────────────────────────────────
        if cfg.angle_speed_mode not in self._VALID_ANGLE_SPEED_MODES:
            issues.append(
                f"angle_speed_mode must be one of {self._VALID_ANGLE_SPEED_MODES}, "
                f"got {cfg.angle_speed_mode!r}"
            )

        # ── Extensions cross-field ────────────────────────────
        if cfg.predator_mode not in self._VALID_PREDATOR_MODES:
            issues.append(
                f"predator_mode must be one of {self._VALID_PREDATOR_MODES}, "
                f"got {cfg.predator_mode!r}"
            )
        if cfg.predator_enabled:
            if (
                _ok("predator_threat_radius")
                and cfg.predator_threat_radius <= 0
            ):
                issues.append(
                    "predator_enabled=True but predator_threat_radius must be > 0"
                )
            if _ok("predator_strength") and cfg.predator_strength <= 0:
                issues.append(
                    "predator_enabled=True but predator_strength must be > 0"
                )

        # ── Spatial index ─────────────────────────────────────
        if cfg.spatial_index not in self._VALID_INDEX_TYPES:
            issues.append(
                f"spatial_index must be one of {self._VALID_INDEX_TYPES}, "
                f"got {cfg.spatial_index!r}"
            )
        if _ok("topological_cap") and cfg.topological_cap < 1:
            issues.append(f"topological_cap must be >= 1, got {cfg.topological_cap}")

        # ── Performance ───────────────────────────────────────
        if cfg.metrics_detail_level not in self._VALID_METRICS_LEVELS:
            issues.append(
                f"metrics_detail_level must be in {self._VALID_METRICS_LEVELS}, "
                f"got {cfg.metrics_detail_level}"
            )
        if _ok("metrics_interval") and cfg.metrics_interval < 1:
            issues.append(f"metrics_interval must be >= 1, got {cfg.metrics_interval}")
        if _ok("parallel_workers") and cfg.parallel_workers < -1:
            issues.append(
                f"parallel_workers must be >= -1, got {cfg.parallel_workers}"
            )

        # ── Visualization ─────────────────────────────────────
        if _ok("fps") and cfg.fps <= 0:
            issues.append(f"viz.fps must be > 0, got {cfg.fps}")
        if _ok("window_width") and cfg.window_width <= 0:
            issues.append(f"viz.window_width must be > 0, got {cfg.window_width}")
        if _ok("window_height") and cfg.window_height <= 0:
            issues.append(f"viz.window_height must be > 0, got {cfg.window_height}")
        if cfg.theme not in self._VALID_THEMES:
            issues.append(
                f"viz.theme must be one of {self._VALID_THEMES}, got {cfg.theme!r}"
            )

        # S4.4a: Validate bird_mesh
        if _ok("bird_mesh") and cfg.bird_mesh not in self._VALID_MESH_NAMES:
            issues.append(
                f"viz.bird_mesh must be one of {self._VALID_MESH_NAMES}, "
                f"got {cfg.bird_mesh!r}"
            )

        # ── Capture ───────────────────────────────────────────
        if _ok("capture_width") and cfg.capture_width <= 0:
            issues.append(f"capture_width must be > 0, got {cfg.capture_width}")
        if _ok("capture_height") and cfg.capture_height <= 0:
            issues.append(f"capture_height must be > 0, got {cfg.capture_height}")
        if _ok("capture_frames") and cfg.capture_frames < 1:
            issues.append(f"capture_frames must be >= 1, got {cfg.capture_frames}")
        if _ok("capture_every") and cfg.capture_every < 1:
            issues.append(f"capture_every must be >= 1, got {cfg.capture_every}")
        if _ok("capture_fps") and cfg.capture_fps <= 0:
            issues.append(f"capture_fps must be > 0, got {cfg.capture_fps}")

        # ── Angle mode ────────────────────────────────────────
        if cfg.mode == "angle":
            if _ok("turn_rate") and cfg.turn_rate <= 0:
                issues.append(f"angle.turn_rate must be > 0, got {cfg.turn_rate}")
            if _ok("max_turn_rate") and cfg.max_turn_rate <= 0:
                issues.append(f"angle.max_turn_rate must be > 0, got {cfg.max_turn_rate}")
            if _ok("turn_threshold") and cfg.turn_threshold < 0:
                issues.append(f"angle.turn_threshold must be >= 0, got {cfg.turn_threshold}")
            if _ok("base_speed") and cfg.base_speed <= 0:
                issues.append(f"angle.base_speed must be > 0, got {cfg.base_speed}")
            if _ok("angle_neighbors") and cfg.angle_neighbors < 1:
                issues.append(f"angle.angle_neighbors must be >= 1, got {cfg.angle_neighbors}")

        # ── MARL mode ─────────────────────────────────────────
        if cfg.mode == "marl":
            if _ok("marl_velocity_cap") and cfg.marl_velocity_cap <= 0:
                issues.append(f"marl.velocity_cap must be > 0, got {cfg.marl_velocity_cap}")
            if _ok("marl_rule_weight") and cfg.marl_rule_weight < 0:
                issues.append(f"marl.rule_weight must be >= 0, got {cfg.marl_rule_weight}")
            if _ok("marl_separation_radius") and cfg.marl_separation_radius <= 0:
                issues.append(
                    f"marl.separation_radius must be > 0, got {cfg.marl_separation_radius}"
                )
            if _ok("marl_action_scale") and cfg.marl_action_scale <= 0:
                issues.append(
                    f"marl.action_scale must be > 0, got {cfg.marl_action_scale}"
                )
            if _ok("marl_episode_steps") and cfg.marl_episode_steps < 1:
                issues.append(
                    f"marl.episode_steps must be >= 1, got {cfg.marl_episode_steps}"
                )

        # ── Ecology cross-field ───────────────────────────────
        if cfg.roosting_enabled:
            rx, ry, rz = cfg.ecology_roost
            domain_ok = (
                _ok("width") and 0 <= rx <= cfg.width
                and _ok("height") and 0 <= ry <= cfg.height
                and _ok("depth") and 0 <= rz <= cfg.depth
            )
            if not domain_ok:
                issues.append(
                    f"roosting_enabled=True but ecology_roost {cfg.ecology_roost} "
                    f"is outside domain bounds ({cfg.width}x{cfg.height}x{cfg.depth})"
                )

        if issues:
            raise ValueError(
                f"SimConfig validation failed with {len(issues)} issue(s):\n"
                + "\n".join(f"  - {i}" for i in issues)
            )

    # ── Copy support (I7.1: deep-copy sub-configs) ────────────

    def __copy__(self) -> "SimConfig":
        """Shallow copy that deep-copies sub-configs.

        Without this, copy.copy(config) shares sub-config objects,
        so copy(config).width = 500 silently mutates config.width too.
        """
        cls = self.__class__
        result = cls.__new__(cls)
        for sub_attr in (
            "_domain", "_flock", "_boundary", "_projection", "_spatial",
            "_field", "_wander", "_vicsek", "_influencer", "_angle", "_marl",
            "_index", "_refinement",
            "_extension", "_predator", "_ecology", "_roost", "_perf", "_viz", "_capture",
        ):
            object.__setattr__(
                result, sub_attr,
                copy.deepcopy(object.__getattribute__(self, sub_attr)),
            )
        for attr in _DIRECT_FIELDS:
            object.__setattr__(result, attr, object.__getattribute__(self, attr))
        return result

    # ── Equality ──────────────────────────────────────────────

    def __eq__(self, other: object) -> bool:
        """Compare all sub-configs and direct fields for equality
        (dataclass-like). Covers nested-only fields such as
        projection.phi_p."""
        if not isinstance(other, SimConfig):
            return NotImplemented
        for sub_attr in (
            "_domain", "_flock", "_boundary", "_projection", "_spatial",
            "_field", "_wander", "_vicsek", "_influencer", "_angle", "_marl",
            "_index", "_refinement", "_extension", "_predator", "_ecology", "_roost", "_perf", "_viz", "_capture",):
            if (object.__getattribute__(self, sub_attr)
                    != object.__getattribute__(other, sub_attr)):
                return False
        for name in _DIRECT_FIELDS:
            if getattr(self, name) != getattr(other, name):
                return False
        return True

    # ── Repr ──────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"SimConfig(mode={self.mode!r}, seed={self.seed!r}, "
            f"num_boids={self.num_boids}, "
            f"domain={self._domain.width:.0f}x{self._domain.height:.0f}"
            f"x{self._domain.depth:.0f})"
        )
