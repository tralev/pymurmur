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
    projection.phi_p → phi_p        extensions    → predator_enabled
    spatial.sep_wt   → separation_weight
    visual.fps       → fps          performance   → metrics_interval
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
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


@dataclass
class BoundaryConfig:
    """Boundary conditions."""
    boundary_mode: str = "toroidal"     # toroidal | open | margin | sphere
    boundary_sphere_radius: float = 300.0
    boundary_avoidance_factor: float = 0.05
    boundary_radius_factor: float = 1.0


@dataclass
class ProjectionConfig:
    """Projection mode weights (Pearce 2014)."""
    phi_p: float = 0.03          # projection weight (δ̂ coherence)
    phi_a: float = 0.80          # alignment weight (neighbor heading)
    sigma: int = 4               # topological neighbor count


@dataclass
class SpatialConfig:
    """Spatial mode weights (Reynolds 1987)."""
    separation_weight: float = 4.5
    alignment_weight: float = 0.65
    cohesion_weight: float = 0.75
    noise_scale: float = 0.0
    acceleration_scale: float = 0.3


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
    field_vacuole_strength: float = 0.0
    field_shell_radius_base: float = 0.22
    field_inner_radius_factor: float = 0.35
    field_num_groups: int = 7
    field_leader_fraction: float = 0.16


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


@dataclass
class InfluencerConfig:
    """Influencer mode parameters."""
    influencer_rank_exponent: float = 1.8
    influencer_substeps: int = 5


@dataclass
class IndexConfig:
    """Spatial index parameters."""
    spatial_index: str = "auto"        # auto | hash_grid | kdtree | none
    topological_cap: int = 50          # cap on k-NN neighbor count
    alignment_radius_ratio: float = 0.75
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


@dataclass
class EcologyConfig:
    """Ecology parameters."""
    ecology_roost: tuple[float, float, float] = (500.0, 350.0, 40.0)
    ecology_critical_mass: int = 500


@dataclass
class PerfConfig:
    """Performance parameters."""
    metrics_detail_level: int = 1     # 0=off, 1=fast, 2=full
    metrics_interval: int = 60        # frames between expensive metric computations
    instance_buffer_chunk: int = 50000
    parallel_workers: int = 1         # n_jobs for occlusion culling (1 = sequential)


@dataclass
class VizConfig:
    """Visualization parameters."""
    fps: int = 60
    window_width: int = 1200
    window_height: int = 800
    show_grid: bool = False
    auto_rotate: bool = False
    theme: str = "ink"                # ink | inverse | paper | graphite


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
    # BoundaryConfig
    "boundary_mode": ("_boundary", "boundary_mode"),
    "boundary_sphere_radius": ("_boundary", "boundary_sphere_radius"),
    "boundary_avoidance_factor": ("_boundary", "boundary_avoidance_factor"),
    # ProjectionConfig
    "phi_p": ("_projection", "phi_p"),
    "phi_a": ("_projection", "phi_a"),
    "sigma": ("_projection", "sigma"),
    # SpatialConfig
    "separation_weight": ("_spatial", "separation_weight"),
    "alignment_weight": ("_spatial", "alignment_weight"),
    "cohesion_weight": ("_spatial", "cohesion_weight"),
    "noise_scale": ("_spatial", "noise_scale"),
    "acceleration_scale": ("_spatial", "acceleration_scale"),
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
    "field_vacuole_strength": ("_field", "field_vacuole_strength"),
    "field_shell_radius_base": ("_field", "field_shell_radius_base"),
    "field_inner_radius_factor": ("_field", "field_inner_radius_factor"),
    "field_num_groups": ("_field", "field_num_groups"),
    "field_leader_fraction": ("_field", "field_leader_fraction"),
    # WanderConfig
    "wander_attractor_speed": ("_wander", "wander_attractor_speed"),
    "wander_attractor_radius": ("_wander", "wander_attractor_radius"),
    # BoundaryConfig extra
    "boundary_radius_factor": ("_boundary", "boundary_radius_factor"),
    # VicsekConfig
    "vicsek_couplage": ("_vicsek", "vicsek_couplage"),
    "vicsek_diffusion": ("_vicsek", "vicsek_diffusion"),
    "vicsek_radius_influence": ("_vicsek", "vicsek_radius_influence"),
    "vicsek_radius_avoid": ("_vicsek", "vicsek_radius_avoid"),
    "vicsek_velocity": ("_vicsek", "vicsek_velocity"),
    "vicsek_time_step": ("_vicsek", "vicsek_time_step"),
    # InfluencerConfig
    "influencer_rank_exponent": ("_influencer", "influencer_rank_exponent"),
    "influencer_substeps": ("_influencer", "influencer_substeps"),
    # IndexConfig
    "spatial_index": ("_index", "spatial_index"),
    "topological_cap": ("_index", "topological_cap"),
    "alignment_radius_ratio": ("_index", "alignment_radius_ratio"),
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
    # EcologyConfig
    "ecology_roost": ("_ecology", "ecology_roost"),
    "ecology_critical_mass": ("_ecology", "ecology_critical_mass"),
    # PerfConfig
    "metrics_detail_level": ("_perf", "metrics_detail_level"),
    "metrics_interval": ("_perf", "metrics_interval"),
    "instance_buffer_chunk": ("_perf", "instance_buffer_chunk"),
    "parallel_workers": ("_perf", "parallel_workers"),
    # VizConfig
    "fps": ("_viz", "fps"),
    "window_width": ("_viz", "window_width"),
    "window_height": ("_viz", "window_height"),
    "show_grid": ("_viz", "show_grid"),
    "auto_rotate": ("_viz", "auto_rotate"),
    "theme": ("_viz", "theme"),
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
}

# Fields not in _FIELD_MAP are stored directly on SimConfig
_DIRECT_FIELDS: set[str] = {"mode", "seed", "position_init"}

# All known field names
_ALL_FIELD_NAMES: set[str] = set(_FIELD_MAP.keys()) | _DIRECT_FIELDS


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
        object.__setattr__(self, "_index", IndexConfig())
        object.__setattr__(self, "_refinement", RefinementConfig())
        object.__setattr__(self, "_extension", ExtensionConfig())
        object.__setattr__(self, "_predator", PredatorConfig())
        object.__setattr__(self, "_ecology", EcologyConfig())
        object.__setattr__(self, "_perf", PerfConfig())
        object.__setattr__(self, "_viz", VizConfig())
        object.__setattr__(self, "_capture", CaptureConfig())

        # Direct fields
        object.__setattr__(self, "mode", kwargs.get("mode", "projection"))
        object.__setattr__(self, "seed", kwargs.get("seed", None))
        object.__setattr__(self, "position_init", kwargs.get("position_init", "box"))

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
    def from_file(cls, path: str | Path) -> "SimConfig":
        """Load config from a YAML file. Nested keys are flattened.

        Raises FileNotFoundError if path doesn't exist.
        Unknown top-level keys are silently ignored for forward-compatibility.
        """
        import yaml

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        raw = yaml.safe_load(path.read_text()) or {}
        flat: dict[str, Any] = {}

        # Flatten nested sections with section-aware key normalisation
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
                    flat[key] = value
            else:
                flat[section_name] = section_data

        # Filter to known fields only
        filtered = {k: v for k, v in flat.items() if k in _ALL_FIELD_NAMES}

        return cls(**filtered)

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
                      "visual_range": self.visual_range},
            "mode": self.mode,
            "projection": {"phi_p": self.phi_p, "phi_a": self.phi_a,
                           "sigma": self.sigma},
            "spatial": {"separation_weight": self.separation_weight,
                        "alignment_weight": self.alignment_weight,
                        "cohesion_weight": self.cohesion_weight,
                        "noise_scale": self.noise_scale,
                        "acceleration_scale": self.acceleration_scale},
            "boundary": {"boundary_mode": self.boundary_mode,
                         "boundary_sphere_radius": self.boundary_sphere_radius,
                         "boundary_avoidance_factor": self.boundary_avoidance_factor},
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
                         "predator_split_gain": self.predator_split_gain},
            "ecology": {"ecology_roost": list(self.ecology_roost),
                        "ecology_critical_mass": self.ecology_critical_mass},
            "vicsek": {"vicsek_couplage": self.vicsek_couplage,
                       "vicsek_diffusion": self.vicsek_diffusion,
                       "vicsek_radius_influence": self.vicsek_radius_influence,
                       "vicsek_radius_avoid": self.vicsek_radius_avoid,
                       "vicsek_velocity": self.vicsek_velocity,
                       "vicsek_time_step": self.vicsek_time_step},
            "influencer": {"influencer_rank_exponent": self.influencer_rank_exponent,
                           "influencer_substeps": self.influencer_substeps},
            "field": {"field_separation": self.field_separation,
                      "field_alignment": self.field_alignment,
                      "field_cohesion": self.field_cohesion,
                      "field_flow": self.field_flow,
                      "field_chase_strength": self.field_chase_strength,
                      "field_noise": self.field_noise,
                      "field_target_pull": self.field_target_pull,
                      "field_drift_pull": self.field_drift_pull,
                      "field_shell_influence": self.field_shell_influence,
                      "field_tangent_pull": self.field_tangent_pull,
                      "field_wave_gain": self.field_wave_gain,
                      "field_inertia": self.field_inertia,
                      "field_vacuole_strength": self.field_vacuole_strength,
                      "field_num_groups": self.field_num_groups,
                      "field_leader_fraction": self.field_leader_fraction},
            "index": {"spatial_index": self.spatial_index,
                      "topological_cap": self.topological_cap,
                      "alignment_radius_ratio": self.alignment_radius_ratio,
                      "use_toroidal_distance": self.use_toroidal_distance},
            "performance": {"metrics_detail_level": self.metrics_detail_level,
                            "metrics_interval": self.metrics_interval,
                            "instance_buffer_chunk": self.instance_buffer_chunk,
                            "parallel_workers": self.parallel_workers},
            "visual": {"fps": self.fps,
                       "window_width": self.window_width,
                       "window_height": self.window_height,
                       "show_grid": self.show_grid,
                       "auto_rotate": self.auto_rotate,
                       "theme": self.theme},
            "capture": {"capture_width": self.capture_width,
                        "capture_height": self.capture_height,
                        "capture_frames": self.capture_frames,
                        "capture_every": self.capture_every,
                        "capture_fps": self.capture_fps,
                        "capture_output": self.capture_output,
                        "capture_metrics_csv": self.capture_metrics_csv,
                        "capture_metrics_json": self.capture_metrics_json,
                        "capture_with_viz": self.capture_with_viz},
            "wander": {"wander_attractor_speed": self.wander_attractor_speed,
                      "wander_attractor_radius": self.wander_attractor_radius},
            "seed": self.seed,
        }

        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    # ── Validation ───────────────────────────────────────────

    _VALID_MODES = {"projection", "spatial", "field", "vicsek", "influencer"}
    _VALID_BOUNDARY_MODES = {"toroidal", "open", "margin", "sphere"}
    _VALID_INDEX_TYPES = {"auto", "hash_grid", "kdtree", "none"}
    _VALID_THEMES = {"ink", "inverse", "paper", "graphite"}
    _VALID_METRICS_LEVELS = {0, 1, 2}
    _VALID_POSITION_INITS = {"box", "random", "sphere", "gaussian", "grid", "sphere_shell", "blob"}

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
            "phi_p", "phi_a", "sigma",
            "separation_weight", "alignment_weight", "cohesion_weight",
            "noise_scale", "acceleration_scale",
            "steric", "blind_deg", "anisotropy",
            "parallel_workers", "metrics_interval", "metrics_detail_level",
            "topological_cap", "alignment_radius_ratio",
            "boundary_sphere_radius",
            "fps", "window_width", "window_height",
            "capture_width", "capture_height", "capture_frames",
            "capture_every", "capture_fps",
            "vicsek_couplage", "vicsek_diffusion",
            "vicsek_radius_influence", "vicsek_radius_avoid",
            "vicsek_velocity", "vicsek_time_step",
            "influencer_rank_exponent", "influencer_substeps",
            "predator_threat_radius", "predator_strength",
            "predator_momentum", "predator_split_gain",
            "field_separation", "field_alignment", "field_cohesion",
            "field_flow", "field_chase_strength",
            "field_noise", "field_target_pull", "field_drift_pull",
            "field_shell_influence", "field_tangent_pull",
            "field_wave_gain", "field_inertia",
            "field_vacuole_strength", "field_shell_radius_base",
            "field_inner_radius_factor", "field_leader_fraction",
            "wander_attractor_speed", "wander_attractor_radius",
            "boundary_avoidance_factor", "boundary_radius_factor",
            "acceleration_scale",
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

        # ── Mode ──────────────────────────────────────────────
        if cfg.mode not in self._VALID_MODES:
            issues.append(
                f"mode must be one of {self._VALID_MODES}, got {cfg.mode!r}"
            )

        # ── Mode-specific constraints ─────────────────────────
        if cfg.mode == "projection":
            if _ok("sigma") and cfg.sigma <= 0:
                issues.append(f"projection.sigma must be > 0, got {cfg.sigma}")
            if _ok("phi_p") and cfg.phi_p < 0:
                issues.append(f"projection.phi_p must be >= 0, got {cfg.phi_p}")
            if _ok("phi_a") and cfg.phi_a < 0:
                issues.append(f"projection.phi_a must be >= 0, got {cfg.phi_a}")

        if cfg.mode == "spatial":
            if _ok("separation_weight") and cfg.separation_weight < 0:
                issues.append(f"spatial.separation_weight >= 0, got {cfg.separation_weight}")
            if _ok("alignment_weight") and cfg.alignment_weight < 0:
                issues.append(f"spatial.alignment_weight >= 0, got {cfg.alignment_weight}")
            if _ok("cohesion_weight") and cfg.cohesion_weight < 0:
                issues.append(f"spatial.cohesion_weight >= 0, got {cfg.cohesion_weight}")
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

        # ── Extensions cross-field ────────────────────────────
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
        if _ok("alignment_radius_ratio") and not (
            0.0 < cfg.alignment_radius_ratio <= 1.0
        ):
            issues.append(
                f"alignment_radius_ratio must be in (0, 1], got {cfg.alignment_radius_ratio}"
            )

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
            "_field", "_wander", "_vicsek", "_influencer", "_index", "_refinement",
            "_extension", "_predator", "_ecology", "_perf", "_viz", "_capture",
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
        """Compare all fields for equality (dataclass-like)."""
        if not isinstance(other, SimConfig):
            return NotImplemented
        for name in _ALL_FIELD_NAMES:
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
