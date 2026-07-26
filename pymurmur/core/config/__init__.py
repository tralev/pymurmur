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

File-size split: sub-config dataclasses live in config_sections.py,
the flat-name field map in config_field_map.py, YAML I/O in
config_io.py, and cross-field validation in config_validation.py.
SimConfig itself (this file) re-exports all of the above so existing
`from pymurmur.core.config import X` call sites keep working.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from .config_field_map import (
    _ALL_FIELD_NAMES,
    _DIRECT_FIELDS,
    _FIELD_MAP,
    _NESTED_ONLY,
)
from .config_io import (
    _NON_FIELD_TOP_LEVEL_LISTS,
    _TUPLE_FIELDS,
    _coerce_tuples,
    load_config_from_file,
    save_config_to_file,
)
from .config_sections import (
    AngleConfig,
    BoidStateMachineConfig,
    BoundaryConfig,
    CaptureConfig,
    DomainConfig,
    DynamicVisionRangeConfig,
    EcologyConfig,
    ExtensionConfig,
    FieldConfig,
    FlockConfig,
    IndexConfig,
    InfluencerConfig,
    MarlConfig,
    NeighborAdaptiveSpeedConfig,
    PerfConfig,
    PredatorConfig,
    ProjectionConfig,
    RefinementConfig,
    RoostConfig,
    SpatialConfig,
    SpeedNoiseConfig,
    VicsekConfig,
    VizConfig,
    WanderConfig,
)
from .config_validation import validate_config

__all__ = [
    "SimConfig",
    "DomainConfig", "FlockConfig", "BoundaryConfig", "ProjectionConfig",
    "SpatialConfig", "FieldConfig", "WanderConfig", "SpeedNoiseConfig",
    "NeighborAdaptiveSpeedConfig", "DynamicVisionRangeConfig",
    "BoidStateMachineConfig", "VicsekConfig",
    "InfluencerConfig", "AngleConfig", "MarlConfig", "IndexConfig",
    "RefinementConfig", "ExtensionConfig", "PredatorConfig", "RoostConfig",
    "EcologyConfig", "PerfConfig", "VizConfig", "CaptureConfig",
    "_FIELD_MAP", "_DIRECT_FIELDS", "_NESTED_ONLY", "_ALL_FIELD_NAMES",
    "_NON_FIELD_TOP_LEVEL_LISTS", "_TUPLE_FIELDS", "_coerce_tuples",
    "validate_config",
]


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
        object.__setattr__(self, "_speed_noise", SpeedNoiseConfig())
        object.__setattr__(self, "_neighbor_adaptive_speed", NeighborAdaptiveSpeedConfig())
        object.__setattr__(self, "_dynamic_vision_range", DynamicVisionRangeConfig())
        object.__setattr__(self, "_boid_state_machine", BoidStateMachineConfig())
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
    def speed_noise(self) -> SpeedNoiseConfig:
        return self._speed_noise

    @property
    def neighbor_adaptive_speed(self) -> NeighborAdaptiveSpeedConfig:
        return self._neighbor_adaptive_speed

    @property
    def dynamic_vision_range(self) -> DynamicVisionRangeConfig:
        return self._dynamic_vision_range

    @property
    def boid_state_machine(self) -> BoidStateMachineConfig:
        return self._boid_state_machine

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
        return load_config_from_file(cls, path, strict=strict)

    def to_file(self, path: str | Path) -> None:
        """Write config to a YAML file. Round-trip preserves all fields."""
        save_config_to_file(self, path)

    # ── Validation ───────────────────────────────────────────

    _VALID_MODES = {"projection", "spatial", "field", "vicsek", "influencer", "angle", "marl"}
    _VALID_BOUNDARY_MODES = {"toroidal", "open", "margin", "sphere", "sphere_soft"}
    _VALID_INDEX_TYPES = {"auto", "hash_grid", "kdtree", "none"}
    # Mirror physics/forces/kernels.py's registry — kept as plain string
    # constants there (Level 0, no config import) and validated here.
    _VALID_SEPARATION_KERNELS = frozenset({
        "sum", "mean", "unit", "exp", "linear_ramp", "asymptotic",
        "velocity_weighted", "cosine_zone", "linear", "nearest_only", "bell_zone",
    })
    _VALID_COHESION_KERNELS = frozenset({"unweighted", "inverse_distance", "bell_zone"})
    _VALID_ALIGNMENT_KERNELS = frozenset({"unweighted", "fov_weighted", "circular_mean_2d"})
    # S4.4a: Valid themes and mesh names — mirror mesh_registry.py values.
    # Defined here statically to avoid core→viz import (forbidden per arch.md).
    _VALID_THEMES = frozenset({"ink", "inverse", "paper", "graphite", "heading"})
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
        issues = validate_config(self)
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
            "_field", "_wander", "_speed_noise", "_neighbor_adaptive_speed",
            "_dynamic_vision_range", "_boid_state_machine",
            "_vicsek", "_influencer", "_angle", "_marl",
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
            "_field", "_wander", "_speed_noise", "_neighbor_adaptive_speed",
            "_dynamic_vision_range", "_boid_state_machine",
            "_vicsek", "_influencer", "_angle", "_marl",
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
