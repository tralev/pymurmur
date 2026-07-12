"""Simulation configuration — the shared parameter contract.

Level 2 — depends on PyYAML (stdlib only otherwise). Every component
reads from SimConfig; only InputControl and __main__ write to it.

YAML nesting convention:
    domain.width      → width
    flock.num_boids   → num_boids
    projection.phi_p  → phi_p
    spatial.separation_weight → separation_weight
    performance.use_numba → use_numba
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

# ── SimConfig ─────────────────────────────────────────────────────

@dataclass
class SimConfig:
    """Shared parameter contract between every subsystem."""

    # ── Domain (static — requires restart) ────────────────────────
    width: float = 1000.0
    height: float = 700.0
    depth: float = 400.0

    # ── Flock (static — requires restart) ─────────────────────────
    num_boids: int = 150
    boid_size: float = 9.0       # body radius for 3D spherical-cap occlusion
    v0: float = 4.0              # cruise speed (units/frame)
    max_force: float = 0.15      # max steering force per frame
    visual_range: float = 70.0   # max distance for neighbor candidate filtering

    # ── Mode (live-mutable) ───────────────────────────────────────
    mode: str = "projection"

    # ── Projection mode weights (Pearce 2014) ─────────────────────
    phi_p: float = 0.03          # projection weight (δ̂ coherence)
    phi_a: float = 0.80          # alignment weight (neighbor heading)
    sigma: int = 4               # topological neighbor count

    # ── Spatial mode weights (Reynolds 1987) ──────────────────────
    separation_weight: float = 4.5
    alignment_weight: float = 0.65
    cohesion_weight: float = 0.75
    noise_scale: float = 0.0
    acceleration_scale: float = 0.3

    # ── Field mode parameters (crs48 blob-anchor) ─────────────────
    field_separation: float = 0.92
    field_alignment: float = 0.90
    field_cohesion: float = 1.80
    field_flow: float = 0.30
    field_chase_strength: float = 0.82

    # ── Vicsek mode parameters ────────────────────────────────────
    vicsek_couplage: float = 0.8       # alignment coupling η ∈ [0,1]
    vicsek_diffusion: float = 0.8      # angular noise D
    vicsek_radius_influence: float = 5.0
    vicsek_radius_avoid: float = 1.0
    vicsek_velocity: float = 1.0       # constant speed

    # ── Influencer mode parameters ────────────────────────────────
    influencer_rank_exponent: float = 1.8
    influencer_substeps: int = 5

    # ── Spatial index ─────────────────────────────────────────────
    topological_cap: int = 50          # cap on k-NN neighbor count
    alignment_radius_ratio: float = 0.75
    use_toroidal_distance: bool = True

    # ── Boundary ──────────────────────────────────────────────────
    boundary_mode: str = "toroidal"     # toroidal | open | margin | sphere
    boundary_sphere_radius: float = 300.0
    boundary_avoidance_factor: float = 0.05

    # ── SI Refinements (live-mutable) ─────────────────────────────
    refinements: bool = True
    steric: float = 0.6               # φ_s: 1/d² repulsion strength (0 = off)
    blind_deg: float = 60.0           # rear blind cone full angle (degrees)
    anisotropy: float = 2.0           # body axis ratio a/b (1.0 = isotropic)

    # ── Extensions ────────────────────────────────────────────────
    predator_enabled: bool = False
    roosting_enabled: bool = False
    wander_enabled: bool = False
    ripple_enabled: bool = False

    # ── Predator params ───────────────────────────────────────────
    predator_threat_radius: float = 12.0
    predator_strength: float = 1.0
    predator_momentum: float = 0.5
    predator_split_gain: float = 0.8

    # ── Ecology params ────────────────────────────────────────────
    ecology_roost: tuple[float, float, float] = (500.0, 350.0, 40.0)
    ecology_critical_mass: int = 500

    # ── Performance ───────────────────────────────────────────────
    use_numba: bool = True
    spatial_index: str = "auto"       # auto | hash_grid | kdtree
    metrics_detail_level: int = 1     # 0=off, 1=fast, 2=full
    metrics_interval: int = 60        # frames between expensive metric computations
    instance_buffer_chunk: int = 50000

    # ── Visualization ─────────────────────────────────────────────
    fps: int = 60
    window_width: int = 1200
    window_height: int = 800
    show_grid: bool = False
    auto_rotate: bool = False
    theme: str = "ink"                # ink | inverse | paper | graphite
    trails: str = "off"               # off | velocity | accumulation
    point_sprites: bool = False

    # ── Capture ──────────────────────────────────────────────────
    capture_width: int = 800
    capture_height: int = 600
    capture_frames: int = 240
    capture_every: int = 3
    capture_fps: int = 20
    capture_output: str = "output/murmuration.gif"
    capture_metrics_csv: str = "output/metrics.csv"
    capture_metrics_json: str = "output/metrics.json"
    capture_with_viz: bool = True

    # ── Seed ──────────────────────────────────────────────────────
    seed: int | None = None

    # ── YAML I/O ──────────────────────────────────────────────────

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

        # Flatten nested sections
        for section_name, section_data in raw.items():
            if isinstance(section_data, dict):
                for key, value in section_data.items():
                    flat[key] = value
            else:
                flat[section_name] = section_data

        # Filter to known fields only
        valid_names = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in flat.items() if k in valid_names}

        return cls(**filtered)

    def to_file(self, path: str | Path) -> None:
        """Write config to a YAML file. Round-trip preserves all fields."""
        import yaml

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data: dict[str, Any] = {
            "domain": {"width": self.width, "height": self.height, "depth": self.depth},
            "flock": {"num_boids": self.num_boids, "boid_size": self.boid_size,
                       "v0": self.v0, "max_force": self.max_force, "visual_range": self.visual_range},
            "mode": self.mode,
            "projection": {"phi_p": self.phi_p, "phi_a": self.phi_a, "sigma": self.sigma},
            "spatial": {"separation_weight": self.separation_weight,
                         "alignment_weight": self.alignment_weight,
                         "cohesion_weight": self.cohesion_weight,
                         "noise_scale": self.noise_scale,
                         "acceleration_scale": self.acceleration_scale},
            "boundary": self.boundary_mode,
            "refinements": {"enabled": self.refinements, "steric": self.steric,
                            "blind_deg": self.blind_deg, "anisotropy": self.anisotropy},
            "extensions": {"predator": self.predator_enabled, "roosting": self.roosting_enabled,
                           "wander": self.wander_enabled, "ripple": self.ripple_enabled},
            "performance": {"use_numba": self.use_numba, "spatial_index": self.spatial_index,
                            "metrics_detail_level": self.metrics_detail_level,
                            "metrics_interval": self.metrics_interval,
                            "instance_buffer_chunk": self.instance_buffer_chunk},
            "visual": {"fps": self.fps, "window_width": self.window_width,
                       "window_height": self.window_height, "show_grid": self.show_grid,
                       "auto_rotate": self.auto_rotate, "theme": self.theme,
                       "trails": self.trails},
            "capture": {"width": self.capture_width, "height": self.capture_height,
                        "frames": self.capture_frames, "every": self.capture_every,
                        "fps": self.capture_fps, "output": self.capture_output,
                        "metrics_csv": self.capture_metrics_csv,
                        "metrics_json": self.capture_metrics_json,
                        "with_viz": self.capture_with_viz},
            "seed": self.seed,
        }

        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
