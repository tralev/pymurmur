"""Named parameter scenarios for quick mode/preset switching.

Level 1 — depends on config only.
"""

PRESETS: dict[str, dict] = {
    "ball": {
        "mode": "projection", "phi_p": 0.05, "phi_a": 0.90, "sigma": 6,
    },
    "storm_cloud": {
        "mode": "projection", "phi_p": 0.02, "phi_a": 0.85, "sigma": 4,
    },
    "stream": {
        "mode": "projection", "phi_p": 0.01, "phi_a": 0.95, "sigma": 8,
    },
    "column": {
        "mode": "projection", "phi_p": 0.04, "phi_a": 0.80, "sigma": 5,
    },
    "acro": {
        "mode": "spatial", "separation_weight": 2.0, "noise_scale": 1.5,
    },
    "spiral_vortex": {
        "mode": "projection", "phi_p": 0.03, "phi_a": 0.70, "sigma": 12,
    },
    "void": {
        "mode": "projection", "phi_p": 0.06, "phi_a": 0.75, "sigma": 3,
    },
}
