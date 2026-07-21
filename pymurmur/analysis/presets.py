"""Named parameter scenarios for quick mode/preset switching.

Level 1 — depends on config only.

P10.1: 8 letter-key presets (a–f, h, w) with labels and descriptions.
Key g is reserved for grid toggle.
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

# P10.1: 8 letter-key presets with labels printed on activation.
# Each tuple is (label, description, params_dict).
# Key g is reserved for grid toggle — not in this map.
# Values match S5.1 spec table (roadmap4.md).
LETTER_PRESETS: dict[str, tuple[str, str, dict]] = {
    "a": ("3D Pearce Default",
          "Pearce projection — \u03c6p=0.04 \u03c6a=0.80 \u03c3=6",
          {"mode": "projection", "phi_p": 0.04, "phi_a": 0.80, "sigma": 6}),
    "b": ("Ball of Birds",
          "Tight ball — \u03c6p=0.18 \u03c6a=0.70 \u03c3=7",
          {"mode": "projection", "phi_p": 0.18, "phi_a": 0.70, "sigma": 7}),
    "c": ("Storm Cloud",
          "Thin shells with hollow interior — \u03c6p=0.06 \u03c6a=0.45 \u03c3=3",
          {"mode": "projection", "phi_p": 0.06, "phi_a": 0.45, "sigma": 3}),
    "d": ("3D Stream",
          "Spatial stream with strong cohesion — sep=0.25 align=0.55 coh=0.80 inf=8",
          {"mode": "spatial", "separation_weight": 0.25, "alignment_weight": 0.55,
           "cohesion_weight": 0.80, "influence_count": 8}),
    "e": ("Vertical Column",
          "Tight column — \u03c6p=0.10 \u03c6a=0.75 \u03c3=6",
          {"mode": "projection", "phi_p": 0.10, "phi_a": 0.75, "sigma": 6}),
    "f": ("3D Acro",
          "Thin ribbon aerobatics — \u03c6p=0.02 \u03c6a=0.85 \u03c3=3",
          {"mode": "projection", "phi_p": 0.02, "phi_a": 0.85, "sigma": 3}),
    "h": ("3D Void",
          "Wide hollow spatial — sep=0.35 align=0.58 coh=0.90 inf=9",
          {"mode": "spatial", "separation_weight": 0.35, "alignment_weight": 0.58,
           "cohesion_weight": 0.90, "influence_count": 9}),
    "w": ("Spiral Vortex",
          "Exploratory spatial with wide stance — sep=0.08 align=0.82 coh=1.0 inf=10",
          {"mode": "spatial", "separation_weight": 0.08, "alignment_weight": 0.82,
           "cohesion_weight": 1.0, "influence_count": 10}),
}


def apply_preset(config, key: str):
    """P10.1: Apply a letter-key preset, returning the label printed.

    Extracted from InputControl._apply_letter_preset so it is
    independently testable (no pygame dependency).

    Returns the preset label, or None if the key is not registered.
    """
    entry = LETTER_PRESETS.get(key)
    if entry is None:
        return None
    label, _desc, params = entry
    for field, value in params.items():
        if field == "mode":
            config.mode = value
        elif field == "phi_p":  # nested-only (flat shim retired)
            config.projection.phi_p = value
        else:
            setattr(config, field, value)
    return label
