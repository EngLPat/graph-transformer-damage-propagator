"""Shared constants and parameter sampling for the deck generators.

Holds the parametric design space (PARAM_RANGES), the fixed geometry defaults,
the elastic constants of the ply, the Abaqus step parameters and the sampler
(sample_parameters: Latin hypercube or random, with a feasibility check) used
by generate_cohesive_laminate_uniform.py and generate_extrapolation_geometry.py.

Authors: Luca Patrignani, Silvestre T. Pinho
Institution: Imperial College London
"""

import random
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════
# PROJECT PATHS  (relative to repository root)
# ═══════════════════════════════════════════════════════════════════════════
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ═══════════════════════════════════════════════════════════════════════════
# PARAMETRIC STUDY CONFIGURATION
# Edit ranges here to change the design space.  Each value is (min, max).
# ═══════════════════════════════════════════════════════════════════════════
PARAM_RANGES = {
    "hole_radius": (2.0, 5.0),          # mm
    "hole_offset_x": (-5.0, 5.0),       # mm, offset from plate centre
    "hole_offset_y": (-5.0, 5.0),       # mm, offset from plate centre
    "displacement": (0.35, 0.7),        # mm, applied on right edge
    "plate_half_x": (14.0, 20.0),        # mm, half-width
    "plate_half_y": (14.0, 20.0),        # mm, half-height
}

N_SAMPLES = 200         # number of simulations to generate
RANDOM_SEED = 42        # for reproducibility (set to None for random)

# ═══════════════════════════════════════════════════════════════════════════
# FIXED GEOMETRY DEFAULTS
# Constant across all samples.
# ═══════════════════════════════════════════════════════════════════════════
PLATE_HALF_SIZE = 16.0          # mm  (plate spans ±PLATE_HALF_SIZE)
PLATE_THICKNESS = 0.2           # mm  (one element through Z)
INNER_SQUARE_FACTOR = 2.0       # feasibility: |offset| + radius × factor + margin < half-size
GEOMETRY_MARGIN = 1.0           # mm  safety margin between inner square and plate edge

# ═══════════════════════════════════════════════════════════════════════════
# MATERIAL: elastic engineering constants  (IM7/8552-like UD ply)
# ═══════════════════════════════════════════════════════════════════════════
MAT_E1 = 171420.0       # MPa
MAT_E2 = 9080.0
MAT_E3 = 9080.0
MAT_NU12 = 0.32
MAT_NU13 = 0.32
MAT_NU23 = 0.32
MAT_G12 = 5290.0
MAT_G13 = 5290.0
MAT_G23 = 3026.0

# ═══════════════════════════════════════════════════════════════════════════
# STEP PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════
STEP_INITIAL_INC = 0.01
STEP_TOTAL_TIME = 1.0
STEP_MIN_INC = 1e-12
STEP_MAX_INC = 0.02
STEP_MAX_NUM_INC = 10000

# ═══════════════════════════════════════════════════════════════════════════
# LAYUP: list of (fibre_angle_deg, ply_thickness_mm) tuples, bottom to top.
# ═══════════════════════════════════════════════════════════════════════════
LAYUP = [(90.0, PLATE_THICKNESS)]   # single-ply default


def _format_set_list(ids, per_line=16):
    """Format a list of integer IDs for *Nset/*Elset data lines."""
    lines = []
    for i in range(0, len(ids), per_line):
        chunk = ids[i : i + per_line]
        lines.append(" " + ", ".join(f"{v:d}" for v in chunk))
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# PARAMETER SAMPLING
# ═══════════════════════════════════════════════════════════════════════════

def _check_feasibility(hole_r, hole_cx, hole_cy, plate_hx, plate_hy,
                       inner_factor, margin):
    """Return True if hole+offsets fit inside the plate (rectangular)."""
    max_offset_x = plate_hx - hole_r * inner_factor - margin
    max_offset_y = plate_hy - hole_r * inner_factor - margin
    if max_offset_x <= 0 or max_offset_y <= 0:
        return False
    return abs(hole_cx) < max_offset_x and abs(hole_cy) < max_offset_y


def sample_parameters(n_samples, param_ranges, plate_half_size,
                      inner_factor, margin, seed=None):
    """Generate n_samples parameter sets via Latin Hypercube Sampling.

    Uses ``scipy.stats.qmc.LatinHypercube`` for space-filling coverage.
    Falls back to plain ``random.uniform`` rejection sampling when scipy
    is unavailable.

    For each sample, the hole must fit entirely inside the plate with
    enough room for the transition square plus a safety margin:
        |hole_cx| + hole_r × inner_factor + margin  <  plate_half_x
        |hole_cy| + hole_r × inner_factor + margin  <  plate_half_y

    If *plate_half_x* and/or *plate_half_y* are included in
    ``param_ranges`` they are sampled together with the other parameters;
    otherwise the scalar ``plate_half_size`` argument is used as a
    constant for both.

    Parameters
    ----------
    n_samples : int
        Number of parameter sets to generate.
    param_ranges : dict
        Ranges for each parameter, keyed by name.
    plate_half_size : float
        Default half-side (mm).  Used for any axis not in ``param_ranges``.
    inner_factor : float
        Transition-square factor.
    margin : float
        Safety margin between inner square and plate edge (mm).
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    list of dict
        Each dict has keys: hole_radius, hole_center_x, hole_center_y,
        displacement, plate_half_x, plate_half_y, thickness.
    """
    variable_plate_x = "plate_half_x" in param_ranges
    variable_plate_y = "plate_half_y" in param_ranges

    # ----- Try LHS first, fall back to random ----
    try:
        from scipy.stats.qmc import LatinHypercube, scale  # noqa: F401 (availability check, scipy ≥ 1.7)
        use_lhs = True
    except ImportError:
        use_lhs = False
        print("  ⚠  scipy.stats.qmc not available; falling back to random sampling")

    # Build ordered dimension list for LHS
    dim_names = ["hole_radius", "hole_offset_x", "hole_offset_y", "displacement"]
    if variable_plate_x:
        dim_names.append("plate_half_x")
    if variable_plate_y:
        dim_names.append("plate_half_y")

    l_bounds = [param_ranges[d][0] for d in dim_names]
    u_bounds = [param_ranges[d][1] for d in dim_names]

    if use_lhs:
        return _sample_lhs(
            n_samples, dim_names, l_bounds, u_bounds,
            plate_half_size, variable_plate_x, variable_plate_y,
            inner_factor, margin, seed,
        )
    else:
        return _sample_random(
            n_samples, param_ranges, plate_half_size,
            variable_plate_x, variable_plate_y,
            inner_factor, margin, seed,
        )


def _sample_lhs(n_samples, dim_names, l_bounds, u_bounds,
                plate_half_size_fixed, variable_plate_x, variable_plate_y,
                inner_factor, margin, seed):
    """Latin Hypercube Sampling with rejection for infeasible points."""
    from scipy.stats.qmc import LatinHypercube, scale

    n_dim = len(dim_names)
    rng_seed = seed if seed is not None else 42

    # Over-sample by a factor to account for feasibility rejections
    oversample = 3
    samples = []
    attempt = 0
    max_attempts = 20

    while len(samples) < n_samples and attempt < max_attempts:
        attempt += 1
        n_generate = (n_samples - len(samples)) * oversample
        sampler = LatinHypercube(d=n_dim, seed=rng_seed + attempt)
        raw = sampler.random(n=n_generate)
        scaled = scale(raw, l_bounds, u_bounds)

        for row in scaled:
            vals = {name: float(row[i]) for i, name in enumerate(dim_names)}
            hole_r = vals["hole_radius"]
            hole_cx = vals["hole_offset_x"]
            hole_cy = vals["hole_offset_y"]
            disp = vals["displacement"]
            plate_hx = vals.get("plate_half_x", plate_half_size_fixed)
            plate_hy = vals.get("plate_half_y", plate_half_size_fixed)

            if not _check_feasibility(hole_r, hole_cx, hole_cy,
                                      plate_hx, plate_hy,
                                      inner_factor, margin):
                continue

            samples.append({
                "hole_radius":    round(hole_r, 4),
                "hole_center_x":  round(hole_cx, 4),
                "hole_center_y":  round(hole_cy, 4),
                "displacement":   round(disp, 4),
                "plate_half_x":   round(plate_hx, 4),
                "plate_half_y":   round(plate_hy, 4),
                "thickness":      round(sum(t for _, t in LAYUP), 6),
                "layup":          LAYUP,
            })
            if len(samples) >= n_samples:
                break

    if len(samples) < n_samples:
        raise RuntimeError(
            f"LHS: only {len(samples)}/{n_samples} feasible after "
            f"{max_attempts} attempts.  Widen PARAM_RANGES or reduce "
            f"INNER_SQUARE_FACTOR."
        )

    print(f"  ✓ Latin Hypercube Sampling: {n_samples} feasible out of "
          f"{n_generate * max_attempts if attempt == max_attempts else n_generate * attempt} candidates")
    return samples


def _sample_random(n_samples, param_ranges, plate_half_size_fixed,
                   variable_plate_x, variable_plate_y,
                   inner_factor, margin, seed):
    """Original random-uniform rejection sampling (fallback)."""
    if seed is not None:
        random.seed(seed)

    r_min, r_max = param_ranges["hole_radius"]
    d_min, d_max = param_ranges["displacement"]
    ox_min, ox_max = param_ranges["hole_offset_x"]
    oy_min, oy_max = param_ranges["hole_offset_y"]
    if variable_plate_x:
        px_min, px_max = param_ranges["plate_half_x"]
    else:
        px_min = px_max = plate_half_size_fixed
    if variable_plate_y:
        py_min, py_max = param_ranges["plate_half_y"]
    else:
        py_min = py_max = plate_half_size_fixed

    samples = []
    max_attempts = n_samples * 100
    attempts = 0

    while len(samples) < n_samples and attempts < max_attempts:
        attempts += 1

        hole_r = random.uniform(r_min, r_max)
        plate_hx = random.uniform(px_min, px_max) if variable_plate_x else plate_half_size_fixed
        plate_hy = random.uniform(py_min, py_max) if variable_plate_y else plate_half_size_fixed
        max_offset_x = plate_hx - hole_r * inner_factor - margin
        max_offset_y = plate_hy - hole_r * inner_factor - margin

        if max_offset_x <= 0 or max_offset_y <= 0:
            continue

        cx_lo = max(ox_min, -max_offset_x)
        cx_hi = min(ox_max, max_offset_x)
        cy_lo = max(oy_min, -max_offset_y)
        cy_hi = min(oy_max, max_offset_y)

        if cx_lo > cx_hi or cy_lo > cy_hi:
            continue

        hole_cx = random.uniform(cx_lo, cx_hi)
        hole_cy = random.uniform(cy_lo, cy_hi)
        displacement = random.uniform(d_min, d_max)

        samples.append({
            "hole_radius":    round(hole_r, 4),
            "hole_center_x":  round(hole_cx, 4),
            "hole_center_y":  round(hole_cy, 4),
            "displacement":   round(displacement, 4),
            "plate_half_x":   round(plate_hx, 4),
            "plate_half_y":   round(plate_hy, 4),
            "thickness":      round(sum(t for _, t in LAYUP), 6),
            "layup":          LAYUP,
        })

    if len(samples) < n_samples:
        raise RuntimeError(
            f"Could only generate {len(samples)}/{n_samples} valid samples "
            f"after {max_attempts} attempts.  Widen PARAM_RANGES or reduce "
            f"INNER_SQUARE_FACTOR."
        )

    return samples
