import numpy as np
from math import sqrt, tan

# --------- PARSEC core ------------
# Surface form: y(x) = sum_{i=1..6} a_i * x^{i/2}
# a1 encodes leading edge radius: rho_LE = a1^2 / 2  => a1 = sqrt(2*rLE); sign + upper, - lower
# Unknowns solved per-surface: a2..a6 from boundary conditions

def _powers_half(n=6):
    """Return the half powers i/2 for i=1..n."""
    return np.array([i/2.0 for i in range(1, n+1)])

def _phi_mat_and_rhs(x, a1, y_val=None, dydx_val=None, d2ydx2_val=None):
    """
    Build matrix rows and RHS contributions for y, y', y'' constraints at a given x,
    considering unknowns a2..a6 (a1 already fixed).
    Returns tuple (rows, rhs) where rows is a dict with keys 'y','dy','d2y' as available.
    """
    p = _powers_half(6)
    # Unknowns correspond to a2..a6 (indices 1..5 relative to p)
    exps = p[1:]                 # [1.0, 1.5, 2.0, 2.5, 3.0]
    d_exps = exps - 1.0          # for derivative powers
    d2_exps = exps - 2.0         # for second derivative powers

    rows = {}
    rhs = {}

    if y_val is not None:
        row_y = np.power(x, exps)
        rhs_y = y_val - a1 * (x**p[0])
        rows["y"] = row_y
        rhs["y"] = rhs_y

    if dydx_val is not None:
        coef = exps
        # derivative of x^{i/2} is (i/2) * x^{i/2 - 1}
        row_dy = coef * np.power(x, d_exps)
        # derivative of a1 * x^{1/2} is (1/2) a1 x^{-1/2}
        rhs_dy = dydx_val - a1 * (0.5) * (x**(p[0] - 1.0))
        rows["dy"] = row_dy
        rhs["dy"] = rhs_dy

    if d2ydx2_val is not None:
        coef = exps * (exps - 1.0)
        row_d2y = coef * np.power(x, d2_exps)
        # second derivative of a1 * x^{1/2} is -(1/4) a1 x^{-3/2}
        rhs_d2y = d2ydx2_val + a1 * 0.25 * (x**(p[0] - 2.0))  # note the sign
        rows["d2y"] = row_d2y
        rhs["d2y"] = rhs_d2y

    return rows, rhs

def solve_parsec_surface(rLE, side, x_crest, y_crest, ypp_crest,
                         y_te, alpha_te_rad):
    """
    Solve a2..a6 for one surface given common rLE and per-surface constraints.
    side: 'upper' or 'lower' (controls sign of a1)
    """
    assert side in ("upper", "lower")
    a1 = sqrt(2.0 * max(rLE, 1e-6))
    if side == "lower":
        a1 *= -1.0

    # Build 5x5 system for [a2..a6]
    rows = []
    rhs = []

    # 1) TE y at x=1
    r_te, b_te = _phi_mat_and_rhs(1.0, a1, y_val=y_te)
    rows.append(r_te["y"]); rhs.append(b_te["y"])

    # 2) TE slope at x=1
    r_te_s, b_te_s = _phi_mat_and_rhs(1.0, a1, dydx_val=np.tan(alpha_te_rad))
    rows.append(r_te_s["dy"]); rhs.append(b_te_s["dy"])

    # 3) crest value y(xc) = yc
    r_c, b_c = _phi_mat_and_rhs(x_crest, a1, y_val=y_crest)
    rows.append(r_c["y"]); rhs.append(b_c["y"])

    # 4) crest slope = 0
    r_c_s, b_c_s = _phi_mat_and_rhs(x_crest, a1, dydx_val=0.0)
    rows.append(r_c_s["dy"]); rhs.append(b_c_s["dy"])

    # 5) crest curvature y''(xc) = ypp_crest
    r_c_2, b_c_2 = _phi_mat_and_rhs(x_crest, a1, d2ydx2_val=ypp_crest)
    rows.append(r_c_2["d2y"]); rhs.append(b_c_2["d2y"])

    A = np.vstack(rows)
    b = np.array(rhs)

    coeffs_2_to_6 = np.linalg.solve(A, b)
    # Full coefficients a1..a6
    a = np.empty(6)
    a[0] = a1
    a[1:] = coeffs_2_to_6
    return a

def eval_surface(a, x):
    """Evaluate y and its first two derivatives for a given coefficient vector a (len=6)."""
    x = np.asarray(x)
    p = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    # y
    y = np.zeros_like(x, dtype=float)
    for i in range(6):
        y += a[i] * np.power(x, p[i])
    # y'
    yp = np.zeros_like(x, dtype=float)
    for i in range(6):
        yp += a[i] * p[i] * np.power(x, p[i]-1.0)
    # y''
    ypp = np.zeros_like(x, dtype=float)
    for i in range(6):
        ypp += a[i] * p[i] * (p[i]-1.0) * np.power(x, p[i]-2.0)
    return y, yp, ypp

def cosine_spacing(n=201):
    """Cosine-clustered points in [0,1]."""
    theta = np.linspace(0.0, np.pi, n)
    return 0.5 * (1.0 - np.cos(theta))

def build_dat_from_parsec(params, n_points=201):
    """
    params must include:
      rLE, x_up, y_up, ypp_up, x_lo, y_lo, ypp_lo, te_thickness, te_camber, te_angle, te_wedge
    We derive per-surface TE y and slopes from these.
    """
    rLE = float(params["rLE"])

    # TE geometry
    t_te = float(params.get("te_thickness", 0.0))
    z_te = float(params.get("te_camber", 0.0))
    alpha_te = float(params.get("te_angle", 0.0))  # radians
    psi = float(params.get("te_wedge", 0.0))       # radians wedge opening

    y_te_up = z_te + 0.5 * t_te
    y_te_lo = z_te - 0.5 * t_te
    alpha_up = alpha_te + 0.5 * psi
    alpha_lo = alpha_te - 0.5 * psi

    # Crest constraints
    x_up = float(params["x_up"]); y_up = float(params["y_up"]); ypp_up = float(params["ypp_up"])
    x_lo = float(params["x_lo"]); y_lo = float(params["y_lo"]); ypp_lo = float(params["ypp_lo"])

    # Solve surfaces
    a_up = solve_parsec_surface(rLE, "upper", x_up, y_up, ypp_up, y_te_up, alpha_up)
    a_lo = solve_parsec_surface(rLE, "lower", x_lo, y_lo, ypp_lo, y_te_lo, alpha_lo)

    # Sample
    x = cosine_spacing(n_points)
    y_up_vals = eval_surface(a_up, x)[0]
    y_lo_vals = eval_surface(a_lo, x)[0]

    # Build .dat order: start at TE upper -> LE -> TE lower
    # Upper surface from x=1 -> 0 (reverse), then lower surface from x=0 -> 1 (forward)
    xu = x[::-1]; yu = y_up_vals[::-1]
    xl = x[1:];   yl = y_lo_vals[1:]   # skip duplicate LE at x=0

    xs = np.concatenate([xu, xl])
    ys = np.concatenate([yu, yl])

    return xs, ys, {"a_up": a_up.tolist(), "a_lo": a_lo.tolist()}
