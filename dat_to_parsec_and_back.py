import os, json, math
import numpy as np
from typing import Tuple, Dict
from parsec_core import build_dat_from_parsec

def read_airfoil_dat(path: str) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    with open(path, "r") as f:
        lines = f.readlines()
    for line in lines:
        s = line.strip()
        if not s:
            continue
        # try parse two floats
        parts = s.replace(",", " ").split()
        if len(parts) < 2:
            continue
        try:
            x = float(parts[0]); y = float(parts[1])
            xs.append(x); ys.append(y)
        except:
            # likely header line, skip
            continue
    x = np.array(xs, dtype=float)
    y = np.array(ys, dtype=float)
    return x, y

def split_upper_lower(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split points into upper and lower chains by detecting the LE (min x index) and
    using file order. This is robust for most .dat (TE->LE upper, LE->TE lower).
    """
    i_le = int(np.argmin(x))
    # segment1: 0 .. i_le  (likely TE->LE on upper)
    # segment2: i_le .. end (likely LE->TE on lower)
    x1, y1 = x[:i_le+1], y[:i_le+1]
    x2, y2 = x[i_le:], y[i_le:]

    # Decide which is upper: higher mean y
    if np.nanmean(y1) > np.nanmean(y2):
        xu_le2te = x1[::-1]; yu_le2te = y1[::-1]  # LE->TE for upper
        xl_le2te = x2;         yl_le2te = y2
    else:
        xu_le2te = x2;         yu_le2te = y2
        xl_le2te = x1[::-1];   yl_le2te = y1[::-1]

    # Ensure both go LE->TE with increasing x
    if xu_le2te[0] > xu_le2te[-1]:
        xu_le2te = xu_le2te[::-1]; yu_le2te = yu_le2te[::-1]
    if xl_le2te[0] > xl_le2te[-1]:
        xl_le2te = xl_le2te[::-1]; yl_le2te = yl_le2te[::-1]

    return xu_le2te, yu_le2te, xl_le2te, yl_le2te

def finite_diff_slope(x, y, k=5):
    """Slope at the right end using last k points least squares."""
    k = min(k, len(x))
    X = x[-k:]; Y = y[-k:]
    A = np.vstack([X, np.ones_like(X)]).T
    m, c = np.linalg.lstsq(A, Y, rcond=None)[0]
    return m

def crest_estimate(x, y, side: str) -> Tuple[float, float, float]:
    """
    Estimate crest location/value/curvature by searching for extremum and fitting a parabola locally.
    side: 'upper' -> max, 'lower' -> min.
    """
    if side == "upper":
        i0 = int(np.nanargmax(y))
    else:
        i0 = int(np.nanargmin(y))
    # fit a parabola y = ax^2 + bx + c over a small window
    r = 7
    i1 = max(0, i0 - r); i2 = min(len(x), i0 + r + 1)
    X = x[i1:i2]; Y = y[i1:i2]
    # Normalize x for conditioning
    xbar = np.mean(X); Xc = X - xbar
    A = np.vstack([Xc**2, Xc, np.ones_like(Xc)]).T
    a,b,c = np.linalg.lstsq(A, Y, rcond=None)[0]
    # Crest at vertex: x* = -b/(2a)  (in centered coords)
    if abs(a) < 1e-10:
        x_crest = x[i0]; y_crest = y[i0]; ypp = 0.0
    else:
        x_crest = -b/(2*a) + xbar
        y_crest = a*(x_crest - xbar)**2 + b*(x_crest - xbar) + c
        # second derivative w.r.t x is 2a
        ypp = 2.0*a
    return float(x_crest), float(y_crest), float(ypp)

def estimate_le_radius(xu, yu, xl, yl) -> float:
    """
    Crude r_LE via 3-point circle near LE on both surfaces; average both.
    """
    def circle_radius(p1, p2, p3):
        # returns radius of circumcircle
        (x1,y1),(x2,y2),(x3,y3) = p1,p2,p3
        A = np.array([[x1, y1, 1],
                      [x2, y2, 1],
                      [x3, y3, 1]], dtype=float)
        a = np.linalg.det(A)
        if abs(a) < 1e-14:
            return np.nan
        Sx = x1**2 + y1**2
        Sy = x2**2 + y2**2
        Sz = x3**2 + y3**2
        Dx = np.linalg.det(np.array([[Sx, y1, 1],
                                     [Sy, y2, 1],
                                     [Sz, y3, 1]]))
        Dy = np.linalg.det(np.array([[x1, Sx, 1],
                                     [x2, Sy, 1],
                                     [x3, Sz, 1]]))
        C  = np.linalg.det(np.array([[x1, y1, Sx],
                                     [x2, y2, Sy],
                                     [x3, Sz, Sz]]))  # minor typo: third col uses Sz twice; corrected below
        # Let's recompute properly:
        C  = np.linalg.det(np.array([[x1, y1, Sx],
                                     [x2, y2, Sy],
                                     [x3, y3, Sz]]))
        # center (ux, uy) = (Dx/(2a), -Dy/(2a)); radius from any point
        ux = Dx/(2*a); uy = -Dy/(2*a)
        r  = np.sqrt((x1-ux)**2 + (y1-uy)**2)
        return float(r)

    # pick 3 points closest to LE on each surface
    k = min(6, len(xu))
    idx_u = np.argsort(xu)[:k]
    idx_l = np.argsort(xl)[:k]
    # pick first, mid, last among those for radius
    def pick_triplet(x, y, idx):
        idx_sorted = idx[np.argsort(x[idx])]
        if len(idx_sorted) < 3:
            return None
        i1 = idx_sorted[0]
        i2 = idx_sorted[len(idx_sorted)//2]
        i3 = idx_sorted[-1]
        return (x[i1], y[i1]), (x[i2], y[i2]), (x[i3], y[i3])
    trip_u = pick_triplet(xu, yu, idx_u)
    trip_l = pick_triplet(xl, yl, idx_l)
    ru = circle_radius(*trip_u) if trip_u else np.nan
    rl = circle_radius(*trip_l) if trip_l else np.nan
    rle = np.nanmean([ru, rl])
    if not np.isfinite(rle) or rle <= 0:
        rle = 0.01
    return float(rle)

def estimate_te_geometry(xu, yu, xl, yl) -> Tuple[float,float,float,float]:
    """
    Returns (t_te, z_te, alpha_te, wedge_psi)
    alpha_te is camber-line angle at TE; psi is wedge opening (upper slope - lower slope).
    """
    # Use last 8 points near TE
    m_u = finite_diff_slope(xu, yu, k=8)
    m_l = finite_diff_slope(xl, yl, k=8)
    alpha_u = math.atan(m_u)
    alpha_l = math.atan(m_l)
    alpha_te = 0.5*(alpha_u + alpha_l)
    psi = (alpha_u - alpha_l)

    yte_u = float(yu[-1])
    yte_l = float(yl[-1])
    t_te = yte_u - yte_l
    z_te = 0.5*(yte_u + yte_l)
    return t_te, z_te, alpha_te, psi

def finite_diff_slope(x, y, k=8):
    k = min(k, len(x))
    X = x[-k:]; Y = y[-k:]
    A = np.vstack([X, np.ones_like(X)]).T
    m, c = np.linalg.lstsq(A, Y, rcond=None)[0]
    return float(m)

def dat_to_parsec_params(dat_path: str) -> Dict[str, float]:
    x, y = read_airfoil_dat(dat_path)
    xu, yu, xl, yl = split_upper_lower(x, y)

    # Crest estimates
    x_up, y_up, ypp_up = crest_estimate(xu, yu, "upper")
    x_lo, y_lo, ypp_lo = crest_estimate(xl, yl, "lower")

    # LE radius
    rLE = estimate_le_radius(xu, yu, xl, yl)

    # TE geometry
    t_te, z_te, alpha_te, psi = estimate_te_geometry(xu, yu, xl, yl)

    params = dict(
        rLE = rLE,
        x_up = x_up, y_up = y_up, ypp_up = ypp_up,
        x_lo = x_lo, y_lo = y_lo, ypp_lo = ypp_lo,
        te_thickness = t_te,
        te_camber = z_te,
        te_angle = alpha_te,
        te_wedge = psi,
    )
    return params

def rebuild_and_write(dat_path: str, out_dir: str) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(dat_path))[0]

    params = dat_to_parsec_params(dat_path)
    xs, ys, extras = build_dat_from_parsec(params, n_points=401)

    # Write outputs
    # Fix brace typo in f-string
    dat_out = os.path.join(out_dir, f"{base}_parsec.dat")
    with open(dat_out, "w") as f:
        f.write(f"{base}  (rebuilt via PARSEC)\n")
        for (xx, yy) in zip(xs, ys):
            f.write(f"{xx:.6f} {yy:.6f}\n")

    json_out = os.path.join(out_dir, f"{base}_parsec_params.json")
    with open(json_out, "w") as f:
        json.dump(dict(params=params, coeffs=extras), f, indent=2)

    return {"dat": dat_out, "json": json_out}

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Fit PARSEC to .dat and rebuild shape")
    ap.add_argument("dat", help="Path to input .dat file")
    ap.add_argument("--out", default="rebuilt_airfoils", help="Output directory")
    args = ap.parse_args()

    paths = rebuild_and_write(args.dat, args.out)
    print("Wrote:")
    for k,v in paths.items():
        print(f"  {k}: {v}")
