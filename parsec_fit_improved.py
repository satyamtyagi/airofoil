import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


class ParsecAirfoil:
    """
    PARSEC airfoil fitter/evaluator with geometric validity checks.

    Parameters modeled (dict self.params):
      rLE           : leading-edge radius (>0)
      Xup, Yup      : x,y of upper crest
      YXXup         : y'' at upper crest (curvature-like, typically <0)
      Xlo, Ylo      : x,y of lower crest
      YXXlo         : y'' at lower crest (typically >0)
      Xte           : trailing-edge x (kept fixed at 1.0 after chord normalization)
      Yte           : trailing-edge y (small camber/thickness offset)
      AlphaTE       : mean TE slope (dy/dx) at x=1 for (upper+lower)/2
      DeltaAlphaTE  : TE slope wedge (upper - lower)

    Geometry: y(x) = sum_{k=1..6} a_k * x^{k/2}.
      a1 = +sqrt(2 rLE) for upper, and -sqrt(2 rLE) for lower.
      a2..a6 solved from 5 constraints per surface:
        - y(Xc) = Yc
        - y'(Xc) = 0
        - y''(Xc) = Y''_c
        - y(1) = Yte
        - y'(1) = AlphaTE ± DeltaAlphaTE/2  (upper uses +, lower uses -)
    """

    # order for saving / reporting
    PARAM_ORDER = [
        "rLE", "Xup", "Yup", "YXXup",
        "Xlo", "Ylo", "YXXlo",
        "Xte", "Yte", "AlphaTE", "DeltaAlphaTE"
    ]
    # variables optimized (we keep Xte fixed at 1.0)
    OPTIM_ORDER = [
        "rLE", "Xup", "Yup", "YXXup",
        "Xlo", "Ylo", "YXXlo",
        "Yte", "AlphaTE", "DeltaAlphaTE"
    ]

    def __init__(self, name="parsec"):
        self.name = name
        self.params = {
            "rLE": 0.01,
            "Xup": 0.4,  "Yup":  0.06, "YXXup": -0.5,
            "Xlo": 0.4,  "Ylo": -0.06, "YXXlo":  0.5,
            "Xte": 1.0,  "Yte":  0.0,
            "AlphaTE": 0.0, "DeltaAlphaTE": 0.0
        }
        self.coeffs_upper = None  # a1..a6
        self.coeffs_lower = None  # a1..a6
        self.error = float('inf')
        self.is_valid = False

    # ---------- math helpers (PARSEC half-power basis) ----------

    @staticmethod
    def _basis_rows(x):
        """
        Build rows for y, y', y'' for unknowns a2..a6 evaluated at x.
        Returns (row_y, row_dy, row_d2) each of length 5 (for a2..a6).
        """
        x = np.asarray(x, dtype=float)
        # exponents for a2..a6 are [1.0, 1.5, 2.0, 2.5, 3.0]
        p = np.arange(2, 7) / 2.0
        row_y = x**p
        row_dy = p * x**(p - 1.0)
        row_d2 = p * (p - 1.0) * x**(p - 2.0)
        return row_y, row_dy, row_d2

    @staticmethod
    def _sqrt_contrib(x):
        """
        Contribution of the a1 * sqrt(x) term to y, y', y'' at x.
        """
        x = np.asarray(x, dtype=float)
        # avoid division at exactly x=0 for derivatives
        x_safe = np.maximum(x, 1e-12)
        y = np.sqrt(x_safe)
        dy = 0.5 * x_safe**(-0.5)
        d2y = 0.5 * (-0.5) * x_safe**(-1.5)
        return y, dy, d2y

    def _solve_surface(self, Xc, Yc, YXXc, Yte, alpha_mean, delta_alpha, sign_upper):
        """
        Solve for a2..a6 on one surface given fixed a1 from rLE.
        sign_upper = +1 for upper, -1 for lower.
        """
        # a1 from rLE with sign
        a1 = np.sqrt(2.0 * max(self.params["rLE"], 1e-12)) * (1.0 if sign_upper > 0 else -1.0)

        A = []
        b = []

        # Crest constraints at Xc
        ry, rdy, rd2 = self._basis_rows(Xc)
        sy, sdy, sd2 = self._sqrt_contrib(Xc)
        A.append(ry);   b.append(Yc   - a1 * sy)
        A.append(rdy);  b.append(0.0  - a1 * sdy)
        A.append(rd2);  b.append(YXXc - a1 * sd2)

        # TE value at x=1
        ry, rdy, rd2 = self._basis_rows(1.0)
        sy, sdy, sd2 = self._sqrt_contrib(1.0)  # (1, 0.5, -0.25)
        A.append(ry);   b.append(Yte - a1 * sy)

        # TE slope: mean ± wedge/2
        slope_target = alpha_mean + 0.5 * sign_upper * delta_alpha
        A.append(rdy);  b.append(slope_target - a1 * sdy)

        A = np.vstack(A)
        b = np.array(b, dtype=float)

        # Solve for a2..a6 (5 unknowns, 5 equations)
        a_rest = np.linalg.solve(A, b)
        # Assemble full a1..a6
        a = np.empty(6)
        a[0] = a1
        a[1:] = a_rest
        return a

    def _calculate_coefficients(self):
        """
        Build a1..a6 for upper/lower from current self.params.
        """
        Xup, Yup, YXXup = self.params["Xup"], self.params["Yup"], self.params["YXXup"]
        Xlo, Ylo, YXXlo = self.params["Xlo"], self.params["Ylo"], self.params["YXXlo"]
        Yte = self.params["Yte"]
        alpha_mean = self.params["AlphaTE"]
        delta_alpha = self.params["DeltaAlphaTE"]

        # Xte is assumed 1.0 after normalization; do not solve for it.

        self.coeffs_upper = self._solve_surface(
            Xup, Yup, YXXup, Yte, alpha_mean, delta_alpha, +1
        )
        self.coeffs_lower = self._solve_surface(
            Xlo, Ylo, YXXlo, Yte, alpha_mean, delta_alpha, -1
        )

    # ---------- public API ----------

    def set_params_from_vector(self, vec):
        """
        Set subset of parameters from a vector following OPTIM_ORDER.
        (Xte remains fixed at 1.0.)
        """
        for k, v in zip(self.OPTIM_ORDER, vec):
            self.params[k] = float(v)
        self.params["Xte"] = 1.0
        self._calculate_coefficients()

    def get_params_vector(self):
        """Return params in OPTIM_ORDER (useful for optimizers)."""
        return np.array([self.params[k] for k in self.OPTIM_ORDER], dtype=float)

    def evaluate(self, x):
        """
        Evaluate upper/lower surfaces at x (array-like), 0<=x<=1.
        """
        if self.coeffs_upper is None or self.coeffs_lower is None:
            self._calculate_coefficients()

        x = np.asarray(x, dtype=float)
        p = np.arange(1, 7) / 2.0  # [0.5, 1, 1.5, 2, 2.5, 3]
        y_up = np.zeros_like(x, dtype=float)
        y_lo = np.zeros_like(x, dtype=float)
        for i in range(6):
            y_up += self.coeffs_upper[i] * x**p[i]
            y_lo += self.coeffs_lower[i] * x**p[i]
        return y_up, y_lo

    # ---------- geometry validation ----------

    def _polyline_for_validation(self, n=400):
        """
        Build a closed polyline (TE→LE on upper, LE→TE on lower) for intersection tests.
        Uses denser sampling near LE/TE.
        """
        n = max(n, 40)
        n3 = n // 3
        x_le = np.linspace(0.0 + 1e-6, 0.10, n3, endpoint=False)
        x_mid = np.linspace(0.10, 0.90, n3, endpoint=False)
        x_te = np.linspace(0.90, 1.0, n - 2 * n3)
        x = np.concatenate([x_le, x_mid, x_te])

        yu, yl = self.evaluate(x)
        # build loop: TE->LE on upper (reverse x), then LE->TE on lower (forward x)
        x_full = np.concatenate([x[::-1], x])
        y_full = np.concatenate([yu[::-1], yl])
        return x_full, y_full

    @staticmethod
    def _seg_intersect(p, q, r, s, eps=1e-12):
        """
        Proper/weak segment intersection test between pq and rs.
        Returns True if they intersect (excluding shared endpoints is handled by caller).
        """
        def orient(a, b, c):
            return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

        def on_seg(a, b, c):
            # c on segment ab (collinear assumed)
            return (min(a[0], b[0]) - eps <= c[0] <= max(a[0], b[0]) + eps and
                    min(a[1], b[1]) - eps <= c[1] <= max(a[1], b[1]) + eps)

        o1 = orient(p, q, r)
        o2 = orient(p, q, s)
        o3 = orient(r, s, p)
        o4 = orient(r, s, q)

        # general case
        if (o1 * o2 < -eps) and (o3 * o4 < -eps):
            return True
        # collinear cases
        if abs(o1) <= eps and on_seg(p, q, r): return True
        if abs(o2) <= eps and on_seg(p, q, s): return True
        if abs(o3) <= eps and on_seg(r, s, p): return True
        if abs(o4) <= eps and on_seg(r, s, q): return True
        return False

    def _has_self_intersection(self, x_full, y_full):
        """
        O(n^2) sweep for segment intersections, ignoring adjacent segments and
        the wrap-around neighbor.
        """
        pts = np.column_stack([x_full, y_full])
        n = len(pts)
        if n < 4:
            return False
        # segments: (i -> i+1), with n-1 -> 0 closing the loop
        for i in range(n):
            p1 = pts[i]
            p2 = pts[(i + 1) % n]
            for j in range(i + 2, n):
                # skip adjacent/shared-vertex pairs and wrap-around neighbor
                if j == i or (j + 1) % n == i:
                    continue
                q1 = pts[j]
                q2 = pts[(j + 1) % n]
                # also skip pair that shares the closing point
                if (i == 0 and (j + 1) % n == 0):
                    continue
                if self._seg_intersect(p1, p2, q1, q2):
                    return True
        return False

    def check_thickness(self, num_points=1000, eps=1e-5):
        """
        Check positive thickness everywhere on [0,1].
        Returns (is_valid, min_thickness).
        """
        n = max(100, int(num_points))
        n3 = n // 3
        x = np.unique(np.concatenate([
            np.linspace(0.0 + 1e-6, 0.10, n3, endpoint=False),
            np.linspace(0.10, 0.90, n3, endpoint=False),
            np.linspace(0.90, 1.0, n - 2 * n3)
        ]))
        yu, yl = self.evaluate(x)
        t = yu - yl
        return bool(np.all(t > eps)), float(np.min(t))

    def check_geometric_validity(self):
        """
        Checks self-intersection of the closed airfoil loop.
        Sets self.is_valid and returns it.
        """
        xf, yf = self._polyline_for_validation(n=500)
        self.is_valid = not self._has_self_intersection(xf, yf)
        return self.is_valid

    # ---------- fitting ----------

    def fit_to_data(self, x_data, y_data, enforce_validity=True):
        """
        Fit PARSEC params to an airfoil coordinate loop.
        Expects a single closed loop sampled TE→upper→LE→lower→TE
        (typical UIUC .dat style is fine). We split at the LE and build
        monotone x interpolants for each surface.
        """
        x_data = np.asarray(x_data, dtype=float)
        y_data = np.asarray(y_data, dtype=float)
        if x_data.ndim != 1 or y_data.ndim != 1 or x_data.size != y_data.size:
            raise ValueError("x_data and y_data must be 1D arrays of equal length")

        # 1) Split at LE (min x)
        idx_le = int(np.argmin(x_data))
        xu_raw, yu_raw = x_data[:idx_le + 1], y_data[:idx_le + 1]  # (upper segment, usually TE->LE)
        xl_raw, yl_raw = x_data[idx_le:],     y_data[idx_le:]      # (lower segment, usually LE->TE)

        # 2) Sort each surface by increasing x
        iu = np.argsort(xu_raw); xu, yu = xu_raw[iu], yu_raw[iu]
        il = np.argsort(xl_raw); xl, yl = xl_raw[il], yl_raw[il]

        # 3) Normalize x to [0,1] chord
        xmin = min(xu.min(), xl.min())
        xmax = max(xu.max(), xl.max())
        scale = xmax - xmin if xmax > xmin else 1.0
        xu = (xu - xmin) / scale
        xl = (xl - xmin) / scale
        # (y stays in original units; PARSEC is chord-normalized only in x)

        # 4) Build monotone interpolants
        f_upper = interp1d(xu, yu, bounds_error=False, fill_value="extrapolate", assume_sorted=True)
        f_lower = interp1d(xl, yl, bounds_error=False, fill_value="extrapolate", assume_sorted=True)

        # 5) Initial guesses
        #    Crest x as argmax/argmin y on each surface; clamp into (0.05, 0.95)
        Xup0 = float(np.clip(xu[np.argmax(yu)], 0.05, 0.95))
        Xlo0 = float(np.clip(xl[np.argmin(yl)], 0.05, 0.95))
        Yup0 = float(np.max(yu))
        Ylo0 = float(np.min(yl))
        init = np.array([
            0.01,         # rLE
            Xup0, Yup0,  -0.5,   # upper crest + curvature
            Xlo0, Ylo0,   0.5,   # lower crest + curvature
            0.0,          # Yte
            0.0,          # AlphaTE (mean slope)
            0.0           # DeltaAlphaTE (wedge)
        ], dtype=float)

        # 6) Bounds (tune as needed)
        bounds = [
            (1e-4, 0.10),      # rLE
            (0.02, 0.98),      # Xup
            (-0.05, 0.30),     # Yup  (allow small negative for cusps)
            (-5.0, -1e-4),     # YXXup (typically negative)
            (0.02, 0.98),      # Xlo
            (-0.30, 0.05),     # Ylo
            (1e-4, 5.0),       # YXXlo (typically positive)
            (-0.05, 0.05),     # Yte
            (-0.5, 0.5),       # AlphaTE
            (0.0, 1.0)         # DeltaAlphaTE (nonnegative wedge)
        ]

        x_eval = np.linspace(0.0, 1.0, 300)

        def error_function(vec):
            # set parameters and coefficients
            self.set_params_from_vector(vec)

            # evaluate model on common grid
            yu_fit, yl_fit = self.evaluate(x_eval)

            # original surfaces resampled on x_eval
            yu_orig = f_upper(x_eval)
            yl_orig = f_lower(x_eval)

            # MSE fit error
            err_u = np.mean((yu_fit - yu_orig) ** 2)
            err_l = np.mean((yl_fit - yl_orig) ** 2)

            # thickness penalty (enforce small epsilon margin)
            t = yu_fit - yl_fit
            neg = np.minimum(t - 1e-5, 0.0)
            thick_pen = 1e5 * np.mean(neg ** 2)

            # optional: extra dense thickness check
            is_pos, tmin = self.check_thickness(num_points=800)
            thick_pen2 = 0.0 if is_pos else 1e6 * (1.0 + max(0.0, -tmin))

            # self-intersection penalty
            x_full, y_full = self._polyline_for_validation(n=400)
            geo_pen = 1e6 if self._has_self_intersection(x_full, y_full) else 0.0

            return err_u + err_l + thick_pen + thick_pen2 + (geo_pen if enforce_validity else 0.0)

        res = minimize(error_function, init, method="L-BFGS-B", bounds=bounds)
        self.set_params_from_vector(res.x)
        self.error = float(res.fun)
        self.check_geometric_validity()
        return self.error

    # ---------- viz / io ----------

    def plot_comparison(self, x_data=None, y_data=None, ax=None, title_extra=""):
        """
        Plot PARSEC fit vs. original points (if provided).
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(9, 4.8))

        if x_data is not None and y_data is not None:
            ax.scatter(x_data, y_data, s=6, label="Original", alpha=0.7)

        x_fit = np.linspace(0.0, 1.0, 400)
        yu, yl = self.evaluate(x_fit)
        ax.plot(x_fit, yu, lw=2, label="PARSEC Upper")
        ax.plot(x_fit, yl, lw=2, label="PARSEC Lower")

        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, ls="--", alpha=0.4)
        ax.set_xlabel("x (chord)")
        ax.set_ylabel("y")
        status = "Valid" if self.is_valid else "INVALID"
        ax.set_title(f"{self.name}  |  err={self.error:.6g}  |  geom={status} {title_extra}")
        ax.legend()
        return ax

    def save_parameters(self, filename):
        """
        Save parameters in a simple text format (one per line).
        """
        with open(filename, "w") as f:
            f.write(f"# PARSEC parameters for {self.name}\n")
            f.write(f"# Fit error: {self.error:.6g}\n")
            f.write(f"# Geometry valid: {self.is_valid}\n")
            for k in self.PARAM_ORDER:
                f.write(f"{k} = {self.params[k]:.9g}\n")
