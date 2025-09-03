#!/usr/bin/env python3
"""
Debug script for PARSEC fitting issues

This script:
1. Loads a single airfoil for focused debugging
2. Provides verbose logging of optimization process
3. Visualizes the fitting steps
4. Tests different optimization parameters
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from parsec_fit_improved import ParsecAirfoil

# Debug settings
DEBUG_AIRFOIL = "ag13.dat"  # One of our top performers
INPUT_DIR = "airfoils_uiuc"
MAX_ITERATIONS = 100  # Limit iterations to prevent hanging
VISUALIZE_STEPS = True
USE_SIMPLIFIED_ERROR = True

def read_airfoil_data(filename):
    """Read airfoil coordinates from a dat file"""
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    coords = []
    for line in lines:
        if line.strip() and len(line.strip().split()) >= 2:
            try:
                x, y = map(float, line.strip().split()[:2])
                coords.append((x, y))
            except ValueError:
                continue
    
    if not coords:
        return None, None
    
    # Convert to arrays
    points = np.array(coords)
    x_data = points[:, 0]
    y_data = points[:, 1]
    
    return x_data, y_data

class DebugParsecAirfoil(ParsecAirfoil):
    """Extended ParsecAirfoil with debugging capabilities"""
    
    def __init__(self, name="debug_parsec"):
        super().__init__(name=name)
        self.iteration = 0
        self.error_history = []
        self.param_history = []
    
    def fit_to_data_with_debug(self, x_data, y_data, enforce_validity=True, method='L-BFGS-B'):
        """Debug version of fit_to_data with additional logging and visualization"""
        x_data = np.asarray(x_data, dtype=float)
        y_data = np.asarray(y_data, dtype=float)
        
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
        
        print(f"Airfoil data processed: {len(xu)} upper points, {len(xl)} lower points")
        print(f"X range normalized from [{xmin}, {xmax}] to [0, 1]")
        
        # 4) Build monotone interpolants
        from scipy.interpolate import interp1d
        f_upper = interp1d(xu, yu, bounds_error=False, fill_value="extrapolate", assume_sorted=True)
        f_lower = interp1d(xl, yl, bounds_error=False, fill_value="extrapolate", assume_sorted=True)

        # 5) Initial guesses
        Xup0 = float(np.clip(xu[np.argmax(yu)], 0.05, 0.95))
        Xlo0 = float(np.clip(xl[np.argmin(yl)], 0.05, 0.95))
        Yup0 = float(np.max(yu))
        Ylo0 = float(np.min(yl))
        
        print(f"Initial guesses: Xup={Xup0}, Yup={Yup0}, Xlo={Xlo0}, Ylo={Ylo0}")
        
        init = np.array([
            0.01,         # rLE
            Xup0, Yup0,  -0.5,   # upper crest + curvature
            Xlo0, Ylo0,   0.5,   # lower crest + curvature
            0.0,          # Yte
            0.0,          # AlphaTE (mean slope)
            0.0           # DeltaAlphaTE (wedge)
        ], dtype=float)

        # 6) Bounds
        bounds = [
            (1e-4, 0.10),      # rLE
            (0.02, 0.98),      # Xup
            (-0.05, 0.30),     # Yup
            (-5.0, -1e-4),     # YXXup
            (0.02, 0.98),      # Xlo
            (-0.30, 0.05),     # Ylo
            (1e-4, 5.0),       # YXXlo
            (-0.05, 0.05),     # Yte
            (-0.5, 0.5),       # AlphaTE
            (0.0, 1.0)         # DeltaAlphaTE
        ]
        
        print("Bounds:")
        for i, (param, bound) in enumerate(zip(self.OPTIM_ORDER, bounds)):
            print(f"  {param}: {bound}")

        # Reset trackers
        self.iteration = 0
        self.error_history = []
        self.param_history = []
        
        # Create evaluation grid
        x_eval = np.linspace(0.0, 1.0, 300)

        # Define simpler error function for debugging
        def error_function_simple(vec):
            self.iteration += 1
            
            # Set parameters and coefficients
            self.set_params_from_vector(vec)
            
            # Evaluate model on common grid
            yu_fit, yl_fit = self.evaluate(x_eval)
            
            # Original surfaces resampled on x_eval
            yu_orig = f_upper(x_eval)
            yl_orig = f_lower(x_eval)
            
            # MSE fit error only - no penalties
            err_u = np.mean((yu_fit - yu_orig) ** 2)
            err_l = np.mean((yl_fit - yl_orig) ** 2)
            err = err_u + err_l
            
            # Log
            if self.iteration % 10 == 0 or self.iteration <= 3:
                print(f"Iteration {self.iteration}: error = {err:.6f}")
            
            # Store history
            self.error_history.append(err)
            self.param_history.append(vec.copy())
            
            # Visualize current fit
            if VISUALIZE_STEPS and (self.iteration == 1 or self.iteration % 20 == 0):
                self.visualize_current_fit(x_data, y_data, x_eval, yu_orig, yl_orig)
            
            # Limit iterations to prevent hanging
            if self.iteration >= MAX_ITERATIONS:
                print("Reached maximum iterations, stopping optimization")
                return err
                
            return err

        # Original error function with all penalties
        def error_function_full(vec):
            self.iteration += 1
            
            # Set parameters and coefficients
            self.set_params_from_vector(vec)
            
            # Evaluate model on common grid
            yu_fit, yl_fit = self.evaluate(x_eval)
            
            # Original surfaces resampled on x_eval
            yu_orig = f_upper(x_eval)
            yl_orig = f_lower(x_eval)
            
            # MSE fit error
            err_u = np.mean((yu_fit - yu_orig) ** 2)
            err_l = np.mean((yl_fit - yl_orig) ** 2)
            
            # Thickness penalty (enforce small epsilon margin)
            t = yu_fit - yl_fit
            neg = np.minimum(t - 1e-5, 0.0)
            thick_pen = 1e5 * np.mean(neg ** 2)
            
            # Optional: extra dense thickness check
            is_pos, tmin = self.check_thickness(num_points=800)
            thick_pen2 = 0.0 if is_pos else 1e6 * (1.0 + max(0.0, -tmin))
            
            # Self-intersection penalty
            x_full, y_full = self._polyline_for_validation(n=400)
            geo_pen = 1e6 if self._has_self_intersection(x_full, y_full) else 0.0
            
            err = err_u + err_l + thick_pen + thick_pen2 + (geo_pen if enforce_validity else 0.0)
            
            # Log
            if self.iteration % 10 == 0 or self.iteration <= 3:
                print(f"Iteration {self.iteration}: error = {err:.6f} (fit_u={err_u:.6f}, fit_l={err_l:.6f}, thick={thick_pen:.6f}, thick2={thick_pen2:.6f}, geo={geo_pen:.6f})")
            
            # Store history
            self.error_history.append(err)
            self.param_history.append(vec.copy())
            
            # Visualize current fit
            if VISUALIZE_STEPS and (self.iteration == 1 or self.iteration % 20 == 0):
                self.visualize_current_fit(x_data, y_data, x_eval, yu_orig, yl_orig)
            
            # Limit iterations to prevent hanging
            if self.iteration >= MAX_ITERATIONS:
                print("Reached maximum iterations, stopping optimization")
                return err
                
            return err
        
        error_function = error_function_simple if USE_SIMPLIFIED_ERROR else error_function_full
        
        # Try different optimization methods
        methods_to_try = ['Nelder-Mead', 'Powell', 'BFGS', 'L-BFGS-B']
        if method in methods_to_try:
            methods_to_try = [method]  # Only use the specified method
        
        print(f"\nTrying {len(methods_to_try)} optimization methods: {methods_to_try}")
        best_error = float('inf')
        best_result = None
        
        for method in methods_to_try:
            print(f"\nOptimization method: {method}")
            self.iteration = 0
            self.error_history = []
            self.param_history = []
            
            options = {'maxiter': MAX_ITERATIONS}
            if method in ['L-BFGS-B', 'TNC']:
                options['maxfun'] = MAX_ITERATIONS * 2
                
            res = minimize(
                error_function, 
                init, 
                method=method,
                bounds=bounds if method in ['L-BFGS-B', 'TNC'] else None,
                options=options
            )
            
            print(f"Result for {method}: success={res.success}, status={res.message}")
            print(f"Final error: {res.fun:.6f}, iterations: {self.iteration}")
            
            if res.fun < best_error:
                best_error = res.fun
                best_result = res
                best_method = method
        
        print(f"\nBest method: {best_method} with error {best_error:.6f}")
        
        # Set final parameters
        self.set_params_from_vector(best_result.x)
        self.error = float(best_result.fun)
        self.check_geometric_validity()
        
        print(f"Final validity: {self.is_valid}")
        print(f"Final params: {dict(zip(self.OPTIM_ORDER, best_result.x))}")
        
        # Final visualization
        self.plot_comparison(x_data, y_data)
        plt.savefig("debug_final_fit.png", dpi=150)
        plt.close()
        
        # Plot error history
        plt.figure(figsize=(10, 5))
        plt.plot(self.error_history)
        plt.yscale('log')
        plt.grid(True)
        plt.xlabel('Iteration')
        plt.ylabel('Error (log scale)')
        plt.title(f'Optimization Convergence - {best_method}')
        plt.savefig("debug_convergence.png", dpi=150)
        plt.close()
        
        return self.error
    
    def visualize_current_fit(self, x_data, y_data, x_eval, yu_orig, yl_orig):
        """Visualize current fit during optimization"""
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Plot original data
        ax.scatter(x_data, y_data, s=2, color='black', alpha=0.5, label='Original')
        
        # Plot current fit
        yu_fit, yl_fit = self.evaluate(x_eval)
        ax.plot(x_eval, yu_fit, 'r-', label=f'Upper (iter {self.iteration})')
        ax.plot(x_eval, yl_fit, 'b-', label=f'Lower (iter {self.iteration})')
        
        # Plot interpolated original
        ax.plot(x_eval, yu_orig, 'r--', alpha=0.5)
        ax.plot(x_eval, yl_orig, 'b--', alpha=0.5)
        
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Iteration {self.iteration} - Error: {self.error_history[-1]:.6f}')
        ax.legend()
        plt.savefig(f"debug_iter_{self.iteration:03d}.png", dpi=100)
        plt.close(fig)

def main():
    """Main debug function"""
    filename = os.path.join(INPUT_DIR, DEBUG_AIRFOIL)
    
    print(f"=== Debug PARSEC Fitting for {DEBUG_AIRFOIL} ===")
    print(f"Max iterations: {MAX_ITERATIONS}")
    print(f"Simplified error function: {USE_SIMPLIFIED_ERROR}")
    print(f"Visualize steps: {VISUALIZE_STEPS}")
    print("")
    
    # Read airfoil data
    x_data, y_data = read_airfoil_data(filename)
    if x_data is None or y_data is None:
        print(f"Error: Could not read data from {filename}")
        return
    
    print(f"Read {len(x_data)} points from {filename}")
    
    # Create debug airfoil object
    airfoil_name = os.path.splitext(os.path.basename(filename))[0]
    airfoil = DebugParsecAirfoil(name=airfoil_name)
    
    # Try both with and without enforcing validity
    for enforce_validity in [True, False]:
        print(f"\n=== Fitting with enforce_validity={enforce_validity} ===")
        try:
            error = airfoil.fit_to_data_with_debug(x_data, y_data, enforce_validity=enforce_validity)
            print(f"Fit completed with error: {error:.6f}")
            print(f"Is valid: {airfoil.is_valid}")
        except Exception as e:
            print(f"Error during fitting: {str(e)}")
    
if __name__ == "__main__":
    main()
