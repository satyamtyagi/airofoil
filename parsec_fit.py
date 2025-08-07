#!/usr/bin/env python3
"""
PARSEC Airfoil Parameterization Tool

This script processes airfoil data files (.dat) and:
1. Fits PARSEC parameters to each airfoil
2. Creates PARSEC parameter files for each airfoil
3. Tracks min/max values for all 11 PARSEC parameters
4. Outputs a summary of parameter ranges

The PARSEC method represents airfoil shapes with 11 parameters:
- rLE    : Leading edge radius
- Xup    : Upper crest position X
- Yup    : Upper crest position Y
- YXXup  : Upper crest curvature
- Xlo    : Lower crest position X
- Ylo    : Lower crest position Y
- YXXlo  : Lower crest curvature
- Xte    : Trailing edge position X (usually 1.0)
- Yte    : Trailing edge position Y (usually 0.0)
- Yte'   : Trailing edge direction
- Δyte'' : Trailing edge wedge angle
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import pandas as pd
from tqdm import tqdm  # For progress bars

# Define directories
INPUT_DIR = "airfoils_uiuc"
OUTPUT_DIR = "airfoils_parsec"
RESULTS_DIR = "parsec_results"

# Create output directories if they don't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "plots"), exist_ok=True)

# PARSEC parameter names for reference
PARSEC_PARAM_NAMES = [
    "rLE", "Xup", "Yup", "YXXup", 
    "Xlo", "Ylo", "YXXlo", 
    "Xte", "Yte", "Yte'", "Δyte''"
]

class ParsecAirfoil:
    """Class for PARSEC airfoil parameterization and fitting"""
    
    def __init__(self, name="parsec"):
        """Initialize with default parameters"""
        self.name = name
        self.params = {
            "rLE": 0.01,      # Leading edge radius
            "Xup": 0.4,       # Upper crest position X
            "Yup": 0.06,      # Upper crest position Y
            "YXXup": -0.5,    # Upper crest curvature
            "Xlo": 0.4,       # Lower crest position X
            "Ylo": -0.06,     # Lower crest position Y
            "YXXlo": 0.5,     # Lower crest curvature
            "Xte": 1.0,       # Trailing edge position X
            "Yte": 0.0,       # Trailing edge position Y
            "Yte'": 0.0,      # Trailing edge direction
            "Δyte''": 0.0     # Trailing edge wedge angle
        }
        self.coeffs_upper = None
        self.coeffs_lower = None
        self.error = float('inf')
    
    def set_params(self, params_array):
        """Set parameters from array"""
        param_names = list(self.params.keys())
        for i, param in enumerate(params_array):
            if i < len(param_names):
                self.params[param_names[i]] = param
        self._calculate_coefficients()
    
    def _calculate_coefficients(self):
        """Calculate polynomial coefficients from PARSEC parameters"""
        # Extract parameters
        rLE = self.params["rLE"]
        Xup = self.params["Xup"]
        Yup = self.params["Yup"]
        YXXup = self.params["YXXup"]
        Xlo = self.params["Xlo"]
        Ylo = self.params["Ylo"]
        YXXlo = self.params["YXXlo"]
        Xte = self.params["Xte"]
        Yte = self.params["Yte"]
        dYte = self.params["Yte'"]
        d2Yte = self.params["Δyte''"]
        
        # For upper surface
        A_up = np.array([
            [1, 0.5, 0.25, 0.125, 0.0625, 0.03125],
            [1, Xup, Xup**2, Xup**3, Xup**4, Xup**5],
            [0, 1, 2*Xup, 3*Xup**2, 4*Xup**3, 5*Xup**4],
            [0, 0, 2, 6*Xup, 12*Xup**2, 20*Xup**3],
            [1, Xte, Xte**2, Xte**3, Xte**4, Xte**5],
            [0, 1, 2*Xte, 3*Xte**2, 4*Xte**3, 5*Xte**4]
        ])
        
        b_up = np.array([
            [0],
            [Yup],
            [0],
            [YXXup],
            [Yte],
            [dYte + 0.5 * d2Yte]
        ])
        
        # For lower surface
        A_lo = np.array([
            [1, 0.5, 0.25, 0.125, 0.0625, 0.03125],
            [1, Xlo, Xlo**2, Xlo**3, Xlo**4, Xlo**5],
            [0, 1, 2*Xlo, 3*Xlo**2, 4*Xlo**3, 5*Xlo**4],
            [0, 0, 2, 6*Xlo, 12*Xlo**2, 20*Xlo**3],
            [1, Xte, Xte**2, Xte**3, Xte**4, Xte**5],
            [0, 1, 2*Xte, 3*Xte**2, 4*Xte**3, 5*Xte**4]
        ])
        
        b_lo = np.array([
            [0],
            [Ylo],
            [0],
            [YXXlo],
            [Yte],
            [dYte - 0.5 * d2Yte]
        ])
        
        # Calculate coefficients by solving linear systems
        try:
            self.coeffs_upper = np.linalg.solve(A_up, b_up).flatten()
            self.coeffs_lower = np.linalg.solve(A_lo, b_lo).flatten()
        except np.linalg.LinAlgError:
            # If matrices are singular, use least squares
            self.coeffs_upper = np.linalg.lstsq(A_up, b_up, rcond=None)[0].flatten()
            self.coeffs_lower = np.linalg.lstsq(A_lo, b_lo, rcond=None)[0].flatten()
        
        # Check leading edge radius constraint
        y2_le = 2 * self.coeffs_upper[2]
        if rLE > 0 and y2_le != 0:
            # Leading edge radius constraint: 1/rLE = 2*a2
            scale = (1/rLE) / y2_le
            self.coeffs_upper = self.coeffs_upper * scale
            self.coeffs_lower = self.coeffs_lower * scale
    
    def evaluate(self, x):
        """Evaluate airfoil shape at given x-coordinates"""
        if self.coeffs_upper is None or self.coeffs_lower is None:
            self._calculate_coefficients()
        
        y_upper = np.zeros_like(x)
        y_lower = np.zeros_like(x)
        
        # Polynomial: y = sum(a_i * x^(i/2)) for i=0..5
        for i in range(6):
            y_upper += self.coeffs_upper[i] * x**(i/2)
            y_lower += self.coeffs_lower[i] * x**(i/2)
        
        return y_upper, y_lower
    
    def fit_to_data(self, x_data, y_data):
        """Fit PARSEC parameters to airfoil coordinate data"""
        # Separate upper and lower surfaces
        idx_le = np.argmin(x_data)
        x_upper = x_data[idx_le::-1]  # Reverse order for upper surface
        y_upper = y_data[idx_le::-1]
        x_lower = x_data[idx_le:]
        y_lower = y_data[idx_le:]
        
        # Initial guess based on data
        initial_params = np.array([
            0.01,                # rLE
            np.mean(x_upper),    # Xup
            np.max(y_upper),     # Yup
            -0.5,                # YXXup
            np.mean(x_lower),    # Xlo
            np.min(y_lower),     # Ylo
            0.5,                 # YXXlo
            1.0,                 # Xte
            0.0,                 # Yte
            0.0,                 # Yte'
            0.0                  # Δyte''
        ])
        
        # Set bounds for parameters
        bounds = [
            (0.0001, 0.1),       # rLE
            (0.1, 0.9),          # Xup
            (0.01, 0.3),         # Yup
            (-5.0, 0.0),         # YXXup
            (0.1, 0.9),          # Xlo
            (-0.3, -0.01),       # Ylo
            (0.0, 5.0),          # YXXlo
            (0.95, 1.05),        # Xte
            (-0.05, 0.05),       # Yte
            (-0.2, 0.2),         # Yte'
            (0.0, 0.5)           # Δyte''
        ]
        
        # Define error function
        def error_function(params):
            self.set_params(params)
            
            # Generate points for comparison
            x_eval = np.linspace(0, 1, 200)
            y_upper_fit, y_lower_fit = self.evaluate(x_eval)
            
            # Interpolate original data for comparison
            from scipy.interpolate import interp1d
            try:
                f_upper = interp1d(x_upper, y_upper, bounds_error=False, fill_value="extrapolate")
                f_lower = interp1d(x_lower, y_lower, bounds_error=False, fill_value="extrapolate")
                
                y_upper_orig = f_upper(x_eval)
                y_lower_orig = f_lower(x_eval)
                
                # Calculate error
                upper_error = np.mean((y_upper_fit - y_upper_orig)**2)
                lower_error = np.mean((y_lower_fit - y_lower_orig)**2)
                
                return upper_error + lower_error
            except:
                return 1e6  # Return large error if interpolation fails
        
        # Perform optimization
        result = minimize(error_function, initial_params, bounds=bounds, method='L-BFGS-B')
        self.set_params(result.x)
        self.error = result.fun
        
        return result.fun
    
    def plot_comparison(self, x_data, y_data, ax=None):
        """Plot comparison between original data and PARSEC fit"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
        
        # Plot original data
        ax.scatter(x_data, y_data, s=5, c='blue', label='Original')
        
        # Plot PARSEC fit
        x_fit = np.linspace(0, 1, 200)
        y_upper_fit, y_lower_fit = self.evaluate(x_fit)
        ax.plot(x_fit, y_upper_fit, 'r-', linewidth=2, label='PARSEC Upper')
        ax.plot(x_fit, y_lower_fit, 'r-', linewidth=2, label='PARSEC Lower')
        
        ax.set_aspect('equal')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_title(f"{self.name} - PARSEC Fit (Error: {self.error:.6f})")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.legend()
        
        return ax
    
    def save_parameters(self, filename):
        """Save PARSEC parameters to a file"""
        with open(filename, 'w') as f:
            f.write(f"# PARSEC parameters for {self.name}\n")
            f.write(f"# Fit error: {self.error:.6f}\n")
            for param_name, value in self.params.items():
                f.write(f"{param_name} = {value:.6f}\n")


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
    
    # Convert to arrays and sort by x
    points = np.array(coords)
    x_data = points[:, 0]
    y_data = points[:, 1]
    
    return x_data, y_data


def process_all_airfoils():
    """Process all airfoil files and collect statistics on PARSEC parameters"""
    # List all .dat files
    airfoil_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.dat')]
    
    # Initialize parameter tracking
    all_params = {name: [] for name in PARSEC_PARAM_NAMES}
    all_errors = []
    successful_fits = 0
    failed_fits = 0
    
    # Process each file
    print(f"Processing {len(airfoil_files)} airfoil files...")
    
    for filename in tqdm(airfoil_files):
        airfoil_name = os.path.splitext(filename)[0]
        input_path = os.path.join(INPUT_DIR, filename)
        output_path = os.path.join(OUTPUT_DIR, f"{airfoil_name}.parsec")
        
        # Read data
        x_data, y_data = read_airfoil_data(input_path)
        if x_data is None or y_data is None:
            print(f"  Skipping {filename}: Could not read data")
            failed_fits += 1
            continue
        
        # Create PARSEC airfoil and fit to data
        airfoil = ParsecAirfoil(name=airfoil_name)
        try:
            fit_error = airfoil.fit_to_data(x_data, y_data)
            
            # Skip bad fits
            if fit_error > 0.01:
                print(f"  Skipping {filename}: Poor fit (error={fit_error:.6f})")
                failed_fits += 1
                continue
            
            # Save parameters
            airfoil.save_parameters(output_path)
            
            # Save comparison plot
            fig, ax = plt.subplots(figsize=(10, 5))
            airfoil.plot_comparison(x_data, y_data, ax=ax)
            plt.savefig(os.path.join(RESULTS_DIR, "plots", f"{airfoil_name}_comparison.png"), dpi=100)
            plt.close(fig)
            
            # Track parameters
            for param_name in PARSEC_PARAM_NAMES:
                all_params[param_name].append(airfoil.params[param_name])
            all_errors.append(fit_error)
            
            successful_fits += 1
            
        except Exception as e:
            print(f"  Error fitting {filename}: {str(e)}")
            failed_fits += 1
            continue
    
    # Compile statistics
    stats = {
        "param": PARSEC_PARAM_NAMES,
        "min": [min(all_params[p]) if all_params[p] else float('nan') for p in PARSEC_PARAM_NAMES],
        "max": [max(all_params[p]) if all_params[p] else float('nan') for p in PARSEC_PARAM_NAMES],
        "mean": [np.mean(all_params[p]) if all_params[p] else float('nan') for p in PARSEC_PARAM_NAMES],
        "std": [np.std(all_params[p]) if all_params[p] else float('nan') for p in PARSEC_PARAM_NAMES]
    }
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(stats)
    df.to_csv(os.path.join(RESULTS_DIR, "parsec_stats.csv"), index=False)
    
    # Create summary text file
    with open(os.path.join(RESULTS_DIR, "parsec_summary.txt"), 'w') as f:
        f.write("PARSEC Parameter Ranges Summary\n")
        f.write("==============================\n\n")
        f.write(f"Total airfoils processed: {successful_fits + failed_fits}\n")
        f.write(f"Successful fits: {successful_fits}\n")
        f.write(f"Failed fits: {failed_fits}\n\n")
        f.write("Parameter Ranges:\n")
        f.write("----------------\n")
        for i, param in enumerate(PARSEC_PARAM_NAMES):
            f.write(f"{param:8s}: Min = {stats['min'][i]:.6f}, Max = {stats['max'][i]:.6f}, ")
            f.write(f"Mean = {stats['mean'][i]:.6f}, StdDev = {stats['std'][i]:.6f}\n")
        f.write("\nFit Errors:\n")
        f.write(f"Min Error: {min(all_errors):.6f}\n")
        f.write(f"Max Error: {max(all_errors):.6f}\n")
        f.write(f"Mean Error: {np.mean(all_errors):.6f}\n")
    
    # Generate visualization of parameter ranges
    fig, axs = plt.subplots(3, 4, figsize=(20, 12))
    axs = axs.flatten()
    
    for i, param in enumerate(PARSEC_PARAM_NAMES):
        if i < len(axs):
            values = all_params[param]
            if values:
                axs[i].hist(values, bins=20, alpha=0.7, color='blue')
                axs[i].set_title(f"{param} Distribution")
                axs[i].axvline(np.mean(values), color='r', linestyle='--', label=f'Mean: {np.mean(values):.4f}')
                axs[i].axvline(np.mean(values) + np.std(values), color='g', linestyle=':', label='+1 StdDev')
                axs[i].axvline(np.mean(values) - np.std(values), color='g', linestyle=':', label='-1 StdDev')
                axs[i].grid(True, alpha=0.3)
                axs[i].legend()
    
    # Last subplot shows error distribution
    if len(axs) > len(PARSEC_PARAM_NAMES):
        i = len(PARSEC_PARAM_NAMES)
        axs[i].hist(all_errors, bins=20, alpha=0.7, color='red')
        axs[i].set_title("Fit Error Distribution")
        axs[i].axvline(np.mean(all_errors), color='b', linestyle='--', label=f'Mean: {np.mean(all_errors):.6f}')
        axs[i].grid(True, alpha=0.3)
        axs[i].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "parsec_parameter_distributions.png"), dpi=150)
    plt.close(fig)
    
    print(f"\nProcessing complete. Results saved to {RESULTS_DIR}")
    print(f"  - Successful fits: {successful_fits}")
    print(f"  - Failed fits: {failed_fits}")
    print(f"  - Parameter statistics saved to: {os.path.join(RESULTS_DIR, 'parsec_stats.csv')}")
    print(f"  - Summary report saved to: {os.path.join(RESULTS_DIR, 'parsec_summary.txt')}")


if __name__ == "__main__":
    process_all_airfoils()
