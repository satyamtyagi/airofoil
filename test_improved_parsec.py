#!/usr/bin/env python3
"""
Test script for the improved PARSEC implementation
Focuses on the top 5 airfoils identified earlier
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from parsec_fit_improved import ParsecAirfoil as ParsecAirfoilImproved
from parsec_fit import ParsecAirfoil as ParsecAirfoilOriginal
from parsec_to_dat import ParsecAirfoil as ParsecGenerator
from airfoil_validation import check_self_intersection

# Test airfoils - our top performers
TEST_AIRFOILS = ["ag13", "a18sm", "ag26", "ag25", "ag27"]
UIUC_DIR = "airfoils_uiuc"
PARSEC_DIR = "airfoils_parsec"
OUTPUT_DIR = "debug_output/improved_parsec"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

def load_parsec_params(filename):
    """Load PARSEC parameters from file"""
    params = {}
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith("#") or "=" not in line:
                continue
            key, value = line.strip().split("=", 1)
            params[key.strip()] = float(value.strip())
    return params

def compare_airfoil_shapes(airfoil_name):
    """
    Compare original, standard PARSEC, and improved PARSEC for a given airfoil
    """
    print(f"Processing airfoil: {airfoil_name}")
    
    # Paths
    dat_file = os.path.join(UIUC_DIR, f"{airfoil_name}.dat")
    parsec_file = os.path.join(PARSEC_DIR, f"{airfoil_name}.parsec")
    
    # 1. Load original coordinates and check validity
    x_orig, y_orig = read_airfoil_data(dat_file)
    original_valid = check_self_intersection(x_orig, y_orig)
    print(f"  1. Original coordinates valid: {original_valid}")
    
    # 2. Load existing PARSEC parameters
    params = load_parsec_params(parsec_file)
    print(f"  2. Loaded PARSEC parameters: {params}")
    
    # 3. Generate coordinates using original PARSEC implementation
    parsec_orig = ParsecAirfoilOriginal(name=f"{airfoil_name}")
    for key, value in params.items():
        parsec_orig.params[key] = value
    
    # 4. Generate coordinates using improved PARSEC implementation
    parsec_improved = ParsecAirfoilImproved(name=f"{airfoil_name}_improved")
    for key, value in params.items():
        parsec_improved.params[key] = value
    
    # 5. Fit improved PARSEC to original coordinates
    parsec_fit = ParsecAirfoilImproved(name=f"{airfoil_name}_fitted")
    fit_error = parsec_fit.fit_to_data(x_orig, y_orig, enforce_validity=True)
    
    # 6. Generate x-coordinates for evaluation
    x_eval = np.linspace(0, 1, 200)
    
    # 7. Evaluate all three PARSEC implementations
    y_upper_orig, y_lower_orig = parsec_orig.evaluate(x_eval)
    y_upper_improved, y_lower_improved = parsec_improved.evaluate(x_eval)
    y_upper_fitted, y_lower_fitted = parsec_fit.evaluate(x_eval)
    
    # 8. Check geometric validity
    # Original PARSEC coordinates
    x_full_orig = np.concatenate([x_eval, x_eval[::-1]])
    y_full_orig = np.concatenate([y_upper_orig, y_lower_orig[::-1]])
    orig_valid = check_self_intersection(x_full_orig, y_full_orig)
    print(f"  3. Original PARSEC implementation valid: {orig_valid}")
    
    # Improved PARSEC with original parameters
    improved_valid = parsec_improved.check_geometric_validity()
    print(f"  4. Improved PARSEC implementation with original parameters valid: {improved_valid}")
    
    # Newly fitted PARSEC parameters
    fitted_valid = parsec_fit.is_valid
    print(f"  5. Improved PARSEC with newly fitted parameters valid: {fitted_valid}")
    print(f"     Fit error: {fit_error:.6f}")
    print(f"     Fitted parameters: {parsec_fit.params}")
    
    # 9. Plot comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Original airfoil data
    ax.scatter(x_orig, y_orig, s=10, alpha=0.5, label="Original Data", color="black")
    
    # Original PARSEC implementation
    ax.plot(x_eval, y_upper_orig, 'r--', linewidth=1, label=f"Original PARSEC (Valid: {orig_valid})")
    ax.plot(x_eval, y_lower_orig, 'r--', linewidth=1)
    
    # Improved PARSEC with original parameters
    ax.plot(x_eval, y_upper_improved, 'g-', linewidth=1, label=f"Improved PARSEC (Valid: {improved_valid})")
    ax.plot(x_eval, y_lower_improved, 'g-', linewidth=1)
    
    # Fitted PARSEC
    if fitted_valid:
        ax.plot(x_eval, y_upper_fitted, 'b-', linewidth=2, label=f"Fitted Improved PARSEC (Valid: {fitted_valid})")
        ax.plot(x_eval, y_lower_fitted, 'b-', linewidth=2)
    
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_title(f"{airfoil_name} - PARSEC Implementation Comparison")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend()
    
    # Save the figure
    fig.savefig(os.path.join(OUTPUT_DIR, f"{airfoil_name}_comparison.png"), dpi=150)
    plt.close(fig)
    
    return {
        'name': airfoil_name,
        'original_valid': original_valid,
        'parsec_orig_valid': orig_valid,
        'parsec_improved_valid': improved_valid,
        'parsec_fitted_valid': fitted_valid,
        'fit_error': fit_error
    }

def main():
    """Main function"""
    # Test all airfoils
    results = []
    for airfoil in TEST_AIRFOILS:
        result = compare_airfoil_shapes(airfoil)
        results.append(result)
    
    # Summarize results
    print("\nSummary of Results:")
    print("------------------")
    
    valid_count = {
        'original': sum(r['original_valid'] for r in results),
        'parsec_orig': sum(r['parsec_orig_valid'] for r in results),
        'parsec_improved': sum(r['parsec_improved_valid'] for r in results),
        'parsec_fitted': sum(r['parsec_fitted_valid'] for r in results)
    }
    
    print(f"Total airfoils tested: {len(results)}")
    print(f"Original coordinates valid: {valid_count['original']}/{len(results)}")
    print(f"Original PARSEC implementation valid: {valid_count['parsec_orig']}/{len(results)}")
    print(f"Improved PARSEC implementation valid: {valid_count['parsec_improved']}/{len(results)}")
    print(f"Improved PARSEC with fitted parameters valid: {valid_count['parsec_fitted']}/{len(results)}")

if __name__ == "__main__":
    main()
