#!/usr/bin/env python3
"""
Debug PARSEC Conversion

This script investigates why 0% parameter variations are failing the self-intersection test.
It examines each step of the conversion process:
1. Original airfoil coordinates (.dat) → Self-intersection check
2. Original coordinates → PARSEC parameters → Self-intersection check  
3. PARSEC parameters → New coordinates → Self-intersection check

This will help identify at which stage the self-intersection is being introduced.
"""

import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from parsec_to_dat import ParsecAirfoil
from airfoil_validation import check_self_intersection
from parsec_fit import ParsecAirfoil as ParsecFitter, read_airfoil_data

# Directories
UIUC_DIR = "airfoils_uiuc"
PARSEC_DIR = "airfoils_parsec"
OUTPUT_DIR = "debug_output"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_airfoil_dat(filename):
    """Load airfoil coordinates from a .dat file"""
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Skip header line
    coords = []
    for line in lines[1:]:
        parts = line.strip().split()
        if len(parts) >= 2:
            try:
                x = float(parts[0])
                y = float(parts[1])
                coords.append((x, y))
            except ValueError:
                continue
    
    x_coords = [p[0] for p in coords]
    y_coords = [p[1] for p in coords]
    return np.array(x_coords), np.array(y_coords)

def debug_single_airfoil(airfoil_name):
    """Debug the conversion process for a single airfoil"""
    print(f"\nDebugging airfoil: {airfoil_name}")
    
    # Step 1: Test original .dat coordinates
    dat_file = os.path.join(UIUC_DIR, f"{airfoil_name}.dat")
    if not os.path.exists(dat_file):
        print(f"  Original .dat file not found: {dat_file}")
        return
    
    x_orig, y_orig = load_airfoil_dat(dat_file)
    is_valid_orig = check_self_intersection(x_orig, y_orig)
    print(f"  1. Original coordinates valid: {is_valid_orig}")
    
    # Step 2: Load existing PARSEC parameters
    parsec_file = os.path.join(PARSEC_DIR, f"{airfoil_name}.parsec")
    if not os.path.exists(parsec_file):
        print(f"  PARSEC parameter file not found: {parsec_file}")
        return
    
    parsec_airfoil = ParsecAirfoil(name=airfoil_name)
    parsec_airfoil.load_from_file(parsec_file)
    print(f"  2. Loaded PARSEC parameters: {parsec_airfoil.params}")
    
    # Step 3: Generate coordinates from PARSEC parameters
    try:
        parsec_airfoil._calculate_coefficients()
        x_parsec, y_parsec = parsec_airfoil.generate_coordinates(num_points=100)
        is_valid_parsec = check_self_intersection(x_parsec, y_parsec)
        print(f"  3. Coordinates from PARSEC valid: {is_valid_parsec}")
    except Exception as e:
        print(f"  Error generating coordinates from PARSEC: {str(e)}")
        is_valid_parsec = False
    
    # Step 4: Fit PARSEC parameters to original coordinates
    try:
        x_coords, y_coords = read_airfoil_data(dat_file)
        
        # Create a new ParsecFitter instance and fit it to the data
        fitter = ParsecFitter(name=f"{airfoil_name}_fitted")
        fit_error = fitter.fit_to_data(x_coords, y_coords)
        fitted_params = fitter.params
        print(f"  4. Fitted PARSEC parameters: {fitted_params}")
        
        # Create airfoil with fitted parameters
        fitted_airfoil = ParsecAirfoil(name=f"{airfoil_name}_fitted")
        for key, value in fitted_params.items():
            if key in fitted_airfoil.params:
                fitted_airfoil.params[key] = value
        
        # Generate coordinates from fitted parameters
        fitted_airfoil._calculate_coefficients()
        x_fitted, y_fitted = fitted_airfoil.generate_coordinates(num_points=100)
        is_valid_fitted = check_self_intersection(x_fitted, y_fitted)
        print(f"  5. Coordinates from fitted PARSEC valid: {is_valid_fitted}")
    except Exception as e:
        print(f"  Error fitting PARSEC parameters: {str(e)}")
    
    # Create visualization if any step fails
    if not is_valid_orig or not is_valid_parsec or not is_valid_fitted:
        plt.figure(figsize=(12, 8))
        
        plt.subplot(3, 1, 1)
        plt.plot(x_orig, y_orig, 'b-', label='Original')
        plt.title(f"Original Coordinates (Valid: {is_valid_orig})")
        plt.axis('equal')
        plt.grid(True)
        
        plt.subplot(3, 1, 2)
        plt.plot(x_parsec, y_parsec, 'r-', label='PARSEC Generated')
        plt.title(f"PARSEC Generated (Valid: {is_valid_parsec})")
        plt.axis('equal')
        plt.grid(True)
        
        plt.subplot(3, 1, 3)
        plt.plot(x_fitted, y_fitted, 'g-', label='Fitted PARSEC')
        plt.title(f"Fitted PARSEC (Valid: {is_valid_fitted})")
        plt.axis('equal')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"{airfoil_name}_debug.png"))
        plt.close()

def main():
    """Main function to debug PARSEC conversion process"""
    # Get top airfoil names
    top_airfoils = ["ag13", "a18sm", "ag26", "ag25", "ag27"]
    
    for airfoil in top_airfoils:
        debug_single_airfoil(airfoil)
    
    print("\nDebug complete. Check debug_output directory for visualizations.")

if __name__ == "__main__":
    main()
