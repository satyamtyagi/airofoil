#!/usr/bin/env python3
"""
Convert PARSEC parameters back to DAT files

This script takes the PARSEC parameters we generated from the top-performing airfoils
and converts them back to airfoil coordinates (.dat files) using build_dat_from_parsec().
"""

import os
import json
import numpy as np
from parsec_core import build_dat_from_parsec

# Top performing airfoils based on L/D ratio
TOP_PERFORMERS = ['ag13', 'a18sm', 'ag26', 'ag25', 'ag27']

# Directories
INPUT_DIR = "converted_parsec"
OUTPUT_DIR = "regenerated_dat"

def main():
    """Convert PARSEC parameters back to airfoil .dat files"""
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Converting PARSEC parameters back to DAT files...\n")
    
    # Process each airfoil
    for airfoil_name in TOP_PERFORMERS:
        print(f"Processing {airfoil_name}...")
        
        # Read PARSEC parameters
        json_path = os.path.join(INPUT_DIR, f"{airfoil_name}_converted.json")
        
        if not os.path.exists(json_path):
            print(f"  Error: {json_path} not found, skipping")
            continue
        
        try:
            # Load PARSEC parameters
            with open(json_path, 'r') as f:
                params = json.load(f)
            
            # Generate airfoil coordinates using build_dat_from_parsec
            xs, ys, extras = build_dat_from_parsec(params, n_points=401)
            
            # Save to DAT file
            dat_path = os.path.join(OUTPUT_DIR, f"{airfoil_name}_regenerated.dat")
            with open(dat_path, 'w') as f:
                f.write(f"{airfoil_name} (regenerated via PARSEC)\n")
                for x, y in zip(xs, ys):
                    f.write(f"{x:.6f} {y:.6f}\n")
            
            print(f"  Regenerated airfoil coordinates saved to {dat_path}")
            print(f"  Airfoil has {len(xs)} points")
            
            # Save coefficients
            coef_path = os.path.join(OUTPUT_DIR, f"{airfoil_name}_coeffs.json")
            with open(coef_path, 'w') as f:
                json.dump(extras, f, indent=2)
            
            print(f"  Coefficients saved to {coef_path}\n")
            
        except Exception as e:
            print(f"  Error processing {airfoil_name}: {e}\n")
    
    print("All conversions completed.")

if __name__ == "__main__":
    main()
