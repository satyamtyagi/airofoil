#!/usr/bin/env python3
"""
Convert Simple PARSEC Variations to DAT Files

This script takes the PARSEC parameter variations in the simple_variations_parsec directory
and converts them to airfoil coordinates (.dat files) using build_dat_from_parsec().
"""

import os
import json
import numpy as np
from pathlib import Path
from parsec_core import build_dat_from_parsec

# Directories
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = BASE_DIR / "simple_variations_parsec"
OUTPUT_DIR = BASE_DIR / "simple_variations_dat"

def main():
    """Convert PARSEC parameter variations to airfoil .dat files"""
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Converting PARSEC parameter variations to DAT files...\n")
    
    # Get all JSON files from the input directory
    json_files = list(INPUT_DIR.glob("*.json"))
    print(f"Found {len(json_files)} variation files to process")
    
    success_count = 0
    error_count = 0
    
    # Process each variation file
    for json_path in json_files:
        variation_name = json_path.stem
        
        try:
            # Load PARSEC parameters
            with open(json_path, 'r') as f:
                params = json.load(f)
            
            # Generate airfoil coordinates using build_dat_from_parsec
            # Note: build_dat_from_parsec returns 3 values: xs, ys, extras
            xs, ys, extras = build_dat_from_parsec(params, n_points=201)
            
            # Save to DAT file
            dat_path = OUTPUT_DIR / f"{variation_name}.dat"
            with open(dat_path, 'w') as f:
                f.write(f"{variation_name}\n")
                for x, y in zip(xs, ys):
                    f.write(f"{x:.6f}  {y:.6f}\n")
            
            success_count += 1
            if success_count % 50 == 0:
                print(f"Processed {success_count} files so far...")
        
        except Exception as e:
            print(f"Error processing {variation_name}: {str(e)}")
            error_count += 1
    
    print(f"\nConversion complete: {success_count} successes, {error_count} errors")
    print(f"DAT files saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
