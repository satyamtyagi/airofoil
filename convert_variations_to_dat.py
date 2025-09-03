#!/usr/bin/env python3
"""
Convert Variations to DAT

This script converts PARSEC parameter variations from simple_variations_parsec directory
to airfoil coordinate files in DAT format using the build_dat_from_parsec function.
"""

import os
import json
import numpy as np
from pathlib import Path
from parsec_core import build_dat_from_parsec

# Directory paths
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
PARSEC_DIR = BASE_DIR / "simple_variations_parsec"
OUTPUT_DIR = BASE_DIR / "simple_variations_dat"

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(exist_ok=True)

def save_dat_file(name, xs, ys):
    """Save coordinates to a DAT file"""
    output_file = OUTPUT_DIR / f"{name}.dat"
    
    with open(output_file, 'w') as f:
        f.write(f"{name}\n")
        for x, y in zip(xs, ys):
            f.write(f"{x:.6f}  {y:.6f}\n")
    
    return output_file

def convert_all_variations():
    """Convert all PARSEC parameter variations to DAT files"""
    parsec_files = list(PARSEC_DIR.glob("*.json"))
    print(f"Found {len(parsec_files)} PARSEC parameter files to convert")
    
    success_count = 0
    error_count = 0
    
    for parsec_file in parsec_files:
        base_name = parsec_file.stem
        
        # Read PARSEC parameters
        with open(parsec_file, 'r') as f:
            params = json.load(f)
        
        try:
            # Generate coordinates using parsec_core function
            # build_dat_from_parsec returns xs and ys as separate arrays, not tuples
            xs, ys = build_dat_from_parsec(params)
            
            # Save to DAT file
            dat_file = save_dat_file(base_name, xs, ys)
            success_count += 1
            
            if success_count % 50 == 0:
                print(f"Converted {success_count} files so far...")
        
        except Exception as e:
            print(f"Error converting {base_name}: {str(e)}")
            error_count += 1
    
    print(f"Conversion complete: {success_count} successes, {error_count} errors")
    print(f"DAT files saved to {OUTPUT_DIR}")

def main():
    print("Converting PARSEC parameter variations to DAT files...")
    convert_all_variations()

if __name__ == "__main__":
    main()
