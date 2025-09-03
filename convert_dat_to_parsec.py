#!/usr/bin/env python3
"""
Convert DAT files to PARSEC parameters

This script takes the top-performing airfoil .dat files and converts them to PARSEC parameters
using the dat_to_parsec_params() function from dat_to_parsec_and_back.py.
"""

import os
import json
from dat_to_parsec_and_back import dat_to_parsec_params

# Top performing airfoils based on L/D ratio
TOP_PERFORMERS = ['ag13', 'a18sm', 'ag26', 'ag25', 'ag27']

# Directory containing airfoil dat files
AIRFOIL_DIR = "airfoils_uiuc"
OUTPUT_DIR = "converted_parsec"

def main():
    """Convert top performing airfoil .dat files to PARSEC parameters"""
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Converting DAT files to PARSEC parameters...\n")
    
    # Process each airfoil
    results = {}
    for airfoil_name in TOP_PERFORMERS:
        print(f"Processing {airfoil_name}...")
        
        # Read airfoil .dat file
        dat_path = os.path.join(AIRFOIL_DIR, f"{airfoil_name}.dat")
        
        if not os.path.exists(dat_path):
            print(f"  Error: {dat_path} not found, skipping")
            continue
        
        try:
            # Convert to PARSEC parameters
            parsec_params = dat_to_parsec_params(dat_path)
            
            # Save to JSON file
            output_path = os.path.join(OUTPUT_DIR, f"{airfoil_name}_converted.json")
            with open(output_path, 'w') as f:
                json.dump(parsec_params, f, indent=2)
            
            print(f"  PARSEC parameters saved to {output_path}")
            
            # Store results
            results[airfoil_name] = parsec_params
            
            # Print parameters
            print("  PARSEC Parameters:")
            for param, value in parsec_params.items():
                print(f"    {param} = {value}")
            print()
            
        except Exception as e:
            print(f"  Error processing {airfoil_name}: {e}")
    
    # Save all results to a combined file
    combined_path = os.path.join(OUTPUT_DIR, "all_converted_parsec.json")
    with open(combined_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"All conversions completed. Combined results saved to {combined_path}")

if __name__ == "__main__":
    main()
