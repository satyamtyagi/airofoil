#!/usr/bin/env python3
"""
Convert All DAT Files to PARSEC Parameters

This script converts all airfoil coordinate files (.dat) from the airfoils_uiuc directory
to PARSEC parameters and saves them as JSON files.
"""

import os
import json
import time
from dat_to_parsec_and_back import dat_to_parsec_params

# Directories
INPUT_DIR = "airfoils_uiuc"
OUTPUT_DIR = "all_converted_parsec"

def main():
    """Convert all DAT files to PARSEC parameters"""
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Converting all DAT files to PARSEC parameters...\n")
    
    # Get all .dat files in the input directory
    dat_files = []
    for file in os.listdir(INPUT_DIR):
        if file.lower().endswith('.dat'):
            dat_files.append(file)
    
    print(f"Found {len(dat_files)} DAT files to process\n")
    
    # Results dictionary to store all parameters
    all_results = {}
    successful_count = 0
    failed_count = 0
    start_time = time.time()
    
    # Process each file
    for i, dat_file in enumerate(dat_files):
        airfoil_name = dat_file.split('.')[0]
        dat_path = os.path.join(INPUT_DIR, dat_file)
        
        print(f"[{i+1}/{len(dat_files)}] Processing {airfoil_name}...")
        
        try:
            # Extract PARSEC parameters
            params = dat_to_parsec_params(dat_path)
            
            # Save to individual JSON file
            json_path = os.path.join(OUTPUT_DIR, f"{airfoil_name}_converted.json")
            with open(json_path, 'w') as f:
                json.dump(params, f, indent=2)
            
            print(f"  PARSEC parameters saved to {json_path}")
            
            # Display parameters
            print("  PARSEC Parameters:")
            for key, value in params.items():
                print(f"    {key} = {value}")
            print()
            
            # Add to results dictionary
            all_results[airfoil_name] = params
            successful_count += 1
            
        except Exception as e:
            print(f"  Error processing {airfoil_name}: {str(e)}\n")
            failed_count += 1
    
    # Save all results to a single JSON file
    combined_path = os.path.join(OUTPUT_DIR, "all_converted_parsec.json")
    with open(combined_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Calculate processing time
    total_time = time.time() - start_time
    
    # Print summary
    print("\nConversion Summary:")
    print(f"Total files processed: {len(dat_files)}")
    print(f"Successfully converted: {successful_count}")
    print(f"Failed to convert: {failed_count}")
    print(f"Processing time: {total_time:.2f} seconds")
    print(f"Combined results saved to {combined_path}")

if __name__ == "__main__":
    main()
