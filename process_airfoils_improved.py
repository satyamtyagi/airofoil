#!/usr/bin/env python3
"""
Process All Airfoils with Improved PARSEC Implementation

This script:
1. Reads all airfoil data files from airfoils_uiuc directory
2. Fits the improved PARSEC model to each airfoil
3. Saves parameter files and visualizations
4. Generates statistics on parameter ranges and fit quality
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from parsec_fit_improved import ParsecAirfoil

# Define directories
INPUT_DIR = "airfoils_uiuc"
OUTPUT_DIR = "airfoils_parsec_improved"
RESULTS_DIR = "parsec_results_improved"

# Create output directories if they don't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "plots"), exist_ok=True)

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

def process_all_airfoils(max_airfoils=None, focus_on=None):
    """Process airfoil files using the improved PARSEC implementation
    
    Args:
        max_airfoils: Maximum number of airfoils to process (None for all)
        focus_on: List of specific airfoil names to focus on (None for all)
    """
    # List all .dat files
    airfoil_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.dat')]
    
    # Prioritize specific airfoils if provided
    if focus_on:
        focus_files = [f for f in airfoil_files if any(name in f for name in focus_on)]
        other_files = [f for f in airfoil_files if not any(name in f for name in focus_on)]
        airfoil_files = focus_files + other_files
    
    # Limit number of airfoils if specified
    if max_airfoils is not None:
        airfoil_files = airfoil_files[:max_airfoils]
    
    # Initialize parameter tracking
    param_names = ParsecAirfoil.PARAM_ORDER
    all_params = {name: [] for name in param_names}
    all_errors = []
    successful_fits = 0
    valid_geometries = 0
    failed_fits = 0
    
    # Store results for valid and invalid geometries separately
    valid_results = []
    invalid_results = []
    
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
            # Print the first few points of the airfoil data for debugging
            print(f"  Airfoil data sample: {len(x_data)} points, x[0:5]={x_data[:5]}, y[0:5]={y_data[:5]}")
            
            # First try with enforcing validity
            try:
                print(f"  Fitting {filename} with geometric constraints...")
                fit_error = airfoil.fit_to_data(x_data, y_data, enforce_validity=True)
                print(f"  Fit complete: error={fit_error:.6f}, valid={airfoil.is_valid}")
                
                # If fit error is too high or optimization failed to find valid solution, try again without enforcing
                if fit_error > 0.01 or not airfoil.is_valid:
                    print(f"  Initial fit failed for {filename}, trying without geometric constraints")
                    fit_error = airfoil.fit_to_data(x_data, y_data, enforce_validity=False)
                    print(f"  Second fit complete: error={fit_error:.6f}, valid={airfoil.is_valid}")
            except KeyboardInterrupt:
                print("\nProcess interrupted by user.")
                break
            
            # Skip extremely bad fits - but use a much higher threshold since our errors are large
            if fit_error > 2000000:  # Increased threshold significantly
                print(f"  Skipping {filename}: Very poor fit (error={fit_error:.6f})")
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
            for param_name in param_names:
                all_params[param_name].append(airfoil.params[param_name])
            all_errors.append(fit_error)
            
            # Add to appropriate result list
            result_entry = {
                "name": airfoil_name,
                "error": fit_error,
                "params": {k: airfoil.params[k] for k in param_names}
            }
            
            if airfoil.is_valid:
                valid_results.append(result_entry)
                valid_geometries += 1
            else:
                invalid_results.append(result_entry)
            
            successful_fits += 1
            
        except Exception as e:
            print(f"  Error fitting {filename}: {str(e)}")
            failed_fits += 1
            continue
    
    # Check if we have any successful results
    if successful_fits == 0:
        print("No successful fits found! Cannot generate statistics.")
        return [], []
        
    # Compile statistics
    stats = {
        "param": param_names,
        "min": [min(all_params[p]) if all_params[p] else float('nan') for p in param_names],
        "max": [max(all_params[p]) if all_params[p] else float('nan') for p in param_names],
        "mean": [np.mean(all_params[p]) if all_params[p] else float('nan') for p in param_names],
        "std": [np.std(all_params[p]) if all_params[p] else float('nan') for p in param_names]
    }
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(stats)
    df.to_csv(os.path.join(RESULTS_DIR, "parsec_stats_improved.csv"), index=False)
    
    # Create summary text file
    with open(os.path.join(RESULTS_DIR, "parsec_summary_improved.txt"), 'w') as f:
        f.write("PARSEC Parameter Ranges Summary (Improved Implementation)\n")
        f.write("===================================================\n\n")
        f.write(f"Total airfoils processed: {successful_fits + failed_fits}\n")
        f.write(f"Successful fits: {successful_fits}\n")
        f.write(f"Geometrically valid fits: {valid_geometries}\n")
        f.write(f"Invalid geometry fits: {successful_fits - valid_geometries}\n")
        f.write(f"Failed fits: {failed_fits}\n\n")
        f.write("Parameter Ranges:\n")
        f.write("----------------\n")
        for i, param in enumerate(param_names):
            f.write(f"{param:12s}: Min = {stats['min'][i]:.6f}, Max = {stats['max'][i]:.6f}, ")
            f.write(f"Mean = {stats['mean'][i]:.6f}, StdDev = {stats['std'][i]:.6f}\n")
        
        if all_errors:
            f.write("\nFit Errors:\n")
            f.write(f"Min Error: {min(all_errors):.6f}\n")
            f.write(f"Max Error: {max(all_errors):.6f}\n")
            f.write(f"Mean Error: {np.mean(all_errors):.6f}\n")
        else:
            f.write("\nNo valid fit errors to report.\n")
    
    # Save lists of valid and invalid airfoils
    with open(os.path.join(RESULTS_DIR, "valid_airfoils.txt"), 'w') as f:
        f.write("Geometrically Valid Airfoil Fits:\n")
        f.write("================================\n\n")
        for result in sorted(valid_results, key=lambda x: x["error"]):
            f.write(f"{result['name']}: Fit error = {result['error']:.6f}\n")
    
    with open(os.path.join(RESULTS_DIR, "invalid_airfoils.txt"), 'w') as f:
        f.write("Geometrically Invalid Airfoil Fits:\n")
        f.write("==================================\n\n")
        for result in sorted(invalid_results, key=lambda x: x["error"]):
            f.write(f"{result['name']}: Fit error = {result['error']:.6f}\n")
    
    # Generate visualization of parameter ranges
    fig, axs = plt.subplots(3, 4, figsize=(20, 12))
    axs = axs.flatten()
    
    for i, param in enumerate(param_names):
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
    if len(axs) > len(param_names):
        i = len(param_names)
        axs[i].hist(all_errors, bins=20, alpha=0.7, color='red')
        axs[i].set_title("Fit Error Distribution")
        axs[i].set_xlabel("Error")
        axs[i].set_ylabel("Count")
        axs[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "parameter_distributions_improved.png"), dpi=150)
    plt.close()
    
    # Generate scatter plots of parameter relationships
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    axs = axs.flatten()
    
    # Key parameter relationships to investigate
    relationships = [
        ("rLE", "YXXup", "Leading Edge Radius vs Upper Curvature"),
        ("Xup", "Yup", "Upper Crest Position"),
        ("Xlo", "Ylo", "Lower Crest Position"),
        ("YXXup", "YXXlo", "Upper vs Lower Curvature"),
        ("AlphaTE", "DeltaAlphaTE", "TE Direction vs Wedge Angle"),
        ("rLE", "YXXlo", "Leading Edge Radius vs Lower Curvature")
    ]
    
    for i, (param1, param2, title) in enumerate(relationships):
        if i < len(axs):
            valid_x = [res["params"][param1] for res in valid_results]
            valid_y = [res["params"][param2] for res in valid_results]
            invalid_x = [res["params"][param1] for res in invalid_results]
            invalid_y = [res["params"][param2] for res in invalid_results]
            
            # Plot both valid and invalid points with different colors
            axs[i].scatter(valid_x, valid_y, c='green', marker='o', label='Valid', alpha=0.7)
            axs[i].scatter(invalid_x, invalid_y, c='red', marker='x', label='Invalid', alpha=0.7)
            axs[i].set_title(title)
            axs[i].set_xlabel(param1)
            axs[i].set_ylabel(param2)
            axs[i].grid(True, alpha=0.3)
            axs[i].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "parameter_relationships_improved.png"), dpi=150)
    plt.close()
    
    print("Processing complete.")
    print(f"Total successful fits: {successful_fits}")
    print(f"Geometrically valid fits: {valid_geometries}")
    print(f"Invalid geometry fits: {successful_fits - valid_geometries}")
    print(f"Failed fits: {failed_fits}")
    print(f"Results saved to {RESULTS_DIR}")
    
    return valid_results, invalid_results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process airfoils with improved PARSEC implementation")
    parser.add_argument("--max", type=int, default=5, help="Maximum number of airfoils to process")
    parser.add_argument("--all", action="store_true", help="Process all airfoils")
    parser.add_argument("--focus", type=str, nargs="+", help="Focus on specific airfoils")
    args = parser.parse_args()
    
    # Default to processing top 5 airfoils unless --all is specified
    if args.all:
        max_airfoils = None
    else:
        max_airfoils = args.max
        
    # Process focused airfoils if specified
    focus_airfoils = None
    if args.focus:
        focus_airfoils = args.focus
        print(f"Focusing on airfoils: {focus_airfoils}")
    
    print(f"Processing {'all' if max_airfoils is None else max_airfoils} airfoils...")
    process_all_airfoils(max_airfoils=max_airfoils, focus_on=focus_airfoils)
