#!/usr/bin/env python3
"""
Generate DAT Files for Top Parameter Sweep Airfoils

This script:
1. Reads the parameter sweep results to find the top-performing airfoils
2. Extracts their PARSEC parameters
3. Generates .dat files for each top performer
4. Creates a visualization of these top performers
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from parsec_to_dat import ParsecAirfoil
# Import geometric validation functions
from airfoil_validation import check_self_intersection, calculate_thickness, check_min_thickness, check_max_thickness

# Define directories and files
SWEEP_RESULTS_FILE = "parameter_sweep_results/parsec_sweep_results_success.csv"
OUTPUT_DIR = "parameter_sweep_top_airfoils"
NUM_TOP_AIRFOILS = 5  # Number of top airfoils to generate

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)


def read_top_performers(n=NUM_TOP_AIRFOILS):
    """Read the top n performers from the parameter sweep results"""
    print(f"Reading top {n} performers from {SWEEP_RESULTS_FILE}...")
    
    if not os.path.exists(SWEEP_RESULTS_FILE):
        print(f"Error: Sweep results file not found: {SWEEP_RESULTS_FILE}")
        return None
    
    # Read the CSV file
    df = pd.read_csv(SWEEP_RESULTS_FILE)
    
    # Sort by lift-to-drag ratio (cl_cd) in descending order
    df_sorted = df.sort_values('cl_cd', ascending=False)
    
    # Get top n performers
    top_performers = df_sorted.head(n).copy()
    
    # Add a rank column for easier reference
    top_performers['rank'] = range(1, n+1)
    
    return top_performers


def validate_airfoil(x_coords, y_coords):
    """Validate airfoil geometry using functions from airfoil_validation.py"""
    # Check for self-intersection
    if not check_self_intersection(x_coords, y_coords):
        return False, "Self-intersection detected"
        
    # Check minimum thickness
    if not check_min_thickness(x_coords, y_coords, min_thickness=0.01):
        return False, "Thickness below minimum (0.01)"
        
    # Check maximum thickness
    if not check_max_thickness(x_coords, y_coords, max_thickness=0.25):
        return False, "Thickness above maximum (0.25)"
    
    # All validations passed
    return True, ""


def generate_dat_files(top_performers):
    """Generate .dat files for each top performer"""
    if top_performers is None or top_performers.empty:
        print("No top performers data available.")
        return
    
    print(f"Generating .dat files for {len(top_performers)} top airfoils...")
    
    # Create a figure for visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # PARSEC parameter names
    param_names = [
        "rLE", "Xup", "Yup", "YXXup", "Xlo", "Ylo", "YXXlo", "Xte", "Yte", "Yte'", "Δyte''"
    ]
    
    # Track validation results
    valid_airfoils = []
    invalid_airfoils = []
    
    # Process each top performer
    for i, (_, airfoil) in enumerate(top_performers.iterrows()):
        # Create an airfoil name based on rank
        rank = airfoil['rank']
        airfoil_name = f"sweep_top{rank}"
        
        # Extract PARSEC parameters from the dataframe
        params = {}
        for param in param_names:
            if param in airfoil:
                params[param] = airfoil[param]
        
        # Create ParsecAirfoil instance
        parsec_airfoil = ParsecAirfoil(name=airfoil_name)
        
        # Update with parameters
        for param, value in params.items():
            parsec_airfoil.params[param] = value
        
        # Calculate coefficients
        try:
            parsec_airfoil._calculate_coefficients()
            
            # Generate coordinates for validation
            x, y = parsec_airfoil.generate_coordinates(200)
            
            # Validate the airfoil geometry
            valid, reason = validate_airfoil(x, y)
            if not valid:
                print(f"  Skipping {airfoil_name} (Rank #{rank}): {reason}")
                invalid_airfoils.append((rank, airfoil['cl_cd'], reason))
                continue
            
            # Generate output file path
            output_file = os.path.join(OUTPUT_DIR, f"{airfoil_name}.dat")
            
            # Save to DAT file
            print(f"  Saving {airfoil_name}.dat (Rank #{rank}, L/D = {airfoil['cl_cd']:.2f})")
            parsec_airfoil.save_to_dat(output_file, num_points=200)
            valid_airfoils.append((rank, airfoil['cl_cd']))
            
            # Plot on the figure if we have space
            if len(valid_airfoils) <= len(axes):
                plot_idx = len(valid_airfoils) - 1
                parsec_airfoil.plot(axes[plot_idx])
                axes[plot_idx].set_title(f"Rank #{rank}: L/D = {airfoil['cl_cd']:.2f}\nCL = {airfoil['cl']:.4f}, CD = {airfoil['cd']:.6f}")
        except Exception as e:
            print(f"  Error processing {airfoil_name} (Rank #{rank}): {str(e)}")
            invalid_airfoils.append((rank, airfoil['cl_cd'], str(e)))

    
    # Add a summary plot showing all valid airfoils together
    if len(valid_airfoils) < len(axes) and len(valid_airfoils) > 0:
        ax_summary = axes[len(valid_airfoils)]
        
        # Plot all valid airfoils on the same axis
        for rank, cl_cd in valid_airfoils:
            airfoil_name = f"sweep_top{rank}"
            
            # Load the saved airfoil from the DAT file
            airfoil_file = os.path.join(OUTPUT_DIR, f"{airfoil_name}.dat")
            if not os.path.exists(airfoil_file):
                continue
                
            # Read the airfoil coordinates
            x_airfoil = []
            y_airfoil = []
            with open(airfoil_file, 'r') as f:
                # Skip the header line
                next(f)
                for line in f:
                    if len(line.strip()) > 0:
                        try:
                            x, y = line.strip().split()
                            x_airfoil.append(float(x))
                            y_airfoil.append(float(y))
                        except ValueError:
                            pass
            
            # Plot with different colors
            ax_summary.plot(x_airfoil, y_airfoil, label=f"Rank #{rank}: L/D = {cl_cd:.2f}")
        
        ax_summary.set_title("Valid Top Performers Comparison")
        ax_summary.legend(loc='lower right', fontsize=8)
        ax_summary.set_aspect('equal')
        ax_summary.grid(True, alpha=0.3)
        
    # Print validation summary
    print("\nValidation Summary:")
    print(f"  Valid airfoils: {len(valid_airfoils)} / {len(top_performers)}")
    print(f"  Invalid airfoils: {len(invalid_airfoils)} / {len(top_performers)}")
    
    if invalid_airfoils:
        print("\nInvalid Airfoils:")
        for rank, cl_cd, reason in invalid_airfoils:
            print(f"  Rank #{rank} (L/D = {cl_cd:.2f}): {reason}")
        
        # If we have axes left, show validation failures
        if len(valid_airfoils) + 1 < len(axes) and len(invalid_airfoils) > 0:
            ax_invalid = axes[len(valid_airfoils) + 1]
            ax_invalid.text(0.5, 0.5, f"{len(invalid_airfoils)} Invalid Airfoils:\n" + 
                         "\n".join([f"Rank #{r}: {reason}" for r, _, reason in invalid_airfoils[:5]]), 
                         ha='center', va='center', fontsize=10)
            ax_invalid.set_title("Validation Failures")
            ax_invalid.axis('off')
    
    # Save the visualization
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "top_sweep_airfoils.png"), dpi=150)
    print(f"  Visualization saved to {os.path.join(OUTPUT_DIR, 'top_sweep_airfoils.png')}")
    plt.close(fig)


def print_parameter_table(top_performers):
    """Print a table of PARSEC parameters for the top performers"""
    if top_performers is None or top_performers.empty:
        return
    
    print("\nPARSEC Parameters for Top Performers:")
    print("====================================")
    
    # PARSEC parameter names to include in the table
    param_names = [
        "rLE", "Xup", "Yup", "YXXup", "Xlo", "Ylo", "YXXlo", "Xte", "Yte", "Yte'", "Δyte''"
    ]
    
    # Print header
    header = "Rank  L/D     "
    for param in param_names:
        header += f"{param:<8} "
    print(header)
    print("-" * len(header))
    
    # Print each airfoil's parameters
    for _, airfoil in top_performers.iterrows():
        rank = airfoil['rank']
        cl_cd = airfoil['cl_cd']
        
        line = f"#{rank:<4} {cl_cd:<7.2f} "
        for param in param_names:
            if param in airfoil:
                line += f"{airfoil[param]:<8.4f} "
            else:
                line += f"{'N/A':<8} "
        print(line)


def create_3d_visualization():
    """Create a 3D visualization of the top airfoils using the new .dat files"""
    # This could be extended to create a 3D visualization similar to visualize_airfoils_3d_simple.py
    # For now, we'll just print instructions for running the existing script
    print("\nTo create a 3D visualization of these airfoils:")
    print(f"1. Copy the .dat files from {OUTPUT_DIR} to airfoils_uiuc directory")
    print("2. Modify visualize_airfoils_3d_simple.py to include these airfoils in TOP_PERFORMERS")
    print("3. Run the visualization script")


def main():
    """Main function"""
    print("\nGenerating DAT Files for Top Parameter Sweep Airfoils\n")
    
    # Read top performers from parameter sweep results
    top_performers = read_top_performers(NUM_TOP_AIRFOILS)
    if top_performers is None:
        return
    
    # Print information about top performers
    print(f"\nFound {len(top_performers)} top performers with the following performance metrics:")
    for i, (_, row) in enumerate(top_performers.iterrows()):
        print(f"  Rank #{i+1}: L/D = {row['cl_cd']:.2f}, CL = {row['cl']:.4f}, CD = {row['cd']:.6f}")
    
    # Print detailed parameter table
    print_parameter_table(top_performers)
    
    # Generate .dat files
    generate_dat_files(top_performers)
    
    # Suggest 3D visualization
    create_3d_visualization()
    
    print("\nDone!")


if __name__ == "__main__":
    main()
