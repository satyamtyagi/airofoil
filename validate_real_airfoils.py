"""
Validate the top real airfoils using the geometric validation checks.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import geometric validation functions
from airfoil_validation import check_self_intersection, calculate_thickness, check_min_thickness, check_max_thickness

# Define directories
RESULTS_DIR = "results"
AIRFOIL_DIR = "airfoils_uiuc"
PARSEC_DIR = "airfoils_parsec"
OUTPUT_DIR = "real_airfoils_validation"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def validate_airfoil(x_coords, y_coords):
    """Validate airfoil geometry using functions from airfoil_validation.py"""
    # Check for self-intersection
    if not check_self_intersection(x_coords, y_coords):
        return False, "Self-intersection detected"
        
    # Check minimum thickness with more lenient threshold
    if not check_min_thickness(x_coords, y_coords, min_thickness=0.005):
        return False, "Thickness below minimum (0.005)"
        
    # Check maximum thickness
    if not check_max_thickness(x_coords, y_coords, max_thickness=0.25):
        return False, "Thickness above maximum (0.25)"
    
    # Calculate actual thickness for reporting
    thickness, pos = calculate_thickness(x_coords, y_coords)
    
    # All validations passed
    return True, f"Max thickness: {thickness:.4f} at x={pos:.2f}"

def read_airfoil_coordinates(file_path):
    """Read airfoil coordinates from a dat file"""
    x_coords = []
    y_coords = []
    
    try:
        with open(file_path, 'r') as f:
            # Skip header line if present
            first_line = f.readline().strip()
            if not any(c.isdigit() for c in first_line):
                # First line was a header, continue reading
                pass
            else:
                # First line had data, process it
                try:
                    x, y = first_line.split()
                    x_coords.append(float(x))
                    y_coords.append(float(y))
                except ValueError:
                    pass
                    
            # Read the rest of the file
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    try:
                        x, y = line.split()
                        x_coords.append(float(x))
                        y_coords.append(float(y))
                    except ValueError:
                        pass
    except Exception as e:
        print(f"Error reading {file_path}: {str(e)}")
        return [], []
        
    return np.array(x_coords), np.array(y_coords)

def main():
    """Validate top real airfoils"""
    print("\nValidating Top Real Airfoils\n")
    
    # Load airfoil performance data
    performance_file = os.path.join(RESULTS_DIR, "airfoil_best_performance.csv")
    if not os.path.exists(performance_file):
        print(f"Error: Could not find {performance_file}")
        return
        
    df = pd.read_csv(performance_file)
    
    # Sort by lift-to-drag ratio (best_ld)
    df = df.sort_values(by="best_ld", ascending=False)
    
    # Select top 10 airfoils
    top_airfoils = df.head(10)
    
    print(f"Top 10 airfoils by L/D ratio:")
    for idx, row in top_airfoils.iterrows():
        print(f"  {row['airfoil']}: L/D = {row['best_ld']:.2f} at α={row['best_ld_alpha']}°")
    print("")
    
    # Set up figure for visualization
    fig, axes = plt.subplots(2, 5, figsize=(15, 8))
    axes = axes.flatten()
    
    # Track validation results
    valid_airfoils = []
    invalid_airfoils = []
    
    # Validate each airfoil
    print("Validating airfoil geometries:")
    for idx, (_, airfoil) in enumerate(top_airfoils.iterrows()):
        name = airfoil['airfoil']
        
        # Construct path to airfoil dat file
        dat_file = os.path.join(AIRFOIL_DIR, f"{name}.dat")
        if not os.path.exists(dat_file):
            print(f"  Skipping {name}: DAT file not found")
            invalid_airfoils.append((name, airfoil['best_ld'], "DAT file not found"))
            continue
            
        # Read coordinates
        x, y = read_airfoil_coordinates(dat_file)
        
        if len(x) < 10 or len(y) < 10:
            print(f"  Skipping {name}: Insufficient coordinates")
            invalid_airfoils.append((name, airfoil['best_ld'], "Insufficient coordinates"))
            continue
            
        # Calculate actual thickness for reporting
        thickness_values = calculate_thickness(x, y)
        max_thickness = max(thickness_values) if isinstance(thickness_values, tuple) else thickness_values
        print(f"  {name}: Measured max thickness = {max_thickness:.6f}")
        
        # Validate the airfoil
        valid, message = validate_airfoil(x, y)
        
        if valid:
            print(f"  ✓ {name}: Valid airfoil ({message})")
            valid_airfoils.append((name, airfoil['best_ld'], message))
            
            # Plot the airfoil
            axes[idx].plot(x, y, 'b-')
            axes[idx].set_title(f"{name}\nL/D = {airfoil['best_ld']:.2f}")
            axes[idx].set_aspect('equal')
            axes[idx].grid(True, alpha=0.3)
        else:
            print(f"  ✗ {name}: Invalid - {message}")
            invalid_airfoils.append((name, airfoil['best_ld'], message))
            
            # Plot the invalid airfoil in red
            axes[idx].plot(x, y, 'r-')
            axes[idx].set_title(f"{name}\nL/D = {airfoil['best_ld']:.2f}\nInvalid: {message}")
            axes[idx].set_aspect('equal')
            axes[idx].grid(True, alpha=0.3)
    
    # Print summary
    print("\nValidation Summary:")
    print(f"  Valid airfoils: {len(valid_airfoils)} / {len(top_airfoils)}")
    print(f"  Invalid airfoils: {len(invalid_airfoils)} / {len(top_airfoils)}")
    
    # Print valid airfoils
    if valid_airfoils:
        print("\nValid Airfoils:")
        for name, l_d, message in valid_airfoils:
            print(f"  {name} (L/D = {l_d:.2f}): {message}")
    
    # Print invalid airfoils
    if invalid_airfoils:
        print("\nInvalid Airfoils:")
        for name, l_d, message in invalid_airfoils:
            print(f"  {name} (L/D = {l_d:.2f}): {message}")
    
    # Save the visualization
    plt.tight_layout()
    output_file = os.path.join(OUTPUT_DIR, "real_airfoil_validation.png")
    plt.savefig(output_file)
    print(f"\nVisualization saved to {output_file}")
    
    # Save results to CSV
    results_df = pd.DataFrame([
        {"airfoil": name, "best_ld": l_d, "valid": True, "message": msg}
        for name, l_d, msg in valid_airfoils
    ] + [
        {"airfoil": name, "best_ld": l_d, "valid": False, "message": msg}
        for name, l_d, msg in invalid_airfoils
    ])
    
    results_csv = os.path.join(OUTPUT_DIR, "validation_results.csv")
    results_df.to_csv(results_csv, index=False)
    print(f"Results saved to {results_csv}")

if __name__ == "__main__":
    main()
