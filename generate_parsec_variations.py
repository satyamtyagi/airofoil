#!/usr/bin/env python3
"""
PARSEC Parameter Variation Generator - Top Performers

This script:
1. Identifies the top 10 real airfoils based on lift-to-drag ratio
2. Extracts their PARSEC parameters from the airfoils_parsec directory
3. For each airfoil, creates systematic variations by:
   - Varying one parameter at a time through its min-max range while keeping others constant
   - Using 5 steps per parameter to create 50 variations per base airfoil
4. Validates each variation geometrically by converting PARSEC to coordinates
5. Runs the surrogate model only on valid shapes
6. Outputs performance metrics and visualizations of the best variations

This approach starts with known high-performing shapes and explores targeted variations
to identify promising design improvements while ensuring physical validity.

Usage:
  python generate_parsec_variations.py [options]

Options:
  -s, --steps N    Number of steps for each parameter variation (default: 5)
  -t, --top N      Number of top airfoils to use as base shapes (default: 10)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import re
import argparse
import torch
from scipy.interpolate import interp1d
from matplotlib.backends.backend_pdf import PdfPages
import h5py

# Import PARSEC functionality
from parsec_to_dat import ParsecAirfoil

# Import validation and surrogate model functionality
from parsec_parameter_sweep import validate_parsec_parameters, normalize_coordinates_for_surrogate, load_surrogate_model
# Import geometric validation functions
from airfoil_validation import check_self_intersection, calculate_thickness, check_min_thickness, check_max_thickness

# Define directories
PARSEC_DIR = "airfoils_parsec"
PERFORMANCE_FILE = "results/airfoil_best_performance.csv"
OUTPUT_DIR_PARSEC = "airfoils_variations_parsec"
OUTPUT_DIR_DAT = "airfoils_variations_dat"
RESULTS_DIR = "variation_results"
MODELS_DIR = "models"

# Create output directories if they don't exist
os.makedirs(OUTPUT_DIR_PARSEC, exist_ok=True)
os.makedirs(OUTPUT_DIR_DAT, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Parameter names and descriptions for better visualization
PARAM_DESCRIPTIONS = {
    "rLE": "Leading Edge Radius",
    "Xup": "Upper Crest X Position",
    "Yup": "Upper Crest Y Position",
    "YXXup": "Upper Crest Curvature",
    "Xlo": "Lower Crest X Position",
    "Ylo": "Lower Crest Y Position",
    "YXXlo": "Lower Crest Curvature",
    "Xte": "Trailing Edge X Position",
    "Yte": "Trailing Edge Y Position",
    "Yte'": "Trailing Edge Direction",
    "Δyte''": "Trailing Edge Wedge Angle"
}

# Parameter bounds for variations - from manual analysis of real airfoils
PARAM_BOUNDS = {
    "rLE": (0.005, 0.15),      # Leading Edge Radius
    "Xup": (0.15, 0.5),       # Upper Crest X Position
    "Yup": (0.01, 0.15),      # Upper Crest Y Position
    "YXXup": (-1.5, 0.5),     # Upper Crest Curvature
    "Xlo": (0.1, 0.5),        # Lower Crest X Position
    "Ylo": (-0.1, -0.005),    # Lower Crest Y Position
    "YXXlo": (0.2, 3.0),      # Lower Crest Curvature
    "Xte": (0.9, 1.0),        # Trailing Edge X Position
    "Yte": (-0.02, 0.02),     # Trailing Edge Y Position
    "Yte'": (-0.2, 0.2),      # Trailing Edge Direction
    "Δyte''": (0.05, 0.5)      # Trailing Edge Wedge Angle
}


def get_top_airfoils(n=10):
    """Get the top N airfoils based on L/D ratio that have PARSEC parameter files available"""
    try:
        # Load airfoil performance data
        performance_file = 'results/airfoil_best_performance.csv'
        airfoil_data = pd.read_csv(performance_file)
        
        # Get list of available PARSEC files
        available_parsec_files = set([f.split('.')[0] for f in os.listdir(PARSEC_DIR) if f.endswith('.parsec')])
        
        # Filter airfoil data for those with available PARSEC files
        airfoil_data['has_parsec'] = airfoil_data['airfoil'].apply(lambda x: x in available_parsec_files)
        filtered_data = airfoil_data[airfoil_data['has_parsec']]
        
        # Sort by best L/D ratio and get top N
        top_airfoils = filtered_data.sort_values(by='best_ld', ascending=False).head(n)
        
        print(f"Top {n} airfoils with PARSEC parameters available:")
        for i, row in top_airfoils.iterrows():
            print(f"{i+1}. {row['airfoil']}: L/D = {row['best_ld']:.2f} at α={row['best_ld_alpha']}°")
            
        return top_airfoils['airfoil'].tolist()
    
    except Exception as e:
        print(f"Error getting top airfoils: {e}")
        return []


def load_parsec_parameters(airfoil_name):
    """Load PARSEC parameters from a file"""
    file_path = os.path.join(PARSEC_DIR, f"{airfoil_name}.parsec")
    
    if not os.path.exists(file_path):
        print(f"Error: PARSEC file not found: {file_path}")
        return None
    
    parameters = {}
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            
            # Parse parameter name and value
            parts = line.split('=')
            if len(parts) == 2:
                param_name = parts[0].strip()
                param_value = float(parts[1].strip())
                parameters[param_name] = param_value
    
    return parameters


def generate_percentage_variations(base_params, percentages=[0.0, 0.05, 0.10, 0.25]):
    """Generate parameter variations based on percentage changes"""
    variations = []
    base_params_copy = base_params.copy()
    
    # Extract airfoil name and remove from parameters to avoid multiplication error
    airfoil_name = base_params_copy.pop('airfoil', 'unknown')
    
    # For each parameter
    for param in base_params_copy.keys():
            
        # Get the base value
        base_value = base_params_copy[param]
        
        # For each percentage change
        for percentage in percentages:
            # For 0% variation, include for each parameter to get the full 770 count
            if percentage == 0.0:
                # Create a base variant (no change to parameters)
                variation = base_params_copy.copy()
                
                # Add airfoil name back for reference
                params_with_airfoil = variation.copy()
                params_with_airfoil['airfoil'] = airfoil_name
                
                # Create a dictionary to store variation details
                variation_details = {
                    'params': params_with_airfoil,
                    'name': f"{airfoil_name}_{param}_0pct",
                    'param': param,
                    'percent': 0.0,
                    'value': base_value
                }
                
                variations.append(variation_details)
                continue
                
            # Create variations with positive and negative percentage changes
            for sign in [1, -1]:
                # Calculate the new value with percentage change
                new_value = base_value * (1 + sign * percentage)
                
                # Create a new set of parameters with this parameter varied
                variation = base_params_copy.copy()
                variation[param] = new_value
                
                # Add airfoil name back for reference
                params_with_airfoil = variation.copy()
                params_with_airfoil['airfoil'] = airfoil_name
                
                # Create a dictionary to store variation details
                variation_details = {
                    'params': params_with_airfoil,
                    'name': f"{airfoil_name}_{param}_{sign * percentage * 100}pct",
                    'param': param,
                    'percent': sign * percentage,
                    'value': new_value
                }
                
                # Add to variations list
                variations.append(variation_details)
    
    return variations


def generate_variations(base_airfoil, base_params, steps=5):
    """Generate parameter variations for the given base airfoil"""
    variations = []
    
    # Add the base airfoil first
    variations.append((f"{base_airfoil}_base", base_params.copy()))
    
    # For each parameter, create steps variations from min to max
    for param, (min_val, max_val) in PARAM_BOUNDS.items():
        if param not in base_params:
            print(f"Warning: Parameter {param} not found in base airfoil {base_airfoil}")
            continue
        
        # Generate evenly spaced values in the parameter range
        param_values = np.linspace(min_val, max_val, steps)
        
        for val in param_values:
            # Skip if the value is very close to the original parameter value
            if abs(val - base_params[param]) < 1e-6:
                continue
                
            # Create a new configuration by copying the base parameters
            new_params = base_params.copy()
            new_params[param] = val
            
            # Create a name for this variation
            # Format: airfoil_param_value
            name = f"{base_airfoil}_{param}_{val:.4f}"
            
            # Add to variations list
            variations.append((name, new_params))
    
    return variations


def create_parsec_files(variations):
    """Create PARSEC parameter files for each variation"""
    print(f"Generating PARSEC parameter files...")
    
    for var in tqdm(variations):
        # Create file path
        file_path = os.path.join(OUTPUT_DIR_PARSEC, f"{var['base_airfoil']}_{var['varied_param']}_{var['percentage']:.2f}.parsec")
        
        # Write parameters to file
        with open(file_path, 'w') as f:
            f.write(f"# PARSEC parameters for {var['base_airfoil']}_{var['varied_param']}_{var['percentage']:.2f}\n")
            for param, value in var['params'].items():
                f.write(f"{param} = {value:.6f}\n")
    
    return [f"{var['base_airfoil']}_{var['varied_param']}_{var['percentage']:.2f}" for var in variations]


def validate_airfoil(params):
    """Validate airfoil parameters and geometry"""
    # Generate airfoil coordinates from PARSEC parameters
    airfoil = ParsecAirfoil()
    for key, value in params.items():
        if key in airfoil.params and key != 'airfoil':
            airfoil.params[key] = value
    
    # Calculate coefficients and generate coordinates
    try:
        airfoil._calculate_coefficients()
        x_coords, y_coords = airfoil.generate_coordinates(num_points=100)
        
        # Check for self-intersection
        if not check_self_intersection(x_coords, y_coords):
            return False, "Self-intersection detected"
            
        # Skip all other validations as requested
        return True, "Passed self-intersection check"
    except Exception as e:
        return False, f"Failed to generate airfoil coordinates: {str(e)}"


def convert_to_dat_files(airfoil_names):
    """Convert PARSEC files to DAT airfoil files"""
    print(f"Converting PARSEC files to airfoil DAT files...")
    
    success_count = 0
    failure_count = 0
    
    for name in tqdm(airfoil_names):
        parsec_file = os.path.join(OUTPUT_DIR_PARSEC, f"{name}.parsec")
        dat_file = os.path.join(OUTPUT_DIR_DAT, f"{name}.dat")
        
        # Create airfoil from PARSEC parameters
        airfoil = ParsecAirfoil(name=name)
        
        try:
            # Load parameters and generate airfoil
            if not airfoil.load_from_file(parsec_file):
                print(f"  Failed to load parameters for {name}")
                failure_count += 1
                continue
            
            # Save to DAT file
            if not airfoil.save_to_dat(dat_file):
                print(f"  Failed to save coordinates for {name}")
                failure_count += 1
                continue
            
            success_count += 1
            
        except Exception as e:
            print(f"  Error processing {name}: {str(e)}")
            failure_count += 1
    
    print(f"Successfully processed {success_count} out of {len(airfoil_names)} variations")
    print(f"Failed: {failure_count} variations")
    
    return success_count


def normalize_coordinates_for_surrogate(x_coords, y_coords, num_points=200):
    """Normalize airfoil coordinates to exactly num_points for surrogate model
    The surrogate model ONLY uses y-coordinates as input
    
    Note: The model expects exactly 200 y-coordinates.
    """
    try:
        # Sort by x-coordinate
        idx = np.argsort(x_coords)
        x = x_coords[idx]
        y = y_coords[idx]
        
        # Parametrize by arc length
        t = np.zeros(len(x))
        for i in range(1, len(x)):
            t[i] = t[i-1] + np.sqrt((x[i] - x[i-1])**2 + (y[i] - y[i-1])**2)
        
        if t[-1] == 0:  # Avoid division by zero
            return None
            
        t = t / t[-1]  # Normalize to [0, 1]
        
        # Interpolate to get evenly spaced points
        fx = interp1d(t, x, kind='linear')
        fy = interp1d(t, y, kind='linear')
        
        # Generate exactly 200 points (to match the model's expected input)
        t_new = np.linspace(0, 1, num_points)
        x_new = fx(t_new)
        y_new = fy(t_new)
        
        # The surrogate model only uses y-coordinates as input (200 points)
        return y_new
    
    except Exception as e:
        print(f"Error normalizing coordinates: {str(e)}")
        return None


def run_surrogate_model(variation_names, variation_parameters):
    """Run the surrogate model on valid airfoil shapes"""
    print("Running surrogate model on valid airfoil shapes...")
    
    # Load surrogate model
    surrogate = load_surrogate_model()
    if surrogate is None:
        print("Error: Failed to load surrogate model")
        return {}
    
    # Track results
    results = {}
    valid_count = 0
    failure_count = 0
    
    # Fixed angle of attack for analysis (matching parsec_parameter_sweep.py)
    alpha = 5.0  # 5 degrees
    
    # Process each variation
    for name, params in tqdm(zip(variation_names, variation_parameters), total=len(variation_names)):
        # Validate the airfoil parameters and shape
        valid, message = validate_airfoil(params)
        
        if not valid:
            # Skip invalid airfoils
            print(f"Skipping {name}: {message}")
            failure_count += 1
            continue
        
        # Generate coordinates
        airfoil = ParsecAirfoil()
        for param, value in params.items():
            airfoil.params[param] = value
        
        try:
            airfoil._calculate_coefficients()
            x, y = airfoil.generate_coordinates(200)  # Generate coordinates
            
            # Normalize coordinates for the surrogate model - use only y coordinates
            normalized = normalize_coordinates_for_surrogate(x, y)
            
            if normalized is None:
                print(f"Skipping {name}: Failed to normalize coordinates")
                failure_count += 1
                continue
            
            # Add angle of attack to the normalized coordinates (matching parsec_parameter_sweep.py)
            input_vector = np.append(normalized, alpha)
            
            # Convert to tensor
            input_tensor = torch.tensor(input_vector, dtype=torch.float32).unsqueeze(0)
            
            # Run surrogate model to get predictions
            with torch.no_grad():
                predictions = surrogate.forward(input_tensor).squeeze().numpy()
            
            # Store results (cl, cd, cm) at AOA=5°
            cl = predictions[0]
            cd = predictions[1]
            cm = predictions[2]
            
            # Apply realistic drag constraint (minimum Cd = 0.003) as in parsec_parameter_sweep.py
            if cd < 0.003:
                cd = 0.003
            
            # Calculate lift-to-drag ratio
            ld_ratio = cl / cd if cd > 0.0001 else 0
            
            # Store results
            results[name] = {
                'cl': cl,
                'cd': cd,
                'cm': cm,
                'ld_ratio': ld_ratio
            }
            
            valid_count += 1
            
        except Exception as e:
            print(f"Error processing {name}: {str(e)}")
            failure_count += 1
    
    print(f"Successfully processed {valid_count} out of {len(variation_names)} variations")
    print(f"Failed variations: {failure_count}")
    
    # Save results to HDF5 file
    with h5py.File(os.path.join(RESULTS_DIR, "variation_results.h5"), 'w') as f:
        for name, metrics in results.items():
            group = f.create_group(name)
            for metric, value in metrics.items():
                group.attrs[metric] = value
    
    print(f"Results saved to {os.path.join(RESULTS_DIR, 'variation_results.h5')}")
    
    return results


def visualize_results(results, top_airfoils):
    """Create visualizations of the variation results"""
    print("Generating result visualizations...")
    
    # Create visualizations directory
    vis_dir = os.path.join(RESULTS_DIR, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Extract data for plotting
    names = list(results.keys())
    ld_ratios = [results[name]['ld_ratio'] for name in names]
    cls = [results[name]['cl'] for name in names]
    cds = [results[name]['cd'] for name in names]
    
    # Sort by lift-to-drag ratio
    sorted_indices = np.argsort(ld_ratios)[::-1]  # Descending order
    
    # Get the top 20 performers
    top_indices = sorted_indices[:20]
    top_names = [names[i] for i in top_indices]
    top_lds = [ld_ratios[i] for i in top_indices]
    
    # Print top 10 performers
    print("\nTop 10 airfoil variations by L/D ratio:")
    for i in range(min(10, len(top_names))):
        name = top_names[i]
        print(f"{i+1}. {name}: L/D = {top_lds[i]:.2f}, CL = {results[name]['cl']:.4f}, CD = {results[name]['cd']:.6f}")
    
    # 1. Bar chart of top performers
    plt.figure(figsize=(14, 8))
    plt.bar(range(len(top_names)), top_lds)
    plt.xticks(range(len(top_names)), top_names, rotation=90)
    plt.xlabel('Airfoil Variation')
    plt.ylabel('Lift-to-Drag Ratio')
    plt.title('Top Performing Airfoil Variations')
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'top_performers.png'), dpi=300)
    plt.close()
    
    # 2. Scatter plot of CL vs CD with L/D as color
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(cds, cls, c=ld_ratios, cmap='viridis', alpha=0.8, s=50)
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Lift-to-Drag Ratio')
    
    # Add labels and title
    plt.xlabel('Drag Coefficient (CD)')
    plt.ylabel('Lift Coefficient (CL)')
    plt.title('Aerodynamic Performance of Airfoil Variations')
    
    # Use log scale for CD
    plt.xscale('log')
    
    # Mark top performers
    for i in top_indices[:5]:
        plt.annotate(names[i], (cds[i], cls[i]))
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'cl_cd_scatter.png'), dpi=300)
    plt.close()
    
    # 3. Parameter influence analysis (which parameters have the biggest impact)
    # Group results by base airfoil and modified parameter
    param_impact = {}
    
    for name in results.keys():
        # Skip base airfoils
        if "_base" in name:
            continue
            
        # Parse the name to get base airfoil and parameter
        # Format: airfoil_param_value
        parts = name.split('_')
        if len(parts) >= 3:
            base_airfoil = parts[0]
            param = parts[1]
            
            # Initialize nested dict if needed
            if base_airfoil not in param_impact:
                param_impact[base_airfoil] = {}
            if param not in param_impact[base_airfoil]:
                param_impact[base_airfoil][param] = []
            
            # Store the impact on L/D ratio
            base_name = f"{base_airfoil}_base"
            if base_name in results:
                base_ld = results[base_name]['ld_ratio']
                variation_ld = results[name]['ld_ratio']
                percent_change = ((variation_ld - base_ld) / base_ld) * 100 if base_ld > 0 else 0
                
                param_impact[base_airfoil][param].append(percent_change)
    
    # Calculate average impact for each parameter
    avg_impact = {}
    for param in PARAM_DESCRIPTIONS.keys():
        values = []
        for airfoil in param_impact:
            if param in param_impact[airfoil]:
                values.extend(param_impact[airfoil][param])
        
        if values:
            avg_impact[param] = np.mean(values)
    
    # Plot parameter impact
    plt.figure(figsize=(12, 8))
    params = list(avg_impact.keys())
    impacts = [avg_impact[p] for p in params]
    
    # Sort by absolute impact
    sorted_idx = np.argsort(np.abs(impacts))[::-1]  # Descending order
    params = [params[i] for i in sorted_idx]
    impacts = [impacts[i] for i in sorted_idx]
    
    colors = ['green' if i >= 0 else 'red' for i in impacts]
    plt.barh(params, impacts, color=colors)
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    plt.xlabel('Average % Change in L/D Ratio')
    plt.ylabel('Parameter')
    plt.title('Impact of Parameter Variations on Lift-to-Drag Ratio')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'parameter_impact.png'), dpi=300)
    plt.close()
    
    print(f"Result visualizations saved to {vis_dir}")


def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="PARSEC Parameter Variation Generator for Top Airfoils")
    
    parser.add_argument(
        "-s", "--steps",
        type=int,
        default=5,
        help="Number of steps for each parameter variation (default: 5)"
    )
    
    parser.add_argument(
        "-t", "--top",
        type=int,
        default=10,
        help="Number of top airfoils to use as base shapes (default: 10)"
    )
    
    parser.add_argument(
        "-p", "--percentage",
        action="store_true",
        help="Use percentage-based variations instead of range-based"
    )
    
    args = parser.parse_args()
    
    # Validate step count
    if args.steps < 2 or args.steps > 10:
        print(f"Warning: Invalid step count ({args.steps}). Using default value of 5.")
        args.steps = 5
    
    # Validate top count
    if args.top < 1 or args.top > 20:
        print(f"Warning: Invalid top airfoil count ({args.top}). Using default value of 10.")
        args.top = 10
    
    return args


def main():
    """Main function to execute airfoil variations study"""
    # Parse command-line arguments
    args = parse_arguments()
    
    # Get top performing airfoils
    top_airfoils = get_top_airfoils(args.top)
    if top_airfoils is None or len(top_airfoils) == 0:
        print("Error: Failed to get top airfoils")
        return
    
    # Process each top airfoil
    all_variations = []
    all_names = []
    all_params = []
    
    print(f"\nGenerating variations for {len(top_airfoils)} top airfoils...")
    for airfoil in top_airfoils:
        # Load PARSEC parameters
        base_params = load_parsec_parameters(airfoil)
        if base_params is None:
            print(f"Skipping {airfoil}: Could not load PARSEC parameters")
            continue
            
        # Add airfoil name to params for reference
        base_params['airfoil'] = airfoil
        
        # Generate parameter variations based on mode
        if args.percentage:
            # Use percentage-based variations
            variations = generate_percentage_variations(base_params, percentages=[0.0, 0.05, 0.10, 0.25])
            all_variations.extend(variations)
            
            # Generate names and params for each variation
            for var in variations:
                # Use the name already created in generate_percentage_variations
                var_name = var['name']
                all_names.append(var_name)
                all_params.append(var['params'])
                
            print(f"Generated {len(variations)} percentage-based variations for {airfoil}")
        else:
            # Use range-based variations (original method)
            variations = generate_variations(airfoil, base_params, args.steps)
            all_variations.extend([{'params': params, 'base_airfoil': airfoil} for _, params in variations])
            print(f"Generated {len(variations)} range-based variations for {airfoil}")
        
    # Create PARSEC parameter files - handle based on variation mode
    if args.percentage:
        # For percentage-based variations
        print(f"Creating PARSEC parameter files for {len(all_variations)} variations...")
        for var in tqdm(all_variations):
            # Create file path
            file_path = os.path.join(OUTPUT_DIR_PARSEC, f"{var['name']}.parsec")
            
            # Write parameters to file
            with open(file_path, 'w') as f:
                airfoil_name = var['params']['airfoil']
                param_name = var['param']
                percent_str = f"{'+' if var['percent'] > 0 else ''}{var['percent']*100:.0f}"
                f.write(f"# PARSEC parameters for {airfoil_name} with {param_name} {percent_str}%\n")
                for param, value in var['params'].items():
                    if param != 'airfoil':  # Skip the airfoil name
                        f.write(f"{param} = {value:.6f}\n")
    else:
        # For range-based variations (original method)
        create_parsec_files(all_variations)
    
    # Convert to DAT files
    convert_to_dat_files(all_names)
    
    # Run surrogate model on valid shapes
    results = run_surrogate_model(all_names, all_params)
    
    # Create visualizations
    if results:
        visualize_results(results, top_airfoils)
        
    print("\nVariation study completed!")
    print(f"PARSEC parameter files: {OUTPUT_DIR_PARSEC} ({len(all_names)} files)")
    print(f"Airfoil DAT files: {OUTPUT_DIR_DAT}")
    print(f"Visualization results: {RESULTS_DIR}")

if __name__ == "__main__":
    main()
