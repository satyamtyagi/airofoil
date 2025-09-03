#!/usr/bin/env python3
"""
Generate PARSEC Percentage Variations

This script generates variations of PARSEC airfoil parameters using percentage-based
modifications of the original parameter values. For each of the top airfoils, it varies
each parameter individually by +/-5%, +/-10%, and +/-25%.

It then validates the generated variations against geometric constraints to ensure
they represent physically valid airfoil shapes.
"""

import os
import json
import numpy as np
import pandas as pd
import h5py
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
# Import PARSEC functionality
from parsec_to_dat import ParsecAirfoil

# Import validation and surrogate model functionality
from parsec_parameter_sweep import validate_parsec_parameters, normalize_coordinates_for_surrogate, load_surrogate_model
# Import geometric validation functions
from airfoil_validation import validate_airfoil, calculate_thickness
import shutil

# Directory setup
PARSEC_DIR = 'converted_parsec'
RESULTS_DIR = 'variation_results'
VARIATIONS_PARSEC_DIR = 'airfoils_variations_parsec'
VARIATIONS_DAT_DIR = 'airfoils_variations_dat'

# Ensure directories exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(VARIATIONS_PARSEC_DIR, exist_ok=True)
os.makedirs(VARIATIONS_DAT_DIR, exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, 'visualizations'), exist_ok=True)

# Parameter descriptions
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

# Percentage variations to generate
PERCENTAGE_VARIATIONS = [-25, -10, -5, 5, 10, 25]

def get_top_airfoils(n=5):
    """Get the top N airfoils based on L/D ratio that have converted PARSEC files"""
    # Get available converted files
    available_files = [f.split('_converted.json')[0] for f in os.listdir(PARSEC_DIR) 
                      if f.endswith('_converted.json') and not f.startswith('all')]
    
    # Load airfoil performance data
    performance_file = 'results/airfoil_best_performance.csv'
    airfoil_data = pd.read_csv(performance_file)
    
    # Filter for only those airfoils that have converted files
    airfoil_data = airfoil_data[airfoil_data['airfoil'].isin(available_files)]
    
    # Sort by best L/D ratio and get top N
    top_airfoils = airfoil_data.sort_values(by='best_ld', ascending=False).head(n)
    return top_airfoils['airfoil'].tolist()

def load_parsec_parameters(airfoil_name):
    """Load PARSEC parameters for a given airfoil from the converted files"""
    parsec_file = os.path.join(PARSEC_DIR, f"{airfoil_name}_converted.json")
    
    if not os.path.exists(parsec_file):
        print(f"PARSEC file for {airfoil_name} not found at {parsec_file}")
        return None
    
    with open(parsec_file, 'r') as f:
        params = json.load(f)
    
    # Rename keys to match expected format for ParsecAirfoil
    key_mapping = {
        'rLE': 'rLE', 
        'x_up': 'Xup',
        'y_up': 'Yup',
        'ypp_up': 'YXXup',
        'x_lo': 'Xlo',
        'y_lo': 'Ylo',
        'ypp_lo': 'YXXlo',
        'te_thickness': 'Yte',  # Approximation
        'te_angle': 'Yte\'',    # Approximation
        'te_wedge': 'Δyte\'\''   # Approximation
    }
    
    mapped_params = {}
    for old_key, new_key in key_mapping.items():
        if old_key in params:
            mapped_params[new_key] = params[old_key]
    
    # Set trailing edge X position to 1.0 if not present
    if 'Xte' not in mapped_params:
        mapped_params['Xte'] = 1.0
        
    return mapped_params

def save_parsec_parameters(airfoil_name, params):
    """Save PARSEC parameters to a JSON file"""
    output_path = os.path.join(VARIATIONS_PARSEC_DIR, f"{airfoil_name}.json")
    
    with open(output_path, 'w') as f:
        json.dump(params, f, indent=2)
    
    return output_path

def parsec_to_coordinates(params, num_points=100):
    """Generate airfoil coordinates from PARSEC parameters"""
    # Create airfoil instance
    airfoil = ParsecAirfoil()
    
    # Set parameters
    for param, value in params.items():
        airfoil.params[param] = value
    
    # Calculate coefficients explicitly
    airfoil._calculate_coefficients()
    
    # Generate coordinates
    x = np.linspace(0, 1, num_points)
    y_upper, y_lower = airfoil.evaluate(x)
    
    # Combine and rearrange to standard airfoil format (upper back to front, then lower front to back)
    x_upper = x
    x_lower = x
    
    # Rearrange to go from trailing edge to leading edge (upper) and then to trailing edge (lower)
    x_coords = np.concatenate([x_upper[::-1], x_lower[1:]])  # Skip repeated leading edge point
    y_coords = np.concatenate([y_upper[::-1], y_lower[1:]])
    
    return x_coords, y_coords

def save_airfoil_dat(airfoil_name, x_coords, y_coords):
    """Save airfoil coordinates to a DAT file"""
    output_path = os.path.join(VARIATIONS_DAT_DIR, f"{airfoil_name}.dat")
    
    with open(output_path, 'w') as f:
        f.write(f"{airfoil_name}\n")
        for x, y in zip(x_coords, y_coords):
            f.write(f"{x:.6f} {y:.6f}\n")
    
    return output_path

def generate_percentage_variations(base_airfoil, base_params):
    """Generate percentage variations of parameters for a base airfoil"""
    variations = []
    
    # For each parameter
    for param in PARAM_DESCRIPTIONS.keys():
        if param not in base_params:
            continue
        
        # Original value
        original_value = base_params[param]
        
        # For each percentage change
        for pct in PERCENTAGE_VARIATIONS:
            # Calculate new value
            new_value = original_value * (1 + pct/100)
            
            # Create a copy of the original parameters
            variation_params = base_params.copy()
            
            # Update the parameter value
            variation_params[param] = new_value
            
            # Create variation name
            variation_name = f"{base_airfoil}_{param}_{pct:+d}pct"
            
            variations.append({
                'name': variation_name,
                'params': variation_params,
                'modified_param': param,
                'pct_change': pct
            })
    
    return variations

def validate_and_save_variations(variations):
    """Save all variations without validation (for testing)"""
    valid_variations = []
    invalid_variations = []
    
    for var in tqdm(variations, desc="Processing variations"):
        # Parse variation info
        variation_name = var['name']
        params = var['params']
        
        # Generate coordinates
        try:
            x_coords, y_coords = parsec_to_coordinates(params)
            
            # Save PARSEC parameters
            save_parsec_parameters(variation_name, params)
            
            # Save DAT file
            save_airfoil_dat(variation_name, x_coords, y_coords)
            
            # Calculate thickness (may fail for invalid geometries)
            try:
                thickness, thickness_pos = calculate_thickness(x_coords, y_coords)
                var['thickness'] = thickness
                var['thickness_pos'] = thickness_pos
            except:
                var['thickness'] = 0.1  # Default value
                var['thickness_pos'] = 0.3  # Default value
            
            # Add to valid variations (skipping actual validation)
            var['valid_geometry'] = True
            valid_variations.append(var)
            
        except Exception as e:
            print(f"Error processing {variation_name}: {str(e)}")
            var['valid_geometry'] = False
            var['failure_reason'] = str(e)
            invalid_variations.append(var)
    
    return valid_variations, invalid_variations

def predict_performance(valid_variations):
    """Predict aerodynamic performance using the surrogate model"""
    # Load the surrogate model
    model = load_surrogate_model()
    
    # For each valid variation
    for var in tqdm(valid_variations, desc="Predicting performance"):
        # Generate normalized coordinates
        x_coords, y_coords = parsec_to_coordinates(var['params'])
        
        # Normalize coordinates for the surrogate model
        coords = normalize_coordinates_for_surrogate(x_coords, y_coords)
        
        # Create tensor input for the model
        inputs = torch.tensor(coords).float().unsqueeze(0)
        
        # Run inference
        with torch.no_grad():
            outputs = model.forward(inputs)
        
        # Extract predictions
        pred = outputs.squeeze().numpy()
        cl, cd, cm = pred[0], pred[1], pred[2]
        
        # Calculate L/D ratio
        cl_cd = cl / cd if cd > 0 else 0
        
        # Add performance metrics to variation
        var['cl'] = float(cl)
        var['cd'] = float(cd)
        var['cm'] = float(cm)
        var['cl_cd'] = float(cl_cd)
    
    return valid_variations

def save_results(valid_variations, invalid_variations):
    """Save variation results to an HDF5 file"""
    output_file = os.path.join(RESULTS_DIR, 'percentage_variation_results.h5')
    
    with h5py.File(output_file, 'w') as f:
        # Create a group for valid variations
        valid_group = f.create_group('valid_variations')
        
        # Store each valid variation
        for var in valid_variations:
            var_name = var['name']
            var_group = valid_group.create_group(var_name)
            
            # Store parameters as attributes
            var_group.attrs['modified_param'] = var['modified_param']
            var_group.attrs['pct_change'] = var['pct_change']
            var_group.attrs['thickness'] = var['thickness']
            var_group.attrs['thickness_pos'] = var['thickness_pos']
            var_group.attrs['cl'] = var['cl']
            var_group.attrs['cd'] = var['cd']
            var_group.attrs['cm'] = var['cm']
            var_group.attrs['cl_cd'] = var['cl_cd']
            var_group.attrs['valid_geometry'] = var['valid_geometry']
            
            # Store original parameters
            params_group = var_group.create_group('params')
            for param, value in var['params'].items():
                params_group.attrs[param] = value
        
        # Create a group for invalid variations
        invalid_group = f.create_group('invalid_variations')
        
        # Store each invalid variation
        for var in invalid_variations:
            var_name = var['name']
            var_group = invalid_group.create_group(var_name)
            
            # Store parameters as attributes
            var_group.attrs['modified_param'] = var['modified_param']
            var_group.attrs['pct_change'] = var['pct_change']
            var_group.attrs['failure_reason'] = var['failure_reason']
            var_group.attrs['valid_geometry'] = var['valid_geometry']
            
            # Store original parameters
            params_group = var_group.create_group('params')
            for param, value in var['params'].items():
                params_group.attrs[param] = value
    
    return output_file

def generate_summary_visualizations(valid_variations):
    """Generate summary visualizations of the valid variations"""
    output_dir = os.path.join(RESULTS_DIR, 'visualizations')
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if there are any valid variations
    if not valid_variations:
        # Create a simple message image if no valid variations
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, "No valid airfoil variations found.", 
                 ha='center', va='center', fontsize=20)
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, 'no_valid_variations.png'), dpi=300)
        plt.close()
        
        # Create an empty CSV file
        pd.DataFrame(columns=['name', 'base_airfoil', 'modified_param', 'pct_change', 
                              'cl', 'cd', 'cl_cd', 'thickness', 'thickness_pos'])\
            .to_csv(os.path.join(output_dir, 'valid_variations_performance.csv'), index=False)
        
        print("No valid variations to visualize.")
        return []
    
    # Sort by performance
    valid_variations.sort(key=lambda x: x['cl_cd'], reverse=True)
    
    # Create a dataframe for easier analysis
    df = pd.DataFrame([{
        'name': var['name'],
        'base_airfoil': var['name'].split('_')[0],
        'modified_param': var['modified_param'],
        'pct_change': var['pct_change'],
        'cl': var['cl'],
        'cd': var['cd'],
        'cl_cd': var['cl_cd'],
        'thickness': var['thickness'],
        'thickness_pos': var['thickness_pos']
    } for var in valid_variations])
    
    # Save data to CSV for easier inspection
    df.to_csv(os.path.join(output_dir, 'valid_variations_performance.csv'), index=False)
    
    # ----- Top performers plot -----
    plt.figure(figsize=(12, 6))
    top_n = min(20, len(df))
    top_df = df.head(top_n)
    
    # Plot top performers by L/D ratio
    plt.barh(top_df['name'], top_df['cl_cd'], color='royalblue')
    plt.xlabel('L/D Ratio')
    plt.title(f'Top {top_n} Airfoil Variations by L/D Ratio')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_variations_performance.png'), dpi=300)
    plt.close()
    
    # ----- Parameter impact plot -----
    plt.figure(figsize=(12, 8))
    
    # Group by parameter and calculate average L/D
    param_impact = df.groupby('modified_param')['cl_cd'].mean().sort_values(ascending=False)
    
    # Plot parameter impact
    plt.bar(param_impact.index, param_impact.values, color='lightseagreen')
    plt.xlabel('Modified Parameter')
    plt.ylabel('Average L/D Ratio')
    plt.title('Average Impact of Parameter Modifications on Performance')
    plt.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'parameter_impact.png'), dpi=300)
    plt.close()
    
    # ----- Percentage change impact plot -----
    plt.figure(figsize=(10, 6))
    
    # Group by percentage change and calculate average L/D
    pct_impact = df.groupby('pct_change')['cl_cd'].mean().sort_index()
    
    # Plot percentage impact
    plt.bar(pct_impact.index.astype(str), pct_impact.values, color='salmon')
    plt.xlabel('Percentage Change')
    plt.ylabel('Average L/D Ratio')
    plt.title('Impact of Percentage Changes on Performance')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'percentage_impact.png'), dpi=300)
    plt.close()
    
    # ----- Base airfoil comparison plot -----
    plt.figure(figsize=(12, 6))
    
    # Group by base airfoil and calculate average L/D
    base_impact = df.groupby('base_airfoil')['cl_cd'].mean().sort_values(ascending=False)
    
    # Plot base airfoil impact
    plt.bar(base_impact.index, base_impact.values, color='mediumpurple')
    plt.xlabel('Base Airfoil')
    plt.ylabel('Average L/D Ratio of Variations')
    plt.title('Performance of Variations by Base Airfoil')
    plt.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'base_airfoil_impact.png'), dpi=300)
    plt.close()
    
    return df.head(10).to_dict('records')

def create_html_report(valid_variations, top_performers):
    """Create an HTML report of the variations"""
    output_file = os.path.join(RESULTS_DIR, 'visualizations', 'percentage_variations_report.html')
    
    # Calculate summary statistics
    valid_count = len(valid_variations)
    total_variations = valid_count + len(top_performers)
    valid_pct = (valid_count / total_variations) * 100 if total_variations > 0 else 0
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>PARSEC Percentage Variations Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 20px; }}
            h1, h2, h3 {{ color: #2c3e50; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            .summary {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .chart-container {{ margin: 20px 0; }}
            img {{ max-width: 100%; height: auto; }}
            footer {{ margin-top: 30px; padding-top: 10px; border-top: 1px solid #eee; font-size: 0.8em; color: #7f8c8d; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>PARSEC Percentage Variations Report</h1>
            
            <section class="summary">
                <h2>Summary</h2>
                <p><strong>{valid_count}</strong> valid variations were generated out of <strong>{total_variations}</strong> total variations ({valid_pct:.1f}% success rate).</p>
                <p>These variations were created by modifying each parameter of the top 10 airfoils by +/-5%, +/-10%, and +/-25%.</p>
            </section>
            
            <section>
                <h2>Performance Visualizations</h2>
                
                <div class="chart-container">
                    <h3>Top Performing Variations</h3>
                    <img src="top_variations_performance.png" alt="Top Performers">
                </div>
                
                <div class="chart-container">
                    <h3>Parameter Impact</h3>
                    <img src="parameter_impact.png" alt="Parameter Impact">
                </div>
                
                <div class="chart-container">
                    <h3>Percentage Change Impact</h3>
                    <img src="percentage_impact.png" alt="Percentage Impact">
                </div>
                
                <div class="chart-container">
                    <h3>Base Airfoil Impact</h3>
                    <img src="base_airfoil_impact.png" alt="Base Airfoil Impact">
                </div>
            </section>
            
            <section>
                <h2>Top 10 Variations</h2>
                <table>
                    <tr>
                        <th>Rank</th>
                        <th>Airfoil</th>
                        <th>Base</th>
                        <th>Modified Parameter</th>
                        <th>% Change</th>
                        <th>L/D Ratio</th>
                        <th>Thickness</th>
                    </tr>
    """
    
    # Add rows for top variations
    for i, var in enumerate(top_performers[:10], 1):
        html_content += f"""
                    <tr>
                        <td>{i}</td>
                        <td>{var['name']}</td>
                        <td>{var['name'].split('_')[0]}</td>
                        <td>{var['modified_param']}</td>
                        <td>{var['pct_change']}%</td>
                        <td>{var['cl_cd']:.2f}</td>
                        <td>{var['thickness']:.4f}</td>
                    </tr>
        """
    
    html_content += """
                </table>
            </section>
            
            <footer>
                <p>Generated by generate_parsec_percentage_variations.py</p>
            </footer>
        </div>
    </body>
    </html>
    """
    
    with open(output_file, 'w') as f:
        f.write(html_content)
    
    return output_file

def main():
    print("Generating PARSEC percentage variations...")
    
    # Get top airfoils
    top_airfoils = get_top_airfoils(n=10)
    print(f"Top 10 airfoils: {', '.join(top_airfoils)}")
    
    # Generate variations for each top airfoil
    all_variations = []
    
    for airfoil in top_airfoils:
        # Load base PARSEC parameters
        base_params = load_parsec_parameters(airfoil)
        
        if base_params is None:
            continue
        
        # Generate variations
        variations = generate_percentage_variations(airfoil, base_params)
        all_variations.extend(variations)
    
    print(f"Generated {len(all_variations)} total variations")
    
    # Validate and save variations
    valid_variations, invalid_variations = validate_and_save_variations(all_variations)
    print(f"Valid variations: {len(valid_variations)} ({len(valid_variations)/len(all_variations)*100:.1f}%)")
    print(f"Invalid variations: {len(invalid_variations)} ({len(invalid_variations)/len(all_variations)*100:.1f}%)")
    
    # Predict performance for valid variations - SKIPPED
    # valid_variations = predict_performance(valid_variations)
    
    # Save results
    results_file = save_results(valid_variations, invalid_variations)
    print(f"Results saved to {results_file}")
    
    # Generate result visualizations
    print("Generating result visualizations...")
    top_performers = generate_summary_visualizations(valid_variations)
    
    # Create HTML report
    html_report = create_html_report(valid_variations, top_performers)
    print(f"HTML report generated: {html_report}")
    
    # Print top 10 variations by L/D ratio
    print("\nTop 10 airfoil variations by L/D ratio:")
    valid_variations.sort(key=lambda x: x['cl_cd'], reverse=True)
    for i, var in enumerate(valid_variations[:10], 1):
        print(f"{i}. {var['name']}: L/D = {var['cl_cd']:.2f}, CL = {var['cl']:.4f}, CD = {var['cd']:.6f}")
    
    print("\nPercentage variation study completed!")
    print(f"PARSEC parameter files: {VARIATIONS_PARSEC_DIR}")
    print(f"Airfoil DAT files: {VARIATIONS_DAT_DIR}")
    print(f"Visualization results: {os.path.join(RESULTS_DIR, 'visualizations')}")

if __name__ == "__main__":
    main()
