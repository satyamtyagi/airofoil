#!/usr/bin/env python3
"""
Visualize Valid Airfoil Variations

This script creates visualizations for the valid airfoil variations that passed
geometric validation with the corrected thickness calculation algorithm.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
from parsec_fit import ParsecAirfoil
from airfoil_validation import calculate_thickness
import json

# Define directories
PARSEC_DIR = 'airfoils_parsec'
VARIATIONS_PARSEC_DIR = 'airfoils_variations_parsec'
VARIATIONS_DAT_DIR = 'airfoils_variations_dat'
RESULTS_DIR = 'variation_results'
VISUALIZATION_DIR = os.path.join(RESULTS_DIR, 'valid_variations_viz')

# Create visualization directory if it doesn't exist
os.makedirs(VISUALIZATION_DIR, exist_ok=True)

def load_variation_results():
    """Load the variation results from the HDF5 file"""
    results_file = os.path.join(RESULTS_DIR, 'variation_results.h5')
    with h5py.File(results_file, 'r') as f:
        # Get valid variations (with valid geometry)
        valid_variations = []
        for var_name in f.keys():
            if 'valid_geometry' in f[var_name].attrs and f[var_name].attrs['valid_geometry']:
                data = {
                    'name': var_name,
                    'cl': f[var_name].attrs.get('cl', 0),
                    'cd': f[var_name].attrs.get('cd', 0),
                    'cl_cd': f[var_name].attrs.get('cl_cd', 0),
                    'base_airfoil': var_name.split('_')[0] if '_' in var_name else None,
                    'varied_param': '_'.join(var_name.split('_')[1:]) if '_' in var_name else 'base'
                }
                valid_variations.append(data)
        
    # Convert to DataFrame for easier manipulation
    return pd.DataFrame(valid_variations)

def get_coordinates(airfoil_name, is_variation=False):
    """Get coordinates for an airfoil"""
    if is_variation:
        # For variations, load from the variations dat directory
        file_path = os.path.join(VARIATIONS_DAT_DIR, f"{airfoil_name}.dat")
    else:
        # For original airfoils, load from the UIUC dat directory
        file_path = os.path.join("airfoils_uiuc", f"{airfoil_name}.dat")
    
    x_coords = []
    y_coords = []
    
    try:
        with open(file_path, 'r') as f:
            # Skip header if present
            first_line = f.readline().strip()
            if not first_line[0].isdigit():
                # It's a header, skip it
                pass
            else:
                # Not a header, parse it
                x, y = first_line.split()[:2]
                x_coords.append(float(x))
                y_coords.append(float(y))
                
            # Read the rest of the coordinates
            for line in f:
                try:
                    x, y = line.strip().split()[:2]
                    x_coords.append(float(x))
                    y_coords.append(float(y))
                except (ValueError, IndexError):
                    # Skip problematic lines
                    continue
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None, None
        
    return np.array(x_coords), np.array(y_coords)

def plot_airfoil_comparison(variation_name, variation_data):
    """Create comparison plot between original airfoil and its variation"""
    # Extract base airfoil name
    base_airfoil = variation_data['base_airfoil']
    varied_param = variation_data['varied_param']
    
    # Get coordinates
    x_var, y_var = get_coordinates(variation_name, is_variation=True)
    x_base, y_base = get_coordinates(base_airfoil, is_variation=False)
    
    # Skip if coordinates are missing
    if x_var is None or x_base is None:
        print(f"Skipping {variation_name}: Missing coordinates")
        return
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot airfoil shapes
    ax1.plot(x_base, y_base, 'b-', label=f'{base_airfoil} (Original)')
    ax1.plot(x_var, y_var, 'r-', label=f'{variation_name} (Variation)')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Airfoil Shape Comparison")
    ax1.legend()
    
    # Calculate thickness information
    thick_var, pos_var = calculate_thickness(x_var, y_var)
    thick_base, pos_base = calculate_thickness(x_base, y_base)
    
    # Prepare performance comparison
    metrics = [
        f"L/D: {variation_data['cl_cd']:.2f}",
        f"CL: {variation_data['cl']:.4f}",
        f"CD: {variation_data['cd']:.6f}",
        f"Thickness: {thick_var:.4f} at x={pos_var:.2f}",
        f"Original Thickness: {thick_base:.4f} at x={pos_base:.2f}"
    ]
    
    # Plot performance metrics as a table
    ax2.axis('tight')
    ax2.axis('off')
    table = ax2.table(
        cellText=[[m] for m in metrics],
        colWidths=[0.8],
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)
    
    param_name = varied_param.split('_')[0] if '_' in varied_param else varied_param
    param_value = varied_param.split('_')[1] if '_' in varied_param else "N/A"
    
    # Set overall title
    plt.suptitle(f"Variation: {variation_name}\nModified Parameter: {param_name} = {param_value}", fontsize=16)
    
    # Save figure
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(VISUALIZATION_DIR, f"{variation_name}_comparison.png"), dpi=300)
    plt.close()
    
    return True

def create_summary_visualization(valid_df):
    """Create summary visualization of all valid variations"""
    # Sort by performance (L/D ratio)
    valid_df = valid_df.sort_values(by='cl_cd', ascending=False)
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Plot L/D ratio
    plt.bar(valid_df['name'], valid_df['cl_cd'], color='royalblue')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Lift-to-Drag Ratio (L/D)')
    plt.title('Performance of Valid Airfoil Variations')
    plt.grid(axis='y', alpha=0.3)
    
    # Add values above bars
    for i, v in enumerate(valid_df['cl_cd']):
        plt.text(i, v + 0.5, f"{v:.2f}", ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATION_DIR, "valid_variations_summary.png"), dpi=300)
    plt.close()
    
    # Create a second figure for the variation parameter impact
    plt.figure(figsize=(12, 10))
    
    # Plot which parameters were successfully varied
    param_counts = valid_df['varied_param'].str.split('_').str[0].value_counts()
    plt.barh(param_counts.index, param_counts.values, color='lightseagreen')
    plt.xlabel('Number of Valid Variations')
    plt.ylabel('Modified Parameter')
    plt.title('Parameters That Produced Valid Variations')
    plt.grid(axis='x', alpha=0.3)
    
    # Add values on bars
    for i, v in enumerate(param_counts.values):
        plt.text(v + 0.1, i, str(v), va='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATION_DIR, "valid_parameter_impact.png"), dpi=300)
    plt.close()

def create_html_report(valid_df):
    """Create an HTML report of the valid variations"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Valid Airfoil Variations</title>
        <style>
            body { font-family: Arial, sans-serif; line-height: 1.6; margin: 20px; }
            h1, h2 { color: #2c3e50; }
            .container { max-width: 1200px; margin: 0 auto; }
            table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            tr:nth-child(even) { background-color: #f9f9f9; }
            .gallery { display: flex; flex-wrap: wrap; gap: 20px; }
            .gallery-item { margin-bottom: 20px; }
            .summary-img { max-width: 100%; height: auto; margin-bottom: 20px; }
            footer { margin-top: 30px; padding-top: 10px; border-top: 1px solid #eee; font-size: 0.8em; color: #7f8c8d; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Valid Airfoil Variations Report</h1>
            
            <section>
                <h2>Summary</h2>
                <p>Out of 534 total variations, <strong>5</strong> passed geometric validation.</p>
                <div>
                    <img class="summary-img" src="valid_variations_summary.png" alt="Performance Summary">
                </div>
                <div>
                    <img class="summary-img" src="valid_parameter_impact.png" alt="Parameter Impact">
                </div>
            </section>
            
            <section>
                <h2>Performance Data</h2>
                <table>
                    <tr>
                        <th>Variation</th>
                        <th>Base Airfoil</th>
                        <th>Modified Parameter</th>
                        <th>L/D Ratio</th>
                        <th>Lift (CL)</th>
                        <th>Drag (CD)</th>
                    </tr>
    """
    
    # Add table rows
    for _, row in valid_df.sort_values(by='cl_cd', ascending=False).iterrows():
        html_content += f"""
                    <tr>
                        <td>{row['name']}</td>
                        <td>{row['base_airfoil']}</td>
                        <td>{row['varied_param']}</td>
                        <td>{row['cl_cd']:.2f}</td>
                        <td>{row['cl']:.4f}</td>
                        <td>{row['cd']:.6f}</td>
                    </tr>
        """
    
    html_content += """
                </table>
            </section>
            
            <section>
                <h2>Individual Comparisons</h2>
                <div class="gallery">
    """
    
    # Add image gallery
    for name in valid_df['name']:
        html_content += f"""
                    <div class="gallery-item">
                        <img src="{name}_comparison.png" alt="{name}" width="600">
                    </div>
        """
    
    html_content += """
                </div>
            </section>
            
            <footer>
                <p>Generated by visualize_valid_variations.py</p>
            </footer>
        </div>
    </body>
    </html>
    """
    
    # Write HTML file
    with open(os.path.join(VISUALIZATION_DIR, "valid_variations_report.html"), 'w') as f:
        f.write(html_content)

def main():
    print("Visualizing valid airfoil variations...")
    
    # Load variation results
    valid_df = load_variation_results()
    print(f"Found {len(valid_df)} valid variations:")
    for idx, (_, row) in enumerate(valid_df.sort_values(by='cl_cd', ascending=False).iterrows(), 1):
        print(f"{idx}. {row['name']}: L/D = {row['cl_cd']:.2f}, CL = {row['cl']:.4f}, CD = {row['cd']:.6f}")
    
    # Create individual comparison plots
    for _, row in valid_df.iterrows():
        print(f"Creating comparison for {row['name']}...")
        plot_airfoil_comparison(row['name'], row)
    
    # Create summary visualization
    print("Creating summary visualizations...")
    create_summary_visualization(valid_df)
    
    # Create HTML report
    print("Generating HTML report...")
    create_html_report(valid_df)
    
    print(f"Visualizations complete! Results saved to {VISUALIZATION_DIR}")
    print(f"Open {os.path.join(VISUALIZATION_DIR, 'valid_variations_report.html')} to view the report")

if __name__ == "__main__":
    main()
