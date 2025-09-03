#!/usr/bin/env python3
"""
Analyze PARSEC Parameter Ranges

This script analyzes the PARSEC parameter values from all converted airfoil files
and calculates the minimum and maximum values for each parameter.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

# Directory containing all converted PARSEC parameters
INPUT_DIR = "all_converted_parsec"
COMBINED_FILE = os.path.join(INPUT_DIR, "all_converted_parsec.json")

def main():
    """Find min and max ranges for all PARSEC parameters"""
    print("Analyzing PARSEC parameter ranges...\n")
    
    # Load all PARSEC parameters from the combined file
    with open(COMBINED_FILE, 'r') as f:
        all_parsec = json.load(f)
    
    print(f"Loaded parameters for {len(all_parsec)} airfoils\n")
    
    # Create a dictionary to store parameter values across all airfoils
    param_values = defaultdict(list)
    
    # Extract parameter values from each airfoil
    for airfoil_name, params in all_parsec.items():
        for param_name, param_value in params.items():
            # Skip 'airfoil' key if present
            if param_name != 'airfoil':
                param_values[param_name].append(param_value)
    
    # Calculate min and max for each parameter
    param_ranges = {}
    extreme_values = {}
    
    print("Parameter ranges across all airfoils:")
    print("====================================")
    
    for param_name, values in param_values.items():
        values = np.array(values)
        min_val = np.min(values)
        max_val = np.max(values)
        median_val = np.median(values)
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        # Store min and max
        param_ranges[param_name] = {
            'min': min_val,
            'max': max_val,
            'median': median_val,
            'mean': mean_val, 
            'std': std_val
        }
        
        # Find airfoils with extreme values
        min_idx = np.argmin(values)
        max_idx = np.argmax(values)
        min_airfoil = list(all_parsec.keys())[min_idx]
        max_airfoil = list(all_parsec.keys())[max_idx]
        
        extreme_values[param_name] = {
            'min_airfoil': min_airfoil,
            'max_airfoil': max_airfoil
        }
        
        print(f"{param_name}:")
        print(f"  Min: {min_val:.6f} (airfoil: {min_airfoil})")
        print(f"  Max: {max_val:.6f} (airfoil: {max_airfoil})")
        print(f"  Median: {median_val:.6f}")
        print(f"  Mean: {mean_val:.6f}")
        print(f"  Std Dev: {std_val:.6f}")
        print()
    
    # Save results to file
    with open("parsec_parameter_ranges.json", 'w') as f:
        json.dump(param_ranges, f, indent=2)
    
    with open("parsec_extreme_airfoils.json", 'w') as f:
        json.dump(extreme_values, f, indent=2)
    
    print("Results saved to parsec_parameter_ranges.json and parsec_extreme_airfoils.json")
    
    # Create a visualization of the parameter ranges
    create_range_visualization(param_ranges, extreme_values)

def create_range_visualization(param_ranges, extreme_values):
    """Create visualizations of parameter ranges"""
    # Create a DataFrame for easier plotting
    df = pd.DataFrame({param: {'min': data['min'], 'max': data['max'], 
                              'median': data['median'], 'mean': data['mean']} 
                      for param, data in param_ranges.items()}).T
    
    # Set up the figure
    plt.figure(figsize=(14, 10))
    
    # Create a bar plot showing the range for each parameter
    params = list(param_ranges.keys())
    
    # We'll normalize the parameters for better visualization
    normalized_df = pd.DataFrame()
    
    for param in params:
        min_val = param_ranges[param]['min']
        max_val = param_ranges[param]['max']
        mean_val = param_ranges[param]['mean']
        median_val = param_ranges[param]['median']
        
        if abs(max_val - min_val) > 1e-6:  # Avoid division by zero
            normalized_df.loc[param, 'min'] = 0
            normalized_df.loc[param, 'max'] = 1
            normalized_df.loc[param, 'mean'] = (mean_val - min_val) / (max_val - min_val)
            normalized_df.loc[param, 'median'] = (median_val - min_val) / (max_val - min_val)
        else:
            normalized_df.loc[param, 'min'] = 0
            normalized_df.loc[param, 'max'] = 1
            normalized_df.loc[param, 'mean'] = 0.5
            normalized_df.loc[param, 'median'] = 0.5
    
    # Create horizontal bar plot of normalized ranges
    ax = normalized_df.plot(kind='barh', figsize=(12, 8), y=['min', 'max'], 
                           color=['blue', 'red'], alpha=0.6)
    
    # Add mean and median markers
    for i, param in enumerate(normalized_df.index):
        plt.scatter(normalized_df.loc[param, 'mean'], i, color='green', s=100, marker='o', label='Mean' if i == 0 else "")
        plt.scatter(normalized_df.loc[param, 'median'], i, color='purple', s=100, marker='D', label='Median' if i == 0 else "")
    
    # Add labels and legend
    plt.xlabel('Normalized Range (0=min, 1=max)')
    plt.ylabel('PARSEC Parameter')
    plt.title('Normalized Ranges of PARSEC Parameters Across All Airfoils')
    plt.legend(loc='lower right')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Add actual min and max values as text
    for i, param in enumerate(normalized_df.index):
        min_val = param_ranges[param]['min']
        max_val = param_ranges[param]['max']
        plt.text(-0.15, i, f"{min_val:.4f}", ha='right', va='center')
        plt.text(1.15, i, f"{max_val:.4f}", ha='left', va='center')
    
    plt.tight_layout()
    plt.savefig("parsec_parameter_ranges.png", dpi=300, bbox_inches='tight')
    print("Visualization saved as parsec_parameter_ranges.png")

    # Create a table with the extreme airfoils
    plt.figure(figsize=(12, 10))
    plt.axis('off')
    
    table_data = []
    table_columns = ['Parameter', 'Min Value', 'Min Airfoil', 'Max Value', 'Max Airfoil']
    
    for param in params:
        min_val = param_ranges[param]['min']
        max_val = param_ranges[param]['max']
        min_airfoil = extreme_values[param]['min_airfoil']
        max_airfoil = extreme_values[param]['max_airfoil']
        
        table_data.append([param, f"{min_val:.6f}", min_airfoil, f"{max_val:.6f}", max_airfoil])
    
    table = plt.table(cellText=table_data, colLabels=table_columns, 
                     loc='center', cellLoc='center', colColours=['#f2f2f2']*5)
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    plt.title('Airfoils with Extreme PARSEC Parameter Values', fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig("parsec_extreme_airfoils.png", dpi=300, bbox_inches='tight')
    print("Extreme airfoils table saved as parsec_extreme_airfoils.png")

if __name__ == "__main__":
    main()
