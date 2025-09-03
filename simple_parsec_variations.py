#!/usr/bin/env python3
"""
Generate Simple PARSEC Percentage Variations

This script generates variations of PARSEC airfoil parameters using percentage-based
modifications of the original parameter values. For each of the top airfoils, it varies
each parameter individually by +/-5%, +/-10%, and +/-25%.
"""

import os
import json
import numpy as np
from pathlib import Path

# Directory paths
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
PARSEC_DIR = BASE_DIR / "converted_parsec"
OUTPUT_DIR = BASE_DIR / "simple_variations_parsec"

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(exist_ok=True)

def load_parsec_parameters(airfoil_name):
    """Load PARSEC parameters for a given airfoil from the converted files"""
    parsec_file = PARSEC_DIR / f"{airfoil_name}_converted.json"
    
    if not parsec_file.exists():
        print(f"No PARSEC parameters found for {airfoil_name}")
        return None
    
    with open(parsec_file, 'r') as f:
        data = json.load(f)
        
    # The parameters are directly at the root level in the JSON file
    return data

def save_parsec_parameters(variation_name, params):
    """Save PARSEC parameters to a JSON file"""
    output_file = OUTPUT_DIR / f"{variation_name}.json"
    
    with open(output_file, 'w') as f:
        json.dump(params, f, indent=4)

def generate_percentage_variations(base_airfoil, base_params):
    """Generate percentage variations of parameters for a base airfoil"""
    variations = []
    
    # Define the percentage variations to try
    percentages = [-25, -10, -5, 5, 10, 25]
    
    # Parameter names to vary - matching the actual JSON file format
    param_names = [
        'rLE', 'x_up', 'y_up', 'ypp_up', 
        'x_lo', 'y_lo', 'ypp_lo', 
        'te_thickness', 'te_camber', 'te_angle', 'te_wedge'
    ]
    
    # Generate variations for each parameter
    for param_name in param_names:
        for percentage in percentages:
            # Create a copy of the base parameters
            params_copy = base_params.copy()
            
            # Skip variation if parameter is not in base_params
            if param_name not in params_copy:
                continue
            
            # Calculate the variation
            original_value = params_copy[param_name]
            variation = original_value * (1 + percentage / 100)
            params_copy[param_name] = variation
            
            # Create variation name
            variation_name = f"{base_airfoil}_{param_name}_{percentage:+d}pct"
            
            # Add to list of variations
            variations.append({
                'name': variation_name,
                'base_airfoil': base_airfoil,
                'varied_param': param_name,
                'percentage': percentage,
                'params': params_copy
            })
    
    return variations

def get_top_airfoils(n=5):
    """Get the top N airfoils based on available converted PARSEC files"""
    # Known top performers from previous analysis
    known_top = ['ag13', 'a18sm', 'ag26', 'ag25', 'ag27']
    
    # Filter to those that have converted PARSEC files
    available_top = []
    for airfoil in known_top:
        if (PARSEC_DIR / f"{airfoil}_converted.json").exists():
            available_top.append(airfoil)
    
    # Return up to n airfoils
    return available_top[:n]

def main():
    print("Generating simple PARSEC percentage variations...")
    
    # Get top airfoils
    top_airfoils = get_top_airfoils(n=5)
    print(f"Top 5 airfoils: {', '.join(top_airfoils)}")
    
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
    
    # Save all variations
    for var in all_variations:
        save_parsec_parameters(var['name'], var['params'])
    
    print(f"All variations saved to {OUTPUT_DIR}")
    
    # Print a summary of variations
    print("\nVariation Summary:")
    for airfoil in top_airfoils:
        airfoil_variations = [v for v in all_variations if v['base_airfoil'] == airfoil]
        if airfoil_variations:
            print(f"  {airfoil}: {len(airfoil_variations)} variations")

if __name__ == "__main__":
    main()
