#!/usr/bin/env python3
"""
PARSEC Parameter Variation Generator

This script:
1. Reads the min/max/mean PARSEC parameter statistics
2. Creates systematic variations of PARSEC parameters:
   - One-at-a-time variations: varies one parameter from min to max while keeping others at mean values
   - Random combinations: creates random combinations within the parameter space
3. Generates PARSEC parameter files for all variations
4. Converts the PARSEC files to airfoil DAT files

This allows exploring the entire PARSEC parameter space in a structured way.

Usage:
  python generate_parsec_variations.py [options]

Options:
  -s, --steps N    Number of steps for each parameter from min to max (range: 2-10, default: 5)
  -r, --random N   Number of random parameter combinations to generate (default: 20)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import re
import random
import subprocess
import argparse
from matplotlib.backends.backend_pdf import PdfPages

# Import the PARSEC to DAT conversion functionality from the previous script
from parsec_to_dat import ParsecAirfoil

# Define directories
STATS_FILE = "parsec_results/parsec_stats.csv"
OUTPUT_DIR_PARSEC = "airfoils_generated_parsec"
OUTPUT_DIR_DAT = "airfoils_generated_dat"
RESULTS_DIR = "parameter_sweep_results"

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


def load_parameter_stats():
    """Load PARSEC parameter statistics from CSV file"""
    try:
        stats = pd.read_csv(STATS_FILE)
        return stats
    except FileNotFoundError:
        print(f"Error: Parameter statistics file not found: {STATS_FILE}")
        print("Make sure you have run parsec_fit.py first to generate statistics.")
        return None


def generate_parameter_variations(stats, steps=5, random_count=20):
    """Generate parameter variations using one-at-a-time and random approaches"""
    variations = []
    
    # Extract parameter names and their min, max, and mean values
    param_names = stats['Parameter'].tolist()
    min_values = stats['Min'].tolist()
    max_values = stats['Max'].tolist()
    mean_values = stats['Mean'].tolist()
    
    # Create a base configuration with all parameters at their mean values
    base_config = {param: mean for param, mean in zip(param_names, mean_values)}
    
    # Method 1: One-at-a-time parameter variations
    print(f"Generating one-at-a-time parameter variations ({steps} steps per parameter)...")
    for i, param in enumerate(param_names):
        param_min = min_values[i]
        param_max = max_values[i]
        
        # Create steps evenly spaced values from min to max
        param_values = np.linspace(param_min, param_max, steps)
        
        for value in param_values:
            # Create a new configuration based on the base config
            config = base_config.copy()
            config[param] = value
            
            # Create a name for this variation
            name = f"oaat_{param}_{value:.4f}"
            
            # Add to variations list
            variations.append((name, config))
    
    # Method 2: Random combinations within the parameter space
    print(f"Generating {random_count} random parameter combinations...")
    for i in range(random_count):
        # Create a new configuration with random values within bounds
        config = {}
        for j, param in enumerate(param_names):
            param_min = min_values[j]
            param_max = max_values[j]
            config[param] = param_min + random.random() * (param_max - param_min)
        
        # Create a name for this random variation
        name = f"random_{i+1:03d}"
        
        # Add to variations list
        variations.append((name, config))
    
    # Method 3: Add the mean configuration as a reference
    variations.append(("mean_config", base_config))
    
    print(f"Total variations generated: {len(variations)}")
    return variations


def create_parsec_files(variations):
    """Create PARSEC parameter files for each variation"""
    print(f"Generating PARSEC parameter files...")
    
    for name, config in tqdm(variations):
        # Create file path
        file_path = os.path.join(OUTPUT_DIR_PARSEC, f"{name}.parsec")
        
        # Write parameters to file
        with open(file_path, 'w') as f:
            f.write(f"# PARSEC parameters for {name}\n")
            for param, value in config.items():
                f.write(f"{param} = {value:.6f}\n")
    
    return [name for name, _ in variations]


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
    
    print(f"Conversion complete:")
    print(f"  - Successfully converted: {success_count}")
    print(f"  - Failed conversions: {failure_count}")


def create_parameter_study_visualizations(stats, variations):
    """Create visualizations of parameter variations"""
    print("Creating parameter study visualizations...")
    
    # Extract parameter names
    param_names = stats['Parameter'].tolist()
    
    # Group variations by parameter
    param_variations = {}
    for name, config in variations:
        if name.startswith("oaat_"):
            # Parse parameter name and value from the variation name
            parts = name.split('_')
            if len(parts) >= 3:
                param = parts[1]
                if param in param_names:
                    if param not in param_variations:
                        param_variations[param] = []
                    param_variations[param].append((name, config))
    
    # Create PDF with visualizations
    pdf_path = os.path.join(RESULTS_DIR, "parameter_study.pdf")
    with PdfPages(pdf_path) as pdf:
        # Create pages for each parameter
        for param in param_names:
            if param in param_variations:
                fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                fig.suptitle(f"Effect of {PARAM_DESCRIPTIONS.get(param, param)} ({param})", fontsize=16)
                
                axes = axes.flatten()
                
                # Sort variations by parameter value
                variations_sorted = sorted(param_variations[param], key=lambda x: [v for k, v in x[1].items() if k == param][0])
                
                for i, (name, config) in enumerate(variations_sorted):
                    if i < len(axes):
                        # Create airfoil and plot it
                        airfoil = ParsecAirfoil(name=name)
                        
                        # Set parameters directly
                        for k, v in config.items():
                            airfoil.params[k] = v
                        
                        # Generate airfoil shape
                        x_airfoil, y_airfoil = airfoil.generate_coordinates(200)
                        
                        # Plot
                        ax = axes[i]
                        ax.plot(x_airfoil, y_airfoil, 'b-', linewidth=2)
                        ax.set_aspect('equal')
                        ax.grid(True, alpha=0.3)
                        
                        # Get the varied parameter value
                        value = config[param]
                        ax.set_title(f"{param} = {value:.4f}")
                        
                        # Highlight the varied parameter effect
                        ymin, ymax = ax.get_ylim()
                        if param.startswith('X'):
                            ax.axvline(x=value, color='r', linestyle='--', alpha=0.5)
                        elif param.startswith('Y'):
                            ax.axhline(y=value, color='r', linestyle='--', alpha=0.5)
                
                # Turn off any unused subplots
                for i in range(len(variations_sorted), len(axes)):
                    axes[i].axis('off')
                
                plt.tight_layout(rect=[0, 0, 1, 0.95])
                pdf.savefig(fig)
                plt.close(fig)
        
        # Create a reference page
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.axis('off')
        ax.set_title("PARSEC Parameter Study Reference", fontsize=16)
        
        # Create a table with parameter descriptions
        cells = []
        for param in param_names:
            min_val = stats[stats['Parameter'] == param]['Min'].values[0]
            max_val = stats[stats['Parameter'] == param]['Max'].values[0]
            mean_val = stats[stats['Parameter'] == param]['Mean'].values[0]
            cells.append([param, PARAM_DESCRIPTIONS.get(param, ""), 
                         f"{min_val:.4f}", f"{max_val:.4f}", f"{mean_val:.4f}"])
        
        # Create the table
        column_labels = ["Parameter", "Description", "Min Value", "Max Value", "Mean Value"]
        table = ax.table(cellText=cells, colLabels=column_labels, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        pdf.savefig(fig)
        plt.close(fig)
    
    # Create an overview image with sample airfoils
    fig, axes = plt.subplots(3, 4, figsize=(15, 10))
    fig.suptitle("Sample Airfoils from Parameter Study", fontsize=16)
    
    axes = axes.flatten()
    
    # Select a diverse set of airfoils
    diverse_airfoils = []
    for param in list(param_variations.keys())[:4]:  # Take first 4 parameters
        if param in param_variations and len(param_variations[param]) >= 3:
            # Add the min, middle, and max variations
            variations_sorted = sorted(param_variations[param], key=lambda x: [v for k, v in x[1].items() if k == param][0])
            diverse_airfoils.extend([variations_sorted[0], variations_sorted[len(variations_sorted)//2], variations_sorted[-1]])
    
    # If we need more airfoils, add random ones
    for name, _ in variations:
        if name.startswith("random_"):
            diverse_airfoils.append((name, dict()))
        if len(diverse_airfoils) >= 12:
            break
    
    # Plot the diverse set
    for i, (name, config) in enumerate(diverse_airfoils[:len(axes)]):
        ax = axes[i]
        
        # Load the airfoil
        parsec_file = os.path.join(OUTPUT_DIR_PARSEC, f"{name}.parsec")
        airfoil = ParsecAirfoil(name=name)
        
        try:
            # Load parameters
            if airfoil.load_from_file(parsec_file):
                # Generate airfoil shape
                x_airfoil, y_airfoil = airfoil.generate_coordinates(200)
                
                # Plot
                ax.plot(x_airfoil, y_airfoil, 'b-', linewidth=2)
                ax.set_aspect('equal')
                ax.grid(True, alpha=0.3)
                ax.set_title(name)
        except:
            ax.text(0.5, 0.5, f"Failed to load {name}", ha='center', va='center')
    
    # Turn off any unused subplots
    for i in range(len(diverse_airfoils), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(RESULTS_DIR, "parameter_study_samples.png"), dpi=150)
    plt.close(fig)
    
    print(f"Visualizations created:")
    print(f"  - Parameter study PDF: {pdf_path}")
    print(f"  - Sample airfoils image: {os.path.join(RESULTS_DIR, 'parameter_study_samples.png')}")


def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Generate airfoil variations by sweeping PARSEC parameters from min to max."
    )
    
    parser.add_argument(
        "-s", "--steps", 
        type=int, 
        default=5,
        help="Number of steps for each parameter from min to max (range: 2-10, default: 5)"
    )
    
    parser.add_argument(
        "-r", "--random", 
        type=int, 
        default=20,
        help="Number of random parameter combinations to generate (default: 20)"
    )
    
    args = parser.parse_args()
    
    # Validate steps parameter
    if args.steps < 2 or args.steps > 10:
        parser.error("Steps must be between 2 and 10")
    
    # Validate random parameter
    if args.random < 0:
        parser.error("Random count must be non-negative")
    
    return args


def main():
    """Main function to execute the parameter study"""
    # Parse command-line arguments
    args = parse_arguments()
    
    print(f"Generating parameter variations with {args.steps} steps and {args.random} random combinations...\n")
    
    # Load parameter statistics
    stats = load_parameter_stats()
    if stats is None:
        return
    
    # Generate parameter variations
    variations = generate_parameter_variations(stats, steps=args.steps, random_count=args.random)
    
    # Create PARSEC parameter files
    airfoil_names = create_parsec_files(variations)
    
    # Convert PARSEC files to DAT files
    convert_to_dat_files(airfoil_names)
    
    # Create parameter study visualizations
    create_parameter_study_visualizations(stats, variations)
    
    print("\nParameter study complete!")
    print(f"PARSEC parameter files: {OUTPUT_DIR_PARSEC} ({len(airfoil_names)} files)")
    print(f"Airfoil DAT files: {OUTPUT_DIR_DAT}")
    print(f"Visualization results: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
