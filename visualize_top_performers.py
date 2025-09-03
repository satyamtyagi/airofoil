#!/usr/bin/env python3
"""
Visualize Top Performing Airfoils

This script creates visualizations comparing the top-performing airfoils
based on their PARSEC parameters and performance metrics.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from parsec_fit import ParsecAirfoil
import matplotlib.gridspec as gridspec

# Top performing airfoils and their lift-to-drag ratios
TOP_PERFORMERS = {
    "ag13": {"L/D": 375.82, "alpha": 10},
    "a18sm": {"L/D": 64.38, "alpha": 5},
    "ag26": {"L/D": 53.83, "alpha": 5},
    "ag25": {"L/D": 53.45, "alpha": 5},
    "ag27": {"L/D": 53.23, "alpha": 5}
}

# Directories
PARSEC_DIR = "airfoils_parsec"
RESULTS_DIR = "results"
OUTPUT_DIR = "visualization_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_parsec_parameters(airfoil_name):
    """Load PARSEC parameters for a given airfoil"""
    param_file = os.path.join(PARSEC_DIR, f"{airfoil_name}.parsec")
    params = {}
    
    with open(param_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if '=' in line and not line.startswith('#'):
                key, value = line.strip().split('=')
                params[key.strip()] = float(value.strip())
    
    return params

def generate_airfoil_coordinates(params):
    """Generate airfoil coordinates from PARSEC parameters"""
    airfoil = ParsecAirfoil()
    
    # Set parameters
    for param, value in params.items():
        airfoil.params[param] = value
    
    # Calculate coefficients
    airfoil._calculate_coefficients()
    
    # Generate x-coordinates (from 0 to 1)
    x_coords = np.linspace(0, 1, 100)
    
    # Calculate upper and lower surface y-coordinates
    y_upper, y_lower = airfoil.evaluate(x_coords)
    
    # Combine coordinates to form a closed shape
    # Starting from trailing edge, go over the upper surface to the leading edge,
    # then from leading edge along the lower surface back to trailing edge
    x_combined = np.concatenate([x_coords, np.flip(x_coords)])
    y_combined = np.concatenate([y_upper, np.flip(y_lower)])
    
    return x_combined, y_combined

def load_performance_data(airfoil_name):
    """Load performance data for a given airfoil"""
    perf_file = os.path.join(RESULTS_DIR, f"{airfoil_name}_results.txt")
    
    alpha = []
    cl = []
    cd = []
    ld = []
    cm = []
    
    with open(perf_file, 'r') as f:
        lines = f.readlines()
        for line in lines[4:]:  # Skip header lines
            if line.strip():
                parts = line.strip().split()
                alpha.append(float(parts[0]))
                cl.append(float(parts[1]))
                cd.append(float(parts[2]))
                ld.append(float(parts[1]) / float(parts[2]))
                cm.append(float(parts[4]))
    
    return {
        'alpha': alpha,
        'cl': cl,
        'cd': cd,
        'ld': ld,
        'cm': cm
    }

def plot_airfoil_shapes():
    """Plot shapes of top performing airfoils"""
    plt.figure(figsize=(12, 8))
    
    for i, (airfoil, info) in enumerate(TOP_PERFORMERS.items()):
        # Load parameters and generate coordinates
        params = load_parsec_parameters(airfoil)
        x_coords, y_coords = generate_airfoil_coordinates(params)
        
        # Plot airfoil shape
        plt.plot(x_coords, y_coords, label=f"{airfoil} (L/D={info['L/D']:.1f} at {info['alpha']}°)")
    
    plt.title('Top Performing Airfoil Shapes')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('equal')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best')
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, 'top_airfoil_shapes.png')
    plt.savefig(output_path, dpi=300)
    print(f"Saved airfoil shapes comparison to {output_path}")
    plt.close()

def plot_parameter_comparison():
    """Create a bar chart comparing PARSEC parameters of top airfoils"""
    # Define key parameters to compare
    key_params = ['rLE', 'Xup', 'Yup', 'YXXup', 'Xlo', 'Ylo', 'YXXlo']
    
    # Load parameters for each airfoil
    param_data = {}
    for airfoil in TOP_PERFORMERS.keys():
        param_data[airfoil] = load_parsec_parameters(airfoil)
    
    # Create plot
    fig, axes = plt.subplots(len(key_params), 1, figsize=(12, 14), sharex=True)
    fig.suptitle('PARSEC Parameter Comparison of Top Airfoils', fontsize=16)
    
    airfoils = list(TOP_PERFORMERS.keys())
    x = np.arange(len(airfoils))
    width = 0.7
    
    for i, param in enumerate(key_params):
        values = [param_data[airfoil].get(param, 0) for airfoil in airfoils]
        bars = axes[i].bar(x, values, width, label=param)
        axes[i].set_ylabel(param)
        axes[i].set_title(f'{param} Comparison')
        axes[i].grid(True, linestyle='--', alpha=0.3)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Set x-tick labels with L/D ratio
    labels = [f"{airfoil}\nL/D={info['L/D']:.1f}" for airfoil, info in TOP_PERFORMERS.items()]
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    output_path = os.path.join(OUTPUT_DIR, 'parameter_comparison.png')
    plt.savefig(output_path, dpi=300)
    print(f"Saved parameter comparison to {output_path}")
    plt.close()

def plot_performance_comparison():
    """Plot performance comparison across angles of attack"""
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2)
    
    ax1 = plt.subplot(gs[0, 0])  # CL plot
    ax2 = plt.subplot(gs[0, 1])  # CD plot
    ax3 = plt.subplot(gs[1, :])  # L/D plot
    
    markers = ['o', 's', 'D', '^', 'v']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, (airfoil, info) in enumerate(TOP_PERFORMERS.items()):
        perf_data = load_performance_data(airfoil)
        
        # Plot lift coefficient
        ax1.plot(perf_data['alpha'], perf_data['cl'], marker=markers[i], 
                 color=colors[i], label=f"{airfoil} (Best L/D={info['L/D']:.1f})")
        
        # Plot drag coefficient
        ax2.plot(perf_data['alpha'], perf_data['cd'], marker=markers[i], 
                 color=colors[i], label=f"{airfoil}")
        
        # Plot lift-to-drag ratio
        ax3.plot(perf_data['alpha'], perf_data['ld'], marker=markers[i], 
                 color=colors[i], linewidth=2, label=f"{airfoil}")
    
    # Customize lift coefficient plot
    ax1.set_title('Lift Coefficient vs Angle of Attack')
    ax1.set_xlabel('Angle of Attack (°)')
    ax1.set_ylabel('Lift Coefficient (CL)')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()
    
    # Customize drag coefficient plot
    ax2.set_title('Drag Coefficient vs Angle of Attack')
    ax2.set_xlabel('Angle of Attack (°)')
    ax2.set_ylabel('Drag Coefficient (CD)')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()
    
    # Customize lift-to-drag ratio plot
    ax3.set_title('Lift-to-Drag Ratio vs Angle of Attack')
    ax3.set_xlabel('Angle of Attack (°)')
    ax3.set_ylabel('Lift-to-Drag Ratio (L/D)')
    ax3.grid(True, linestyle='--', alpha=0.7)
    ax3.legend(loc='upper left')
    
    # Mark the best L/D points
    for i, (airfoil, info) in enumerate(TOP_PERFORMERS.items()):
        alpha = info['alpha']
        perf_data = load_performance_data(airfoil)
        alpha_idx = perf_data['alpha'].index(alpha) if alpha in perf_data['alpha'] else None
        
        if alpha_idx is not None:
            ld_value = perf_data['ld'][alpha_idx]
            ax3.plot([alpha], [ld_value], 'o', markersize=10, 
                     markerfacecolor='none', markeredgewidth=2, 
                     markeredgecolor=colors[i])
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'performance_comparison.png')
    plt.savefig(output_path, dpi=300)
    print(f"Saved performance comparison to {output_path}")
    plt.close()

if __name__ == "__main__":
    plot_airfoil_shapes()
    plot_parameter_comparison()
    plot_performance_comparison()
    print("All visualization completed!")
