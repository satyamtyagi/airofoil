#!/usr/bin/env python3
"""
Simple 3D Visualization of Top Performing Airfoils (Real and Parameter Sweep)

This script creates a 3D visualization of both real and parameter sweep airfoils
and saves it as a standalone HTML file that can be opened directly in a browser.
Includes lift-to-drag ratio in the legend for easy performance comparison.
"""

import plotly.graph_objects as go
import plotly.io as pio
import numpy as np
import os
import webbrowser
import pandas as pd
import h5py
from threading import Timer

# Top performing airfoils based on L/D ratio
REAL_AIRFOILS = ['ag13', 'hs1606', 'davissm', 'a18sm', 'cr001sm'] # Top 5 from our analysis
VARIATION_AIRFOILS = [] # Will be populated dynamically from variation results file

# Directories containing airfoil dat files and performance data
REAL_AIRFOIL_DIR = "airfoils_uiuc"
VARIATION_AIRFOIL_DIR = "airfoils_variations_dat"
REAL_PERFORMANCE_FILE = "results/airfoil_best_performance.csv"
VARIATION_RESULTS_FILE = "variation_results/variation_results.h5"
OUTPUT_FILE = "visualization_output/unified_airfoils_variations_3d.html"

# Ensure output directory exists
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)


def load_performance_data():
    """Load performance data for both real and variation airfoils"""
    global VARIATION_AIRFOILS
    real_airfoil_data = {}
    variation_airfoil_data = {}
    
    # Load real airfoil data from CSV
    if os.path.exists(REAL_PERFORMANCE_FILE):
        try:
            df = pd.read_csv(REAL_PERFORMANCE_FILE)
            for _, row in df.iterrows():
                if row['airfoil'] in REAL_AIRFOILS:
                    real_airfoil_data[row['airfoil']] = row['best_ld']
        except Exception as e:
            print(f"Error loading real airfoil data: {str(e)}")
    
    # Load variation results data
    if os.path.exists(VARIATION_RESULTS_FILE):
        try:
            with h5py.File(VARIATION_RESULTS_FILE, 'r') as f:
                # Get all variation names and their L/D ratios
                variation_names = []
                ld_ratios = []
                
                for name in f.keys():
                    ld_ratio = f[name].attrs.get('ld_ratio', 0)
                    variation_names.append(name)
                    ld_ratios.append(ld_ratio)
                
                # Sort by L/D ratio
                sorted_indices = np.argsort(ld_ratios)[::-1]  # Descending order
                
                # Get top 5 variations
                top_indices = sorted_indices[:5]
                VARIATION_AIRFOILS = [variation_names[i] for i in top_indices]
                
                # Store L/D ratios
                for i, idx in enumerate(top_indices):
                    name = variation_names[idx]
                    variation_airfoil_data[name] = ld_ratios[idx]
                    print(f"{i+1}. {name}: L/D = {ld_ratios[idx]:.2f}")
            
        except Exception as e:
            print(f"Error loading variation results: {str(e)}")
    
    return real_airfoil_data, variation_airfoil_data


def read_airfoil_dat(filename):
    """Read airfoil coordinates from .dat file"""
    try:
        # Read the file
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        # Skip header lines
        data_lines = []
        for line in lines:
            # Skip empty lines and check if line contains two float values
            if line.strip() and len(line.strip().split()) >= 2:
                try:
                    x, y = map(float, line.strip().split()[:2])
                    data_lines.append((x, y))
                except ValueError:
                    # Skip lines that can't be parsed as two floats
                    continue
        
        # Convert to numpy arrays
        if data_lines:
            coords = np.array(data_lines)
            x_coords = coords[:, 0]
            y_coords = coords[:, 1]
            return x_coords, y_coords
        else:
            print(f"Warning: No valid coordinate data found in {filename}")
            return None, None
    
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return None, None

def create_3d_airfoil_figure():
    """Create a 3D figure of airfoil shapes"""
    # Load performance data for legend
    global real_airfoil_data, variation_airfoil_data
    real_airfoil_data, variation_airfoil_data = load_performance_data()
    
    # Create figure
    fig = go.Figure()
    
    # Define colors for real airfoils (warm colors)
    real_colors = ['red', 'orange', 'brown', 'darkred', 'coral']
    
    # Define colors for variation airfoils (cool colors)
    variation_colors = ['blue', 'cyan', 'teal', 'navy', 'turquoise']
    
    # Process each real airfoil
    print("Loading real airfoils...")
    for idx, airfoil_name in enumerate(REAL_AIRFOILS):
        # Determine color for this airfoil
        color = real_colors[idx % len(real_colors)]
        
        # Read airfoil coordinates
        airfoil_path = os.path.join(REAL_AIRFOIL_DIR, f"{airfoil_name}.dat")
        x_coords, y_coords = read_airfoil_dat(airfoil_path)
        
        if x_coords is None or y_coords is None:
            print(f"Skipping {airfoil_name} due to data loading error")
            continue
        
        # Create simple 3D visualization
        # Create a span for the wing
        span = 1.0  # 1 unit span
        z_spacing = 2.0  # Spacing between different airfoils
        z_offset = idx * z_spacing
        
        # Plot a 3D line for the airfoil profile at the root
        legend_name = f"Real: {airfoil_name}"
        if airfoil_name in real_airfoil_data:
            legend_name = f"Real: {airfoil_name} (L/D={real_airfoil_data[airfoil_name]:.2f})"
        
        fig.add_trace(go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=[z_offset] * len(x_coords),
            mode='lines',
            line=dict(color=color, width=5),
            name=legend_name
        ))
        
        # Plot a 3D line for the airfoil profile at the tip
        fig.add_trace(go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=[z_offset + span] * len(x_coords),
            mode='lines',
            line=dict(color=color, width=5),
            showlegend=False
        ))
        
        # Add mesh surface connecting root and tip profiles
        x_surf, y_surf, z_surf = [], [], []
        
        # Add first profile (at z=z_offset)
        for i in range(len(x_coords)):
            x_surf.append(x_coords[i])
            y_surf.append(y_coords[i])
            z_surf.append(z_offset)
        
        # Add second profile (at z=z_offset+span)
        for i in range(len(x_coords)-1, -1, -1):
            x_surf.append(x_coords[i])
            y_surf.append(y_coords[i])
            z_surf.append(z_offset + span)
        
        # Close the surface by adding the first point again
        x_surf.append(x_surf[0])
        y_surf.append(y_surf[0])
        z_surf.append(z_surf[0])
        
        # Add surface
        fig.add_trace(go.Mesh3d(
            x=x_surf,
            y=y_surf,
            z=z_surf,
            color=color,
            opacity=0.5,
            name=f"Real: {airfoil_name}",
            showlegend=False
        ))
    
    # Process each variation airfoil
    print("Loading variation airfoils...")
    for idx, airfoil_name in enumerate(VARIATION_AIRFOILS):
        # Determine color for this airfoil
        color = variation_colors[idx % len(variation_colors)]
        
        # Read airfoil coordinates
        airfoil_path = os.path.join(VARIATION_AIRFOIL_DIR, f"{airfoil_name}.dat")
        if not os.path.exists(airfoil_path):
            print(f"Warning: {airfoil_path} not found")
            continue
            
        x_coords, y_coords = read_airfoil_dat(airfoil_path)
        
        if x_coords is None or y_coords is None:
            print(f"Skipping {airfoil_name} due to data loading error")
            continue
        
        # Create simple 3D visualization
        # Create a span for the wing
        span = 1.0  # 1 unit span
        z_spacing = 2.0  # Spacing between different airfoils
        z_offset = (idx + len(REAL_AIRFOILS)) * z_spacing  # Offset after real airfoils
        
        # Plot a 3D line for the airfoil profile at the root
        legend_name = f"Variation: {airfoil_name}"
        if airfoil_name in variation_airfoil_data:
            legend_name = f"Variation: {airfoil_name} (L/D={variation_airfoil_data[airfoil_name]:.2f})"
        
        fig.add_trace(go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=[z_offset] * len(x_coords),
            mode='lines',
            line=dict(color=color, width=5),
            name=legend_name
        ))
        
        # Plot a 3D line for the airfoil profile at the tip
        fig.add_trace(go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=[z_offset + span] * len(x_coords),
            mode='lines',
            line=dict(color=color, width=5),
            showlegend=False
        ))
        
        # Add mesh surface connecting root and tip profiles
        x_surf, y_surf, z_surf = [], [], []
        
        # Add first profile (at z=z_offset)
        for i in range(len(x_coords)):
            x_surf.append(x_coords[i])
            y_surf.append(y_coords[i])
            z_surf.append(z_offset)
        
        # Add second profile (at z=z_offset+span)
        for i in range(len(x_coords)-1, -1, -1):
            x_surf.append(x_coords[i])
            y_surf.append(y_coords[i])
            z_surf.append(z_offset + span)
        
        # Close the surface by adding the first point again
        x_surf.append(x_surf[0])
        y_surf.append(y_surf[0])
        z_surf.append(z_surf[0])
        
        # Add surface
        fig.add_trace(go.Mesh3d(
            x=x_surf,
            y=y_surf,
            z=z_surf,
            color=color,
            opacity=0.5,
            name=f"Variation: {airfoil_name}",
            showlegend=False
        ))
    
    # Update 3D layout
    fig.update_layout(
        title="Unified 3D Visualization: Real vs Variation Airfoils",
        scene=dict(
            aspectmode='data',
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Airfoil Index')
        )
    )
    
    return fig

def open_browser():
    """Open the browser to view the visualization"""
    # Use a timer to allow the file to be saved first
    def open_file():
        webbrowser.open('file://' + os.path.abspath(OUTPUT_FILE), new=2)
    
    Timer(1.0, open_file).start()

def main():
    """Main function to generate and save the visualization"""
    print("\nGenerating unified 3D visualization of real and parameter sweep airfoils...")
    
    # Create the figure
    fig = create_3d_airfoil_figure()
    
    # Save the figure to an HTML file
    fig.write_html(OUTPUT_FILE, include_plotlyjs='cdn', full_html=True)
    print(f"Saved visualization to {OUTPUT_FILE}")
    
    # Open the browser to view the file
    print("Opening visualization in browser...")
    open_browser()
    
    print("Done!")

if __name__ == "__main__":
    main()
