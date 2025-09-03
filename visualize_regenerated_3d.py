#!/usr/bin/env python3
"""
Simple 3D Visualization of Regenerated Airfoils

This script creates a 3D visualization of the regenerated airfoils
and saves it as a standalone HTML file that can be opened directly in a browser.
"""

import plotly.graph_objects as go
import plotly.io as pio
import numpy as np
import os
import webbrowser
from threading import Timer

# Top performing airfoils based on L/D ratio
TOP_PERFORMERS = ['ag13', 'a18sm', 'ag26', 'ag25', 'ag27']

# Directory containing regenerated airfoil dat files
AIRFOIL_DIR = "regenerated_dat"
OUTPUT_FILE = "regenerated_airfoils_3d.html"

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
    # Create figure
    fig = go.Figure()
    
    # Color palette for different airfoils
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    # Process each airfoil
    for idx, airfoil_name in enumerate(TOP_PERFORMERS):
        # Determine color for this airfoil
        color = colors[idx % len(colors)]
        
        # Read airfoil coordinates
        airfoil_path = os.path.join(AIRFOIL_DIR, f"{airfoil_name}_regenerated.dat")
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
        fig.add_trace(go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=[z_offset] * len(x_coords),
            mode='lines',
            line=dict(color=color, width=5),
            name=f"{airfoil_name} (Regenerated)"
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
        
        # Connect the root and tip profiles with lines to create a wireframe
        num_points = len(x_coords)
        sample_points = range(0, num_points, max(1, num_points // 20))  # Sample some points to avoid too many lines
        
        for i in sample_points:
            fig.add_trace(go.Scatter3d(
                x=[x_coords[i], x_coords[i]],
                y=[y_coords[i], y_coords[i]],
                z=[z_offset, z_offset + span],
                mode='lines',
                line=dict(color=color, width=2),
                showlegend=False
            ))
    
    # Set up the layout
    fig.update_layout(
        title='3D Visualization of Regenerated Airfoils',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        width=1000,
        height=800
    )
    
    return fig

def open_browser():
    """Open the browser to view the visualization"""
    webbrowser.open('file://' + os.path.realpath(OUTPUT_FILE))

def main():
    """Main function to generate and save the visualization"""
    print("Creating 3D visualization of regenerated airfoils...")
    fig = create_3d_airfoil_figure()
    
    print(f"Saving visualization to {OUTPUT_FILE}...")
    pio.write_html(fig, file=OUTPUT_FILE, auto_open=False)
    
    print(f"Opening visualization in browser...")
    Timer(1.0, open_browser).start()
    
    print("Done!")

if __name__ == "__main__":
    main()
