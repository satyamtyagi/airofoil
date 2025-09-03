#!/usr/bin/env python3
"""
3D Visualization of Top Performing Airfoil Variations

This script creates a 3D visualization of the top 10 performing airfoil variations
from the surrogate model evaluation and saves it as a standalone HTML file.
"""

import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from pathlib import Path
import webbrowser
from threading import Timer

# Directories and files
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
RESULTS_FILE = BASE_DIR / "variation_results" / "surrogate_variation_results.csv"
DAT_DIR = BASE_DIR / "simple_variations_dat"
OUTPUT_FILE = BASE_DIR / "variation_results" / "top_variations_3d.html"
TOP_N = 10  # Number of top performers to visualize

def read_airfoil_dat(filename):
    """Read airfoil coordinates from .dat file"""
    try:
        # Read the file
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        # Skip header lines
        data_lines = []
        for line in lines[1:]:  # Skip the first line (header)
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

def get_top_performers():
    """Get the top N performers from the results CSV"""
    try:
        results_df = pd.read_csv(RESULTS_FILE)
        # Sort by lift-to-drag ratio (descending)
        results_df = results_df.sort_values('cl_cd', ascending=False)
        # Get top N
        top_performers = results_df.head(TOP_N)['airfoil'].tolist()
        top_data = results_df.head(TOP_N)
        return top_performers, top_data
    except Exception as e:
        print(f"Error loading results: {e}")
        return [], pd.DataFrame()

def create_3d_airfoil_figure(top_performers, top_data):
    """Create a 3D figure of airfoil shapes"""
    # Create figure
    fig = go.Figure()
    
    # Color palette for different airfoils - using a colorful spectrum
    colors = ['red', 'orange', 'gold', 'green', 'cyan', 'blue', 'indigo', 'violet', 'magenta', 'crimson']
    
    # Process each airfoil
    for idx, airfoil_name in enumerate(top_performers):
        # Determine color for this airfoil
        color = colors[idx % len(colors)]
        
        # Get performance data
        perf_row = top_data.iloc[idx]
        cl = perf_row['cl']
        cd = perf_row['cd']
        cl_cd = perf_row['cl_cd']
        
        # Read airfoil coordinates
        airfoil_path = DAT_DIR / f"{airfoil_name}.dat"
        x_coords, y_coords = read_airfoil_dat(airfoil_path)
        
        if x_coords is None or y_coords is None:
            print(f"Skipping {airfoil_name} due to data loading error")
            continue
        
        # Create simple 3D visualization
        # Create a span for the wing
        span = 1.0  # 1 unit span
        z_spacing = 2.0  # Spacing between different airfoils
        z_offset = idx * z_spacing
        
        # Add performance data to the name for the legend
        display_name = f"{idx+1}. {airfoil_name}: L/D={cl_cd:.2f} (CL={cl:.4f}, CD={cd:.6f})"
        
        # Plot a 3D line for the airfoil profile at the root
        fig.add_trace(go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=[z_offset] * len(x_coords),
            mode='lines',
            line=dict(color=color, width=5),
            name=display_name
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
        
        # Create a mesh surface connecting root and tip profiles
        x_surface = []
        y_surface = []
        z_surface = []
        
        for i in range(len(x_coords)):
            # Add point at root
            x_surface.append(x_coords[i])
            y_surface.append(y_coords[i])
            z_surface.append(z_offset)
            
            # Add point at tip
            x_surface.append(x_coords[i])
            y_surface.append(y_coords[i])
            z_surface.append(z_offset + span)
            
        fig.add_trace(go.Mesh3d(
            x=x_surface,
            y=y_surface,
            z=z_surface,
            opacity=0.7,
            color=color,
            showlegend=False
        ))
    
    # Set up the layout for better visualization
    fig.update_layout(
        title=f"Top {TOP_N} Airfoil Variations by Lift-to-Drag Ratio",
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'  # Preserve the shape of the airfoils
        ),
        width=1000,
        height=800
    )
    
    # Add buttons to control the view
    camera = dict(
        up=dict(x=0, y=1, z=0),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=2, y=0, z=0)  # Side view
    )
    
    # Add buttons for different views
    fig.update_layout(
        updatemenus=[
            dict(
                type='buttons',
                showactive=False,
                buttons=[
                    dict(
                        label='Side View',
                        method='relayout',
                        args=['scene.camera', dict(
                            up=dict(x=0, y=1, z=0),
                            center=dict(x=0, y=0, z=0),
                            eye=dict(x=2, y=0, z=0)
                        )]
                    ),
                    dict(
                        label='Top View',
                        method='relayout',
                        args=['scene.camera', dict(
                            up=dict(x=0, y=0, z=-1),
                            center=dict(x=0, y=0, z=0),
                            eye=dict(x=0, y=1.5, z=0)
                        )]
                    ),
                    dict(
                        label='Front View',
                        method='relayout',
                        args=['scene.camera', dict(
                            up=dict(x=0, y=1, z=0),
                            center=dict(x=0, y=0, z=0),
                            eye=dict(x=0, y=0, z=2)
                        )]
                    ),
                    dict(
                        label='Isometric View',
                        method='relayout',
                        args=['scene.camera', dict(
                            up=dict(x=0, y=1, z=0),
                            center=dict(x=0, y=0, z=0),
                            eye=dict(x=1.25, y=1.25, z=1.25)
                        )]
                    )
                ]
            )
        ]
    )
    
    return fig

def save_figure(fig):
    """Save the figure as a standalone HTML file"""
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    # Save as HTML
    pio.write_html(fig, file=OUTPUT_FILE, auto_open=False)
    print(f"Saved visualization to {OUTPUT_FILE}")

def open_browser():
    """Open the browser to view the visualization"""
    # Use a timer to open browser after a short delay to ensure file is saved
    def open_html():
        webbrowser.open(f'file://{OUTPUT_FILE}', new=2)
        
    Timer(1.0, open_html).start()

def main():
    """Main function to generate and save visualization"""
    print("Creating 3D visualization of top airfoil variations...")
    
    # Get top performers
    top_performers, top_data = get_top_performers()
    
    if not top_performers:
        print("Error: No top performers found.")
        return
    
    print(f"Visualizing top {len(top_performers)} airfoil variations...")
    
    # Create the 3D figure
    fig = create_3d_airfoil_figure(top_performers, top_data)
    
    # Save the figure
    save_figure(fig)
    
    # Open browser to view the visualization
    open_browser()
    
    print("Done!")

if __name__ == "__main__":
    main()
