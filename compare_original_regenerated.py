#!/usr/bin/env python3
"""
Compare Original and Regenerated Airfoils

This script visualizes both the original and PARSEC-regenerated airfoil shapes
to verify the accuracy of the bidirectional conversion process.
"""

import os
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from threading import Timer
import webbrowser

# Top performing airfoils
TOP_PERFORMERS = ['ag13', 'a18sm', 'ag26', 'ag25', 'ag27']

# Directories
ORIGINAL_DIR = "airfoils_uiuc"
REGENERATED_DIR = "regenerated_dat"
OUTPUT_FILE = "comparison_airfoils_3d.html"

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

def create_comparison_figure():
    """Create a 3D figure comparing original and regenerated airfoil shapes"""
    # Create figure
    fig = go.Figure()
    
    # Color palette
    original_colors = ['red', 'blue', 'green', 'orange', 'purple']
    regenerated_colors = ['darkred', 'darkblue', 'darkgreen', 'darkorange', 'darkviolet']
    
    # Z spacing
    z_spacing = 2.0
    
    # Process each airfoil
    for idx, airfoil_name in enumerate(TOP_PERFORMERS):
        # Z position for this airfoil
        z_offset = idx * z_spacing
        
        # Read original airfoil coordinates
        original_path = os.path.join(ORIGINAL_DIR, f"{airfoil_name}.dat")
        x_original, y_original = read_airfoil_dat(original_path)
        
        if x_original is None:
            print(f"Skipping original {airfoil_name} due to data loading error")
            continue
            
        # Read regenerated airfoil coordinates
        regenerated_path = os.path.join(REGENERATED_DIR, f"{airfoil_name}_regenerated.dat")
        x_regenerated, y_regenerated = read_airfoil_dat(regenerated_path)
        
        if x_regenerated is None:
            print(f"Skipping regenerated {airfoil_name} due to data loading error")
            continue
        
        # Plot original airfoil profile
        fig.add_trace(go.Scatter3d(
            x=x_original,
            y=y_original,
            z=[z_offset] * len(x_original),
            mode='lines',
            line=dict(color=original_colors[idx % len(original_colors)], width=4),
            name=f"{airfoil_name} (Original)"
        ))
        
        # Plot regenerated airfoil profile
        fig.add_trace(go.Scatter3d(
            x=x_regenerated,
            y=y_regenerated,
            z=[z_offset + 0.5] * len(x_regenerated),  # Slight z-offset for visibility
            mode='lines',
            line=dict(color=regenerated_colors[idx % len(regenerated_colors)], width=2, dash='dot'),
            name=f"{airfoil_name} (Regenerated)"
        ))

    # Calculate squared error between original and regenerated for each airfoil
    error_text = []
    for airfoil_name in TOP_PERFORMERS:
        # Read coordinates
        original_path = os.path.join(ORIGINAL_DIR, f"{airfoil_name}.dat")
        regenerated_path = os.path.join(REGENERATED_DIR, f"{airfoil_name}_regenerated.dat")
        
        x_orig, y_orig = read_airfoil_dat(original_path)
        x_regen, y_regen = read_airfoil_dat(regenerated_path)
        
        if x_orig is None or x_regen is None:
            error_text.append(f"{airfoil_name}: Error reading coordinates")
            continue
            
        # For proper comparison, we need to interpolate to common x-coordinates
        # This is a simplified approach - in practice, more sophisticated comparison might be needed
        # Using 100 points between 0 and 1
        x_common = np.linspace(0, 1, 100)
        
        # Simple linear interpolation for both upper and lower surfaces
        # First split the airfoils into upper and lower surfaces
        def split_airfoil(x, y):
            # Find leading edge (min x)
            le_idx = np.argmin(x)
            
            # Upper surface (from TE to LE)
            if le_idx > 0:
                upper_x = x[:le_idx+1]
                upper_y = y[:le_idx+1]
            else:
                upper_x = np.array([x[0]])
                upper_y = np.array([y[0]])
                
            # Lower surface (from LE to TE)
            if le_idx < len(x) - 1:
                lower_x = x[le_idx:]
                lower_y = y[le_idx:]
            else:
                lower_x = np.array([x[-1]])
                lower_y = np.array([y[-1]])
                
            return upper_x, upper_y, lower_x, lower_y
        
        orig_up_x, orig_up_y, orig_lo_x, orig_lo_y = split_airfoil(x_orig, y_orig)
        regen_up_x, regen_up_y, regen_lo_x, regen_lo_y = split_airfoil(x_regen, y_regen)
        
        # For proper comparison, ensure x coordinates are in the right order
        if len(orig_up_x) > 1 and orig_up_x[0] > orig_up_x[-1]:
            orig_up_x = orig_up_x[::-1]
            orig_up_y = orig_up_y[::-1]
        
        if len(orig_lo_x) > 1 and orig_lo_x[0] > orig_lo_x[-1]:
            orig_lo_x = orig_lo_x[::-1]
            orig_lo_y = orig_lo_y[::-1]
            
        if len(regen_up_x) > 1 and regen_up_x[0] > regen_up_x[-1]:
            regen_up_x = regen_up_x[::-1]
            regen_up_y = regen_up_y[::-1]
        
        if len(regen_lo_x) > 1 and regen_lo_x[0] > regen_lo_x[-1]:
            regen_lo_x = regen_lo_x[::-1]
            regen_lo_y = regen_lo_y[::-1]
        
        # Interpolate to common x-coordinates
        import scipy.interpolate as interp
        
        try:
            orig_up_interp = interp.interp1d(orig_up_x, orig_up_y, bounds_error=False, fill_value="extrapolate")
            orig_lo_interp = interp.interp1d(orig_lo_x, orig_lo_y, bounds_error=False, fill_value="extrapolate")
            regen_up_interp = interp.interp1d(regen_up_x, regen_up_y, bounds_error=False, fill_value="extrapolate")
            regen_lo_interp = interp.interp1d(regen_lo_x, regen_lo_y, bounds_error=False, fill_value="extrapolate")
            
            # Generate y-coordinates at common x-points
            orig_up_y_common = orig_up_interp(x_common)
            orig_lo_y_common = orig_lo_interp(x_common)
            regen_up_y_common = regen_up_interp(x_common)
            regen_lo_y_common = regen_lo_interp(x_common)
            
            # Calculate mean squared error
            upper_mse = np.mean((orig_up_y_common - regen_up_y_common) ** 2)
            lower_mse = np.mean((orig_lo_y_common - regen_lo_y_common) ** 2)
            total_mse = (upper_mse + lower_mse) / 2
            
            error_text.append(f"{airfoil_name}: Mean Squared Error = {total_mse:.8f}")
        except Exception as e:
            error_text.append(f"{airfoil_name}: Error calculating MSE - {str(e)}")
    
    # Set up the layout
    fig.update_layout(
        title='Comparison of Original vs Regenerated Airfoils',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=50),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        width=1000,
        height=800
    )
    
    # Add annotation with error information
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.5, y=0,
        text="<br>".join(error_text),
        showarrow=False,
        font=dict(size=12),
        bgcolor="rgba(255, 255, 255, 0.7)",
        bordercolor="black",
        borderwidth=1,
        borderpad=4,
        align="left"
    )
    
    return fig

def open_browser():
    """Open the browser to view the visualization"""
    webbrowser.open('file://' + os.path.realpath(OUTPUT_FILE))

def main():
    """Main function to generate and save the visualization"""
    print("Creating comparison visualization...")
    fig = create_comparison_figure()
    
    print(f"Saving visualization to {OUTPUT_FILE}...")
    pio.write_html(fig, file=OUTPUT_FILE, auto_open=False)
    
    print(f"Opening visualization in browser...")
    Timer(1.0, open_browser).start()
    
    print("Done!")

if __name__ == "__main__":
    main()
