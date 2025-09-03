#!/usr/bin/env python3
"""
3D Visualization of Top Performing Airfoils

This script creates an interactive 3D visualization dashboard showing the shapes
and performance metrics of the top-performing airfoils from the analysis.
"""

import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os
import dash
from dash import html, dcc
from dash.dependencies import Input, Output
import webbrowser
from threading import Timer

# Top performing airfoils based on L/D ratio
TOP_PERFORMERS = ['ag13', 'a18sm', 'ag26', 'ag25', 'ag27']

# Directory containing airfoil dat files
AIRFOIL_DIR = "airfoils_uiuc"
RESULTS_DIR = "results"

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

def get_airfoil_performance(airfoil_name):
    """Get airfoil performance data from the result files"""
    try:
        # Try to parse from the individual results file
        results_file = os.path.join(RESULTS_DIR, f"{airfoil_name}_results.txt")
        if os.path.exists(results_file):
            print(f"Loading performance data for {airfoil_name} from {results_file}")
            with open(results_file, 'r') as f:
                lines = f.readlines()
            
            # Initialize data dictionary
            data = {'airfoil': airfoil_name, 'angles': [], 'cl': [], 'cd': [], 'ld': [], 'cm': []}
            
            # Parse data lines
            for i, line in enumerate(lines):
                # Skip header lines
                if i < 4 or not line.strip():
                    continue
                    
                # Parse data line
                parts = line.strip().split()
                if len(parts) >= 5:
                    try:
                        alpha = float(parts[0])
                        cl = float(parts[1])
                        cd = float(parts[2])
                        cm = float(parts[4])
                        ld = cl / cd if cd != 0 else 0
                        
                        data['angles'].append(alpha)
                        data['cl'].append(cl)
                        data['cd'].append(cd)
                        data['ld'].append(ld)
                        data['cm'].append(cm)
                    except (ValueError, IndexError) as e:
                        print(f"Error parsing line: {line.strip()} - {e}")
            
            # Find best L/D and corresponding angle
            if data['ld']:
                best_ld_idx = np.argmax(data['ld'])
                data['best_ld'] = data['ld'][best_ld_idx]
                data['best_ld_angle'] = data['angles'][best_ld_idx]
                data['best_cl'] = data['cl'][best_ld_idx]
                data['best_cd'] = data['cd'][best_ld_idx]
                
                print(f"Successfully loaded performance data for {airfoil_name} - Best L/D: {data['best_ld']:.2f} at {data['best_ld_angle']}°")
                return data
            else:
                print(f"No valid performance data found for {airfoil_name}")
                
        else:
            print(f"Results file not found for {airfoil_name}: {results_file}")
    
    except Exception as e:
        print(f"Error getting performance data for {airfoil_name}: {e}")
    
    return None

def create_3d_airfoil_figure(airfoil_names):
    """Create a 3D figure of airfoil shapes with performance metrics"""
    # Create figure
    fig = go.Figure()
    
    # Color palette for different airfoils
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'pink', 'brown']
    
    # Process each airfoil
    for idx, airfoil_name in enumerate(airfoil_names):
        # Determine color for this airfoil
        color = colors[idx % len(colors)]
        
        # Read airfoil coordinates
        airfoil_path = os.path.join(AIRFOIL_DIR, f"{airfoil_name}.dat")
        x_coords, y_coords = read_airfoil_dat(airfoil_path)
        
        if x_coords is None or y_coords is None:
            print(f"Skipping {airfoil_name} due to data loading error")
            continue
        
        # Get performance data
        perf_data = get_airfoil_performance(airfoil_name)
        if perf_data:
            ld_ratio = perf_data.get('best_ld', 0)
            ld_angle = perf_data.get('best_ld_angle', 0)
            cl = perf_data.get('best_cl', 0)
            cd = perf_data.get('best_cd', 0)
        else:
            ld_ratio = 0
            ld_angle = 0
            cl = 0
            cd = 0
        
        # Create 3D wing by extruding airfoil profile
        span = 1.0
        num_sections = 20
        z_positions = np.linspace(0, span, num_sections)
        z_offset = idx * span * 1.5  # Separate different airfoils along z-axis
        
        # Create wing mesh
        for i in range(len(z_positions)-1):
            z1 = z_positions[i] + z_offset
            z2 = z_positions[i+1] + z_offset
            
            # Create faces connecting consecutive airfoil sections
            for j in range(len(x_coords)-1):
                # Define the 4 corners of a quad face
                x_quad = [x_coords[j], x_coords[j+1], x_coords[j+1], x_coords[j]]
                y_quad = [y_coords[j], y_coords[j+1], y_coords[j+1], y_coords[j]]
                z_quad = [z1, z1, z2, z2]
                
                # Plot this quad face
                fig.add_trace(go.Mesh3d(
                    x=x_quad,
                    y=y_quad,
                    z=z_quad,
                    color=color,
                    opacity=0.7,
                    name=f"{airfoil_name} (L/D={ld_ratio:.1f} at {ld_angle}°)"
                ))
        
        # Add a line for the leading edge
        idx_le = np.argmin(x_coords)
        le_x = x_coords[idx_le]
        le_y = y_coords[idx_le]
        fig.add_trace(go.Scatter3d(
            x=[le_x] * num_sections,
            y=[le_y] * num_sections,
            z=z_positions + z_offset,
            mode='lines',
            line=dict(color='black', width=5),
            showlegend=False
        ))
        
        # Add a label for the airfoil
        fig.add_trace(go.Scatter3d(
            x=[np.mean(x_coords)],
            y=[np.mean(y_coords)],
            z=[z_offset + span/2],
            mode='text',
            text=f"{airfoil_name}<br>L/D={ld_ratio:.1f} at {ld_angle}°",
            textposition='middle center',
            textfont=dict(size=12, color=color),
            showlegend=False
        ))
    
    # Set up the layout
    fig.update_layout(
        title='3D Visualization of Top Performing Airfoils',
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
        )
    )
    
    return fig

def create_performance_comparison():
    """Create a comparison figure of performance metrics across angles of attack"""
    fig = go.Figure()
    
    # Color palette for different airfoils
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'pink', 'brown']
    
    # Process each airfoil
    for idx, airfoil_name in enumerate(TOP_PERFORMERS):
        # Get performance data
        perf_data = get_airfoil_performance(airfoil_name)
        if not perf_data or 'angles' not in perf_data:
            print(f"Skipping {airfoil_name} due to missing performance data")
            continue
        
        # Extract performance metrics across angles
        angles = perf_data['angles']
        ld_values = perf_data['ld']
        
        # Plot L/D ratio vs angle of attack
        fig.add_trace(go.Scatter(
            x=angles,
            y=ld_values,
            mode='lines+markers',
            name=f"{airfoil_name}",
            line=dict(color=colors[idx % len(colors)], width=2),
            marker=dict(size=8)
        ))
    
    fig.update_layout(
        title='Lift-to-Drag Ratio vs Angle of Attack',
        xaxis_title='Angle of Attack (degrees)',
        yaxis_title='Lift-to-Drag Ratio',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig

def open_browser():
    """Open the browser after a short delay"""
    Timer(1.5, lambda: webbrowser.open('http://127.0.0.1:8050')).start()

def run_dashboard():
    """Run the interactive dashboard"""
    # Initialize Dash app
    app = dash.Dash(__name__, suppress_callback_exceptions=True)
    
    # Create 3D airfoil figure
    fig_3d = create_3d_airfoil_figure(TOP_PERFORMERS)
    
    # Create performance comparison
    fig_performance = create_performance_comparison()
    
    # Define layout
    app.layout = html.Div([
        html.H1("Top Performing Airfoil Analysis - 3D Visualization"),
        
        html.Div([
            html.Div([
                html.H2("3D Airfoil Shapes"),
                html.P("Drag to rotate. Scroll to zoom."),
                dcc.Graph(figure=fig_3d, style={'height': '700px'})
            ], style={'width': '100%', 'display': 'inline-block', 'padding': '0 20px'}),
            
            html.Div([
                html.H2("Performance Comparison"),
                dcc.Graph(figure=fig_performance)
            ], style={'width': '100%', 'display': 'inline-block', 'padding': '0 20px', 'marginTop': '30px'})
        ])
    ])
    
    # Open browser
    open_browser()
    
    # Run the server
    app.run(debug=False)

if __name__ == "__main__":
    print("Starting 3D visualization dashboard...")
    run_dashboard()
