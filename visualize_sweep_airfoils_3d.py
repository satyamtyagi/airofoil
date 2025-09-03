#!/usr/bin/env python3
"""
3D Visualization of Parameter Sweep Airfoils

This script creates an interactive 3D visualization of the top-performing
airfoils from the parameter sweep, allowing comparison of their shapes
and performance metrics.
"""

import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html
import re

# Define directories and files
SWEEP_AIRFOILS_DIR = "parameter_sweep_top_airfoils"
SWEEP_RESULTS_FILE = "parameter_sweep_results/parsec_sweep_results_success.csv"
OUTPUT_HTML = "visualization_output/parameter_sweep_airfoils_3d.html"

# Ensure output directory exists
os.makedirs("visualization_output", exist_ok=True)

# Colors for different airfoils
COLORS = ['rgb(31, 119, 180)', 'rgb(255, 127, 14)', 'rgb(44, 160, 44)',
          'rgb(214, 39, 40)', 'rgb(148, 103, 189)', 'rgb(140, 86, 75)',
          'rgb(227, 119, 194)', 'rgb(127, 127, 127)', 'rgb(188, 189, 34)']


def read_airfoil_dat(filename):
    """Read airfoil coordinates from a .dat file"""
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Extract the airfoil name from the first line
    name = lines[0].strip()
    
    # Skip any header lines
    data_start = 1
    while data_start < len(lines) and not re.match(r'^[\s]*[0-9]', lines[data_start]):
        data_start += 1
    
    # Read coordinates
    coordinates = []
    for line in lines[data_start:]:
        try:
            x, y = line.strip().split()
            coordinates.append((float(x), float(y)))
        except ValueError:
            continue
    
    if not coordinates:
        return None, None, None
    
    # Convert to numpy arrays
    coords = np.array(coordinates)
    x_coords = coords[:, 0]
    y_coords = coords[:, 1]
    
    return name, x_coords, y_coords


def load_airfoil_data():
    """Load airfoil coordinates from .dat files and performance data from CSV"""
    airfoils = {}
    
    # Find all .dat files in the directory
    dat_files = [f for f in os.listdir(SWEEP_AIRFOILS_DIR) if f.endswith('.dat')]
    
    if not dat_files:
        print(f"No .dat files found in {SWEEP_AIRFOILS_DIR}")
        return None
    
    print(f"Loading {len(dat_files)} airfoil files...")
    
    # Load coordinates for each airfoil
    for dat_file in dat_files:
        file_path = os.path.join(SWEEP_AIRFOILS_DIR, dat_file)
        name, x_coords, y_coords = read_airfoil_dat(file_path)
        
        if x_coords is None:
            print(f"  Failed to read coordinates from {dat_file}")
            continue
        
        # Extract rank number from filename (sweep_topN.dat)
        rank_match = re.search(r'top(\d+)', dat_file)
        if rank_match:
            rank = int(float(rank_match.group(1)))
            name = f"Rank #{rank}"
        else:
            name = os.path.splitext(dat_file)[0]
        
        airfoils[name] = {
            'x': x_coords,
            'y': y_coords,
            'filename': dat_file,
        }
    
    # Load performance data from CSV
    try:
        performance_data = pd.read_csv(SWEEP_RESULTS_FILE)
        
        # Sort by lift-to-drag ratio
        performance_data = performance_data.sort_values('cl_cd', ascending=False)
        
        # Add rank information
        performance_data['rank'] = range(1, len(performance_data) + 1)
        
        # Get data for the loaded airfoils
        for name, airfoil in airfoils.items():
            rank_match = re.search(r'#(\d+)', name)
            if rank_match:
                rank = int(rank_match.group(1))
                row = performance_data[performance_data['rank'] == rank]
                
                if not row.empty:
                    airfoil['cl'] = float(row['cl'].values[0])
                    airfoil['cd'] = float(row['cd'].values[0])
                    airfoil['cm'] = float(row['cm'].values[0])
                    airfoil['cl_cd'] = float(row['cl_cd'].values[0])
                    airfoil['alpha'] = float(row['alpha'].values[0])
    
    except Exception as e:
        print(f"Warning: Could not load performance data: {str(e)}")
    
    return airfoils


def create_static_3d_plot(airfoils, output_file):
    """Create a static 3D visualization of the airfoils and save to HTML"""
    # Create a figure with 2 subplots side by side
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'surface'}, {'type': 'xy'}]],
        subplot_titles=('3D Airfoil Shapes', 'Performance Comparison'),
        column_widths=[0.7, 0.3]
    )
    
    # Add 3D plots of airfoils
    for i, (name, airfoil) in enumerate(airfoils.items()):
        # Create z values (for visualization only)
        x = airfoil['x']
        y = airfoil['y']
        z_spacing = 0.05  # Spacing between airfoils in z-axis
        z = np.ones_like(x) * i * z_spacing
        
        # Get color for this airfoil
        color = COLORS[i % len(COLORS)]
        
        # Add 3D line
        fig.add_trace(
            go.Scatter3d(
                x=x, y=y, z=z,
                mode='lines',
                line=dict(color=color, width=5),
                name=f"{name} (L/D={airfoil.get('cl_cd', 'N/A'):.1f})",
                showlegend=True
            ),
            row=1, col=1
        )
        
        # Add marker for CL/CD ratio on the performance chart
        if 'cl' in airfoil and 'cd' in airfoil:
            fig.add_trace(
                go.Scatter(
                    x=[airfoil['cl']],
                    y=[airfoil['cd']],
                    mode='markers+text',
                    marker=dict(color=color, size=15),
                    text=[name],
                    textposition="top center",
                    name=name,
                    showlegend=False
                ),
                row=1, col=2
            )
    
    # Update 3D layout
    fig.update_scenes(
        aspectmode='data',
        xaxis=dict(title='X'),
        yaxis=dict(title='Y'),
        zaxis=dict(title='Airfoil')
    )
    
    # Update performance chart layout
    fig.update_xaxes(title_text="Lift Coefficient (CL)", row=1, col=2)
    fig.update_yaxes(title_text="Drag Coefficient (CD)", row=1, col=2)
    
    # Update overall layout
    fig.update_layout(
        title_text="Parameter Sweep Top Airfoil Comparison",
        height=800,
        width=1200,
        scene=dict(
            aspectmode='manual',
            aspectratio=dict(x=1, y=0.5, z=0.2)
        )
    )
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save to HTML
    fig.write_html(output_file)
    print(f"Static 3D visualization saved to {output_file}")
    
    return fig


def create_interactive_dashboard(airfoils):
    """Create an interactive Dash dashboard for 3D visualization"""
    app = dash.Dash(__name__)
    
    # Create a 3D figure
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'surface'}, {'type': 'xy'}]],
        subplot_titles=('3D Airfoil Shapes', 'Performance Comparison'),
        column_widths=[0.7, 0.3]
    )
    
    # Add all airfoils to the plot initially
    for i, (name, airfoil) in enumerate(airfoils.items()):
        # Create z values (for visualization only)
        x = airfoil['x']
        y = airfoil['y']
        z_spacing = 0.05
        z = np.ones_like(x) * i * z_spacing
        
        # Get color for this airfoil
        color = COLORS[i % len(COLORS)]
        
        # Add 3D line
        fig.add_trace(
            go.Scatter3d(
                x=x, y=y, z=z,
                mode='lines',
                line=dict(color=color, width=5),
                name=f"{name} (L/D={airfoil.get('cl_cd', 'N/A'):.1f})",
                showlegend=True
            ),
            row=1, col=1
        )
        
        # Add marker for CL/CD ratio on the performance chart
        if 'cl' in airfoil and 'cd' in airfoil:
            fig.add_trace(
                go.Scatter(
                    x=[airfoil['cl']],
                    y=[airfoil['cd']],
                    mode='markers+text',
                    marker=dict(color=color, size=15),
                    text=[name],
                    textposition="top center",
                    name=name,
                    showlegend=False
                ),
                row=1, col=2
            )
    
    # Update layout
    fig.update_layout(
        title_text="Parameter Sweep Top Airfoil Comparison",
        height=800,
        width=1200,
        scene=dict(
            aspectmode='manual',
            aspectratio=dict(x=1, y=0.5, z=0.2)
        )
    )
    
    # Update 3D layout
    fig.update_scenes(
        aspectmode='data',
        xaxis=dict(title='X'),
        yaxis=dict(title='Y'),
        zaxis=dict(title='Airfoil')
    )
    
    # Update performance chart layout
    fig.update_xaxes(title_text="Lift Coefficient (CL)", row=1, col=2)
    fig.update_yaxes(title_text="Drag Coefficient (CD)", row=1, col=2)
    
    # Create the app layout
    app.layout = html.Div([
        html.H1("Parameter Sweep Top Airfoil Comparison"),
        html.Div([
            html.Label("Select Airfoils to Display:"),
            dcc.Checklist(
                id='airfoil-checklist',
                options=[{'label': name, 'value': name} for name in airfoils.keys()],
                value=list(airfoils.keys()),  # All selected initially
                inline=True
            ),
        ]),
        dcc.Graph(id='airfoil-3d-graph', figure=fig),
    ])
    
    # Define callback to update the figure based on selections
    @app.callback(
        dash.Output('airfoil-3d-graph', 'figure'),
        [dash.Input('airfoil-checklist', 'value')]
    )
    def update_figure(selected_airfoils):
        # Create a new figure with the selected airfoils
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{'type': 'surface'}, {'type': 'xy'}]],
            subplot_titles=('3D Airfoil Shapes', 'Performance Comparison'),
            column_widths=[0.7, 0.3]
        )
        
        for i, name in enumerate(airfoils.keys()):
            if name in selected_airfoils:
                airfoil = airfoils[name]
                
                # Create z values (for visualization only)
                x = airfoil['x']
                y = airfoil['y']
                z_spacing = 0.05
                z = np.ones_like(x) * i * z_spacing
                
                # Get color for this airfoil
                color = COLORS[i % len(COLORS)]
                
                # Add 3D line
                fig.add_trace(
                    go.Scatter3d(
                        x=x, y=y, z=z,
                        mode='lines',
                        line=dict(color=color, width=5),
                        name=f"{name} (L/D={airfoil.get('cl_cd', 'N/A'):.1f})",
                        showlegend=True
                    ),
                    row=1, col=1
                )
                
                # Add marker for CL/CD ratio on the performance chart
                if 'cl' in airfoil and 'cd' in airfoil:
                    fig.add_trace(
                        go.Scatter(
                            x=[airfoil['cl']],
                            y=[airfoil['cd']],
                            mode='markers+text',
                            marker=dict(color=color, size=15),
                            text=[name],
                            textposition="top center",
                            name=name,
                            showlegend=False
                        ),
                        row=1, col=2
                    )
        
        # Update layout
        fig.update_layout(
            title_text="Parameter Sweep Top Airfoil Comparison",
            height=800,
            width=1200,
            scene=dict(
                aspectmode='manual',
                aspectratio=dict(x=1, y=0.5, z=0.2)
            )
        )
        
        # Update 3D layout
        fig.update_scenes(
            aspectmode='data',
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Airfoil')
        )
        
        # Update performance chart layout
        fig.update_xaxes(title_text="Lift Coefficient (CL)", row=1, col=2)
        fig.update_yaxes(title_text="Drag Coefficient (CD)", row=1, col=2)
        
        return fig
    
    return app


def main():
    """Main function"""
    print("\n3D Visualization of Parameter Sweep Airfoils\n")
    
    # Load airfoil data
    airfoils = load_airfoil_data()
    
    if not airfoils:
        print("No airfoils were loaded. Exiting.")
        return
    
    print(f"Loaded {len(airfoils)} airfoils.")
    
    # Create static 3D plot
    create_static_3d_plot(airfoils, OUTPUT_HTML)
    
    # Create interactive dashboard
    print("\nCreating interactive dashboard...")
    app = create_interactive_dashboard(airfoils)
    
    # Run the app
    print("Starting Dash server. Press Ctrl+C to stop.")
    print("You can view the interactive visualization at: http://localhost:8050")
    app.run(debug=True, port=8050)


if __name__ == "__main__":
    main()
