#!/usr/bin/env python3
"""
Unified 3D Visualization of Airfoils

This script creates a 3D visualization that includes both real airfoils from
the airfoils_uiuc directory and parameter sweep airfoils, allowing for
direct comparison of their shapes and performance metrics.
"""

import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, callback
import re

# Define directories and files
REAL_AIRFOILS_DIR = "airfoils_uiuc"
SWEEP_AIRFOILS_DIR = "parameter_sweep_top_airfoils"
REAL_PERFORMANCE_FILE = "results/airfoil_best_performance.csv"
SWEEP_RESULTS_FILE = "parameter_sweep_results/parsec_sweep_results_success.csv"
OUTPUT_HTML = "visualization_output/unified_airfoils_3d.html"

# Top performing real airfoils to include
TOP_REAL_AIRFOILS = ["ag13.dat", "a18sm.dat", "ag26.dat", "ag25.dat", "ag27.dat"]

# Ensure output directory exists
os.makedirs("visualization_output", exist_ok=True)

# Colors for different airfoil groups
REAL_COLORS = ['rgb(31, 119, 180)', 'rgb(255, 127, 14)', 'rgb(44, 160, 44)', 
              'rgb(214, 39, 40)', 'rgb(148, 103, 189)']
SWEEP_COLORS = ['rgb(140, 86, 75)', 'rgb(227, 119, 194)', 'rgb(127, 127, 127)', 
               'rgb(188, 189, 34)', 'rgb(23, 190, 207)']


def read_airfoil_dat(filename):
    """Read airfoil coordinates from a .dat file"""
    try:
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
                parts = line.strip().split()
                if len(parts) >= 2:
                    x, y = parts[0], parts[1]
                    coordinates.append((float(x), float(y)))
            except ValueError:
                continue
        
        if not coordinates:
            print(f"  Warning: No valid coordinates found in {filename}")
            return None, None, None
        
        # Convert to numpy arrays
        coords = np.array(coordinates)
        x_coords = coords[:, 0]
        y_coords = coords[:, 1]
        
        return name, x_coords, y_coords
    
    except Exception as e:
        print(f"  Error reading {filename}: {str(e)}")
        return None, None, None


def load_airfoil_data():
    """Load both real airfoils and parameter sweep airfoils with their performance data"""
    airfoils = {}
    
    # Load real airfoils
    print("Loading real airfoil data...")
    for dat_file in TOP_REAL_AIRFOILS:
        file_path = os.path.join(REAL_AIRFOILS_DIR, dat_file)
        if not os.path.exists(file_path):
            print(f"  File not found: {file_path}")
            continue
            
        name, x_coords, y_coords = read_airfoil_dat(file_path)
        
        if x_coords is None:
            continue
        
        # Use the filename without extension as the airfoil name
        airfoil_name = os.path.splitext(dat_file)[0]
        airfoils[f"Real: {airfoil_name}"] = {
            'x': x_coords,
            'y': y_coords,
            'filename': dat_file,
            'type': 'real'
        }
    
    # Load real airfoils performance data
    try:
        if os.path.exists(REAL_PERFORMANCE_FILE):
            real_perf = pd.read_csv(REAL_PERFORMANCE_FILE)
            for name, airfoil in list(airfoils.items()):
                if name.startswith("Real:"):
                    # Extract airfoil name without the "Real: " prefix
                    airfoil_name = name.split(": ")[1]
                    # Find matching row in performance data
                    row = real_perf[real_perf['airfoil'] == airfoil_name]
                    if not row.empty:
                        airfoil['cl'] = float(row['cl'].values[0])
                        airfoil['cd'] = float(row['cd'].values[0])
                        airfoil['cl_cd'] = float(row['cl_cd'].values[0])
                        airfoil['alpha'] = float(row['alpha'].values[0])
    except Exception as e:
        print(f"  Warning: Could not load real performance data: {str(e)}")
    
    # Load parameter sweep airfoils
    print("Loading parameter sweep airfoil data...")
    if os.path.exists(SWEEP_AIRFOILS_DIR):
        sweep_files = [f for f in os.listdir(SWEEP_AIRFOILS_DIR) if f.endswith('.dat')]
        
        for dat_file in sweep_files:
            file_path = os.path.join(SWEEP_AIRFOILS_DIR, dat_file)
            name, x_coords, y_coords = read_airfoil_dat(file_path)
            
            if x_coords is None:
                continue
            
            # Extract rank number from filename (sweep_topN.dat)
            rank_match = re.search(r'top(\d+)', dat_file)
            if rank_match:
                rank = int(float(rank_match.group(1)))
                display_name = f"Sweep: Rank #{rank}"
            else:
                display_name = f"Sweep: {os.path.splitext(dat_file)[0]}"
            
            airfoils[display_name] = {
                'x': x_coords,
                'y': y_coords,
                'filename': dat_file,
                'type': 'sweep',
                'rank': rank if rank_match else None
            }
    
    # Load parameter sweep performance data
    try:
        if os.path.exists(SWEEP_RESULTS_FILE):
            sweep_perf = pd.read_csv(SWEEP_RESULTS_FILE)
            sweep_perf = sweep_perf.sort_values('cl_cd', ascending=False)
            sweep_perf['rank'] = range(1, len(sweep_perf) + 1)
            
            for name, airfoil in list(airfoils.items()):
                if airfoil.get('type') == 'sweep' and airfoil.get('rank') is not None:
                    rank = airfoil['rank']
                    row = sweep_perf[sweep_perf['rank'] == rank]
                    if not row.empty:
                        airfoil['cl'] = float(row['cl'].values[0])
                        airfoil['cd'] = float(row['cd'].values[0])
                        airfoil['cm'] = float(row['cm'].values[0]) if 'cm' in row else None
                        airfoil['cl_cd'] = float(row['cl_cd'].values[0])
                        airfoil['alpha'] = float(row['alpha'].values[0]) if 'alpha' in row else 5.0
    except Exception as e:
        print(f"  Warning: Could not load sweep performance data: {str(e)}")
    
    print(f"Total airfoils loaded: {len(airfoils)}")
    return airfoils


def create_static_3d_plot(airfoils, output_file):
    """Create a static 3D visualization of all airfoils and save to HTML"""
    # Create a figure with 2 subplots side by side
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'surface'}, {'type': 'xy'}]],
        subplot_titles=('3D Airfoil Shapes', 'Performance Comparison (CL vs CD)'),
        column_widths=[0.7, 0.3]
    )
    
    # Sort airfoils by type (real first, then sweep)
    sorted_airfoils = {}
    # First add real airfoils
    for name, airfoil in airfoils.items():
        if airfoil.get('type') == 'real':
            sorted_airfoils[name] = airfoil
    # Then add sweep airfoils
    for name, airfoil in airfoils.items():
        if airfoil.get('type') == 'sweep':
            sorted_airfoils[name] = airfoil
    
    # Add 3D plots of airfoils
    real_count = 0
    sweep_count = 0
    
    for i, (name, airfoil) in enumerate(sorted_airfoils.items()):
        # Create z values (for visualization only)
        x = airfoil['x']
        y = airfoil['y']
        z_spacing = 0.05  # Spacing between airfoils in z-axis
        z = np.ones_like(x) * i * z_spacing
        
        # Choose color based on airfoil type
        if airfoil.get('type') == 'real':
            color = REAL_COLORS[real_count % len(REAL_COLORS)]
            real_count += 1
        else:
            color = SWEEP_COLORS[sweep_count % len(SWEEP_COLORS)]
            sweep_count += 1
        
        # Format name with performance data
        display_name = name
        if 'cl_cd' in airfoil:
            display_name = f"{name} (L/D={airfoil['cl_cd']:.2f})"
        
        # Add 3D line
        fig.add_trace(
            go.Scatter3d(
                x=x, y=y, z=z,
                mode='lines',
                line=dict(color=color, width=5),
                name=display_name,
                showlegend=True
            ),
            row=1, col=1
        )
        
        # Add marker for CL/CD ratio on the performance chart
        if 'cl' in airfoil and 'cd' in airfoil:
            marker_size = 12
            # Make sweep markers triangles and real markers circles
            marker_symbol = 'triangle-up' if airfoil.get('type') == 'sweep' else 'circle'
            
            fig.add_trace(
                go.Scatter(
                    x=[airfoil['cl']],
                    y=[airfoil['cd']],
                    mode='markers+text',
                    marker=dict(
                        color=color, 
                        size=marker_size,
                        symbol=marker_symbol,
                        line=dict(width=1, color='black')
                    ),
                    text=[name.split(": ")[1] if ": " in name else name],
                    textposition="top center",
                    name=display_name,
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
    
    # Use logarithmic scale for drag coefficient due to large range differences
    fig.update_yaxes(type="log", title_text="Drag Coefficient (CD) - Log Scale", row=1, col=2)
    fig.update_xaxes(title_text="Lift Coefficient (CL)", row=1, col=2)
    
    # Add annotations to differentiate the airfoil types
    fig.add_annotation(
        x=0.1, y=0.95,
        xref="paper", yref="paper",
        text="● Real Airfoils",
        showarrow=False,
        font=dict(color=REAL_COLORS[0], size=12),
        bgcolor="rgba(255, 255, 255, 0.7)",
        bordercolor="black",
        borderwidth=1,
        borderpad=4
    )
    
    fig.add_annotation(
        x=0.1, y=0.9,
        xref="paper", yref="paper",
        text="▲ Parameter Sweep Airfoils",
        showarrow=False,
        font=dict(color=SWEEP_COLORS[0], size=12),
        bgcolor="rgba(255, 255, 255, 0.7)",
        bordercolor="black",
        borderwidth=1,
        borderpad=4
    )
    
    # Update overall layout
    fig.update_layout(
        title_text="Unified Airfoil Comparison (Real vs Parameter Sweep)",
        height=800,
        width=1200,
        scene=dict(
            aspectmode='manual',
            aspectratio=dict(x=1, y=0.5, z=0.2)
        ),
        legend=dict(
            groupclick="toggleitem",
            itemsizing="constant"
        )
    )
    
    # Save to HTML
    fig.write_html(output_file)
    print(f"Static 3D visualization saved to {output_file}")
    
    return fig


def create_interactive_dashboard(airfoils):
    """Create an interactive Dash dashboard for 3D visualization"""
    app = dash.Dash(__name__)
    
    # Group airfoils by type
    real_airfoils = {k: v for k, v in airfoils.items() if v.get('type') == 'real'}
    sweep_airfoils = {k: v for k, v in airfoils.items() if v.get('type') == 'sweep'}
    
    # Create initial 3D figure
    fig = create_static_3d_plot(airfoils, OUTPUT_HTML)
    
    # Create the app layout
    app.layout = html.Div([
        html.H1("Unified Airfoil Comparison (Real vs Parameter Sweep)", style={'textAlign': 'center'}),
        
        html.Div([
            html.Div([
                html.H3("Display Options:", style={'marginBottom': '10px'}),
                
                html.Div([
                    html.Label("Real Airfoils:"),
                    dcc.Checklist(
                        id='real-airfoil-checklist',
                        options=[{'label': name.split(": ")[1], 'value': name} for name in real_airfoils.keys()],
                        value=list(real_airfoils.keys()),  # All selected initially
                        inline=True,
                        style={'marginBottom': '10px'}
                    ),
                ]),
                
                html.Div([
                    html.Label("Parameter Sweep Airfoils:"),
                    dcc.Checklist(
                        id='sweep-airfoil-checklist',
                        options=[{'label': name.split(": ")[1], 'value': name} for name in sweep_airfoils.keys()],
                        value=list(sweep_airfoils.keys()),  # All selected initially
                        inline=True,
                        style={'marginBottom': '10px'}
                    ),
                ]),
                
                html.Div([
                    html.Button('Select All Real', id='select-all-real', n_clicks=0),
                    html.Button('Deselect All Real', id='deselect-all-real', n_clicks=0),
                    html.Button('Select All Sweep', id='select-all-sweep', n_clicks=0),
                    html.Button('Deselect All Sweep', id='deselect-all-sweep', n_clicks=0),
                ], style={'marginBottom': '20px'})
            ], style={'marginBottom': '20px', 'padding': '10px', 'border': '1px solid #ddd', 'borderRadius': '5px'}),
            
            dcc.Graph(id='airfoil-3d-graph', figure=fig, style={'height': '800px'}),
            
            html.Div([
                html.H3("Performance Comparison Table"),
                html.Div(id='performance-table')
            ], style={'marginTop': '20px'})
        ], style={'maxWidth': '1200px', 'margin': 'auto', 'padding': '20px'})
    ])
    
    # Define callbacks
    @app.callback(
        [dash.Output('real-airfoil-checklist', 'value')],
        [dash.Input('select-all-real', 'n_clicks'),
         dash.Input('deselect-all-real', 'n_clicks')],
        [dash.State('real-airfoil-checklist', 'value')]
    )
    def update_real_checklist(select_clicks, deselect_clicks, current_values):
        ctx = dash.callback_context
        if not ctx.triggered:
            return [current_values]
        
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if button_id == 'select-all-real':
            return [list(real_airfoils.keys())]
        elif button_id == 'deselect-all-real':
            return [[]]
        
        return [current_values]
    
    @app.callback(
        [dash.Output('sweep-airfoil-checklist', 'value')],
        [dash.Input('select-all-sweep', 'n_clicks'),
         dash.Input('deselect-all-sweep', 'n_clicks')],
        [dash.State('sweep-airfoil-checklist', 'value')]
    )
    def update_sweep_checklist(select_clicks, deselect_clicks, current_values):
        ctx = dash.callback_context
        if not ctx.triggered:
            return [current_values]
        
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if button_id == 'select-all-sweep':
            return [list(sweep_airfoils.keys())]
        elif button_id == 'deselect-all-sweep':
            return [[]]
        
        return [current_values]
    
    @app.callback(
        [dash.Output('airfoil-3d-graph', 'figure'),
         dash.Output('performance-table', 'children')],
        [dash.Input('real-airfoil-checklist', 'value'),
         dash.Input('sweep-airfoil-checklist', 'value')]
    )
    def update_figure(selected_real, selected_sweep):
        # Combine selected airfoils
        selected_airfoils = {}
        for name in selected_real:
            if name in airfoils:
                selected_airfoils[name] = airfoils[name]
        
        for name in selected_sweep:
            if name in airfoils:
                selected_airfoils[name] = airfoils[name]
        
        # Create updated figure with selected airfoils
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{'type': 'surface'}, {'type': 'xy'}]],
            subplot_titles=('3D Airfoil Shapes', 'Performance Comparison (CL vs CD)'),
            column_widths=[0.7, 0.3]
        )
        
        # Sort by airfoil type (real first, then sweep)
        sorted_airfoils = {}
        # First add real airfoils
        for name in selected_real:
            if name in airfoils:
                sorted_airfoils[name] = airfoils[name]
        # Then add sweep airfoils
        for name in selected_sweep:
            if name in airfoils:
                sorted_airfoils[name] = airfoils[name]
        
        # Add 3D plots of airfoils
        real_count = 0
        sweep_count = 0
        
        for i, (name, airfoil) in enumerate(sorted_airfoils.items()):
            # Create z values (for visualization only)
            x = airfoil['x']
            y = airfoil['y']
            z_spacing = 0.05  # Spacing between airfoils in z-axis
            z = np.ones_like(x) * i * z_spacing
            
            # Choose color based on airfoil type
            if airfoil.get('type') == 'real':
                color = REAL_COLORS[real_count % len(REAL_COLORS)]
                real_count += 1
            else:
                color = SWEEP_COLORS[sweep_count % len(SWEEP_COLORS)]
                sweep_count += 1
            
            # Format name with performance data
            display_name = name
            if 'cl_cd' in airfoil:
                display_name = f"{name} (L/D={airfoil['cl_cd']:.2f})"
            
            # Add 3D line
            fig.add_trace(
                go.Scatter3d(
                    x=x, y=y, z=z,
                    mode='lines',
                    line=dict(color=color, width=5),
                    name=display_name,
                    showlegend=True
                ),
                row=1, col=1
            )
            
            # Add marker for CL/CD ratio on the performance chart
            if 'cl' in airfoil and 'cd' in airfoil:
                marker_size = 12
                # Make sweep markers triangles and real markers circles
                marker_symbol = 'triangle-up' if airfoil.get('type') == 'sweep' else 'circle'
                
                fig.add_trace(
                    go.Scatter(
                        x=[airfoil['cl']],
                        y=[airfoil['cd']],
                        mode='markers+text',
                        marker=dict(
                            color=color, 
                            size=marker_size,
                            symbol=marker_symbol,
                            line=dict(width=1, color='black')
                        ),
                        text=[name.split(": ")[1] if ": " in name else name],
                        textposition="top center",
                        name=display_name,
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
        
        # Use logarithmic scale for drag coefficient due to large range differences
        fig.update_yaxes(type="log", title_text="Drag Coefficient (CD) - Log Scale", row=1, col=2)
        fig.update_xaxes(title_text="Lift Coefficient (CL)", row=1, col=2)
        
        # Add annotations to differentiate the airfoil types
        fig.add_annotation(
            x=0.1, y=0.95,
            xref="paper", yref="paper",
            text="● Real Airfoils",
            showarrow=False,
            font=dict(color=REAL_COLORS[0], size=12),
            bgcolor="rgba(255, 255, 255, 0.7)",
            bordercolor="black",
            borderwidth=1,
            borderpad=4
        )
        
        fig.add_annotation(
            x=0.1, y=0.9,
            xref="paper", yref="paper",
            text="▲ Parameter Sweep Airfoils",
            showarrow=False,
            font=dict(color=SWEEP_COLORS[0], size=12),
            bgcolor="rgba(255, 255, 255, 0.7)",
            bordercolor="black",
            borderwidth=1,
            borderpad=4
        )
        
        # Update overall layout
        fig.update_layout(
            title_text="Unified Airfoil Comparison (Real vs Parameter Sweep)",
            height=800,
            width=1200,
            scene=dict(
                aspectmode='manual',
                aspectratio=dict(x=1, y=0.5, z=0.2)
            ),
            legend=dict(
                groupclick="toggleitem",
                itemsizing="constant"
            )
        )
        
        # Create performance table HTML
        if not selected_airfoils:
            table_html = html.P("No airfoils selected.")
        else:
            # Create table header
            table_header = [
                html.Thead(html.Tr([
                    html.Th("Airfoil"),
                    html.Th("Type"),
                    html.Th("L/D Ratio"),
                    html.Th("CL"),
                    html.Th("CD"),
                    html.Th("Alpha (°)")
                ]))
            ]
            
            # Create table rows
            rows = []
            for name, airfoil in sorted_airfoils.items():
                airfoil_name = name.split(": ")[1] if ": " in name else name
                airfoil_type = "Real" if airfoil.get('type') == 'real' else "Sweep"
                cl_cd = f"{airfoil.get('cl_cd', 'N/A'):.2f}" if 'cl_cd' in airfoil else "N/A"
                cl = f"{airfoil.get('cl', 'N/A'):.4f}" if 'cl' in airfoil else "N/A"
                cd = f"{airfoil.get('cd', 'N/A'):.6f}" if 'cd' in airfoil else "N/A"
                alpha = f"{airfoil.get('alpha', 'N/A')}" if 'alpha' in airfoil else "N/A"
                
                row = html.Tr([
                    html.Td(airfoil_name),
                    html.Td(airfoil_type),
                    html.Td(cl_cd),
                    html.Td(cl),
                    html.Td(cd),
                    html.Td(alpha)
                ])
                rows.append(row)
            
            table_body = [html.Tbody(rows)]
            
            table_html = html.Table(
                table_header + table_body,
                style={
                    'borderCollapse': 'collapse',
                    'width': '100%',
                    'textAlign': 'center'
                }
            )
        
        return fig, table_html
    
    return app


def main():
    """Main function"""
    print("\nUnified 3D Visualization of Airfoils (Real and Parameter Sweep)\n")
    
    # Load airfoil data
    airfoils = load_airfoil_data()
    
    if not airfoils:
        print("No airfoils were loaded. Exiting.")
        return
    
    # Create static 3D plot
    create_static_3d_plot(airfoils, OUTPUT_HTML)
    
    # Create interactive dashboard
    print("\nCreating interactive dashboard...")
    app = create_interactive_dashboard(airfoils)
    
    # Run the app
    print("Starting Dash server. Press Ctrl+C to stop.")
    print("You can view the interactive visualization at: http://localhost:8050")
    # Use a different port to avoid conflict
    app.run(debug=True, port=8051)


if __name__ == "__main__":
    main()
