#!/usr/bin/env python3
"""
3D Visualization of PARSEC Parameter Sweep Results

This script creates a 3D interactive dashboard to visualize the top-performing airfoils
from the PARSEC parameter sweep, leveraging the existing 3D visualization functionality.
"""

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import webbrowser
from threading import Timer
import h5py
from parsec_to_dat import ParsecAirfoil

# Define directories and files
RESULTS_DIR = "parameter_sweep_results"
SWEEP_RESULTS_FILE = os.path.join(RESULTS_DIR, "parsec_sweep_results_success.csv")
SWEEP_DB_FILE = os.path.join(RESULTS_DIR, "parsec_sweep.h5")
VISUALIZATION_DIR = os.path.join(RESULTS_DIR, "web_visualizations")

# Create output directory
os.makedirs(VISUALIZATION_DIR, exist_ok=True)


def load_sweep_results():
    """Load the results from the parameter sweep"""
    if not os.path.exists(SWEEP_RESULTS_FILE):
        print(f"Error: Sweep results file not found: {SWEEP_RESULTS_FILE}")
        return None
    
    return pd.read_csv(SWEEP_RESULTS_FILE)


def get_top_performers(df, n=10, metric='cl_cd'):
    """Get the top n performers based on a specified metric"""
    if df is None or df.empty:
        return None
    
    return df.nlargest(n, metric).copy().reset_index(drop=True)


def get_airfoil_coordinates(params):
    """Generate airfoil coordinates from PARSEC parameters"""
    airfoil = ParsecAirfoil()
    
    # Update parameters
    for param, value in params.items():
        if param in airfoil.params:
            airfoil.params[param] = value
    
    try:
        x_coords, y_coords = airfoil.generate_coordinates(200)
        return x_coords, y_coords
    except Exception as e:
        print(f"Error generating coordinates: {str(e)}")
        return None, None


def create_3d_airfoil_figure(top_airfoils, params_to_show):
    """Create a 3D figure of airfoil shapes from parameter sweep results"""
    # Create the 3D figure
    fig = go.Figure()
    
    # Process each airfoil
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'pink', 'brown']
    span = 1.0
    num_sections = 10
    
    for idx, (_, airfoil_data) in enumerate(top_airfoils.iterrows()):
        # Determine color for this airfoil
        color = colors[idx % len(colors)]
        rank = idx + 1
        
        # Extract parameters for this airfoil
        params = {param: airfoil_data[param] for param in params_to_show if param in airfoil_data}
        
        # Generate airfoil coordinates from parameters
        x_coords, y_coords = get_airfoil_coordinates(params)
        
        if x_coords is None or y_coords is None:
            print(f"Skipping airfoil rank {rank} due to coordinate generation error")
            continue
            
        # Performance metrics
        cl = airfoil_data['cl']
        cd = airfoil_data['cd']
        cm = airfoil_data['cm']
        cl_cd = airfoil_data['cl_cd']
        
        # Create 3D wing by extruding airfoil profile
        z_positions = np.linspace(0, span, num_sections)
        z_offset = idx * span * 1.5  # Separate different airfoils along z-axis
        
        # Create hover text with parameter and performance data
        hover_text = f"Rank: #{rank}<br>"
        hover_text += f"CL: {cl:.4f}<br>"
        hover_text += f"CD: {cd:.6f}<br>"
        hover_text += f"CM: {cm:.4f}<br>"
        hover_text += f"L/D: {cl_cd:.2f}<br>"
        hover_text += "<br>Parameters:<br>"
        for param, value in params.items():
            hover_text += f"{param}: {value:.6f}<br>"

        # Add profile sections along the span
        for z_pos in [0, span/2, span]:
            z = z_pos + z_offset
            fig.add_trace(go.Scatter3d(
                x=x_coords,
                y=y_coords,
                z=[z] * len(x_coords),
                mode='lines',
                line=dict(color=color, width=4),
                name=f"Rank #{rank}",
                text=hover_text,
                hovertemplate=hover_text + '<extra></extra>',
                showlegend=(z_pos == 0)  # Only show in legend once
            ))
        
        # Add visible labels at root
        fig.add_trace(go.Scatter3d(
            x=[0.5],
            y=[0.0],
            z=[z_offset - 0.1],
            mode='text',
            text=[f"Rank #{rank}: L/D = {cl_cd:.2f}"],
            textfont=dict(size=12, color=color),
            showlegend=False
        ))

    # Update layout
    fig.update_layout(
        scene=dict(
            xaxis_title='X (Chord)',
            yaxis_title='Y (Thickness)',
            zaxis_title='Z (Span)',
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.0),
            )
        ),
        title='3D Comparison of Top-Performing Airfoils from Parameter Sweep',
        margin=dict(l=0, r=0, b=0, t=40),
        height=700,
    )
    
    return fig


def create_parameters_heatmap(top_airfoils, params_to_show):
    """Create a heatmap showing parameter values for top airfoils"""
    # Extract parameters data
    param_data = top_airfoils[params_to_show].copy()
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=param_data.values,
        x=param_data.columns,
        y=[f"Rank #{i+1}" for i in range(len(param_data))],
        colorscale='Viridis',
        text=param_data.values.round(6),
        texttemplate="%{text}",
        textfont={"size": 10},
    ))
    
    fig.update_layout(
        title='Parameter Values for Top Airfoils',
        xaxis_title='Parameter',
        yaxis_title='Airfoil Rank',
        height=400,
    )
    
    return fig


def create_performance_radar_chart(top_airfoils):
    """Create a radar chart comparing key performance metrics for top airfoils"""
    fig = go.Figure()
    
    # Normalize metrics for radar chart
    metrics = ['cl', 'cd', 'cm', 'cl_cd']
    normalized_data = top_airfoils[metrics].copy()
    
    # Handle positive and negative values differently
    for col in normalized_data.columns:
        values = normalized_data[col]
        min_val = values.min()
        max_val = values.max()
        
        # If all values are positive, normalize to [0,1]
        if min_val >= 0:
            normalized_data[col] = (values - min_val) / (max_val - min_val) if max_val > min_val else values
        # If we have mixed signs, handle differently
        else:
            abs_max = max(abs(min_val), abs(max_val))
            normalized_data[col] = values / abs_max
    
    # Color palette for different airfoils
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'pink', 'brown']
    
    # Add traces for each airfoil
    for i, (_, airfoil) in enumerate(top_airfoils.iterrows()):
        rank = i + 1
        color = colors[i % len(colors)]
        
        # Prepare data for radar chart
        radar_data = normalized_data.iloc[i].tolist()
        # Close the polygon by repeating the first value
        radar_data.append(radar_data[0])
        
        # Add trace
        fig.add_trace(go.Scatterpolar(
            r=radar_data,
            theta=metrics + [metrics[0]],  # Close the polygon
            fill='toself',
            name=f"Rank #{rank}",
            line_color=color,
        ))
    
    # Update layout
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        title="Normalized Performance Comparison",
        height=500,
    )
    
    return fig


def create_dashboard(top_airfoils, params_to_show):
    """Create an interactive dashboard with 3D airfoils and performance metrics"""
    # Create app
    app = dash.Dash(__name__, suppress_callback_exceptions=True)
    
    # 3D figure of airfoil shapes
    airfoil_fig = create_3d_airfoil_figure(top_airfoils, params_to_show)
    
    # Parameter heatmap
    heatmap_fig = create_parameters_heatmap(top_airfoils, params_to_show)
    
    # Performance radar chart
    radar_fig = create_performance_radar_chart(top_airfoils)
    
    # Create the layout
    app.layout = html.Div([
        html.H1("PARSEC Parameter Sweep: Top Performers", style={'textAlign': 'center'}),
        
        html.Div([
            html.H2("3D Airfoil Shapes", style={'textAlign': 'center'}),
            dcc.Graph(id='airfoil-3d', figure=airfoil_fig),
            
            html.H2("Performance Metrics", style={'textAlign': 'center'}),
            html.Div([
                html.Div([
                    dcc.Graph(id='radar-chart', figure=radar_fig),
                ], style={'width': '50%', 'display': 'inline-block', 'vertical-align': 'top'}),
                
                html.Div([
                    dcc.Graph(id='param-heatmap', figure=heatmap_fig),
                ], style={'width': '50%', 'display': 'inline-block', 'vertical-align': 'top'}),
            ]),
            
            html.Div([
                html.H2("Top Performers Data Table", style={'textAlign': 'center'}),
                html.Table([
                    html.Thead(
                        html.Tr([html.Th("Rank")] + 
                                [html.Th(col) for col in params_to_show] +
                                [html.Th("CL"), html.Th("CD"), html.Th("CM"), html.Th("L/D")])
                    ),
                    html.Tbody([
                        html.Tr(
                            [html.Td(f"#{i+1}")] +
                            [html.Td(f"{row[param]:.6f}") for param in params_to_show] +
                            [html.Td(f"{row['cl']:.4f}"), 
                             html.Td(f"{row['cd']:.6f}"), 
                             html.Td(f"{row['cm']:.4f}"), 
                             html.Td(f"{row['cl_cd']:.2f}")]
                        ) for i, (_, row) in enumerate(top_airfoils.iterrows())
                    ])
                ], style={'width': '100%', 'borderCollapse': 'collapse'})
            ], style={'marginTop': '40px'})
            
        ], style={'maxWidth': '1200px', 'margin': '0 auto', 'padding': '20px', 
                 'backgroundColor': '#f5f5f5', 'borderRadius': '10px', 
                 'boxShadow': '0 4px 8px rgba(0, 0, 0, 0.1)'})
    ])
    
    return app


def open_browser():
    """Open the browser after a short delay"""
    webbrowser.open_new("http://127.0.0.1:8050/")


def main():
    """Main function to run the visualization"""
    # Load parameter sweep results
    print("Loading parameter sweep results...")
    df = load_sweep_results()
    
    if df is None:
        print("Failed to load sweep results. Exiting.")
        return
    
    # Get top performers
    top_n = 10
    print(f"Extracting top {top_n} performers by L/D ratio...")
    top_airfoils = get_top_performers(df, n=top_n, metric='cl_cd')
    
    if top_airfoils is None or top_airfoils.empty:
        print("No valid top performers found. Exiting.")
        return
    
    # Parameters to visualize
    params_to_show = ['rLE', 'Xup', 'Yup', 'Xlo', 'Ylo']
    
    # Create the dashboard app
    print("Creating 3D visualization dashboard...")
    app = create_dashboard(top_airfoils, params_to_show)
    
    # Open browser after a delay
    print("Starting server...")
    Timer(1, open_browser).start()
    
    # Run the app
    app.run(debug=False)


if __name__ == "__main__":
    main()
