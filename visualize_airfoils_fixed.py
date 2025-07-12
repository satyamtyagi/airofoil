import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os
import dash
from dash import html, dcc
import webbrowser
from threading import Timer

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
    """Get airfoil performance data from the summary file"""
    try:
        performance_file = os.path.join(RESULTS_DIR, "airfoil_best_performance.csv")
        if os.path.exists(performance_file):
            df = pd.read_csv(performance_file)
            airfoil_data = df[df['airfoil'] == airfoil_name]
            if not airfoil_data.empty:
                return airfoil_data.iloc[0].to_dict()
    except Exception as e:
        print(f"Error getting performance data for {airfoil_name}: {e}")
    
    return None

def get_top_airfoils(n=5):
    """Get the top N airfoils by L/D ratio"""
    try:
        performance_file = os.path.join(RESULTS_DIR, "airfoil_best_performance.csv")
        if os.path.exists(performance_file):
            df = pd.read_csv(performance_file)
            top_airfoils = df.sort_values('best_ld', ascending=False).head(n)
            return top_airfoils['airfoil'].tolist()
    except Exception as e:
        print(f"Error getting top airfoils: {e}")
    
    # Default fallback
    return ['ag13', 'ag03', 'ag08', 'ag23', 'ag36']

def create_2d_profile_figure(airfoil_names):
    """Create a 2D figure of airfoil profiles with performance metrics"""
    # Create figure
    fig = go.Figure()
    
    # Color palette for different airfoils
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'pink', 'brown']
    
    # Performance table data for annotation
    table_data = []
    header = ['Airfoil', 'Thickness', 'Camber', 'Best L/D', 'Max CL']
    table_data.append(header)
    
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
        performance_data = get_airfoil_performance(airfoil_name)
        
        # Add airfoil to table
        if performance_data:
            thickness = performance_data.get('max_thickness', 'N/A')
            camber = performance_data.get('max_camber', 'N/A')
            best_ld = performance_data.get('best_ld', 'N/A')
            max_cl = performance_data.get('max_cl', 'N/A')
            table_data.append([airfoil_name, f"{thickness:.4f}", f"{camber:.4f}", f"{best_ld:.2f}", f"{max_cl:.4f}"])
        else:
            table_data.append([airfoil_name, 'N/A', 'N/A', 'N/A', 'N/A'])
        
        # Create hover text
        hover_text = f"Airfoil: {airfoil_name}<br>"
        if performance_data:
            hover_text += f"Thickness: {thickness:.4f}<br>"
            hover_text += f"Camber: {camber:.4f}<br>"
            hover_text += f"Best L/D: {best_ld:.2f}<br>"
            hover_text += f"Max CL: {max_cl:.4f}"
        
        # Add a line trace showing the airfoil profile
        fig.add_trace(go.Scatter(
            x=x_coords,
            y=y_coords,
            mode='lines',
            line=dict(color=color, width=2),
            name=airfoil_name,
            text=hover_text,
            hovertemplate=hover_text + '<extra></extra>'
        ))

    # Update layout
    fig.update_layout(
        xaxis_title='X (Chord)',
        yaxis_title='Y (Thickness)',
        title='Airfoil Profile Comparison',
        title_x=0.5,
        yaxis=dict(
            scaleanchor='x',
            scaleratio=1,
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )
    
    return fig, table_data

def create_3d_airfoil_figure(airfoil_names):
    """Create a 3D figure of airfoil shapes"""
    # Create the 3D figure
    fig = go.Figure()
    
    # Process each airfoil
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'pink', 'brown']
    span = 1.0
    num_sections = 10
    
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
        performance_data = get_airfoil_performance(airfoil_name)
        
        # Create 3D wing by extruding airfoil profile
        z_positions = np.linspace(0, span, num_sections)
        z_offset = idx * span * 1.5  # Separate different airfoils along z-axis
        
        # Create hover text with performance data
        hover_text = f"Airfoil: {airfoil_name}<br>"
        if performance_data:
            hover_text += f"Thickness: {performance_data.get('max_thickness', 'N/A'):.4f}<br>"
            hover_text += f"Camber: {performance_data.get('max_camber', 'N/A'):.4f}<br>"
            hover_text += f"Best L/D: {performance_data.get('best_ld', 'N/A'):.2f}<br>"
            hover_text += f"Max CL: {performance_data.get('max_cl', 'N/A'):.4f}"

        # For simplicity, just add profile at root, mid, and tip
        for z_pos in [0, span/2, span]:
            z = z_pos + z_offset
            fig.add_trace(go.Scatter3d(
                x=x_coords,
                y=y_coords,
                z=[z] * len(x_coords),
                mode='lines',
                line=dict(color=color, width=4),
                name=airfoil_name,
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
            text=[airfoil_name],
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
        title='3D Airfoil Shape Comparison',
        margin=dict(l=0, r=0, b=0, t=40),
    )
    
    return fig

def create_performance_radar_chart(airfoil_names):
    """Create a radar chart comparing key performance metrics"""
    fig = go.Figure()
    
    # Color palette for different airfoils
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'pink', 'brown']
    
    # Get all performance data first to normalize
    all_performance = []
    for airfoil_name in airfoil_names:
        performance_data = get_airfoil_performance(airfoil_name)
        if performance_data:
            all_performance.append(performance_data)
    
    if not all_performance:
        return None
        
    # Calculate max values for normalization
    max_ld = max([p.get('best_ld', 0) for p in all_performance])
    max_cl = max([p.get('max_cl', 0) for p in all_performance])
    min_cd = min([p.get('min_cd', 1) for p in all_performance])
    
    # Categories for radar chart
    categories = ['L/D Ratio', 'Max Lift', 'Min Drag', 'Alpha for best L/D', 'Thickness']
    
    # Add trace for each airfoil
    for idx, airfoil_name in enumerate(airfoil_names):
        performance_data = get_airfoil_performance(airfoil_name)
        if not performance_data:
            continue
            
        # Normalize values to 0-1 range
        ld_norm = performance_data.get('best_ld', 0) / max_ld if max_ld else 0
        cl_norm = performance_data.get('max_cl', 0) / max_cl if max_cl else 0
        cd_norm = min_cd / performance_data.get('min_cd', 1) if performance_data.get('min_cd', 0) > 0 else 0
        alpha_norm = 1.0 if performance_data.get('best_ld_alpha', 0) == 5 else 0.5  # Optimal is usually 5
        thickness_norm = 1.0 - abs(0.12 - performance_data.get('max_thickness', 0)) / 0.12  # Normalize around 12%
        
        values = [ld_norm, cl_norm, cd_norm, alpha_norm, thickness_norm]
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name=airfoil_name,
            line_color=colors[idx % len(colors)]
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        title="Airfoil Performance Comparison",
        showlegend=True
    )
    
    return fig

def open_browser():
    webbrowser.open_new("http://127.0.0.1:8050/")

# Main function to run the dashboard
def run_dashboard():
    # Get top performing airfoils
    airfoil_names = get_top_airfoils(5)
    print(f"Creating visualization for airfoils: {', '.join(airfoil_names)}")
    
    # Create figures
    profile_fig, table_data = create_2d_profile_figure(airfoil_names)
    shape_fig = create_3d_airfoil_figure(airfoil_names)
    radar_fig = create_performance_radar_chart(airfoil_names)
    
    # Create Dash app
    app = dash.Dash(__name__)
    
    # Create the layout
    app.layout = html.Div(style={'maxWidth': '1200px', 'margin': '0 auto'}, children=[
        html.H1("Airfoil Analysis Dashboard", style={'textAlign': 'center'}),
        
        html.Div([
            html.H2("Performance Metrics", style={'textAlign': 'center'}),
            html.Table([
                html.Tr([html.Th(col) for col in table_data[0]]),
                *[html.Tr([html.Td(cell) for cell in row]) for row in table_data[1:]]
            ], style={'width': '100%', 'textAlign': 'center', 'border': '1px solid black'})
        ]),
        
        html.Div([
            html.Div([
                html.H2("2D Airfoil Profiles", style={'textAlign': 'center'}),
                dcc.Graph(figure=profile_fig)
            ], style={'width': '48%', 'display': 'inline-block'}),
            
            html.Div([
                html.H2("Performance Comparison", style={'textAlign': 'center'}),
                dcc.Graph(figure=radar_fig) if radar_fig else html.P("No performance data available")
            ], style={'width': '48%', 'display': 'inline-block'}),
        ]),
        
        html.Div([
            html.H2("3D Airfoil Shapes", style={'textAlign': 'center'}),
            dcc.Graph(figure=shape_fig, style={'height': '600px'})
        ]),
        
        html.Div([
            html.P("Hover over the airfoil shapes and plots for detailed performance metrics", 
                  style={'textAlign': 'center', 'fontStyle': 'italic'})
        ]),
    ])
    
    # Open browser after a short delay
    def open_browser_custom():
        webbrowser.open_new("http://127.0.0.1:8051/")
    
    Timer(1, open_browser_custom).start()
    
    # Run the server - FIXED: use app.run() instead of app.run_server()
    print("Starting dashboard server...")
    print("Please wait for your browser to open automatically.")
    print("If it doesn't open, navigate to http://127.0.0.1:8051/")
    
    app.run(debug=False, use_reloader=False, port=8051)

if __name__ == "__main__":
    run_dashboard()
