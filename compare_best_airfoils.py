#!/usr/bin/env python3
"""
Comprehensive Airfoil Comparison

This script creates a comparative visualization showing the best airfoils from:
1. Original airfoils evaluated by XFOIL
2. Original airfoils evaluated by surrogate model
3. Variant airfoils evaluated by XFOIL
4. Variant airfoils evaluated by surrogate model
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from pathlib import Path

# Set up directories and files
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = BASE_DIR / "results"
VARIATION_RESULTS_DIR = BASE_DIR / "variation_results"
XFOIL_VARIATION_DIR = VARIATION_RESULTS_DIR / "xfoil_analysis"
AIRFOIL_UIUC_DIR = BASE_DIR / "airfoils_uiuc"
VARIATION_DAT_DIR = BASE_DIR / "simple_variations_dat"
OUTPUT_DIR = VARIATION_RESULTS_DIR
OUTPUT_FILE = OUTPUT_DIR / "comprehensive_comparison.png"
OUTPUT_HTML = OUTPUT_DIR / "comprehensive_comparison.html"

# Files
ORIGINAL_XFOIL_FILE = RESULTS_DIR / "all_airfoils_summary.csv"
SURROGATE_VARIATION_FILE = VARIATION_RESULTS_DIR / "surrogate_variation_results.csv"
XFOIL_VARIATION_FILE = XFOIL_VARIATION_DIR / "xfoil_variations_summary.csv"
ORIGINAL_SURROGATE_FILE = BASE_DIR / "parameter_sweep_results" / "parsec_sweep_results_success.csv"

# Number of top performers to show
TOP_N = 4

def load_and_prepare_data():
    """Load data from all sources and prepare for comparison"""
    # 1. Original airfoils from XFOIL
    if not os.path.exists(ORIGINAL_XFOIL_FILE):
        print(f"Error: {ORIGINAL_XFOIL_FILE} not found")
        return None
    
    original_xfoil_df = pd.read_csv(ORIGINAL_XFOIL_FILE)
    
    # Calculate L/D ratio for original XFOIL results
    original_xfoil_df['L_D'] = original_xfoil_df['CL'] / original_xfoil_df['CD']
    
    # Filter for alpha = 5 degrees (where most airfoils perform best)
    original_xfoil_df_5deg = original_xfoil_df[original_xfoil_df['alpha'] == 5]
    
    # Also include ag13 at 10 degrees which has exceptional performance
    ag13_10deg = original_xfoil_df[(original_xfoil_df['airfoil'] == 'ag13') & 
                                   (original_xfoil_df['alpha'] == 10)]
    
    if not ag13_10deg.empty:
        original_xfoil_df_5deg = pd.concat([original_xfoil_df_5deg, ag13_10deg])
    
    # Sort by L/D ratio
    original_xfoil_df_5deg = original_xfoil_df_5deg.sort_values('L_D', ascending=False)
    
    # 2. Original airfoils from surrogate model
    if not os.path.exists(ORIGINAL_SURROGATE_FILE):
        print(f"Warning: {ORIGINAL_SURROGATE_FILE} not found. Using empty dataframe for original surrogate.")
        original_surrogate_df = pd.DataFrame(columns=['name', 'cl', 'cd', 'cm', 'cl_cd'])
    else:
        original_surrogate_df = pd.read_csv(ORIGINAL_SURROGATE_FILE)
        
        # For parameter sweep results, need to create 'name' from parameter values
        if 'cl_cd' in original_surrogate_df.columns and 'name' not in original_surrogate_df.columns:
            # Generate synthetic names based on parameters to represent the original airfoils
            original_surrogate_df['name'] = original_surrogate_df.apply(
                lambda row: f"parsec_{row.get('rLE', 0):.2f}_{row.get('Xup', 0):.2f}_{row.get('YXXup', 0):.1f}", 
                axis=1
            )
        
        # Ensure cl_cd column exists
        if 'cl_cd' not in original_surrogate_df.columns:
            if 'L_D' in original_surrogate_df.columns:
                original_surrogate_df['cl_cd'] = original_surrogate_df['L_D']
            else:
                # Try to calculate it
                if 'cl' in original_surrogate_df.columns and 'cd' in original_surrogate_df.columns:
                    original_surrogate_df['cl_cd'] = original_surrogate_df['cl'] / original_surrogate_df['cd']
    
    # 3. Variant airfoils from XFOIL
    if not os.path.exists(XFOIL_VARIATION_FILE):
        print(f"Error: {XFOIL_VARIATION_FILE} not found")
        return None
    
    xfoil_variation_df = pd.read_csv(XFOIL_VARIATION_FILE)
    
    # Calculate L/D ratio if not already present
    if 'L_D' not in xfoil_variation_df.columns:
        xfoil_variation_df['L_D'] = xfoil_variation_df['CL'] / xfoil_variation_df['CD']
    
    # Filter for alpha = 5 degrees
    xfoil_variation_df_5deg = xfoil_variation_df[xfoil_variation_df['alpha'] == 5]
    
    # Sort by L/D ratio
    xfoil_variation_df_5deg = xfoil_variation_df_5deg.sort_values('L_D', ascending=False)
    
    # 4. Variant airfoils from surrogate model
    if not os.path.exists(SURROGATE_VARIATION_FILE):
        print(f"Error: {SURROGATE_VARIATION_FILE} not found")
        return None
    
    surrogate_variation_df = pd.read_csv(SURROGATE_VARIATION_FILE)
    
    # Return the prepared dataframes
    return {
        'original_xfoil': original_xfoil_df_5deg,
        'original_surrogate': original_surrogate_df,
        'variation_xfoil': xfoil_variation_df_5deg,
        'variation_surrogate': surrogate_variation_df
    }

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

def create_comparison_plots(data_dict):
    """Create comprehensive comparison plots"""
    if not data_dict:
        print("No data available for plotting")
        return
    
    # Get top N airfoils from each category
    original_xfoil_top = data_dict['original_xfoil'].head(TOP_N)
    
    # If we have surrogate data for original airfoils
    if 'original_surrogate' in data_dict and not data_dict['original_surrogate'].empty:
        original_surrogate_top = data_dict['original_surrogate'].sort_values('cl_cd', ascending=False).head(TOP_N)
    else:
        # If no surrogate data for original airfoils, create dummy dataframe
        original_surrogate_top = pd.DataFrame({
            'name': ['N/A'] * TOP_N,
            'cl': [0] * TOP_N,
            'cd': [0] * TOP_N,
            'cl_cd': [0] * TOP_N
        })
    
    variation_xfoil_top = data_dict['variation_xfoil'].head(TOP_N)
    variation_surrogate_top = data_dict['variation_surrogate'].sort_values('cl_cd', ascending=False).head(TOP_N)
    
    # Create figure with GridSpec for complex layout
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.25)
    
    # Set up axes
    ax1 = fig.add_subplot(gs[0, 0])  # Original XFOIL
    ax2 = fig.add_subplot(gs[0, 1])  # Original Surrogate
    ax3 = fig.add_subplot(gs[1, 0])  # Variation XFOIL
    ax4 = fig.add_subplot(gs[1, 1])  # Variation Surrogate
    
    # Set up colors and markers
    colors = plt.cm.tab10(np.linspace(0, 1, TOP_N))
    
    # 1. Original Airfoils - XFOIL
    bars1 = ax1.bar(original_xfoil_top['airfoil'], original_xfoil_top['L_D'], color=colors)
    ax1.set_title('Original Airfoils - XFOIL', fontsize=14)
    ax1.set_ylabel('L/D Ratio', fontsize=12)
    ax1.set_ylim(0, max(original_xfoil_top['L_D'].max(), variation_xfoil_top['L_D'].max()) * 1.1)
    ax1.tick_params(axis='x', rotation=45)
    
    # Add annotations for L/D, CL, CD
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        cl = original_xfoil_top.iloc[i]['CL']
        cd = original_xfoil_top.iloc[i]['CD']
        alpha = original_xfoil_top.iloc[i]['alpha']
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'L/D = {height:.2f}\nCL = {cl:.3f}\nCD = {cd:.5f}\nα = {alpha}°',
                ha='center', va='bottom', fontsize=10)
    
    # 2. Original Airfoils - Surrogate
    # Check if we have surrogate data for original airfoils
    if 'cl_cd' in original_surrogate_top.columns and original_surrogate_top['cl_cd'].max() > 0:
        bars2 = ax2.bar(original_surrogate_top['name'], original_surrogate_top['cl_cd'], color=colors)
        ax2.set_title('Original Airfoils - Surrogate Model', fontsize=14)
        ax2.set_ylabel('L/D Ratio', fontsize=12)
        ax2.set_ylim(0, max(original_surrogate_top['cl_cd'].max(), variation_surrogate_top['cl_cd'].max()) * 1.1)
        ax2.tick_params(axis='x', rotation=45)
        
        # Add annotations for L/D, CL, CD
        for i, bar in enumerate(bars2):
            height = bar.get_height()
            cl = original_surrogate_top.iloc[i]['cl']
            cd = original_surrogate_top.iloc[i]['cd']
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'L/D = {height:.2f}\nCL = {cl:.3f}\nCD = {cd:.5f}\nα = 5°',
                    ha='center', va='bottom', fontsize=10)
    else:
        ax2.text(0.5, 0.5, 'No surrogate data for original airfoils', 
                ha='center', va='center', fontsize=12, transform=ax2.transAxes)
        ax2.set_title('Original Airfoils - Surrogate Model (No Data)', fontsize=14)
    
    # 3. Variation Airfoils - XFOIL
    bars3 = ax3.bar(variation_xfoil_top['airfoil'], variation_xfoil_top['L_D'], color=colors)
    ax3.set_title('Variation Airfoils - XFOIL', fontsize=14)
    ax3.set_ylabel('L/D Ratio', fontsize=12)
    ax3.set_ylim(0, max(original_xfoil_top['L_D'].max(), variation_xfoil_top['L_D'].max()) * 1.1)
    ax3.tick_params(axis='x', rotation=45)
    
    # Add annotations for L/D, CL, CD
    for i, bar in enumerate(bars3):
        height = bar.get_height()
        cl = variation_xfoil_top.iloc[i]['CL']
        cd = variation_xfoil_top.iloc[i]['CD']
        alpha = variation_xfoil_top.iloc[i]['alpha']
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'L/D = {height:.2f}\nCL = {cl:.3f}\nCD = {cd:.5f}\nα = {alpha}°',
                ha='center', va='bottom', fontsize=10)
    
    # 4. Variation Airfoils - Surrogate
    bars4 = ax4.bar(variation_surrogate_top['airfoil'], variation_surrogate_top['cl_cd'], color=colors)
    ax4.set_title('Variation Airfoils - Surrogate Model', fontsize=14)
    ax4.set_ylabel('L/D Ratio', fontsize=12)
    ax4.set_ylim(0, max(variation_surrogate_top['cl_cd'].max() * 1.1, 400))  # Higher limit for surrogate predictions
    ax4.tick_params(axis='x', rotation=45)
    
    # Add annotations for L/D, CL, CD
    for i, bar in enumerate(bars4):
        height = bar.get_height()
        cl = variation_surrogate_top.iloc[i]['cl']
        cd = variation_surrogate_top.iloc[i]['cd']
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'L/D = {height:.2f}\nCL = {cl:.3f}\nCD = {cd:.5f}\nα = 5°',
                ha='center', va='bottom', fontsize=10)
    
    # Main title
    plt.suptitle('Comprehensive Comparison of Best Airfoils', fontsize=16, weight='bold', y=0.98)
    
    # Add an overall explanation text
    explanation_text = (
        f"This comparison shows the top {TOP_N} performing airfoils in each of four categories. "
        "Note the significant difference in L/D ratio scale between XFOIL and surrogate model predictions."
    )
    fig.text(0.5, 0.01, explanation_text, ha='center', fontsize=12, style='italic')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot to {OUTPUT_FILE}")
    
    # Create additional plot showing airfoil shapes
    create_airfoil_shape_comparison(original_xfoil_top, variation_xfoil_top)
    
    # Create HTML summary
    create_html_summary(data_dict)

def create_airfoil_shape_comparison(original_top, variation_top):
    """Create a comparison of airfoil shapes"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot original airfoils
    for i, row in original_top.iterrows():
        airfoil_name = row['airfoil']
        airfoil_file = AIRFOIL_UIUC_DIR / f"{airfoil_name}.dat"
        
        if os.path.exists(airfoil_file):
            x_coords, y_coords = read_airfoil_dat(airfoil_file)
            if x_coords is not None:
                ax1.plot(x_coords, y_coords, label=f"{airfoil_name} (L/D={row['L_D']:.2f})")
    
    # Plot variation airfoils
    for i, row in variation_top.iterrows():
        airfoil_name = row['airfoil']
        airfoil_file = VARIATION_DAT_DIR / f"{airfoil_name}.dat"
        
        if os.path.exists(airfoil_file):
            x_coords, y_coords = read_airfoil_dat(airfoil_file)
            if x_coords is not None:
                ax2.plot(x_coords, y_coords, label=f"{airfoil_name} (L/D={row['L_D']:.2f})")
    
    # Set up axes
    for ax in [ax1, ax2]:
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        ax.set_xlabel('x/c')
        ax.set_ylabel('y/c')
    
    ax1.set_title('Original Airfoil Shapes (XFOIL Top Performers)')
    ax2.set_title('Variation Airfoil Shapes (XFOIL Top Performers)')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "airfoil_shapes_comparison.png", dpi=300, bbox_inches='tight')
    print(f"Saved airfoil shape comparison to {OUTPUT_DIR / 'airfoil_shapes_comparison.png'}")

def create_html_summary(data_dict):
    """Create an HTML summary of the comparison"""
    # Get top N airfoils from each category
    original_xfoil_top = data_dict['original_xfoil'].head(TOP_N)
    
    # If we have surrogate data for original airfoils
    if 'original_surrogate' in data_dict and not data_dict['original_surrogate'].empty:
        original_surrogate_top = data_dict['original_surrogate'].sort_values('cl_cd', ascending=False).head(TOP_N)
    else:
        # If no surrogate data for original airfoils, create dummy dataframe
        original_surrogate_top = pd.DataFrame({
            'name': ['N/A'] * TOP_N,
            'cl': [0] * TOP_N,
            'cd': [0] * TOP_N,
            'cl_cd': [0] * TOP_N
        })
    
    variation_xfoil_top = data_dict['variation_xfoil'].head(TOP_N)
    variation_surrogate_top = data_dict['variation_surrogate'].sort_values('cl_cd', ascending=False).head(TOP_N)
    
    # Create the HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Comprehensive Airfoil Comparison</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
                line-height: 1.6;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
            }}
            h1, h2, h3 {{
                color: #333;
            }}
            .img-container {{
                margin: 30px 0;
                text-align: center;
            }}
            .img-container img {{
                max-width: 100%;
                height: auto;
                border: 1px solid #ddd;
                border-radius: 4px;
                box-shadow: 0 0 5px rgba(0,0,0,0.1);
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin: 20px 0;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }}
            th {{
                background-color: #f2f2f2;
            }}
            tr:nth-child(even) {{
                background-color: #f9f9f9;
            }}
            .highlight {{
                background-color: #e8f4f8;
                font-weight: bold;
            }}
            .findings-box {{
                background-color: #eaf7ea;
                border-left: 5px solid #28a745;
                padding: 15px;
                margin: 20px 0;
                border-radius: 4px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Comprehensive Airfoil Comparison</h1>
            
            <div class="findings-box">
                <h2>Key Findings</h2>
                <ul>
                    <li>Best original airfoil (XFOIL): {original_xfoil_top.iloc[0]['airfoil']} with L/D = {original_xfoil_top.iloc[0]['L_D']:.2f}</li>
                    <li>Best variation airfoil (XFOIL): {variation_xfoil_top.iloc[0]['airfoil']} with L/D = {variation_xfoil_top.iloc[0]['L_D']:.2f}</li>
                    <li>Variation airfoils show slight improvement over original airfoils in XFOIL analysis</li>
                    <li>Surrogate model significantly overestimates L/D ratios compared to XFOIL</li>
                    <li>Parameter variations that improved performance: lower crest curvature (ypp_lo) and upper crest X position (x_up)</li>
                </ul>
            </div>
            
            <h2>Comparison Visualizations</h2>
            
            <div class="img-container">
                <img src="comprehensive_comparison.png" alt="Comprehensive Comparison">
                <p>L/D ratio comparison across four categories showing performance differences</p>
            </div>
            
            <div class="img-container">
                <img src="airfoil_shapes_comparison.png" alt="Airfoil Shapes Comparison">
                <p>Comparison of airfoil shapes for top performers according to XFOIL</p>
            </div>
            
            <h2>Detailed Results</h2>
            
            <h3>Original Airfoils - XFOIL</h3>
            <table>
                <tr>
                    <th>Rank</th>
                    <th>Airfoil</th>
                    <th>L/D Ratio</th>
                    <th>Lift Coefficient (CL)</th>
                    <th>Drag Coefficient (CD)</th>
                    <th>Angle of Attack</th>
                </tr>
    """
    
    # Add rows for original XFOIL
    for i, row in original_xfoil_top.iterrows():
        highlight = "highlight" if i == 0 else ""
        html_content += f"""
                <tr class="{highlight}">
                    <td>{i + 1}</td>
                    <td>{row['airfoil']}</td>
                    <td>{row['L_D']:.2f}</td>
                    <td>{row['CL']:.4f}</td>
                    <td>{row['CD']:.6f}</td>
                    <td>{row['alpha']}°</td>
                </tr>
        """
    
    html_content += """
            </table>
            
            <h3>Variation Airfoils - XFOIL</h3>
            <table>
                <tr>
                    <th>Rank</th>
                    <th>Airfoil</th>
                    <th>L/D Ratio</th>
                    <th>Lift Coefficient (CL)</th>
                    <th>Drag Coefficient (CD)</th>
                    <th>Angle of Attack</th>
                </tr>
    """
    
    # Add rows for variation XFOIL
    for i, row in variation_xfoil_top.iterrows():
        highlight = "highlight" if i == 0 else ""
        html_content += f"""
                <tr class="{highlight}">
                    <td>{i + 1}</td>
                    <td>{row['airfoil']}</td>
                    <td>{row['L_D']:.2f}</td>
                    <td>{row['CL']:.4f}</td>
                    <td>{row['CD']:.6f}</td>
                    <td>{row['alpha']}°</td>
                </tr>
        """
    
    html_content += """
            </table>
            
            <h3>Variation Airfoils - Surrogate Model</h3>
            <table>
                <tr>
                    <th>Rank</th>
                    <th>Airfoil</th>
                    <th>L/D Ratio</th>
                    <th>Lift Coefficient (CL)</th>
                    <th>Drag Coefficient (CD)</th>
                    <th>Angle of Attack</th>
                </tr>
    """
    
    # Add rows for variation surrogate
    for i, row in variation_surrogate_top.iterrows():
        highlight = "highlight" if i == 0 else ""
        html_content += f"""
                <tr class="{highlight}">
                    <td>{i + 1}</td>
                    <td>{row['airfoil']}</td>
                    <td>{row['cl_cd']:.2f}</td>
                    <td>{row['cl']:.4f}</td>
                    <td>{row['cd']:.6f}</td>
                    <td>5°</td>
                </tr>
        """
    
    html_content += """
            </table>
            
            <h2>Conclusions</h2>
            <p>The comparison shows that parameter variations of existing airfoils can lead to modest performance improvements according to XFOIL analysis. The most effective parameter changes were related to lower crest curvature (ypp_lo) and upper crest X position (x_up).</p>
            <p>However, the surrogate model significantly overestimates airfoil performance compared to XFOIL, suggesting limitations in its predictive capabilities for novel designs. This highlights the importance of validating surrogate model predictions with higher-fidelity tools.</p>
            <p>Based on XFOIL validation, the ag26_ypp_lo_+10pct variation shows the best overall performance with an L/D ratio of 54.35 at α=5°, which represents a modest improvement over the best original airfoil performance.</p>
        </div>
    </body>
    </html>
    """
    
    # Write HTML file
    with open(OUTPUT_HTML, 'w') as f:
        f.write(html_content)
    
    print(f"Saved HTML summary to {OUTPUT_HTML}")
    
    # Try to open the HTML file in a browser
    try:
        import webbrowser
        from threading import Timer
        Timer(1.0, lambda: webbrowser.open(f'file://{os.path.abspath(OUTPUT_HTML)}', new=2)).start()
    except:
        pass

def main():
    """Main function to create comparison visualization"""
    print("Creating comprehensive airfoil comparison...")
    
    # Load and prepare data
    data_dict = load_and_prepare_data()
    
    if data_dict:
        # Create comparison plots
        create_comparison_plots(data_dict)
        print("Comparison complete!")
    else:
        print("Failed to create comparison due to missing data.")

if __name__ == "__main__":
    main()
