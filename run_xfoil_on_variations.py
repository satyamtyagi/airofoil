#!/usr/bin/env python3
"""
XFOIL Analysis of Top Airfoil Variations

This script runs XFOIL on the top 10 airfoil variations from our surrogate model
to validate their performance and compare with surrogate model predictions.
"""

import os
import subprocess
import time
import re
import datetime
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# CONFIG
# Path to XFOIL executable
XFOIL_PATH = "/Users/satyamtyagi/CascadeProjects/xfoil-mac/bin/xfoil"
AIRFOIL_DIR = "simple_variations_dat"  # Directory with variation DAT files
RESULTS_DIR = "variation_results/xfoil_analysis"  # Directory for results
ANGLES = [-5, 0, 5, 10]  # Analyze at these angles
REYNOLDS = 100000
MACH = 0.0
TIMEOUT = 20  # Timeout in seconds per angle of attack

# Top 10 performing airfoils from surrogate model
TOP_PERFORMERS = [
    "a18sm_y_up_-25pct",
    "ag27_rLE_+10pct",
    "ag27_x_up_-5pct",
    "ag26_ypp_up_+25pct",
    "ag26_ypp_lo_+10pct",
    "ag27_y_up_-5pct",
    "ag27_te_wedge_-25pct",
    "ag27_ypp_lo_+5pct",
    "ag27_x_lo_+10pct",
    "ag25_x_up_+10pct"
]

# Load surrogate model predictions for comparison
def load_surrogate_predictions():
    """Load surrogate model predictions from CSV file"""
    surrogate_file = Path("variation_results/surrogate_variation_results.csv")
    if not surrogate_file.exists():
        print("Warning: Surrogate model results file not found")
        return {}
    
    df = pd.read_csv(surrogate_file)
    
    # Create a dictionary mapping airfoil names to their surrogate predictions
    surrogate_data = {}
    for _, row in df.iterrows():
        airfoil_name = row["airfoil"]
        if airfoil_name in TOP_PERFORMERS:
            surrogate_data[airfoil_name] = {
                "cl": row["cl"],
                "cd": row["cd"],
                "cl_cd": row["cl_cd"]
            }
    
    return surrogate_data

def setup():
    """Set up directories and files"""
    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Create a summary file
    summary_file = os.path.join(RESULTS_DIR, "xfoil_variations_summary.csv")
    with open(summary_file, "w") as f:
        f.write("airfoil,alpha,CL,CD,CM,L_D\n")
        
    return summary_file

def run_xfoil_analysis(summary_file, surrogate_data):
    """Run XFOIL analysis on top airfoil variations"""
    # Check if all DAT files exist
    missing_files = []
    for airfoil in TOP_PERFORMERS:
        dat_file = os.path.join(AIRFOIL_DIR, f"{airfoil}.dat")
        if not os.path.exists(dat_file):
            missing_files.append(airfoil)
    
    if missing_files:
        print(f"Warning: {len(missing_files)} DAT files not found:")
        for file in missing_files:
            print(f"  - {file}.dat")
        print("Please make sure all variation DAT files have been generated.")
        return
    
    # Start processing
    total_files = len(TOP_PERFORMERS)
    print(f"Analyzing {total_files} top-performing airfoil variations")
    print(f"Angles to analyze: {ANGLES}")
    print(f"Reynolds number: {REYNOLDS}, Mach: {MACH}")
    print("-" * 60)
    
    # Track statistics
    successful = 0
    failed = 0
    timeout_count = 0
    start_time = time.time()
    
    # Results to compare with surrogate model
    comparison_data = []
    
    # Process each airfoil
    for idx, airfoil_name in enumerate(TOP_PERFORMERS, 1):
        airfoil_file = os.path.join(AIRFOIL_DIR, f"{airfoil_name}.dat")
        results_file = os.path.join(RESULTS_DIR, f"{airfoil_name}_results.txt")
        
        print(f"\n[{idx}/{total_files}] Analyzing {airfoil_name}...")
        
        # Create a results file with headers
        with open(results_file, "w") as f:
            f.write(f"# Airfoil: {airfoil_name}\n")
            f.write(f"# Reynolds: {REYNOLDS}\n")
            f.write(f"# Mach: {MACH}\n")
            f.write(f"# {'Alpha':>8}{'CL':>10}{'CD':>10}{'CDp':>10}{'CM':>10}{'L/D':>10}\n")
        
        airfoil_success = False
        airfoil_results = []
        
        # Process each angle of attack separately
        for angle in ANGLES:
            sys.stdout.write(f"  Processing angle: {angle}° ... ")
            sys.stdout.flush()
            
            # Create a command file for this angle
            with open("xfoil_temp_cmd.txt", "w") as f:
                f.write("PLOP\n")       # Plot options
                f.write("G F\n")        # Graphics off
                f.write("\n")           # Exit menu
                f.write(f"LOAD {airfoil_file}\n")
                f.write("PANE\n")
                f.write("OPER\n")
                f.write(f"VISC {REYNOLDS}\n")
                f.write(f"MACH {MACH}\n")
                f.write("ITER 200\n")   # More iterations for better convergence
                f.write(f"ALFA {angle}\n")
                f.write("QUIT\n")

            try:
                # Run XFOIL with the temp file
                result = subprocess.run(
                    [XFOIL_PATH],
                    stdin=open("xfoil_temp_cmd.txt", "r"),
                    capture_output=True,
                    text=True,
                    timeout=TIMEOUT
                )
                
                output = result.stdout
                
                # Regex patterns to match XFOIL output
                cl_pattern = r"a =\s+[-+]?\d*\.\d+\s+CL =\s+([-+]?\d*\.\d+)"
                cd_pattern = r"CD =\s+([-+]?\d*\.\d+)"
                cdp_pattern = r"CDp =\s+([-+]?\d*\.\d+)"
                cm_pattern = r"Cm =\s+([-+]?\d*\.\d+)"
                
                # Find all matches in case there are multiple iterations
                cl_matches = re.findall(cl_pattern, output)
                cd_matches = re.findall(cd_pattern, output)
                cdp_matches = re.findall(cdp_pattern, output)
                cm_matches = re.findall(cm_pattern, output)
                
                # Take the last match if it exists (final converged value)
                if cl_matches and cd_matches and cm_matches:
                    cl = float(cl_matches[-1])
                    cd = float(cd_matches[-1])
                    cm = float(cm_matches[-1])
                    cdp = float(cdp_matches[-1]) if cdp_matches else 0.0
                    
                    # Calculate lift-to-drag ratio
                    ld_ratio = cl / cd if cd > 0 else 0
                    
                    print(f"CL: {cl:.4f}  CD: {cd:.6f}  L/D: {ld_ratio:.2f}")
                    airfoil_success = True
                    
                    # Add to results list
                    result_data = {
                        'alpha': angle,
                        'CL': cl,
                        'CD': cd,
                        'CDp': cdp,
                        'CM': cm,
                        'L/D': ld_ratio
                    }
                    airfoil_results.append(result_data)
                    
                    # Append to results file
                    with open(results_file, "a") as f:
                        f.write(f"{angle:8.2f}{cl:10.4f}{cd:10.6f}{cdp:10.6f}{cm:10.4f}{ld_ratio:10.2f}\n")
                    
                    # Append to summary file
                    with open(summary_file, "a") as f:
                        f.write(f"{airfoil_name},{angle},{cl},{cd},{cm},{ld_ratio}\n")
                        
                else:
                    print(f"Failed to extract data")
            
            except subprocess.TimeoutExpired:
                print(f"Timeout")
                timeout_count += 1
            except Exception as e:
                print(f"Error: {str(e)}")
        
        # Clean up
        try:
            os.remove("xfoil_temp_cmd.txt")
        except:
            pass
        
        # Update statistics
        if airfoil_success:
            successful += 1
            
            # Find the max L/D ratio and the corresponding angle
            if airfoil_results:
                # Find best L/D value
                best_ld_entry = max(airfoil_results, key=lambda r: r['L/D'])
                best_ld_angle = best_ld_entry['alpha']
                best_ld_value = best_ld_entry['L/D']
                best_cl = best_ld_entry['CL']
                best_cd = best_ld_entry['CD']
                
                # Add to comparison data for surrogate vs XFOIL comparison
                if airfoil_name in surrogate_data:
                    comparison_data.append({
                        'airfoil': airfoil_name,
                        'xfoil_ld': best_ld_value,
                        'xfoil_cl': best_cl,
                        'xfoil_cd': best_cd,
                        'xfoil_best_angle': best_ld_angle,
                        'surrogate_ld': surrogate_data[airfoil_name]['cl_cd'],
                        'surrogate_cl': surrogate_data[airfoil_name]['cl'],
                        'surrogate_cd': surrogate_data[airfoil_name]['cd']
                    })
            
            # Print summary for this airfoil
            print(f"  Results for {airfoil_name}:")
            print(f"  {'Alpha':>8}{'CL':>10}{'CD':>10}{'L/D':>10}")
            print(f"  {'-'*38}")
            
            for r in airfoil_results:
                print(f"  {r['alpha']:8.2f}{r['CL']:10.4f}{r['CD']:10.6f}{r['L/D']:10.2f}")
            
            if airfoil_name in surrogate_data:
                sur_ld = surrogate_data[airfoil_name]['cl_cd']
                sur_cl = surrogate_data[airfoil_name]['cl']
                sur_cd = surrogate_data[airfoil_name]['cd']
                print(f"\n  Surrogate prediction (at α=5°):")
                print(f"  {'CL':>10}{'CD':>10}{'L/D':>10}")
                print(f"  {sur_cl:10.4f}{sur_cd:10.6f}{sur_ld:10.2f}")
            
        else:
            failed += 1
            print(f"  ❌ No valid results for {airfoil_name}")
        
        # Progress and time estimate
        elapsed = time.time() - start_time
        avg_time_per_file = elapsed / idx
        remaining_files = total_files - idx
        estimated_remaining = avg_time_per_file * remaining_files
        
        print(f"\n  Progress: {idx}/{total_files} ({idx/total_files*100:.1f}%)")
        print(f"  Elapsed: {elapsed/60:.1f} min, Est. remaining: {estimated_remaining/60:.1f} min")
    
    # Print final summary
    end_time = time.time()
    total_time = end_time - start_time
    
    print("\n" + "="*60)
    print(f"Analysis complete!")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Processed {total_files} airfoils")
    print(f"Success: {successful}, Failed: {failed}, Timeouts: {timeout_count}")
    print(f"Results saved to {RESULTS_DIR} directory")
    print(f"Summary file: {summary_file}")
    print("="*60)
    
    return comparison_data

def generate_comparison_plots(comparison_data):
    """Generate plots comparing XFOIL and surrogate model results"""
    if not comparison_data:
        print("No comparison data available for plotting")
        return
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Save comparison data
    comparison_csv = os.path.join(RESULTS_DIR, "xfoil_vs_surrogate_comparison.csv")
    comparison_df.to_csv(comparison_csv, index=False)
    print(f"Saved comparison data to {comparison_csv}")
    
    # Create plots directory
    plots_dir = os.path.join(RESULTS_DIR, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. Bar chart comparing L/D ratios
    plt.figure(figsize=(12, 8))
    
    # Sort by surrogate L/D
    comparison_df = comparison_df.sort_values('surrogate_ld', ascending=False)
    
    # Set up bar positions
    airfoils = comparison_df['airfoil'].tolist()
    x = np.arange(len(airfoils))
    width = 0.35
    
    # Create bars
    plt.bar(x - width/2, comparison_df['surrogate_ld'], width, label='Surrogate Model (α=5°)')
    plt.bar(x + width/2, comparison_df['xfoil_ld'], width, label=f'XFOIL (Best α)')
    
    # Add text labels on the bars for XFOIL best angles
    for i, angle in enumerate(comparison_df['xfoil_best_angle']):
        plt.text(i + width/2, comparison_df['xfoil_ld'].iloc[i] + 5, 
                 f"α={angle}°", ha='center', va='bottom', fontsize=9)
    
    # Labels and title
    plt.xlabel('Airfoil Variation')
    plt.ylabel('Lift-to-Drag Ratio (L/D)')
    plt.title('Comparison of L/D Ratios: Surrogate Model vs. XFOIL')
    plt.xticks(x, airfoils, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(plots_dir, 'ld_comparison.png'), dpi=300)
    
    # 2. Scatter plot of surrogate L/D vs XFOIL L/D
    plt.figure(figsize=(10, 8))
    
    # Create scatter plot
    plt.scatter(comparison_df['surrogate_ld'], comparison_df['xfoil_ld'], s=100, alpha=0.7)
    
    # Add a diagonal line for reference (y=x)
    max_val = max(comparison_df['surrogate_ld'].max(), comparison_df['xfoil_ld'].max())
    plt.plot([0, max_val], [0, max_val], 'k--', alpha=0.5)
    
    # Add labels to points
    for i, txt in enumerate(comparison_df['airfoil']):
        plt.annotate(txt, 
                     (comparison_df['surrogate_ld'].iloc[i], comparison_df['xfoil_ld'].iloc[i]),
                     fontsize=9, alpha=0.8, xytext=(5, 5), textcoords='offset points')
    
    # Labels and title
    plt.xlabel('Surrogate Model L/D')
    plt.ylabel('XFOIL L/D')
    plt.title('Correlation Between Surrogate and XFOIL L/D Predictions')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(plots_dir, 'ld_correlation.png'), dpi=300)
    
    print(f"Saved comparison plots to {plots_dir}")
    
    # Create an HTML file summarizing the results
    create_summary_html(comparison_df, plots_dir)

def create_summary_html(comparison_df, plots_dir):
    """Create an HTML summary of XFOIL vs surrogate results"""
    html_file = os.path.join(RESULTS_DIR, "xfoil_validation_summary.html")
    
    # Calculate statistics
    avg_surrogate_ld = comparison_df['surrogate_ld'].mean()
    avg_xfoil_ld = comparison_df['xfoil_ld'].mean()
    max_surrogate_ld = comparison_df['surrogate_ld'].max()
    max_xfoil_ld = comparison_df['xfoil_ld'].max()
    
    # Calculate ratio of XFOIL to surrogate prediction
    comparison_df['ratio'] = comparison_df['xfoil_ld'] / comparison_df['surrogate_ld']
    avg_ratio = comparison_df['ratio'].mean()
    
    # Find best performer in XFOIL
    best_xfoil_idx = comparison_df['xfoil_ld'].idxmax()
    best_xfoil_airfoil = comparison_df.loc[best_xfoil_idx, 'airfoil']
    best_xfoil_ld = comparison_df.loc[best_xfoil_idx, 'xfoil_ld']
    best_xfoil_angle = comparison_df.loc[best_xfoil_idx, 'xfoil_best_angle']
    
    # Create the HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>XFOIL Validation Results</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
            }}
            h1, h2 {{
                color: #333;
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
            .plot-container {{
                margin: 30px 0;
                text-align: center;
            }}
            .plot-container img {{
                max-width: 100%;
                height: auto;
                box-shadow: 0 0 5px rgba(0,0,0,0.2);
            }}
            .summary-box {{
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
            <h1>XFOIL Validation of Top Performing Airfoil Variations</h1>
            
            <div class="summary-box">
                <h2>Key Findings</h2>
                <ul>
                    <li><strong>Best performer in XFOIL:</strong> {best_xfoil_airfoil} with L/D = {best_xfoil_ld:.2f} at α = {best_xfoil_angle}°</li>
                    <li><strong>Average surrogate L/D:</strong> {avg_surrogate_ld:.2f}</li>
                    <li><strong>Average XFOIL L/D:</strong> {avg_xfoil_ld:.2f}</li>
                    <li><strong>Ratio of XFOIL to surrogate:</strong> {avg_ratio:.3f} (average)</li>
                </ul>
            </div>
            
            <h2>Comparison Plots</h2>
            
            <div class="plot-container">
                <h3>L/D Ratio Comparison</h3>
                <img src="plots/ld_comparison.png" alt="L/D Ratio Comparison">
                <p>Comparison of L/D ratios between surrogate model predictions and XFOIL results.</p>
            </div>
            
            <div class="plot-container">
                <h3>Correlation Between Surrogate and XFOIL</h3>
                <img src="plots/ld_correlation.png" alt="L/D Correlation">
                <p>Scatter plot showing the correlation between surrogate model and XFOIL L/D predictions.</p>
            </div>
            
            <h2>Detailed Results</h2>
            
            <table>
                <tr>
                    <th>Airfoil</th>
                    <th>XFOIL L/D</th>
                    <th>XFOIL Best Angle</th>
                    <th>XFOIL CL</th>
                    <th>XFOIL CD</th>
                    <th>Surrogate L/D</th>
                    <th>Surrogate CL</th>
                    <th>Surrogate CD</th>
                    <th>Ratio<br>(XFOIL/Surrogate)</th>
                </tr>
    """
    
    # Add a row for each airfoil
    for _, row in comparison_df.sort_values('xfoil_ld', ascending=False).iterrows():
        ratio = row['xfoil_ld'] / row['surrogate_ld']
        html_content += f"""
                <tr>
                    <td>{row['airfoil']}</td>
                    <td>{row['xfoil_ld']:.2f}</td>
                    <td>{row['xfoil_best_angle']}°</td>
                    <td>{row['xfoil_cl']:.4f}</td>
                    <td>{row['xfoil_cd']:.6f}</td>
                    <td>{row['surrogate_ld']:.2f}</td>
                    <td>{row['surrogate_cl']:.4f}</td>
                    <td>{row['surrogate_cd']:.6f}</td>
                    <td>{ratio:.3f}</td>
                </tr>
        """
    
    html_content += """
            </table>
        </div>
    </body>
    </html>
    """
    
    # Write to file
    with open(html_file, "w") as f:
        f.write(html_content)
    
    print(f"Created summary HTML: {html_file}")
    
    # Try to open the HTML file in a browser
    try:
        import webbrowser
        from threading import Timer
        Timer(1.0, lambda: webbrowser.open(f'file://{os.path.abspath(html_file)}', new=2)).start()
    except:
        pass

def main():
    """Main function to run the analysis"""
    print("Starting XFOIL analysis of top airfoil variations...")
    
    # Load surrogate predictions for comparison
    surrogate_data = load_surrogate_predictions()
    
    # Set up directories and files
    summary_file = setup()
    
    # Run XFOIL analysis
    comparison_data = run_xfoil_analysis(summary_file, surrogate_data)
    
    # Generate comparison plots
    if comparison_data:
        generate_comparison_plots(comparison_data)
    
    print("Analysis complete!")

if __name__ == "__main__":
    main()
