#!/usr/bin/env python3
"""
Visualize Parameter Relationships from PARSEC Sweep Results

This script:
1. Reads the CSV results from the PARSEC parameter sweep
2. Creates correlation plots between parameters and performance metrics
3. Generates 2D contour plots showing performance landscapes
4. Creates interactive plots for parameter exploration

Usage:
  python visualize_parameter_relationships.py [--file RESULTS_CSV]
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from matplotlib.backends.backend_pdf import PdfPages
from parsec_to_dat import ParsecAirfoil

# Directory for results
RESULTS_DIR = "parameter_sweep_results"
DEFAULT_RESULTS_FILE = os.path.join(RESULTS_DIR, "parsec_sweep_results_success.csv")
OUTPUT_DIR = os.path.join(RESULTS_DIR, "parameter_relationships")

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Parameter descriptions for better labels
PARAM_DESCRIPTIONS = {
    "rLE": "Leading Edge Radius",
    "Xup": "Upper Crest X Position",
    "Yup": "Upper Crest Y Position",
    "YXXup": "Upper Crest Curvature",
    "Xlo": "Lower Crest X Position",
    "Ylo": "Lower Crest Y Position",
    "YXXlo": "Lower Crest Curvature",
    "Xte": "Trailing Edge X Position",
    "Yte": "Trailing Edge Y Position",
    "Yte'": "Trailing Edge Direction",
    "Δyte''": "Trailing Edge Wedge Angle"
}

# Performance metrics descriptions
METRIC_DESCRIPTIONS = {
    "cl": "Lift Coefficient (CL)",
    "cd": "Drag Coefficient (CD)",
    "cm": "Moment Coefficient (CM)",
    "cl_cd": "Lift-to-Drag Ratio (L/D)"
}


def create_correlation_matrix(df, output_file):
    """Create correlation matrix between parameters and performance metrics"""
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Remove status column if present
    if 'status' in numeric_df.columns:
        numeric_df = numeric_df.drop('status', axis=1)
    
    # Calculate correlation matrix
    corr_matrix = numeric_df.corr()
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Plot heatmap
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', 
                linewidths=0.5, vmin=-1, vmax=1)
    
    plt.title('Correlation Matrix of PARSEC Parameters and Performance Metrics', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    
    return corr_matrix


def create_parameter_performance_plots(df, output_pdf):
    """Create plots showing relationship between each parameter and performance metrics"""
    # Get parameter columns (exclude performance metrics and status)
    param_columns = [col for col in df.columns if col in PARAM_DESCRIPTIONS]
    metric_columns = list(METRIC_DESCRIPTIONS.keys())
    
    with PdfPages(output_pdf) as pdf:
        # Create scatter plots for each parameter-metric combination
        for param in param_columns:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f"Effect of {PARAM_DESCRIPTIONS.get(param, param)} on Performance", fontsize=16)
            
            axes = axes.flatten()
            
            for i, metric in enumerate(metric_columns):
                ax = axes[i]
                
                # Create scatter plot
                ax.scatter(df[param], df[metric], alpha=0.5, s=10)
                
                # Add trend line
                try:
                    z = np.polyfit(df[param], df[metric], 1)
                    p = np.poly1d(z)
                    ax.plot(df[param], p(df[param]), "r--", linewidth=2)
                except:
                    pass
                
                ax.set_xlabel(PARAM_DESCRIPTIONS.get(param, param))
                ax.set_ylabel(METRIC_DESCRIPTIONS.get(metric, metric))
                ax.grid(True, alpha=0.3)
                
                # Add correlation coefficient
                corr = df[param].corr(df[metric])
                ax.annotate(f"Correlation: {corr:.3f}", 
                           xy=(0.05, 0.95), 
                           xycoords='axes fraction', 
                           fontsize=12, 
                           bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            pdf.savefig(fig)
            plt.close(fig)


def create_contour_plots(df, output_pdf):
    """Create 2D contour plots showing performance landscapes for pairs of parameters"""
    # Get parameter columns
    param_columns = [col for col in df.columns if col in PARAM_DESCRIPTIONS]
    
    # Only use parameters actually in the DataFrame (those that were varied)
    param_columns = [p for p in param_columns if df[p].nunique() > 1]
    
    # If we have more than 2 parameters, create all possible pairs
    if len(param_columns) > 1:
        param_pairs = [(param_columns[i], param_columns[j]) 
                     for i in range(len(param_columns)) 
                     for j in range(i+1, len(param_columns))]
    else:
        # Not enough parameters to create pairs
        return
    
    with PdfPages(output_pdf) as pdf:
        # For each pair of parameters, create contour plots for L/D ratio
        for param1, param2 in param_pairs:
            fig = plt.figure(figsize=(10, 8))
            
            # Check if we have enough unique values for a meaningful contour plot
            if df[param1].nunique() < 3 or df[param2].nunique() < 3:
                plt.text(0.5, 0.5, f"Not enough unique values for\n{param1} and {param2}\nto create a contour plot",
                       ha='center', va='center', fontsize=14)
                plt.title(f"Performance Landscape: {param1} vs {param2}")
                plt.axis('off')
                pdf.savefig(fig)
                plt.close(fig)
                continue
                
            # Create pivot table
            try:
                pivot = df.pivot_table(
                    values='cl_cd',
                    index=param1,
                    columns=param2,
                    aggfunc=np.mean
                )
                
                # Create contour plot
                X, Y = np.meshgrid(pivot.columns, pivot.index)
                
                plt.contourf(X, Y, pivot.values, 20, cmap='viridis')
                plt.colorbar(label='Lift-to-Drag Ratio (L/D)')
                
                plt.xlabel(PARAM_DESCRIPTIONS.get(param2, param2))
                plt.ylabel(PARAM_DESCRIPTIONS.get(param1, param1))
                plt.title(f"L/D Ratio Landscape: {param1} vs {param2}")
                
                # Mark the best point
                best_idx = df['cl_cd'].idxmax()
                best_x = df.loc[best_idx, param2]
                best_y = df.loc[best_idx, param1]
                plt.plot(best_x, best_y, 'r*', markersize=15)
                plt.annotate(f"Best L/D: {df.loc[best_idx, 'cl_cd']:.1f}",
                           (best_x, best_y), 
                           xytext=(10, 10), 
                           textcoords='offset points',
                           bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.8))
                
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)
                
            except Exception as e:
                print(f"Error creating contour plot for {param1} vs {param2}: {e}")
                plt.text(0.5, 0.5, f"Error creating contour plot for\n{param1} and {param2}\n{str(e)}",
                       ha='center', va='center', fontsize=12)
                plt.title(f"Performance Landscape: {param1} vs {param2}")
                plt.axis('off')
                pdf.savefig(fig)
                plt.close(fig)


def plot_top_performers_with_parameter_values(df, output_pdf, top_n=10):
    """Plot top performers with their parameter values and airfoil shapes"""
    # Get top n airfoils by L/D ratio
    top_airfoils = df.nlargest(top_n, 'cl_cd')
    
    # Parameters to plot
    param_columns = [col for col in df.columns if col in PARAM_DESCRIPTIONS]
    param_columns = [p for p in param_columns if p in top_airfoils.columns]
    
    with PdfPages(output_pdf) as pdf:
        # Title page
        fig = plt.figure(figsize=(12, 10))
        plt.axis('off')
        plt.text(0.5, 0.6, f"Top {top_n} Airfoils by Lift-to-Drag Ratio", 
                ha='center', fontsize=24, weight='bold')
        plt.text(0.5, 0.5, f"PARSEC Parameter Sweep Results", 
                ha='center', fontsize=20)
        plt.text(0.5, 0.4, f"Total combinations analyzed: {len(df)}", 
                ha='center', fontsize=16)
        plt.text(0.5, 0.3, f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d')}", 
                ha='center', fontsize=16)
        pdf.savefig(fig)
        plt.close(fig)
        
        # For each top airfoil
        for i, (_, row) in enumerate(top_airfoils.iterrows()):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8), 
                                         gridspec_kw={'width_ratios': [1, 1.5]})
            
            # Plot airfoil shape
            airfoil = ParsecAirfoil()
            for param in param_columns:
                if param in row.index:
                    airfoil.params[param] = row[param]
            
            try:
                x_coords, y_coords = airfoil.generate_coordinates(200)
                ax1.plot(x_coords, y_coords, 'b-', linewidth=3)
                ax1.set_aspect('equal')
                ax1.grid(True, alpha=0.3)
                ax1.set_title(f"#{i+1}: L/D = {row['cl_cd']:.2f}, CL = {row['cl']:.4f}, CD = {row['cd']:.6f}")
                ax1.set_xlabel('x/c')
                ax1.set_ylabel('y/c')
            except Exception as e:
                ax1.text(0.5, 0.5, f"Failed to generate airfoil:\n{str(e)}", 
                       ha='center', va='center', fontsize=12)
            
            # Create table of parameter values
            ax2.axis('off')
            ax2.set_title("Parameter Values", fontsize=14)
            
            # Format parameter table data
            param_data = []
            for param in param_columns:
                if param in row.index:
                    desc = PARAM_DESCRIPTIONS.get(param, param)
                    param_data.append([desc, f"{row[param]:.6f}"])
            
            # Add performance metrics
            param_data.append(["-" * 20, "-" * 10])  # Separator
            for metric in METRIC_DESCRIPTIONS:
                if metric in row.index:
                    desc = METRIC_DESCRIPTIONS.get(metric, metric)
                    param_data.append([desc, f"{row[metric]:.6f}"])
            
            # Create table
            table = ax2.table(cellText=param_data,
                           colLabels=["Parameter", "Value"],
                           loc='center',
                           cellLoc='left',
                           bbox=[0.1, 0.1, 0.8, 0.8])
            
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(1.2, 1.5)
            
            # Set properties for headers
            for k, cell in table._cells.items():
                if k[0] == 0:  # Header row
                    cell.set_text_props(weight='bold', color='white')
                    cell.set_facecolor('#4472C4')
                elif k[1] == 0:  # Parameter column
                    cell.set_text_props(weight='bold')
            
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Visualize parameter relationships from PARSEC sweep results"
    )
    parser.add_argument(
        "--file",
        type=str,
        default=DEFAULT_RESULTS_FILE,
        help=f"Results CSV file (default: {DEFAULT_RESULTS_FILE})"
    )
    return parser.parse_args()


def main():
    """Main function"""
    # Parse arguments
    args = parse_arguments()
    
    # Check if file exists
    if not os.path.exists(args.file):
        print(f"Error: Results file not found: {args.file}")
        return
    
    print(f"Loading results from: {args.file}")
    df = pd.read_csv(args.file)
    
    # Filter out any rows with NaN values in key metrics
    df = df.dropna(subset=['cl', 'cd', 'cm', 'cl_cd'])
    print(f"Analyzing {len(df)} valid results")
    
    # Create correlation matrix
    print("Creating correlation matrix...")
    output_file = os.path.join(OUTPUT_DIR, "parameter_correlation_matrix.png")
    create_correlation_matrix(df, output_file)
    print(f"Saved correlation matrix to: {output_file}")
    
    # Create parameter-performance plots
    print("Creating parameter-performance relationship plots...")
    output_pdf = os.path.join(OUTPUT_DIR, "parameter_performance_plots.pdf")
    create_parameter_performance_plots(df, output_pdf)
    print(f"Saved parameter-performance plots to: {output_pdf}")
    
    # Create 2D contour plots
    print("Creating 2D performance landscape contour plots...")
    output_pdf = os.path.join(OUTPUT_DIR, "performance_contour_plots.pdf")
    create_contour_plots(df, output_pdf)
    print(f"Saved performance contour plots to: {output_pdf}")
    
    # Plot top performers with parameter values
    print("Creating detailed view of top performers...")
    output_pdf = os.path.join(OUTPUT_DIR, "top_performers_detailed.pdf")
    plot_top_performers_with_parameter_values(df, output_pdf, top_n=10)
    print(f"Saved detailed top performers to: {output_pdf}")
    
    print("All visualizations complete!")
    print(f"Results available in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
