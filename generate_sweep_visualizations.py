#!/usr/bin/env python3
"""
Generate visualizations for parameter sweep results to be displayed in an HTML page.
This creates a set of plots for the top-performing airfoils from the parameter sweep.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import h5py
import seaborn as sns
from parsec_to_dat import ParsecAirfoil
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Constants
RESULTS_DIR = "parameter_sweep_results"
SWEEP_RESULTS_FILE = os.path.join(RESULTS_DIR, "parsec_sweep_results_success.csv")
SWEEP_DB_FILE = os.path.join(RESULTS_DIR, "parsec_sweep.h5")
VISUALIZATION_DIR = os.path.join(RESULTS_DIR, "web_visualizations")

# Create output directory
os.makedirs(VISUALIZATION_DIR, exist_ok=True)

def plot_airfoil_shape(params, title, filename):
    """Generate an airfoil shape plot from PARSEC parameters"""
    try:
        airfoil = ParsecAirfoil()
        for param, value in params.items():
            if param in airfoil.params:
                airfoil.params[param] = value
        
        x_coords, y_coords = airfoil.generate_coordinates(200)
        
        plt.figure(figsize=(8, 4))
        plt.plot(x_coords, y_coords, 'b-', linewidth=2)
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.title(title)
        plt.xlabel('x/c')
        plt.ylabel('y/c')
        
        # Add parameter values as text
        param_text = "\n".join([f"{p}: {v:.4f}" for p, v in params.items() 
                              if p in ['rLE', 'Xup', 'Yup', 'Xlo', 'Ylo']])
        plt.figtext(0.02, 0.02, param_text, fontsize=8)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        plt.close()
        return True
    except Exception as e:
        print(f"Error plotting airfoil: {e}")
        return False

def create_performance_bar_charts(top_n_airfoils, metrics, filename):
    """Create bar charts comparing performance metrics across top airfoils"""
    plt.figure(figsize=(12, 8))
    
    # Create a subplot for each metric
    for i, metric in enumerate(metrics):
        plt.subplot(len(metrics), 1, i+1)
        
        # Extract metric values and airfoil indices
        values = top_n_airfoils[metric].values
        indices = np.arange(len(values))
        
        # Create bar chart
        bars = plt.bar(indices, values, color=plt.cm.viridis(np.linspace(0, 1, len(values))))
        
        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., 
                    height + (0.01 * max(values)),
                    f'{value:.4f}',
                    ha='center', va='bottom', rotation=0, fontsize=8)
        
        plt.title(f"Top {len(values)} Airfoils - {metric}")
        plt.ylabel(metric)
        plt.xlabel("Airfoil Rank")
        plt.xticks(indices, [str(i+1) for i in indices])
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

def create_parameter_heatmap(top_n_airfoils, params, filename):
    """Create a heatmap showing parameter values for top airfoils"""
    # Extract parameters data
    param_data = top_n_airfoils[params].copy()
    
    # Normalize each parameter column for better visualization
    for param in params:
        if param in param_data.columns:
            min_val = param_data[param].min()
            max_val = param_data[param].max()
            if max_val > min_val:  # Avoid division by zero
                param_data[param] = (param_data[param] - min_val) / (max_val - min_val)
    
    plt.figure(figsize=(12, len(param_data) * 0.5 + 2))
    
    # Create heatmap
    ax = sns.heatmap(param_data, annot=top_n_airfoils[params].round(4), 
                    fmt='.4f', cmap='viridis', linewidths=0.5, cbar_kws={'label': 'Normalized Value'})
    
    plt.title(f"Parameter Values for Top {len(param_data)} Airfoils")
    plt.ylabel("Airfoil Rank")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

def create_correlation_plot(df, filename):
    """Create correlation plot between parameters and performance metrics"""
    # Select parameters and metrics
    params = ['rLE', 'Xup', 'Yup', 'Xlo', 'Ylo']
    metrics = ['cl', 'cd', 'cm', 'cl_cd']
    
    # Create correlation matrix
    corr_data = df[params + metrics].corr().loc[params, metrics]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_data, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f')
    plt.title('Parameter-Performance Correlation')
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

def create_distribution_plot(df, filename):
    """Create a grid of histograms showing distribution of performance metrics"""
    metrics = ['cl', 'cd', 'cm', 'cl_cd']
    
    plt.figure(figsize=(12, 10))
    
    for i, metric in enumerate(metrics):
        plt.subplot(2, 2, i+1)
        sns.histplot(df[metric].dropna(), kde=True, bins=30)
        plt.title(f"{metric} Distribution")
        plt.xlabel(metric)
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

def create_pairplot(df, filename):
    """Create a pairplot showing relationships between key parameters"""
    params = ['rLE', 'Xup', 'Yup', 'Xlo', 'Ylo']
    
    # Sample data to keep plot manageable
    sample_size = min(1000, len(df))
    sample_df = df.sample(sample_size)
    
    g = sns.pairplot(sample_df[params], diag_kind='kde', plot_kws={'alpha': 0.6, 's': 10})
    g.fig.suptitle('Parameter Relationships', y=1.02)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

def create_contour_plots(df, filename):
    """Create contour plots for lift-to-drag ratio against parameter pairs"""
    params = ['rLE', 'Xup', 'Yup', 'Xlo', 'Ylo']
    
    # Create figure with subplots for each parameter pair
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    plot_idx = 0
    for i in range(len(params)):
        for j in range(i+1, len(params)):
            if plot_idx < len(axes):
                param1 = params[i]
                param2 = params[j]
                
                try:
                    # Create pivot table for contour plot
                    pivot = pd.pivot_table(
                        df, values='cl_cd', 
                        index=param1, columns=param2,
                        aggfunc='mean'
                    )
                    
                    # Create contour plot
                    if not pivot.empty:
                        X, Y = np.meshgrid(pivot.columns, pivot.index)
                        cont = axes[plot_idx].contourf(X, Y, pivot.values, 20, cmap='viridis')
                        
                        # Add labels
                        axes[plot_idx].set_xlabel(param2)
                        axes[plot_idx].set_ylabel(param1)
                        axes[plot_idx].set_title(f'{param1} vs {param2}')
                        
                        # Add colorbar
                        fig.colorbar(cont, ax=axes[plot_idx], label='L/D Ratio')
                        
                        plot_idx += 1
                except Exception as e:
                    print(f"Error creating contour plot for {param1} vs {param2}: {e}")
                    axes[plot_idx].text(0.5, 0.5, 'Error creating plot', 
                                      ha='center', va='center')
                    plot_idx += 1
    
    # Hide any unused subplots
    for i in range(plot_idx, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

def main():
    # Load parameter sweep results
    print("Loading parameter sweep results...")
    df = pd.read_csv(SWEEP_RESULTS_FILE)
    
    # Get top performers by lift-to-drag ratio
    top_n = 10
    top_airfoils = df.nlargest(top_n, 'cl_cd').copy()
    top_airfoils.reset_index(drop=True, inplace=True)
    
    # Key parameters and metrics
    params = ['rLE', 'Xup', 'Yup', 'Xlo', 'Ylo']
    metrics = ['cl', 'cd', 'cm', 'cl_cd']
    
    print(f"Generating visualizations for top {top_n} airfoils...")
    
    # Plot top airfoil shapes
    for i, (_, airfoil) in enumerate(top_airfoils.iterrows()):
        rank = i + 1
        ld_ratio = airfoil['cl_cd']
        title = f"Rank #{rank}: L/D = {ld_ratio:.2f}"
        filename = os.path.join(VISUALIZATION_DIR, f"airfoil_rank_{rank}.png")
        
        # Extract parameters for this airfoil
        airfoil_params = {}
        for param in params:
            airfoil_params[param] = airfoil[param]
        
        plot_airfoil_shape(airfoil_params, title, filename)
        print(f"Generated airfoil shape {rank}/{top_n}")
    
    # Create performance comparison chart
    print("Creating performance comparison chart...")
    filename = os.path.join(VISUALIZATION_DIR, "performance_comparison.png")
    create_performance_bar_charts(top_airfoils, metrics, filename)
    
    # Create parameter heatmap
    print("Creating parameter heatmap...")
    filename = os.path.join(VISUALIZATION_DIR, "parameter_heatmap.png")
    create_parameter_heatmap(top_airfoils, params, filename)
    
    # Create correlation plot
    print("Creating correlation plot...")
    filename = os.path.join(VISUALIZATION_DIR, "parameter_correlation.png")
    create_correlation_plot(df, filename)
    
    # Create distribution plot
    print("Creating distribution plots...")
    filename = os.path.join(VISUALIZATION_DIR, "metric_distributions.png")
    create_distribution_plot(df, filename)
    
    # Create parameter relationships plot
    print("Creating parameter relationship plot...")
    filename = os.path.join(VISUALIZATION_DIR, "parameter_relationships.png")
    create_pairplot(df, filename)
    
    # Create contour plots
    print("Creating contour plots...")
    filename = os.path.join(VISUALIZATION_DIR, "contour_plots.png")
    create_contour_plots(df, filename)
    
    print(f"All visualizations saved to {VISUALIZATION_DIR}")

if __name__ == "__main__":
    main()
