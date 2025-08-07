#!/usr/bin/env python3
"""
Compare Original Airfoil Data with PARSEC-Generated Airfoil Shapes

This script:
1. Selects a random sample of airfoils
2. Loads both the original .dat file and the PARSEC-generated .dat file
3. Creates side-by-side visual comparisons
4. Calculates error metrics to quantify the difference
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.interpolate import interp1d
from matplotlib.backends.backend_pdf import PdfPages

# Directory paths
ORIGINAL_DIR = "airfoils_uiuc"
GENERATED_DIR = "airfoils_parsec_dat"
OUTPUT_DIR = "comparison_results"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def read_airfoil_data(filename):
    """Read airfoil coordinates from a dat file"""
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Skip header if present (first line)
    start_line = 1
    
    # Try to parse first line - if it fails, it's a header
    try:
        x, y = map(float, lines[0].strip().split()[:2])
        start_line = 0
    except:
        pass
    
    coords = []
    for line in lines[start_line:]:
        if line.strip() and len(line.strip().split()) >= 2:
            try:
                x, y = map(float, line.strip().split()[:2])
                coords.append((x, y))
            except ValueError:
                continue
    
    if not coords:
        return None, None
    
    # Convert to arrays
    points = np.array(coords)
    x_data = points[:, 0]
    y_data = points[:, 1]
    
    return x_data, y_data


def normalize_airfoil_data(x, y):
    """Normalize airfoil coordinates to start at trailing edge and go counterclockwise"""
    # Find leading edge (minimum x)
    idx_le = np.argmin(x)
    
    # Split into upper and lower surfaces
    x_upper = x[:idx_le+1][::-1]  # Reverse to go from LE to TE
    y_upper = y[:idx_le+1][::-1]
    x_lower = x[idx_le:]
    y_lower = y[idx_le:]
    
    # Ensure we go counterclockwise: TE -> lower -> LE -> upper -> TE
    x_airfoil = np.concatenate([x_lower[::-1], x_upper[1:]])
    y_airfoil = np.concatenate([y_lower[::-1], y_upper[1:]])
    
    return x_airfoil, y_airfoil


def compute_error_metrics(x_orig, y_orig, x_gen, y_gen):
    """Compute error metrics between original and generated airfoil"""
    # Interpolate both curves to common x points
    x_common = np.linspace(0, 1, 200)
    
    # Split each airfoil into upper and lower surfaces
    idx_le_orig = np.argmin(x_orig)
    idx_le_gen = np.argmin(x_gen)
    
    # Original airfoil surfaces
    x_upper_orig = x_orig[:idx_le_orig+1]
    y_upper_orig = y_orig[:idx_le_orig+1]
    x_lower_orig = x_orig[idx_le_orig:]
    y_lower_orig = y_orig[idx_le_orig:]
    
    # Generated airfoil surfaces
    x_upper_gen = x_gen[:idx_le_gen+1]
    y_upper_gen = y_gen[:idx_le_gen+1]
    x_lower_gen = x_gen[idx_le_gen:]
    y_lower_gen = y_gen[idx_le_gen:]
    
    # Create interpolation functions
    try:
        f_upper_orig = interp1d(x_upper_orig, y_upper_orig, bounds_error=False, fill_value="extrapolate")
        f_lower_orig = interp1d(x_lower_orig, y_lower_orig, bounds_error=False, fill_value="extrapolate")
        f_upper_gen = interp1d(x_upper_gen, y_upper_gen, bounds_error=False, fill_value="extrapolate")
        f_lower_gen = interp1d(x_lower_gen, y_lower_gen, bounds_error=False, fill_value="extrapolate")
        
        # Evaluate at common points
        y_upper_orig_common = f_upper_orig(x_common)
        y_lower_orig_common = f_lower_orig(x_common)
        y_upper_gen_common = f_upper_gen(x_common)
        y_lower_gen_common = f_lower_gen(x_common)
        
        # Calculate mean squared error
        upper_mse = np.mean((y_upper_orig_common - y_upper_gen_common)**2)
        lower_mse = np.mean((y_lower_orig_common - y_lower_gen_common)**2)
        total_mse = (upper_mse + lower_mse) / 2
        
        # Calculate maximum absolute error
        upper_max_err = np.max(np.abs(y_upper_orig_common - y_upper_gen_common))
        lower_max_err = np.max(np.abs(y_lower_orig_common - y_lower_gen_common))
        total_max_err = max(upper_max_err, lower_max_err)
        
        return {
            'upper_mse': upper_mse,
            'lower_mse': lower_mse, 
            'total_mse': total_mse,
            'upper_max_err': upper_max_err,
            'lower_max_err': lower_max_err,
            'total_max_err': total_max_err
        }
    except:
        return {
            'upper_mse': float('nan'),
            'lower_mse': float('nan'),
            'total_mse': float('nan'),
            'upper_max_err': float('nan'),
            'lower_max_err': float('nan'),
            'total_max_err': float('nan')
        }


def compare_airfoils(sample_size=20):
    """Compare original and generated airfoils for a sample"""
    # Find common airfoils (exist in both directories)
    original_files = {os.path.splitext(f)[0] for f in os.listdir(ORIGINAL_DIR) if f.endswith('.dat')}
    generated_files = {os.path.splitext(f)[0] for f in os.listdir(GENERATED_DIR) if f.endswith('.dat')}
    common_airfoils = list(original_files & generated_files)
    
    # If we don't have enough common airfoils, reduce the sample size
    sample_size = min(sample_size, len(common_airfoils))
    if sample_size == 0:
        print("No common airfoils found between original and generated directories.")
        return
    
    # Select random sample
    random.seed(42)  # For reproducibility
    sample_airfoils = random.sample(common_airfoils, sample_size)
    print(f"Comparing {sample_size} random airfoils...")
    
    # Setup PDF file for saving all comparisons
    pdf_filename = os.path.join(OUTPUT_DIR, "airfoil_comparisons.pdf")
    with PdfPages(pdf_filename) as pdf:
        # Create a multi-page figure layout
        num_rows = 4
        num_cols = 4
        airfoils_per_page = num_rows * num_cols
        
        # Track overall errors
        all_mse = []
        all_max_err = []
        
        for page_idx in range((sample_size + airfoils_per_page - 1) // airfoils_per_page):
            # Create a new figure for this page
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 12))
            axes = axes.flatten()
            
            # Process airfoils for this page
            start_idx = page_idx * airfoils_per_page
            end_idx = min((page_idx + 1) * airfoils_per_page, sample_size)
            
            for i in range(start_idx, end_idx):
                airfoil_name = sample_airfoils[i]
                ax = axes[i % airfoils_per_page]
                
                # Load original airfoil data
                orig_path = os.path.join(ORIGINAL_DIR, f"{airfoil_name}.dat")
                x_orig, y_orig = read_airfoil_data(orig_path)
                if x_orig is None:
                    ax.text(0.5, 0.5, f"Failed to load {airfoil_name}", 
                           ha='center', va='center')
                    continue
                
                # Load generated airfoil data
                gen_path = os.path.join(GENERATED_DIR, f"{airfoil_name}.dat")
                x_gen, y_gen = read_airfoil_data(gen_path)
                if x_gen is None:
                    ax.text(0.5, 0.5, f"Failed to load generated {airfoil_name}", 
                           ha='center', va='center')
                    continue
                
                # Normalize data for plotting
                x_orig_norm, y_orig_norm = normalize_airfoil_data(x_orig, y_orig)
                
                # Compute error metrics
                errors = compute_error_metrics(x_orig, y_orig, x_gen, y_gen)
                all_mse.append(errors['total_mse'])
                all_max_err.append(errors['total_max_err'])
                
                # Plot comparison
                ax.plot(x_orig_norm, y_orig_norm, 'b-', linewidth=1.5, label='Original')
                ax.plot(x_gen, y_gen, 'r--', linewidth=1.5, label='PARSEC')
                
                ax.set_aspect('equal')
                ax.grid(True, alpha=0.3, linestyle='--')
                ax.set_title(f"{airfoil_name}")
                
                # Add error metrics to plot
                if not np.isnan(errors['total_mse']):
                    err_text = f"MSE: {errors['total_mse']:.2e}\nMax Err: {errors['total_max_err']:.2e}"
                    ax.text(0.95, 0.95, err_text, transform=ax.transAxes, 
                           ha='right', va='top', fontsize=8, 
                           bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
            
            # Turn off any unused subplots
            for j in range(end_idx - start_idx, airfoils_per_page):
                axes[j].axis('off')
            
            # Add legend to the first subplot
            if axes.size > 0:
                axes[0].legend(loc='lower right', fontsize=8)
            
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
        
        # Create a summary page
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot MSE distribution
        ax1.hist(all_mse, bins=20, color='blue', alpha=0.7)
        ax1.set_title('Distribution of Mean Squared Error')
        ax1.set_xlabel('MSE')
        ax1.set_ylabel('Count')
        ax1.grid(True, linestyle='--', alpha=0.5)
        
        # Plot Max Error distribution
        ax2.hist(all_max_err, bins=20, color='red', alpha=0.7)
        ax2.set_title('Distribution of Maximum Absolute Error')
        ax2.set_xlabel('Max Error')
        ax2.set_ylabel('Count')
        ax2.grid(True, linestyle='--', alpha=0.5)
        
        # Add summary statistics
        summary_text = (f"Summary Statistics (n={len(all_mse)}):\n"
                       f"Mean MSE: {np.mean(all_mse):.2e} ± {np.std(all_mse):.2e}\n"
                       f"Mean Max Error: {np.mean(all_max_err):.2e} ± {np.std(all_max_err):.2e}")
        fig.text(0.5, 0.01, summary_text, ha='center', fontsize=12)
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        pdf.savefig(fig)
        plt.close(fig)
    
    # Create a single display image with a few samples
    num_display_samples = min(15, sample_size)
    display_samples = sample_airfoils[:num_display_samples]
    
    # Create a grid for display
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    axes = axes.flatten()
    
    for i, airfoil_name in enumerate(display_samples):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        # Load original airfoil data
        orig_path = os.path.join(ORIGINAL_DIR, f"{airfoil_name}.dat")
        x_orig, y_orig = read_airfoil_data(orig_path)
        
        # Load generated airfoil data
        gen_path = os.path.join(GENERATED_DIR, f"{airfoil_name}.dat")
        x_gen, y_gen = read_airfoil_data(gen_path)
        
        if x_orig is not None and x_gen is not None:
            # Normalize data for plotting
            x_orig_norm, y_orig_norm = normalize_airfoil_data(x_orig, y_orig)
            
            # Compute error metrics
            errors = compute_error_metrics(x_orig, y_orig, x_gen, y_gen)
            
            # Plot comparison
            ax.plot(x_orig_norm, y_orig_norm, 'b-', linewidth=1.5, label='Original')
            ax.plot(x_gen, y_gen, 'r--', linewidth=1.5, label='PARSEC')
            
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_title(f"{airfoil_name}")
            
            # Add error metrics to plot
            if not np.isnan(errors['total_mse']):
                err_text = f"MSE: {errors['total_mse']:.2e}\nMax Err: {errors['total_max_err']:.2e}"
                ax.text(0.95, 0.95, err_text, transform=ax.transAxes, 
                       ha='right', va='top', fontsize=8, 
                       bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
    
    # Turn off any unused subplots
    for j in range(num_display_samples, len(axes)):
        axes[j].axis('off')
    
    # Add legend to the first subplot
    if axes.size > 0:
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.02), 
                  ncol=2, fontsize=12)
    
    plt.tight_layout(rect=[0, 0.04, 1, 0.98])
    plt.savefig(os.path.join(OUTPUT_DIR, "airfoil_comparison_sample.png"), dpi=150)
    plt.close(fig)  # Close the figure instead of showing it
    
    print(f"Comparison complete. Results saved to:")
    print(f"  - Detailed PDF: {pdf_filename}")
    print(f"  - Sample image: {os.path.join(OUTPUT_DIR, 'airfoil_comparison_sample.png')}")


if __name__ == "__main__":
    compare_airfoils(sample_size=16)
