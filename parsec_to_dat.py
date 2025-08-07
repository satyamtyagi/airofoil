#!/usr/bin/env python3
"""
PARSEC to Airfoil Coordinates Converter

This script takes PARSEC parameter files and generates airfoil coordinate (.dat) files
that can be used by XFOIL or surrogate models.

The script:
1. Reads PARSEC parameter files from airfoils_parsec directory
2. Generates airfoil coordinates by evaluating the PARSEC polynomials
3. Saves the coordinates in standard airfoil .dat format in a new directory
4. Ensures proper coordinate density and formatting for XFOIL compatibility
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import re

# Define directories
INPUT_DIR = "airfoils_parsec"
OUTPUT_DIR = "airfoils_parsec_dat"
RESULTS_DIR = "parsec_results"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

class ParsecAirfoil:
    """Class for generating airfoil coordinates from PARSEC parameters"""
    
    def __init__(self, name="parsec"):
        """Initialize with default parameters"""
        self.name = name
        self.params = {
            "rLE": 0.01,      # Leading edge radius
            "Xup": 0.4,       # Upper crest position X
            "Yup": 0.06,      # Upper crest position Y
            "YXXup": -0.5,    # Upper crest curvature
            "Xlo": 0.4,       # Lower crest position X
            "Ylo": -0.06,     # Lower crest position Y
            "YXXlo": 0.5,     # Lower crest curvature
            "Xte": 1.0,       # Trailing edge position X
            "Yte": 0.0,       # Trailing edge position Y
            "Yte'": 0.0,      # Trailing edge direction
            "Δyte''": 0.0     # Trailing edge wedge angle
        }
        self.coeffs_upper = None
        self.coeffs_lower = None
    
    def load_from_file(self, filename):
        """Load PARSEC parameters from a file"""
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        # Skip comments
        param_lines = [line for line in lines if not line.strip().startswith('#')]
        
        # Extract parameters using regex
        for line in param_lines:
            match = re.match(r'([^\s=]+)\s*=\s*([0-9.-]+)', line)
            if match:
                param_name, param_value = match.groups()
                if param_name in self.params:
                    self.params[param_name] = float(param_value)
        
        # Calculate coefficients after loading parameters
        self._calculate_coefficients()
        
        return True
    
    def _calculate_coefficients(self):
        """Calculate polynomial coefficients from PARSEC parameters"""
        # Extract parameters
        rLE = self.params["rLE"]
        Xup = self.params["Xup"]
        Yup = self.params["Yup"]
        YXXup = self.params["YXXup"]
        Xlo = self.params["Xlo"]
        Ylo = self.params["Ylo"]
        YXXlo = self.params["YXXlo"]
        Xte = self.params["Xte"]
        Yte = self.params["Yte"]
        dYte = self.params["Yte'"]
        d2Yte = self.params["Δyte''"]
        
        # For upper surface
        A_up = np.array([
            [1, 0.5, 0.25, 0.125, 0.0625, 0.03125],
            [1, Xup, Xup**2, Xup**3, Xup**4, Xup**5],
            [0, 1, 2*Xup, 3*Xup**2, 4*Xup**3, 5*Xup**4],
            [0, 0, 2, 6*Xup, 12*Xup**2, 20*Xup**3],
            [1, Xte, Xte**2, Xte**3, Xte**4, Xte**5],
            [0, 1, 2*Xte, 3*Xte**2, 4*Xte**3, 5*Xte**4]
        ])
        
        b_up = np.array([
            [0],
            [Yup],
            [0],
            [YXXup],
            [Yte],
            [dYte + 0.5 * d2Yte]
        ])
        
        # For lower surface
        A_lo = np.array([
            [1, 0.5, 0.25, 0.125, 0.0625, 0.03125],
            [1, Xlo, Xlo**2, Xlo**3, Xlo**4, Xlo**5],
            [0, 1, 2*Xlo, 3*Xlo**2, 4*Xlo**3, 5*Xlo**4],
            [0, 0, 2, 6*Xlo, 12*Xlo**2, 20*Xlo**3],
            [1, Xte, Xte**2, Xte**3, Xte**4, Xte**5],
            [0, 1, 2*Xte, 3*Xte**2, 4*Xte**3, 5*Xte**4]
        ])
        
        b_lo = np.array([
            [0],
            [Ylo],
            [0],
            [YXXlo],
            [Yte],
            [dYte - 0.5 * d2Yte]
        ])
        
        # Calculate coefficients by solving linear systems
        try:
            self.coeffs_upper = np.linalg.solve(A_up, b_up).flatten()
            self.coeffs_lower = np.linalg.solve(A_lo, b_lo).flatten()
        except np.linalg.LinAlgError:
            # If matrices are singular, use least squares
            self.coeffs_upper = np.linalg.lstsq(A_up, b_up, rcond=None)[0].flatten()
            self.coeffs_lower = np.linalg.lstsq(A_lo, b_lo, rcond=None)[0].flatten()
        
        # Check leading edge radius constraint
        y2_le = 2 * self.coeffs_upper[2]
        if rLE > 0 and y2_le != 0:
            # Leading edge radius constraint: 1/rLE = 2*a2
            scale = (1/rLE) / y2_le
            self.coeffs_upper = self.coeffs_upper * scale
            self.coeffs_lower = self.coeffs_lower * scale
    
    def evaluate(self, x):
        """Evaluate airfoil shape at given x-coordinates"""
        if self.coeffs_upper is None or self.coeffs_lower is None:
            self._calculate_coefficients()
        
        y_upper = np.zeros_like(x)
        y_lower = np.zeros_like(x)
        
        # Polynomial: y = sum(a_i * x^(i/2)) for i=0..5
        for i in range(6):
            y_upper += self.coeffs_upper[i] * x**(i/2)
            y_lower += self.coeffs_lower[i] * x**(i/2)
        
        return y_upper, y_lower
    
    def generate_coordinates(self, num_points=100):
        """Generate airfoil coordinates with proper distribution"""
        # Generate x-coordinates with cosine spacing for better LE resolution
        beta = np.linspace(0, np.pi, num_points)
        x = 0.5 * (1 - np.cos(beta))  # Cosine spacing from 0 to 1
        
        # Evaluate upper and lower surfaces
        y_upper, y_lower = self.evaluate(x)
        
        # Organize coordinates to go around the airfoil counterclockwise
        # Starting from trailing edge, going to leading edge on the lower surface
        # and back to trailing edge on the upper surface
        x_airfoil = np.concatenate([x[::-1], x[1:]])
        y_airfoil = np.concatenate([y_lower[::-1], y_upper[1:]])
        
        return x_airfoil, y_airfoil
    
    def save_to_dat(self, filename, num_points=100):
        """Save airfoil coordinates to a .dat file in XFOIL format"""
        x_airfoil, y_airfoil = self.generate_coordinates(num_points)
        
        with open(filename, 'w') as f:
            # Write header
            f.write(f"{self.name}\n")
            
            # Write coordinates
            for i in range(len(x_airfoil)):
                f.write(f"{x_airfoil[i]:.6f}  {y_airfoil[i]:.6f}\n")
        
        return True
    
    def plot(self, ax=None):
        """Plot the airfoil shape"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4))
        
        x_airfoil, y_airfoil = self.generate_coordinates(200)
        ax.plot(x_airfoil, y_airfoil, 'b-', linewidth=2)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(f"PARSEC Airfoil: {self.name}")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
        return ax


def convert_parsec_files_to_dat():
    """Convert all PARSEC parameter files to DAT airfoil files"""
    # Check if input directory exists
    if not os.path.exists(INPUT_DIR):
        print(f"Error: Input directory '{INPUT_DIR}' does not exist.")
        print("Make sure you have run parsec_fit.py first to generate PARSEC parameter files.")
        return
    
    # List all .parsec files
    parsec_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.parsec')]
    if not parsec_files:
        print(f"No PARSEC parameter files found in '{INPUT_DIR}'.")
        return
    
    print(f"Converting {len(parsec_files)} PARSEC files to airfoil coordinates...")
    
    # Create a visualization grid for sample airfoils
    num_sample = min(12, len(parsec_files))
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    axes = axes.flatten()
    sample_indices = np.linspace(0, len(parsec_files)-1, num_sample, dtype=int)
    
    successful = 0
    failed = 0
    
    for i, filename in enumerate(tqdm(parsec_files)):
        airfoil_name = os.path.splitext(filename)[0]
        input_path = os.path.join(INPUT_DIR, filename)
        output_path = os.path.join(OUTPUT_DIR, f"{airfoil_name}.dat")
        
        try:
            # Create airfoil from PARSEC parameters
            airfoil = ParsecAirfoil(name=airfoil_name)
            if not airfoil.load_from_file(input_path):
                print(f"  Failed to load parameters for {filename}")
                failed += 1
                continue
            
            # Save to DAT file
            if not airfoil.save_to_dat(output_path):
                print(f"  Failed to save coordinates for {filename}")
                failed += 1
                continue
            
            # Add to visualization if this is a sample airfoil
            if i in sample_indices:
                sample_idx = np.where(sample_indices == i)[0][0]
                if sample_idx < len(axes):
                    airfoil.plot(axes[sample_idx])
            
            successful += 1
            
        except Exception as e:
            print(f"  Error processing {filename}: {str(e)}")
            failed += 1
    
    # Save visualization
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "parsec_generated_airfoils.png"), dpi=150)
    plt.close(fig)
    
    print(f"\nConversion complete:")
    print(f"  - Successfully converted: {successful}")
    print(f"  - Failed conversions: {failed}")
    print(f"  - Airfoil DAT files saved to: {OUTPUT_DIR}")
    print(f"  - Sample visualization saved to: {os.path.join(RESULTS_DIR, 'parsec_generated_airfoils.png')}")


def generate_parametric_airfoil(output_path, params=None):
    """Generate a single airfoil with specific PARSEC parameters"""
    airfoil = ParsecAirfoil(name="parametric_airfoil")
    
    # Update with provided parameters if any
    if params:
        for key, value in params.items():
            if key in airfoil.params:
                airfoil.params[key] = value
    
    # Recalculate coefficients with updated parameters
    airfoil._calculate_coefficients()
    
    # Save to DAT file
    airfoil.save_to_dat(output_path)
    
    # Create a visualization
    fig, ax = plt.subplots(figsize=(10, 5))
    airfoil.plot(ax)
    plt.savefig(f"{output_path}.png")
    plt.close(fig)
    
    return True


if __name__ == "__main__":
    # Convert all PARSEC files to DAT format
    convert_parsec_files_to_dat()
    
    print("\nExample: Generating a custom parametric airfoil...")
    
    # Example of generating a custom parametric airfoil
    custom_params = {
        "rLE": 0.02,      # Leading edge radius
        "Xup": 0.4,       # Upper crest position X
        "Yup": 0.06,      # Upper crest position Y
        "YXXup": -0.5,    # Upper crest curvature
        "Xlo": 0.4,       # Lower crest position X
        "Ylo": -0.04,     # Lower crest position Y
        "YXXlo": 0.6,     # Lower crest curvature
        "Xte": 1.0,       # Trailing edge position X
        "Yte": 0.0,       # Trailing edge position Y
        "Yte'": 0.0,      # Trailing edge direction
        "Δyte''": 0.1     # Trailing edge wedge angle
    }
    
    generate_parametric_airfoil(os.path.join(OUTPUT_DIR, "custom_parametric.dat"), custom_params)
