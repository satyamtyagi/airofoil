#!/usr/bin/env python3
"""
Run Surrogate Model on PARSEC Variations

This script evaluates the airfoil performance of all generated PARSEC parameter variations
using the trained surrogate model. It identifies and reports the top 10 performing airfoils
based on the predicted lift-to-drag ratio.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import interp1d

# Directory paths
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DAT_DIR = BASE_DIR / "simple_variations_dat"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "variation_results"
RESULTS_FILE = RESULTS_DIR / "surrogate_variation_results.csv"

# Create results directory if it doesn't exist
RESULTS_DIR.mkdir(exist_ok=True)

# Angle of attack for evaluation (5 degrees)
ALPHA = 5.0

# Surrogate model class (from parsec_parameter_sweep.py)
class SurrogateModel:
    def __init__(self, weights):
        self.weights = weights
    
    def forward(self, x):
        x = torch.matmul(x, self.weights['net.0.weight'].t()) + self.weights['net.0.bias']
        x = torch.relu(x)
        x = torch.matmul(x, self.weights['net.2.weight'].t()) + self.weights['net.2.bias']
        x = torch.relu(x)
        x = torch.matmul(x, self.weights['net.4.weight'].t()) + self.weights['net.4.bias']
        return x

def load_surrogate_model():
    """Load the surrogate model from file"""
    try:
        model_path = MODELS_DIR / "surrogate_model_retrained.pt"
        if not model_path.exists():
            model_path = MODELS_DIR / "surrogate_model.pt"
            
        if not model_path.exists():
            print(f"Error: No surrogate model found in {MODELS_DIR}")
            print("Using random values as placeholder. Add model files to 'models/' directory.")
            return None
            
        surrogate_weights = torch.load(model_path, map_location=torch.device('cpu'))
        surrogate = SurrogateModel(surrogate_weights)
        print(f"Loaded surrogate model from {model_path}")
        return surrogate
    except Exception as e:
        print(f"Error loading surrogate model: {str(e)}")
        return None

def normalize_coordinates_for_surrogate(x_coords, y_coords, num_points=200):
    """Normalize airfoil coordinates to exactly num_points for surrogate model"""
    try:
        # Combine x and y coordinates
        coords = np.column_stack((x_coords, y_coords))
        
        # Sort by x-coordinate
        idx = np.argsort(coords[:, 0])
        coords = coords[idx]
        
        x = coords[:, 0]
        y = coords[:, 1]
        
        # Parametrize by arc length
        t = np.zeros(len(x))
        for i in range(1, len(x)):
            t[i] = t[i-1] + np.sqrt((x[i] - x[i-1])**2 + (y[i] - y[i-1])**2)
        
        if t[-1] == 0:  # Avoid division by zero
            return None
            
        t = t / t[-1]  # Normalize to [0, 1]
        
        # Interpolate to get evenly spaced points
        fx = interp1d(t, x, kind='linear')
        fy = interp1d(t, y, kind='linear')
        
        t_new = np.linspace(0, 1, num_points)
        x_new = fx(t_new)
        y_new = fy(t_new)
        
        # The surrogate model only uses y-coordinates as input
        normalized = y_new
        
        return normalized
    
    except Exception as e:
        print(f"Error normalizing coordinates: {str(e)}")
        return None

def read_dat_file(file_path):
    """Read airfoil coordinates from DAT file"""
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()[1:]  # Skip header line
            
        x_coords = []
        y_coords = []
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    x = float(parts[0])
                    y = float(parts[1])
                    x_coords.append(x)
                    y_coords.append(y)
                except ValueError:
                    continue
        
        return np.array(x_coords), np.array(y_coords)
    
    except Exception as e:
        print(f"Error reading DAT file {file_path}: {str(e)}")
        return None, None

def run_surrogate_on_airfoil(surrogate_model, dat_file):
    """Run surrogate model on a single airfoil"""
    try:
        # Read airfoil coordinates
        x_coords, y_coords = read_dat_file(dat_file)
        
        if x_coords is None or len(x_coords) < 3:
            return None
        
        # Normalize coordinates for surrogate model
        normalized = normalize_coordinates_for_surrogate(x_coords, y_coords)
        
        if normalized is None:
            return None
        
        # Prepare input for surrogate model (y-coordinates + angle of attack)
        inputs = np.append(normalized, ALPHA)
        
        # Convert to torch tensor
        inputs_tensor = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        
        # Run surrogate model
        with torch.no_grad():
            outputs = surrogate_model.forward(inputs_tensor)
        
        # Extract predictions
        cl = float(outputs[0, 0])
        cd = float(outputs[0, 1])
        cm = float(outputs[0, 2])
        
        # Calculate lift-to-drag ratio
        cl_cd = cl / cd if cd > 0 else 0
        
        return {
            'cl': cl,
            'cd': cd,
            'cm': cm,
            'cl_cd': cl_cd
        }
    
    except Exception as e:
        print(f"Error running surrogate on {dat_file}: {str(e)}")
        return None

def main():
    """Main function to run surrogate model on all variations"""
    print("Running surrogate model on PARSEC variations...")
    
    # Load surrogate model
    surrogate_model = load_surrogate_model()
    if surrogate_model is None:
        print("Failed to load surrogate model. Exiting.")
        return
    
    # Get all DAT files in the variations directory
    dat_files = list(DAT_DIR.glob("*.dat"))
    print(f"Found {len(dat_files)} DAT files to evaluate")
    
    results = []
    
    # Process each DAT file
    for dat_file in tqdm(dat_files):
        airfoil_name = dat_file.stem
        
        # Run surrogate model
        predictions = run_surrogate_on_airfoil(surrogate_model, dat_file)
        
        if predictions is None:
            continue
        
        # Add to results
        result = {
            'airfoil': airfoil_name,
            'cl': predictions['cl'],
            'cd': predictions['cd'],
            'cm': predictions['cm'],
            'cl_cd': predictions['cl_cd']
        }
        results.append(result)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Sort by lift-to-drag ratio (descending)
    results_df = results_df.sort_values('cl_cd', ascending=False)
    
    # Save all results to CSV
    results_df.to_csv(RESULTS_FILE, index=False)
    print(f"Saved all results to {RESULTS_FILE}")
    
    # Print top 10 performers
    top_10 = results_df.head(10)
    print("\nTop 10 airfoil variations by lift-to-drag ratio:")
    for i, (_, row) in enumerate(top_10.iterrows()):
        print(f"{i+1}. {row['airfoil']}: L/D = {row['cl_cd']:.2f} (CL = {row['cl']:.4f}, CD = {row['cd']:.6f})")
    
    # Create a simple bar chart of the top 10
    plt.figure(figsize=(12, 6))
    plt.bar(top_10['airfoil'], top_10['cl_cd'])
    plt.xlabel('Airfoil Variation')
    plt.ylabel('Lift-to-Drag Ratio')
    plt.title('Top 10 Airfoil Variations by Lift-to-Drag Ratio')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "top_10_variations.png")
    print(f"Saved chart to {RESULTS_DIR / 'top_10_variations.png'}")

if __name__ == "__main__":
    main()
