import torch
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import argparse
from scipy.interpolate import interp1d

# Directory paths
AIRFOIL_DIR = "./airfoils_uiuc"
MODELS_DIR = "./models"
RESULTS_DIR = "./results"
DEMO_OUTPUT_DIR = "./demo_results"

# Function to read and normalize airfoil coordinates - same as in demonstrate_models.py
def read_and_normalize_airfoil(filename, num_points=200):
    """Read airfoil coordinates and normalize to exactly num_points"""
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        coords = []
        for line in lines:
            if line.strip() and len(line.strip().split()) >= 2:
                try:
                    x, y = map(float, line.strip().split()[:2])
                    coords.append((x, y))
                except ValueError:
                    continue
        
        if not coords:
            print(f"No valid data found in {filename}")
            return None
        
        coords = np.array(coords)
        idx = np.argsort(coords[:, 0])
        coords = coords[idx]
        
        x = coords[:, 0]
        y = coords[:, 1]
        
        try:
            t = np.zeros(len(x))
            for i in range(1, len(x)):
                t[i] = t[i-1] + np.sqrt((x[i] - x[i-1])**2 + (y[i] - y[i-1])**2)
            t = t / t[-1]
            
            fx = interp1d(t, x, kind='linear')
            fy = interp1d(t, y, kind='linear')
            
            t_new = np.linspace(0, 1, num_points)
            x_new = fx(t_new)
            y_new = fy(t_new)
            
            normalized = np.zeros(num_points)
            for i in range(num_points):
                normalized[i] = y_new[i]
            return normalized
        except Exception as e:
            print(f"Error during interpolation: {e}")
            return None
            
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return None

# Load surrogate models
def load_surrogate_models():
    try:
        original_surrogate = torch.load(os.path.join(MODELS_DIR, "surrogate_model.pt"), map_location=torch.device('cpu'))
        retrained_surrogate = torch.load(os.path.join(MODELS_DIR, "surrogate_model_retrained.pt"), map_location=torch.device('cpu'))
        print("Surrogate models loaded successfully")
        return original_surrogate, retrained_surrogate
    except Exception as e:
        print(f"Error loading surrogate models: {e}")
        return None, None

# Classes for running inference with the models
class SurrogateModelOriginal:
    def __init__(self, weights):
        self.weights = weights
    
    def forward(self, x):
        x = torch.matmul(x, self.weights['net.0.weight'].t()) + self.weights['net.0.bias']
        x = torch.relu(x)
        x = torch.matmul(x, self.weights['net.2.weight'].t()) + self.weights['net.2.bias']
        x = torch.relu(x)
        x = torch.matmul(x, self.weights['net.4.weight'].t()) + self.weights['net.4.bias']
        return x

class SurrogateModelRetrained:
    def __init__(self, weights):
        self.weights = weights
    
    def forward(self, x):
        x = torch.matmul(x, self.weights['net.0.weight'].t()) + self.weights['net.0.bias']
        x = torch.relu(x)
        x = torch.matmul(x, self.weights['net.2.weight'].t()) + self.weights['net.2.bias']
        x = torch.relu(x)
        x = torch.matmul(x, self.weights['net.4.weight'].t()) + self.weights['net.4.bias']
        return x

# Generate predictions from both models
def generate_predictions(airfoil_name, angles, original_model, retrained_model):
    airfoil_file = os.path.join(AIRFOIL_DIR, f"{airfoil_name}.dat")
    if not os.path.exists(airfoil_file):
        print(f"Airfoil file not found: {airfoil_file}")
        return None
    
    coords = read_and_normalize_airfoil(airfoil_file)
    if coords is None:
        print(f"Failed to process coordinates for {airfoil_name}")
        return None
    
    # Load XFOIL results
    xfoil_results = {}
    try:
        xfoil_data = pd.read_csv(os.path.join(RESULTS_DIR, 'all_airfoils_summary.csv'))
        airfoil_data = xfoil_data[xfoil_data['airfoil'] == airfoil_name]
        
        for _, row in airfoil_data.iterrows():
            alpha = row['alpha']
            if alpha in angles:
                xfoil_results[alpha] = {
                    'CL': row['CL'],
                    'CD': row['CD'],
                    'CM': row['CM']
                }
    except Exception as e:
        print(f"Error loading XFOIL results: {e}")
    
    # Generate predictions
    original_predictions = {}
    retrained_predictions = {}
    
    for angle in angles:
        # Prepare input
        input_vector = np.append(coords, angle)
        input_tensor = torch.tensor(input_vector, dtype=torch.float32).unsqueeze(0)
        
        # Original model predictions
        with torch.no_grad():
            output = original_model.forward(input_tensor)
            cl, cd, cm = output.numpy().flatten()
            original_predictions[angle] = {'CL': cl, 'CD': cd, 'CM': cm}
        
        # Retrained model predictions
        with torch.no_grad():
            output = retrained_model.forward(input_tensor)
            cl, cd, cm = output.numpy().flatten()
            retrained_predictions[angle] = {'CL': cl, 'CD': cd, 'CM': cm}
    
    return {
        'xfoil': xfoil_results,
        'original': original_predictions,
        'retrained': retrained_predictions
    }

# Create comparison plots
def create_comparison_plots(airfoil_name, predictions, angles):
    os.makedirs(DEMO_OUTPUT_DIR, exist_ok=True)
    
    # Create arrays for plotting
    xfoil_cl = []
    xfoil_cd = []
    xfoil_cm = []
    original_cl = []
    original_cd = []
    original_cm = []
    retrained_cl = []
    retrained_cd = []
    retrained_cm = []
    
    for angle in angles:
        if angle in predictions['xfoil']:
            xfoil_cl.append(predictions['xfoil'][angle]['CL'])
            xfoil_cd.append(predictions['xfoil'][angle]['CD'])
            xfoil_cm.append(predictions['xfoil'][angle]['CM'])
        else:
            xfoil_cl.append(np.nan)
            xfoil_cd.append(np.nan)
            xfoil_cm.append(np.nan)
            
        original_cl.append(predictions['original'][angle]['CL'])
        original_cd.append(predictions['original'][angle]['CD'])
        original_cm.append(predictions['original'][angle]['CM'])
        
        retrained_cl.append(predictions['retrained'][angle]['CL'])
        retrained_cd.append(predictions['retrained'][angle]['CD'])
        retrained_cm.append(predictions['retrained'][angle]['CM'])
    
    # Plot lift coefficient (CL)
    plt.figure(figsize=(15, 10))
    
    plt.subplot(3, 1, 1)
    plt.plot(angles, xfoil_cl, 'bo-', label='XFOIL')
    plt.plot(angles, original_cl, 'ro--', label='Original Model')
    plt.plot(angles, retrained_cl, 'go--', label='Retrained Model')
    plt.title(f'Lift Coefficient (CL) - {airfoil_name}')
    plt.xlabel('Angle of Attack (degrees)')
    plt.ylabel('CL')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(3, 1, 2)
    plt.plot(angles, xfoil_cd, 'bo-', label='XFOIL')
    plt.plot(angles, retrained_cd, 'go--', label='Retrained Model')
    plt.title(f'Drag Coefficient (CD) - {airfoil_name}')
    plt.xlabel('Angle of Attack (degrees)')
    plt.ylabel('CD')
    plt.grid(True)
    plt.legend()
    
    # Add a separate plot for original CD values which might be way off scale
    plt.subplot(3, 1, 3)
    plt.plot(angles, xfoil_cm, 'bo-', label='XFOIL')
    plt.plot(angles, original_cm, 'ro--', label='Original Model')
    plt.plot(angles, retrained_cm, 'go--', label='Retrained Model')
    plt.title(f'Moment Coefficient (CM) - {airfoil_name}')
    plt.xlabel('Angle of Attack (degrees)')
    plt.ylabel('CM')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(DEMO_OUTPUT_DIR, f"{airfoil_name}_coefficient_comparison.png"))
    print(f"Saved comparison plot to {os.path.join(DEMO_OUTPUT_DIR, airfoil_name + '_coefficient_comparison.png')}")
    
    # Create separate plots with appropriate scaling for CD
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 1, 1)
    plt.plot(angles, xfoil_cd, 'bo-', label='XFOIL')
    plt.plot(angles, retrained_cd, 'go--', label='Retrained Model')
    plt.title(f'Drag Coefficient (CD) - {airfoil_name} - Detailed View')
    plt.xlabel('Angle of Attack (degrees)')
    plt.ylabel('CD')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(DEMO_OUTPUT_DIR, f"{airfoil_name}_drag_detailed.png"))
    
    # Create a table of values
    plt.figure(figsize=(12, 3))
    plt.axis('off')
    table_data = []
    table_data.append(['Angle', 'XFOIL CL', 'Original CL', 'Retrained CL', 
                      'XFOIL CD', 'Original CD', 'Retrained CD',
                      'XFOIL CM', 'Original CM', 'Retrained CM'])
    
    for i, angle in enumerate(angles):
        row = [f"{angle}°"]
        # CL values
        if np.isnan(xfoil_cl[i]):
            row.append('-')
        else:
            row.append(f"{xfoil_cl[i]:.4f}")
        row.append(f"{original_cl[i]:.4f}")
        row.append(f"{retrained_cl[i]:.4f}")
        
        # CD values
        if np.isnan(xfoil_cd[i]):
            row.append('-')
        else:
            row.append(f"{xfoil_cd[i]:.6f}")
        row.append(f"{original_cd[i]:.6f}")
        row.append(f"{retrained_cd[i]:.6f}")
        
        # CM values
        if np.isnan(xfoil_cm[i]):
            row.append('-')
        else:
            row.append(f"{xfoil_cm[i]:.4f}")
        row.append(f"{original_cm[i]:.4f}")
        row.append(f"{retrained_cm[i]:.4f}")
        
        table_data.append(row)
    
    plt.table(cellText=table_data, loc='center', cellLoc='center', colWidths=[0.08]*10)
    plt.title(f'Aerodynamic Coefficient Comparison - {airfoil_name}', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(DEMO_OUTPUT_DIR, f"{airfoil_name}_coefficient_table.png"))
    
    # Return a formatted string with the results
    result_str = f"=== {airfoil_name} Comparison ===\n"
    result_str += "Angle  |   XFOIL   |  Original  | Retrained \n"
    result_str += "-------+-----------+------------+-----------\n"
    
    for i, angle in enumerate(angles):
        xfoil_cl_str = f"{xfoil_cl[i]:.4f}" if not np.isnan(xfoil_cl[i]) else "   -   "
        result_str += f"{angle:+3d}°  | CL: {xfoil_cl_str} | {original_cl[i]:+.4f} | {retrained_cl[i]:+.4f}\n"
        
        xfoil_cd_str = f"{xfoil_cd[i]:.6f}" if not np.isnan(xfoil_cd[i]) else "   -   "
        result_str += f"      | CD: {xfoil_cd_str} | {original_cd[i]:+.6f} | {retrained_cd[i]:+.6f}\n"
        
        xfoil_cm_str = f"{xfoil_cm[i]:.4f}" if not np.isnan(xfoil_cm[i]) else "   -   "
        result_str += f"      | CM: {xfoil_cm_str} | {original_cm[i]:+.4f} | {retrained_cm[i]:+.4f}\n"
        result_str += "-------+-----------+------------+-----------\n"
    
    return result_str

# Main function
def main():
    parser = argparse.ArgumentParser(description="Compare predictions from original and retrained models")
    parser.add_argument("--airfoils", type=str, nargs="+", default=["a18sm", "ag13", "hs1606"],
                        help="Names of airfoils to analyze")
    args = parser.parse_args()
    
    # Load surrogate models
    original_weights, retrained_weights = load_surrogate_models()
    if original_weights is None or retrained_weights is None:
        print("Failed to load surrogate models")
        return
    
    # Initialize model classes
    original_model = SurrogateModelOriginal(original_weights)
    retrained_model = SurrogateModelRetrained(retrained_weights)
    
    # Angles of attack to test
    angles = [-5, 0, 5, 10]
    
    # Process each airfoil
    for airfoil_name in args.airfoils:
        print(f"\nProcessing airfoil: {airfoil_name}")
        predictions = generate_predictions(airfoil_name, angles, original_model, retrained_model)
        
        if predictions is not None:
            result_str = create_comparison_plots(airfoil_name, predictions, angles)
            print(result_str)

if __name__ == "__main__":
    main()
