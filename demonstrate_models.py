import torch
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import argparse
import sys
from scipy.interpolate import interp1d

# Directory paths - updated for airfoil-analysis project
AIRFOIL_DIR = "./airfoils_uiuc"
MODELS_DIR = "./models"
RESULTS_DIR = "./results"
DEMO_OUTPUT_DIR = "./demo_results"

# Function to read and normalize airfoil coordinates
def read_and_normalize_airfoil(filename, num_points=200):
    """Read airfoil coordinates and normalize to exactly num_points"""
    try:
        # Read the file
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        # Extract coordinates
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
        
        # Convert to numpy array and sort by x-coordinate
        coords = np.array(coords)
        # Sort to ensure we go from trailing edge to trailing edge
        idx = np.argsort(coords[:, 0])
        coords = coords[idx]
        
        # Interpolate to get exactly num_points
        x = coords[:, 0]
        y = coords[:, 1]
        
        # Create interpolation function
        try:
            # Create parameter t that goes from 0 to 1 along the airfoil contour
            t = np.zeros(len(x))
            for i in range(1, len(x)):
                t[i] = t[i-1] + np.sqrt((x[i] - x[i-1])**2 + (y[i] - y[i-1])**2)
            t = t / t[-1]  # Normalize to [0, 1]
            
            # Create interpolation functions
            fx = interp1d(t, x, kind='linear')
            fy = interp1d(t, y, kind='linear')
            
            # Generate new points
            t_new = np.linspace(0, 1, num_points)
            x_new = fx(t_new)
            y_new = fy(t_new)
            
            # Return as flattened array of alternating x,y coordinates
            normalized = np.zeros(num_points)
            for i in range(num_points):
                normalized[i] = y_new[i]  # Just use y-coordinates as the input
            return normalized
        except Exception as e:
            print(f"Error during interpolation: {e}")
            return None
            
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return None

# Load models
def load_models(surrogate_type="retrained"):
    print("Loading models...")
    try:
        encoder = torch.load(os.path.join(MODELS_DIR, "encoder.pt"), map_location=torch.device('cpu'))
        decoder = torch.load(os.path.join(MODELS_DIR, "decoder.pt"), map_location=torch.device('cpu'))
        
        # Load the appropriate surrogate model
        if surrogate_type == "original":
            print("Using original surrogate model")
            surrogate_file = "surrogate_model.pt"
        elif surrogate_type == "retrained":
            print("Using retrained surrogate model")
            surrogate_file = "surrogate_model_retrained.pt"
        else:
            print(f"Unknown surrogate type '{surrogate_type}', defaulting to retrained")
            surrogate_file = "surrogate_model_retrained.pt"
            
        surrogate = torch.load(os.path.join(MODELS_DIR, surrogate_file), map_location=torch.device('cpu'))
        print("Models loaded successfully")
        return encoder, decoder, surrogate
    except Exception as e:
        print(f"Error loading models: {e}")
        return None, None, None

# Create a simple class for running inference
class AirfoilEncoder:
    def __init__(self, encoder_weights):
        self.weights = encoder_weights
    
    def forward(self, x):
        # First layer
        x = torch.matmul(x, self.weights['0.weight'].t()) + self.weights['0.bias']
        x = torch.relu(x)
        # Second layer
        x = torch.matmul(x, self.weights['2.weight'].t()) + self.weights['2.bias']
        return x

class AirfoilDecoder:
    def __init__(self, decoder_weights):
        self.weights = decoder_weights
    
    def forward(self, x):
        # First layer
        x = torch.matmul(x, self.weights['0.weight'].t()) + self.weights['0.bias']
        x = torch.relu(x)
        # Second layer
        x = torch.matmul(x, self.weights['2.weight'].t()) + self.weights['2.bias']
        return x

class SurrogateModel:
    def __init__(self, surrogate_weights):
        self.weights = surrogate_weights
    
    def forward(self, x):
        # First layer
        x = torch.matmul(x, self.weights['net.0.weight'].t()) + self.weights['net.0.bias']
        x = torch.relu(x)
        # Second layer
        x = torch.matmul(x, self.weights['net.2.weight'].t()) + self.weights['net.2.bias']
        x = torch.relu(x)
        # Third layer
        x = torch.matmul(x, self.weights['net.4.weight'].t()) + self.weights['net.4.bias']
        return x

# Main function
def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Demonstrate airfoil ML models")
    parser.add_argument("--model", type=str, choices=["original", "retrained"], 
                        default="retrained", help="Which surrogate model to use")
    parser.add_argument("--airfoils", type=str, nargs="*",
                        default=["a18sm", "ag13", "hs1606"], 
                        help="Names of airfoils to analyze")
    parser.add_argument("--all", action="store_true",
                        help="Analyze all available airfoils")
    args = parser.parse_args()
    
    # Create demo output directory if it doesn't exist
    os.makedirs(DEMO_OUTPUT_DIR, exist_ok=True)
    
    # Load models
    encoder_weights, decoder_weights, surrogate_weights = load_models(args.model)
    if encoder_weights is None or decoder_weights is None or surrogate_weights is None:
        print("Failed to load one or more models.")
        return
    
    # Create model instances
    encoder = AirfoilEncoder(encoder_weights)
    decoder = AirfoilDecoder(decoder_weights)
    surrogate = SurrogateModel(surrogate_weights)
    
    # Get airfoils to analyze
    if args.all:
        # Get all available airfoils
        airfoil_names = [os.path.splitext(f)[0] for f in os.listdir(AIRFOIL_DIR) 
                        if f.endswith('.dat') and os.path.isfile(os.path.join(AIRFOIL_DIR, f))]
        print(f"Found {len(airfoil_names)} airfoils to analyze")
    else:
        airfoil_names = args.airfoils
        
    angles = [-5, 0, 5, 10]  # Angles of attack
    
    # Load XFOIL results for comparison
    try:
        xfoil_results = pd.read_csv(os.path.join(RESULTS_DIR, 'all_airfoils_summary.csv'))
    except Exception as e:
        print(f"Error loading XFOIL results: {e}")
        xfoil_results = None
        
    # Process each airfoil
    for airfoil_name in airfoil_names:
        print(f"\n=== Processing airfoil: {airfoil_name} ===")
        
        # Find and load the airfoil file
        airfoil_file = os.path.join(AIRFOIL_DIR, f"{airfoil_name}.dat")
        if not os.path.exists(airfoil_file):
            print(f"Airfoil file not found: {airfoil_file}")
            continue
            
        # Read and normalize coordinates
        normalized_coords = read_and_normalize_airfoil(airfoil_file)
        if normalized_coords is None:
            print("Failed to process airfoil coordinates")
            continue
            
        # Convert to tensor
        airfoil_tensor = torch.tensor(normalized_coords, dtype=torch.float32).unsqueeze(0)
        
        # Encode to latent space
        with torch.no_grad():
            latent = encoder.forward(airfoil_tensor)
            print(f"Latent representation (8 dimensions): {latent.numpy().flatten()}")
            
            # Decode back to airfoil shape
            reconstructed = decoder.forward(latent)
            
            # Calculate reconstruction error
            error = torch.mean((reconstructed - airfoil_tensor) ** 2).item()
            print(f"Reconstruction MSE: {error:.6f}")
            
            # Make predictions for different angles of attack
            print("\nAerodynamic predictions:")
            print("Alpha    CL       CD       CM      | XFOIL CL  XFOIL CD  XFOIL CM")
            print("------------------------------------------------------------------")
            
            for angle in angles:
                # Prepare input for surrogate model: concatenate airfoil and angle
                angle_tensor = torch.tensor([[angle]], dtype=torch.float32)
                surrogate_input = torch.cat([airfoil_tensor, angle_tensor], dim=1)
                
                # Predict aerodynamic coefficients
                pred = surrogate.forward(surrogate_input)
                cl_pred, cd_pred, cm_pred = pred.numpy().flatten()
                
                # Get XFOIL results for comparison if available
                if xfoil_results is not None:
                    xfoil_row = xfoil_results[(xfoil_results['airfoil'] == airfoil_name) & 
                                              (xfoil_results['alpha'] == angle)]
                    if len(xfoil_row) > 0:
                        cl_xfoil = xfoil_row['CL'].values[0]
                        cd_xfoil = xfoil_row['CD'].values[0]
                        cm_xfoil = xfoil_row['CM'].values[0]
                        print(f"{angle:+2d}°    {cl_pred:.4f}  {cd_pred:.6f}  {cm_pred:.4f}  | {cl_xfoil:.4f}   {cd_xfoil:.6f}  {cm_xfoil:.4f}")
                    else:
                        print(f"{angle:+2d}°    {cl_pred:.4f}  {cd_pred:.6f}  {cm_pred:.4f}  | (No XFOIL data)")
                else:
                    print(f"{angle:+2d}°    {cl_pred:.4f}  {cd_pred:.6f}  {cm_pred:.4f}")
                    
            # Plot the original and reconstructed airfoil
            plt.figure(figsize=(10, 6))
            plt.subplot(211)
            plt.title(f"Airfoil: {airfoil_name}")
            plt.plot(np.linspace(0, 1, len(normalized_coords)), normalized_coords, 'b-', label='Original')
            plt.plot(np.linspace(0, 1, len(normalized_coords)), reconstructed.numpy().flatten(), 'r--', label='Reconstructed')
            plt.legend()
            plt.grid(True)
            
            plt.subplot(212)
            plt.title("Reconstruction Error")
            plt.plot(np.linspace(0, 1, len(normalized_coords)), 
                    (reconstructed.numpy().flatten() - normalized_coords), 'g-')
            plt.grid(True)
            
            plt.tight_layout()
            output_path = os.path.join(DEMO_OUTPUT_DIR, f"{airfoil_name}_{args.model}_reconstruction.png")
            plt.savefig(output_path)
            plt.close()
            print(f"Saved visualization to {output_path}")

if __name__ == "__main__":
    main()
