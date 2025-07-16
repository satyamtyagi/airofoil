import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from tqdm import tqdm
import random

# Directory paths
AIRFOIL_DIR = "./airfoils_uiuc"
MODELS_DIR = "./models"
RESULTS_DIR = "./results"
SUMMARY_FILE = os.path.join(RESULTS_DIR, "all_airfoils_summary.csv")

# Function to read and normalize airfoil coordinates (same as in demonstrate_models.py)
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

# Load existing encoder model
def load_encoder():
    try:
        encoder_weights = torch.load(os.path.join(MODELS_DIR, "encoder.pt"), map_location=torch.device('cpu'))
        return encoder_weights
    except Exception as e:
        print(f"Error loading encoder: {e}")
        return None

# Simple neural network for the surrogate model
class SurrogateNet(nn.Module):
    def __init__(self, input_size=201, hidden_size=256, output_size=3):
        super(SurrogateNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        
    def forward(self, x):
        return self.net(x)

# Function to prepare training data
def prepare_training_data():
    print("Preparing training data...")
    
    # Load XFOIL results
    try:
        xfoil_data = pd.read_csv(SUMMARY_FILE)
    except Exception as e:
        print(f"Error loading XFOIL data: {e}")
        return None, None
    
    # Get unique airfoil names from the results
    airfoil_names = xfoil_data['airfoil'].unique()
    
    # Prepare lists for inputs and targets
    inputs = []
    targets = []
    
    for airfoil_name in tqdm(airfoil_names, desc="Processing airfoils"):
        # Get airfoil file
        airfoil_file = os.path.join(AIRFOIL_DIR, f"{airfoil_name}.dat")
        if not os.path.exists(airfoil_file):
            continue
            
        # Get airfoil coordinates
        coords = read_and_normalize_airfoil(airfoil_file)
        if coords is None:
            continue
            
        # Get performance data for this airfoil
        airfoil_data = xfoil_data[xfoil_data['airfoil'] == airfoil_name]
        
        # For each angle of attack, create a training example
        for _, row in airfoil_data.iterrows():
            alpha = row['alpha']
            cl = row['CL']
            cd = row['CD']
            cm = row['CM']
            
            # Skip invalid or extremely large values
            if not np.isfinite(cl) or not np.isfinite(cd) or not np.isfinite(cm) or cd > 1.0:
                continue
                
            # Create input: airfoil coords + angle of attack
            input_vector = np.append(coords, alpha)
            
            # Create target: [CL, CD, CM]
            target_vector = np.array([cl, cd, cm])
            
            inputs.append(input_vector)
            targets.append(target_vector)
    
    if not inputs:
        print("No valid training examples found")
        return None, None
        
    # Convert to numpy arrays
    inputs = np.array(inputs)
    targets = np.array(targets)
    
    print(f"Prepared {len(inputs)} training examples from {len(airfoil_names)} airfoils")
    return inputs, targets

# Train the surrogate model
def train_surrogate_model(inputs, targets, epochs=500, batch_size=32, learning_rate=0.001, val_split=0.2):
    # Convert to tensors
    inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
    targets_tensor = torch.tensor(targets, dtype=torch.float32)
    
    # Split into training and validation sets
    dataset_size = len(inputs)
    val_size = int(dataset_size * val_split)
    train_size = dataset_size - val_size
    
    indices = list(range(dataset_size))
    random.shuffle(indices)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_inputs = inputs_tensor[train_indices]
    train_targets = targets_tensor[train_indices]
    val_inputs = inputs_tensor[val_indices]
    val_targets = targets_tensor[val_indices]
    
    # Create model, loss function, and optimizer
    model = SurrogateNet(input_size=inputs.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # For tracking loss
    train_losses = []
    val_losses = []
    
    # Check if MPS (Metal Performance Shaders) is available
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model.to(device)
    
    print("Training surrogate model...")
    for epoch in range(epochs):
        # Training
        model.train()
        epoch_loss = 0
        
        # Process in batches
        for i in range(0, train_size, batch_size):
            batch_inputs = train_inputs[i:i+batch_size].to(device)
            batch_targets = train_targets[i:i+batch_size].to(device)
            
            # Forward pass
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_targets)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * len(batch_inputs)
        
        train_loss = epoch_loss / train_size
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(val_inputs.to(device))
            val_loss = criterion(val_outputs, val_targets.to(device)).item()
            val_losses.append(val_loss)
        
        # Print progress
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
    
    # Save model
    model.to(torch.device("cpu"))  # Move back to CPU for saving
    model_state = {}
    
    # Extract weights in the same format as the original surrogate model
    for name, param in model.named_parameters():
        model_state[name] = param
    
    torch.save(model_state, os.path.join(MODELS_DIR, "surrogate_model_retrained.pt"))
    print(f"Model saved to {os.path.join(MODELS_DIR, 'surrogate_model_retrained.pt')}")
    
    # Plot training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig("surrogate_training_loss.png")
    plt.close()
    print("Training plot saved to surrogate_training_loss.png")
    
    return model

# Function to evaluate the model on a few test cases
def evaluate_model(model):
    print("\nEvaluating model on test cases...")
    
    # Load XFOIL results
    xfoil_data = pd.read_csv(SUMMARY_FILE)
    
    # Select a few test airfoils
    test_airfoils = ['a18sm', 'ag13', 'hs1606']
    angles = [-5, 0, 5, 10]
    
    for airfoil_name in test_airfoils:
        print(f"\n=== Testing airfoil: {airfoil_name} ===")
        
        # Get airfoil file
        airfoil_file = os.path.join(AIRFOIL_DIR, f"{airfoil_name}.dat")
        if not os.path.exists(airfoil_file):
            print(f"Airfoil file not found: {airfoil_file}")
            continue
            
        # Get airfoil coordinates
        coords = read_and_normalize_airfoil(airfoil_file)
        if coords is None:
            print("Failed to process airfoil coordinates")
            continue
        
        # Make predictions for different angles of attack
        print("Alpha    CL       CD       CM      | XFOIL CL  XFOIL CD  XFOIL CM")
        print("------------------------------------------------------------------")
        
        for angle in angles:
            # Create input vector
            input_vector = np.append(coords, angle)
            input_tensor = torch.tensor(input_vector, dtype=torch.float32).unsqueeze(0)
            
            # Make prediction
            with torch.no_grad():
                output = model(input_tensor)
                cl_pred, cd_pred, cm_pred = output.numpy().flatten()
            
            # Get XFOIL results for comparison
            xfoil_row = xfoil_data[(xfoil_data['airfoil'] == airfoil_name) & 
                                  (xfoil_data['alpha'] == angle)]
            if len(xfoil_row) > 0:
                cl_xfoil = xfoil_row['CL'].values[0]
                cd_xfoil = xfoil_row['CD'].values[0]
                cm_xfoil = xfoil_row['CM'].values[0]
                print(f"{angle:+2d}°    {cl_pred:.4f}  {cd_pred:.6f}  {cm_pred:.4f}  | {cl_xfoil:.4f}   {cd_xfoil:.6f}  {cm_xfoil:.4f}")
            else:
                print(f"{angle:+2d}°    {cl_pred:.4f}  {cd_pred:.6f}  {cm_pred:.4f}  | (No XFOIL data)")

# Main function
def main():
    # Check for PyTorch MPS acceleration
    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS (Metal Performance Shaders) available: {torch.backends.mps.is_available()}\n")
    
    # Prepare data
    inputs, targets = prepare_training_data()
    if inputs is None or targets is None:
        print("Failed to prepare training data")
        return
    
    # Train model
    model = train_surrogate_model(inputs, targets)
    
    # Evaluate model
    evaluate_model(model)

if __name__ == "__main__":
    main()
