#!/usr/bin/env python3
"""
PARSEC Parameter Sweep with Database Storage

This script:
1. Reads the min/max/mean PARSEC parameter statistics
2. Generates all combinations of parameters with n steps for each parameter
3. Stores all combinations in an HDF5 database
4. Provides functions to convert PARSEC parameters to airfoil coordinates in memory
5. Processes batches through the surrogate model without creating intermediate files
6. Stores results back in the database and generates a CSV report

Usage:
  python parsec_parameter_sweep.py [options]

Options:
  -s, --steps N    Number of steps for each parameter from min to max (range: 2-10, default: 3)
  -p, --params P   Comma-separated list of parameters to vary (default: all 11 parameters)
  -b, --batch N    Batch size for processing (default: 1000)
  -o, --output F   Output database file (default: parsec_sweep.h5)
"""

import os
import sys
import numpy as np
import pandas as pd
import h5py
import argparse
import time
import itertools
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import torch
from scipy.interpolate import interp1d

# Import the ParsecAirfoil class
from parsec_to_dat import ParsecAirfoil
# Import airfoil validation functions
from airfoil_validation import check_self_intersection, calculate_thickness, check_min_thickness, check_max_thickness
# Imports already added above

# Define directories and files
STATS_FILE = "parsec_results/parsec_stats.csv"
RESULTS_DIR = "parameter_sweep_results"
DEFAULT_DB_FILE = "parsec_sweep.h5"
MODELS_DIR = "./models"

# Create output directory if it doesn't exist
os.makedirs(RESULTS_DIR, exist_ok=True)

# Parameter names and descriptions for better visualization
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


def load_parameter_stats():
    """Load PARSEC parameter statistics from CSV file"""
    try:
        stats = pd.read_csv(STATS_FILE)
        return stats
    except FileNotFoundError:
        print(f"Error: Parameter statistics file not found: {STATS_FILE}")
        print("Make sure you have run parsec_fit.py first to generate statistics.")
        return None


def generate_parameter_combinations(stats, steps=3, selected_params=None):
    """Generate all combinations of parameter values with n steps each"""
    print(f"Generating parameter combinations using ranges focused on top-performing airfoils...")
    
    # Use fixed parameter ranges based on real top-performing airfoils
    # instead of using the statistics from the dataset
    param_ranges = {
        'rLE': np.linspace(0.01, 0.15, steps),        # Leading edge radius (0.005-0.2)
        'Xup': np.linspace(0.2, 0.45, steps),         # Upper crest X position (0.1-0.7)
        'Yup': np.linspace(0.02, 0.09, steps),        # Upper crest Y position (0.01-0.15)
        'YXXup': np.linspace(-2.0, 0.0, steps),       # Upper crest curvature (-5.0-1.0)
        'Xlo': np.linspace(0.1, 0.3, steps),          # Lower crest X position (0.1-0.7)
        'Ylo': np.linspace(-0.05, -0.01, steps),      # Lower crest Y position (-0.1--0.005)
        'YXXlo': np.linspace(0.5, 2.5, steps),        # Lower crest curvature (0.0-5.0)
        'Xte': np.linspace(0.95, 1.0, steps),         # Trailing edge X position (0.9-1.05)
        'Yte': np.linspace(-0.01, 0.01, steps),       # Trailing edge Y position (near 0)
        "Yte'": np.linspace(-0.1, 0.1, steps),         # Trailing edge direction (-0.2-0.2)
        "Δyte''": np.linspace(0.1, 0.3, steps)        # Trailing edge wedge angle (0.05-0.5)
    }
    
    # Determine parameters to vary
    if selected_params is None:
        # Vary all parameters
        selected_params = list(param_ranges.keys())
    
    # Get parameter values for selected parameters
    param_names = []
    param_values = []
    
    for param in selected_params:
        if param in param_ranges:
            param_names.append(param)
            param_values.append(param_ranges[param])
        else:
            # Fallback to statistics for any parameters not in our defined ranges
            # Find the row index for this parameter
            idx = stats.index[stats['param'] == param].tolist()[0]
            
            # Get mean and standard deviation
            mean = stats.iloc[idx]['mean']
            std = stats.iloc[idx]['std']
            
            # Create range centered on mean with +/- 1 standard deviation (narrower range)
            min_val = mean - std
            max_val = mean + std
            
            # Special handling for bounded parameters
            if param == 'rLE':
                min_val = max(0.005, min_val)  # Ensure positive
            
            # Generate steps values
            values = np.linspace(min_val, max_val, steps)
            
            param_names.append(param)
            param_values.append(values)
    # Generate all combinations
    combinations = list(itertools.product(*param_values))
    
    # Create a list of parameter dictionaries
    param_dicts = []
    for combo in combinations:
        param_dict = {}
        for i, param in enumerate(param_names):
            param_dict[param] = combo[i]
        
        # Add mean values for any parameters not being varied
        if selected_params:
            for i, param in enumerate(stats['param'].tolist()):
                if param not in selected_params:
                    param_dict[param] = stats['mean'].iloc[i]
        
        param_dicts.append(param_dict)
    
    return param_names, param_dicts


def create_parameter_database(param_names, param_combinations, db_file):
    """Create HDF5 database with all parameter combinations"""
    # Restore the original function signature to match what's expected in main()
    print(f"Creating parameter database with {len(param_combinations)} combinations...")
    start_time = time.time()
    
    # Create HDF5 file
    with h5py.File(db_file, 'w') as hf:
        # Create dataset for parameter names
        dt_params = h5py.special_dtype(vlen=str)
        param_names_ds = hf.create_dataset('param_names', (len(param_names),), dtype=dt_params)
        for i, param in enumerate(param_names):
            param_names_ds[i] = param
        
        # Create dataset for parameter values
        param_values = hf.create_dataset('param_values', 
                                         (len(param_combinations), len(param_names)), 
                                         dtype='float32')
        
        # Fill parameter values
        for i, combo in enumerate(param_combinations):
            for j, param in enumerate(param_names):
                param_values[i, j] = combo[param]
        
        # Create dataset for results (to be filled later)
        hf.create_dataset('results', 
                          (len(param_combinations), 5), 
                          dtype='float32', 
                          fillvalue=np.nan)
        
        # Create status dataset to track processing
        hf.create_dataset('status', 
                          (len(param_combinations),), 
                          dtype='int8', 
                          fillvalue=0)
    
    elapsed_time = time.time() - start_time
    print(f"Database created in {elapsed_time:.2f} seconds")
    return db_file


def parsec_to_coordinates(params):
    """Convert PARSEC parameters to airfoil coordinates in memory"""
    # Create airfoil from parameters
    airfoil = ParsecAirfoil()
    
    # Set parameters
    for param, value in params.items():
        airfoil.params[param] = value
    
    # Calculate coefficients
    airfoil._calculate_coefficients()
    
    # Generate coordinates
    x_coords, y_coords = airfoil.generate_coordinates(100)
    
    return x_coords, y_coords


def validate_parsec_parameters(params):
    """Validate PARSEC parameters based on reasonable ranges"""
    # Leading edge radius (rLE) - must be positive and within reasonable range
    # Real airfoils typically have rLE between 0.005 and 0.2
    if not (0.005 <= params.get('rLE', 0) <= 0.2):
        return False, f"Leading edge radius out of range: {params.get('rLE', 0)}"
    
    # Upper crest X position (Xup) - must be between 0 and Xte
    # Typically between 0.1 and 0.7 for real airfoils
    if not (0.1 <= params.get('Xup', 0) <= 0.7):
        return False, f"Upper crest X position out of range: {params.get('Xup', 0)}"
    
    # Upper crest Y position (Yup) - must be positive for standard airfoils
    # Typically between 0.01 and 0.15 for real airfoils
    if not (0.01 <= params.get('Yup', 0) <= 0.15):
        return False, f"Upper crest Y position out of range: {params.get('Yup', 0)}"
    
    # Upper crest curvature (YXXup) - typically between -5.0 and 1.0
    if not (-5.0 <= params.get('YXXup', 0) <= 1.0):
        return False, f"Upper crest curvature out of range: {params.get('YXXup', 0)}"
    
    # Lower crest X position (Xlo) - must be between 0 and Xte
    # Typically between 0.1 and 0.7 for real airfoils
    if not (0.1 <= params.get('Xlo', 0) <= 0.7):
        return False, f"Lower crest X position out of range: {params.get('Xlo', 0)}"
    
    # Lower crest Y position (Ylo) - must be negative for standard airfoils
    # Typically between -0.1 and -0.005 for real airfoils
    if not (-0.1 <= params.get('Ylo', 0) <= -0.005):
        return False, f"Lower crest Y position out of range: {params.get('Ylo', 0)}"
    
    # Lower crest curvature (YXXlo) - typically between 0.0 and 5.0
    if not (0.0 <= params.get('YXXlo', 0) <= 5.0):
        return False, f"Lower crest curvature out of range: {params.get('YXXlo', 0)}"
    
    # Trailing edge X position (Xte) - typically around 0.95-1.0
    if not (0.9 <= params.get('Xte', 1.0) <= 1.05):
        return False, f"Trailing edge X position out of range: {params.get('Xte', 0)}"
    
    # Trailing edge Y position (Yte) - typically close to 0
    if abs(params.get('Yte', 0)) > 0.02:
        return False, f"Trailing edge Y position too far from 0: {params.get('Yte', 0)}"
    
    # Trailing edge direction (Yte') - typically between -0.2 and 0.2
    if abs(params.get("Yte'", 0)) > 0.2:
        return False, f"Trailing edge direction out of range: {params.get('Yte\'', 0)}"
    
    # Trailing edge wedge angle (Δyte'') - typically between 0.05 and 0.5
    if not (0.05 <= abs(params.get("Δyte''", 0.2)) <= 0.5):
        return False, f"Trailing edge wedge angle out of range: {params.get('Δyte\'', 0)}"
    
    # Upper must be above lower surface for physically realistic airfoils
    if not (params.get('Yup', 0) > abs(params.get('Ylo', 0))):
        return False, "Upper surface not above lower surface"
    
    # Check relationship between parameters (additional physical constraints)
    # Maximum thickness should be reasonable (approximation check)
    approx_thickness = params.get('Yup', 0) + abs(params.get('Ylo', 0))
    if approx_thickness < 0.01 or approx_thickness > 0.25:
        return False, f"Approximate thickness out of reasonable range: {approx_thickness}"
    
    return True, ""


# Surrogate model class (from compare_predictions.py)
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

# Function to normalize airfoil coordinates to the format expected by the surrogate model
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

# Load surrogate model
def load_surrogate_model():
    """Load the surrogate model from file"""
    try:
        model_path = os.path.join(MODELS_DIR, "surrogate_model_retrained.pt")
        if not os.path.exists(model_path):
            model_path = os.path.join(MODELS_DIR, "surrogate_model.pt")
            
        if not os.path.exists(model_path):
            print(f"Error: No surrogate model found in {MODELS_DIR}")
            print("Using random values as placeholder. Add model files to 'models/' directory.")
            return None
            
        surrogate_weights = torch.load(model_path, map_location=torch.device('cpu'))
        surrogate = SurrogateModel(surrogate_weights)
        print(f"Loaded surrogate model from {model_path}")
        return surrogate
    except Exception as e:
        print(f"Error loading surrogate model: {str(e)}")
        print("Using random values as placeholder. Add model files to 'models/' directory.")
        return None

def process_with_surrogate(param_batch, batch_indices):
    """Process a batch of parameters with the surrogate model"""
    # Load the surrogate model (only once)
    global _surrogate_model
    if '_surrogate_model' not in globals():
        _surrogate_model = load_surrogate_model()
    
    # Fixed angle of attack for analysis
    alpha = 5.0  # 5 degrees
    
    results = []
    validation_results = []
    
    for params in param_batch:
        # First validate the parameters directly
        valid, reason = validate_parsec_parameters(params)
        
        # Skip surrogate model if parameter validation fails
        if not valid:
            validation_results.append((valid, reason))
            results.append((False, [np.nan, np.nan, np.nan, np.nan, alpha]))
            continue
        
        # Convert parameters to coordinates
        try:
            x_coords, y_coords = parsec_to_coordinates(params)
            
            # ADDED: Geometric validation
            # Check for self-intersection
            if not check_self_intersection(x_coords, y_coords):
                validation_results.append((False, "Self-intersection detected"))
                results.append((False, [np.nan, np.nan, np.nan, np.nan, alpha]))
                continue
                
            # Check minimum thickness
            if not check_min_thickness(x_coords, y_coords, min_thickness=0.01):
                validation_results.append((False, "Thickness below minimum (0.01)"))
                results.append((False, [np.nan, np.nan, np.nan, np.nan, alpha]))
                continue
                
            # Check maximum thickness
            if not check_max_thickness(x_coords, y_coords, max_thickness=0.25):
                validation_results.append((False, "Thickness above maximum (0.25)"))
                results.append((False, [np.nan, np.nan, np.nan, np.nan, alpha]))
                continue
                
            # All validations passed
            validation_results.append((True, ""))
            
            # Normalize coordinates for surrogate input
            normalized_coords = normalize_coordinates_for_surrogate(x_coords, y_coords)
            if normalized_coords is None:
                raise ValueError("Failed to normalize coordinates")
            
            # Add angle of attack to input vector
            input_vector = np.append(normalized_coords, alpha)
            
            # If we have a surrogate model, use it; otherwise generate random values
            if _surrogate_model is not None:
                # Prepare input for PyTorch
                input_tensor = torch.tensor(input_vector, dtype=torch.float32).unsqueeze(0)
                
                # Run prediction
                with torch.no_grad():
                    output = _surrogate_model.forward(input_tensor)
                    cl, cd, cm = output.numpy().flatten()
                    
                    # Apply realistic drag constraint (minimum Cd = 0.003)
                    if cd < 0.003:
                        cd = 0.003
                    
                    cl_cd = cl/cd if cd > 0 else 0
                    
                    # Cap maximum L/D ratio at a realistic value (150)
                    if cl_cd > 150:
                        cl_cd = 150
            else:
                # Generate random results if no model is available
                cl = np.random.uniform(0.1, 1.5)
                cd = np.random.uniform(0.005, 0.05)
                cm = np.random.uniform(-0.2, 0.1)
                cl_cd = cl/cd
            
            # Results: [cl, cd, cm, cl/cd, alpha]
            results.append((True, [cl, cd, cm, cl_cd, alpha]))
            
        except Exception as e:
            print(f"Error processing parameters: {str(e)}")
            validation_results.append((False, f"Error: {str(e)}"))
            results.append((False, [np.nan, np.nan, np.nan, np.nan, alpha]))
    
    # Print validation statistics
    total = len(validation_results)
    valid_count = sum(1 for valid, _ in validation_results if valid)
    print(f"Validation summary: {valid_count}/{total} ({valid_count/total*100:.1f}%) valid parameter sets")
    
    # Count rejection reasons
    reasons = {}
    for valid, reason in validation_results:
        if not valid:
            reasons[reason] = reasons.get(reason, 0) + 1
    
    # Print top rejection reasons
    if reasons:
        print("Top rejection reasons:")
        for reason, count in sorted(reasons.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {reason}: {count}")
    
    return batch_indices, results


def run_parameter_sweep(db_file, batch_size=1000):
    """Run the parameter sweep using the database and surrogate model"""
    print(f"Running parameter sweep from database: {db_file}")
    
    # Open the database
    with h5py.File(db_file, 'r+') as hf:
        param_names = [name.decode('utf-8') for name in hf['param_names']]
        param_values = hf['param_values']
        results = hf['results']
        status = hf['status']
        
        # Create a new dataset for validation status if it doesn't exist
        if 'validation_status' not in hf:
            validation_status = hf.create_dataset('validation_status', 
                                    (param_values.shape[0],), 
                                    dtype='int8', 
                                    fillvalue=0)
            # Create a dataset for validation reasons
            dt_reasons = h5py.special_dtype(vlen=str)
            validation_reasons = hf.create_dataset('validation_reasons', 
                                    (param_values.shape[0],), 
                                    dtype=dt_reasons)
        else:
            validation_status = hf['validation_status']
            validation_reasons = hf['validation_reasons']
        
        total_combinations = param_values.shape[0]
        print(f"Total parameter combinations to process: {total_combinations}")
        
        valid_count = 0
        processed_count = 0
        
        # Process in batches
        for batch_start in tqdm(range(0, total_combinations, batch_size)):
            batch_end = min(batch_start + batch_size, total_combinations)
            batch_indices = list(range(batch_start, batch_end))
            
            # Get parameter batch
            param_batch = []
            for i in batch_indices:
                param_dict = {param_names[j]: param_values[i, j] for j in range(len(param_names))}
                param_batch.append(param_dict)
            
            # First validate all parameters directly
            for i, params in zip(batch_indices, param_batch):
                valid, reason = validate_parsec_parameters(params)
                validation_status[i] = 1 if valid else 0
                validation_reasons[i] = reason if not valid else ""
                if valid:
                    valid_count += 1
            
            # Filter batch to include only valid parameter sets
            valid_indices = []
            valid_params = []
            for i, params in zip(batch_indices, param_batch):
                if validation_status[i] == 1:
                    valid_indices.append(i)
                    valid_params.append(params)
            
            # Process only valid parameter sets with surrogate model
            if valid_params:
                processed_count += len(valid_params)
                batch_indices, batch_results = process_with_surrogate(valid_params, valid_indices)
                
                # Store results
                for i, (success, result) in zip(batch_indices, batch_results):
                    if success:
                        results[i] = result
                        status[i] = 1  # Processed successfully
                    else:
                        status[i] = -1  # Error occurred
            
        print(f"Parameter sweep complete: {valid_count}/{total_combinations} ({valid_count/total_combinations*100:.1f}%) valid parameter sets")
        print(f"Successfully processed: {processed_count} parameter sets")
    
    print("Parameter sweep completed")
    return db_file


def export_results(db_file):
    """Export results from the database to CSV"""
    print(f"Exporting results from database: {db_file}")
    
    # Open the database
    with h5py.File(db_file, 'r') as hf:
        param_names = [name.decode('utf-8') for name in hf['param_names']]
        param_values = hf['param_values']
        results = hf['results']
        status = hf['status']
        
        # Get validation status if available
        validation_status = hf.get('validation_status', None)
        validation_reasons = hf.get('validation_reasons', None)
        
        total_combinations = param_values.shape[0]
        print(f"Total parameter combinations: {total_combinations}")
        
        # Count successfully processed combinations
        successful = np.sum(status[:] == 1)
        print(f"Successfully processed: {successful}/{total_combinations} combinations")
        
        # Count valid airfoils if validation data exists
        if validation_status is not None:
            valid_count = np.sum(validation_status[:] == 1)
            print(f"Valid airfoils: {valid_count}/{total_combinations} ({valid_count/total_combinations*100:.1f}%)")
            
            # Count top rejection reasons
            if validation_reasons is not None:
                reasons = {}
                for i in range(total_combinations):
                    if validation_status[i] == 0:
                        reason = validation_reasons[i]
                        reasons[reason] = reasons.get(reason, 0) + 1
                
                if reasons:
                    print("Top rejection reasons:")
                    for reason, count in sorted(reasons.items(), key=lambda x: x[1], reverse=True)[:5]:
                        print(f"  {reason}: {count}")
        
        # Create dataframe for export
        data = []
        for i in range(total_combinations):
            if status[i] == 1:  # Only include successful results
                row = {param: param_values[i, j] for j, param in enumerate(param_names)}
                row['cl'] = results[i, 0]
                row['cd'] = results[i, 1]
                row['cm'] = results[i, 2]
                row['cl_cd'] = results[i, 3]
                row['alpha'] = results[i, 4]
                
                # Add validation status if available
                if validation_status is not None:
                    row['valid'] = validation_status[i]
                
                data.append(row)
        
        # Convert to dataframe
        if data:
            df = pd.DataFrame(data)
            
            # Sort by lift-to-drag ratio
            df_sorted = df.sort_values('cl_cd', ascending=False)
            
            # Save to CSV files
            all_file = os.path.join(RESULTS_DIR, "parsec_sweep_results.csv")
            success_file = os.path.join(RESULTS_DIR, "parsec_sweep_results_success.csv")
            
            df.to_csv(all_file, index=False)
            df_sorted.to_csv(success_file, index=False)
            
            print(f"Exported all results to: {all_file}")
            print(f"Exported sorted results to: {success_file}")
            
            # Print top performers
            print("\nTop 10 performers by lift-to-drag ratio:")
            top_10 = df_sorted.head(10)[['cl_cd', 'cl', 'cd'] + param_names]
            print(top_10.to_string())
            
            return df_sorted
        else:
            print("No successful results to export")
            return None


def create_visualizations(db_file):
    """Create visualizations of parameter effects on performance"""
    print("Creating result visualizations...")
    
    # Load results from CSV (more convenient for analysis)
    csv_file = os.path.join(RESULTS_DIR, 'parsec_sweep_results_success.csv')
    df = pd.read_csv(csv_file)
    
    # Create performance histograms
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Performance Metrics Distribution", fontsize=16)
    
    axes = axes.flatten()
    
    metrics = [
        ('cl', 'Lift Coefficient'),
        ('cd', 'Drag Coefficient'),
        ('cm', 'Moment Coefficient'),
        ('cl_cd', 'Lift-to-Drag Ratio')
    ]
    
    for i, (metric, title) in enumerate(metrics):
        ax = axes[i]
        ax.hist(df[metric].dropna(), bins=50, alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel(metric)
        ax.set_ylabel("Frequency")
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    hist_file = os.path.join(RESULTS_DIR, 'parsec_sweep_histograms.png')
    plt.savefig(hist_file, dpi=150)
    plt.close(fig)
    
    # Find best performers by lift-to-drag ratio
    top_performers = df.nlargest(20, 'cl_cd')
    
    # Create PDF with top performers
    pdf_path = os.path.join(RESULTS_DIR, "top_performers.pdf")
    with PdfPages(pdf_path) as pdf:
        # Create summary page
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis('off')
        ax.set_title("Top 20 Airfoils by Lift-to-Drag Ratio", fontsize=16)
        
        # Create the table
        cell_text = []
        for i, row in top_performers.iterrows():
            cell = [f"{row['cl_cd']:.2f}", f"{row['cl']:.4f}", f"{row['cd']:.6f}", f"{row['cm']:.4f}"]
            cell_text.append(cell)
        
        column_labels = ["L/D", "CL", "CD", "CM"]
        ax_table = ax.table(cellText=cell_text, colLabels=column_labels, 
                         loc='center', cellLoc='center')
        ax_table.auto_set_font_size(False)
        ax_table.set_fontsize(10)
        ax_table.scale(1.2, 1.5)
        
        pdf.savefig(fig)
        plt.close(fig)
        
        # Create individual pages for top performers
        for i, row in top_performers.iterrows():
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            
            # Plot airfoil shape
            # Only include parameters that exist in the results DataFrame
            param_dict = {}
            for param in PARAM_DESCRIPTIONS.keys():
                if param in row.index:
                    param_dict[param] = row[param]
                else:
                    # Use default mean value for missing parameters
                    with h5py.File(db_file, 'r') as hf:
                        param_names = [name.decode('utf-8') for name in hf['param_names']]
                        if param not in param_names:
                            param_dict[param] = 0.0  # Will be replaced with mean later
            
            # Get mean values for any missing parameters
            stats = pd.read_csv(STATS_FILE)
            for param in PARAM_DESCRIPTIONS.keys():
                if param not in param_dict or param_dict[param] == 0.0:
                    idx = stats[stats['param'] == param].index
                    if len(idx) > 0:
                        param_dict[param] = stats.loc[idx[0], 'mean']
                    
            try:
                x_coords, y_coords = parsec_to_coordinates(param_dict)
                ax1.plot(x_coords, y_coords, 'b-', linewidth=2)
                ax1.set_aspect('equal')
                ax1.grid(True, alpha=0.3)
                ax1.set_title("Airfoil Shape")
            except Exception as e:
                print(f"Error plotting airfoil: {str(e)}")
                ax1.text(0.5, 0.5, f"Failed to generate shape\n{str(e)}", ha='center', va='center')
            
            # Create parameter table
            ax2.axis('off')
            ax2.set_title(f"Performance: L/D = {row['cl_cd']:.2f}, CL = {row['cl']:.4f}, CD = {row['cd']:.6f}")
            
            # Create table with only parameters that exist in the results DataFrame
            # plus any parameters used to generate the airfoil shape (with mean values)
            param_text = []
            
            # First add the parameters that were varied
            varied_params = [p for p in row.index if p in PARAM_DESCRIPTIONS]
            for param in varied_params:
                desc = PARAM_DESCRIPTIONS.get(param, param)
                param_text.append([desc, f"{row[param]:.6f}"])
            
            # Add a note about other parameters
            if len(varied_params) < len(PARAM_DESCRIPTIONS):
                param_text.append(["Note", "Other parameters at mean values"])
            
            param_table = ax2.table(cellText=param_text, 
                                  colLabels=["Parameter", "Value"],
                                  loc='center', cellLoc='center')
            param_table.auto_set_font_size(False)
            param_table.set_fontsize(9)
            param_table.scale(1.2, 1.5)
            
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
    
    print(f"Visualizations created:")
    print(f"  - Histograms: {hist_file}")
    print(f"  - Top performers: {pdf_path}")


def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Generate and analyze all combinations of PARSEC parameters."
    )
    
    parser.add_argument(
        "-s", "--steps", 
        type=int, 
        default=3,
        help="Number of steps for each parameter from min to max (range: 2-10, default: 3)"
    )
    
    parser.add_argument(
        "-p", "--params", 
        type=str, 
        default="",
        help="Comma-separated list of parameters to vary (default: all 11 parameters)"
    )
    
    parser.add_argument(
        "-b", "--batch", 
        type=int, 
        default=1000,
        help="Batch size for processing (default: 1000)"
    )
    
    parser.add_argument(
        "-o", "--output", 
        type=str, 
        default=DEFAULT_DB_FILE,
        help=f"Output database file (default: {DEFAULT_DB_FILE})"
    )
    
    args = parser.parse_args()
    
    # Validate steps parameter
    if args.steps < 2 or args.steps > 10:
        parser.error("Steps must be between 2 and 10")
    
    # Validate batch size
    if args.batch < 1:
        parser.error("Batch size must be at least 1")
    
    # Process selected parameters
    selected_params = None
    if args.params:
        selected_params = [p.strip() for p in args.params.split(',')]
        # Validate parameter names
        valid_params = list(PARAM_DESCRIPTIONS.keys())
        for param in selected_params:
            if param not in valid_params:
                parser.error(f"Invalid parameter name: {param}. Valid parameters: {', '.join(valid_params)}")
    
    return args


def main():
    """Main function to execute the parameter sweep"""
    # Parse command-line arguments
    args = parse_arguments()
    
    # Process selected parameters
    selected_params = None
    if args.params:
        selected_params = [p.strip() for p in args.params.split(',')]
    
    # Calculate number of combinations
    num_params = len(selected_params) if selected_params else 11
    num_combinations = args.steps ** num_params
    
    print(f"PARSEC Parameter Sweep")
    print(f"=====================")
    print(f"Steps per parameter: {args.steps}")
    print(f"Parameters to vary: {num_params} ({'all' if not selected_params else ', '.join(selected_params)})")
    print(f"Total combinations: {num_combinations}")
    print(f"Batch size: {args.batch}")
    print(f"Database file: {args.output}")
    print()
    
    # Ask for confirmation if combinations > 1 million
    if num_combinations > 1000000:
        response = input(f"Warning: This will generate {num_combinations:,} combinations. Continue? (y/n): ")
        if response.lower() != 'y':
            print("Operation cancelled.")
            return
    
    # Start timing
    overall_start = time.time()
    
    # Load parameter statistics
    stats = load_parameter_stats()
    if stats is None:
        return
    
    # Generate parameter combinations
    param_names, param_combinations = generate_parameter_combinations(
        stats, steps=args.steps, selected_params=selected_params
    )
    
    # Create database
    db_file = create_parameter_database(param_names, param_combinations, args.output)
    
    # Run parameter sweep
    run_parameter_sweep(db_file, batch_size=args.batch)
    
    # Export results
    export_results(db_file)
    
    # Create visualizations
    create_visualizations(db_file)
    
    # Report total time
    overall_time = time.time() - overall_start
    print(f"Total processing time: {overall_time:.2f} seconds")
    print(f"Average time per combination: {overall_time / len(param_combinations):.6f} seconds")


if __name__ == "__main__":
    main()
