import os
import csv
import numpy as np
import re
import pandas as pd

# Directories
AIRFOIL_DIR = "airfoils_uiuc"
RESULTS_DIR = "results"
OUTPUT_FILE = os.path.join(RESULTS_DIR, "combined_airfoil_data.csv")
SUMMARY_FILE = os.path.join(RESULTS_DIR, "all_airfoils_summary.csv")

# Load the existing results data
print("Loading existing XFOIL results...")
if not os.path.exists(SUMMARY_FILE):
    print(f"Error: Summary file {SUMMARY_FILE} not found!")
    exit(1)
    
results_df = pd.read_csv(SUMMARY_FILE)
print(f"Loaded results for {len(results_df['airfoil'].unique())} airfoils")

# Function to extract airfoil geometry parameters from DAT file
def analyze_airfoil_geometry(filepath):
    """
    Extract key geometry parameters from airfoil coordinates file
    """
    try:
        # Read airfoil coordinates
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        # Skip header line(s)
        start_line = 0
        for i, line in enumerate(lines):
            if re.match(r'^[-+]?\d*\.?\d+\s+[-+]?\d*\.?\d+', line.strip()):
                start_line = i
                break
        
        coords = []
        for line in lines[start_line:]:
            try:
                parts = line.strip().split()
                if len(parts) >= 2:
                    x, y = float(parts[0]), float(parts[1])
                    coords.append((x, y))
            except ValueError:
                # Skip lines that don't contain valid coordinate pairs
                continue
                
        if not coords:
            return None
            
        # Convert to numpy array for easier manipulation
        coords = np.array(coords)
        x_coords = coords[:, 0]
        y_coords = coords[:, 1]
        
        # Extract key geometric parameters
        
        # 1. Maximum thickness and its location
        thickness = []
        for i in range(len(x_coords)//2):
            idx = i
            idx_bottom = len(x_coords) - i - 1
            if idx < len(x_coords) and idx_bottom < len(x_coords):
                t = abs(y_coords[idx] - y_coords[idx_bottom])
                thickness.append((x_coords[idx], t))
        
        if not thickness:
            # Alternative approach if first method fails
            upper = [(x, y) for x, y in zip(x_coords, y_coords) if y >= 0]
            lower = [(x, y) for x, y in zip(x_coords, y_coords) if y < 0]
            
            if not upper or not lower:
                return None
                
            upper_x = np.array([p[0] for p in upper])
            upper_y = np.array([p[1] for p in upper])
            lower_x = np.array([p[0] for p in lower])
            lower_y = np.array([p[1] for p in lower])
            
            # Simple thickness calculation
            max_thickness = 0
            thickness_loc = 0
            for x in np.linspace(0.1, 0.9, 50):
                # Find upper y at this x
                if upper_x.size > 0 and lower_x.size > 0:
                    upper_y_at_x = np.interp(x, upper_x, upper_y)
                    lower_y_at_x = np.interp(x, lower_x, lower_y)
                    t = upper_y_at_x - lower_y_at_x
                    if t > max_thickness:
                        max_thickness = t
                        thickness_loc = x
        else:
            thickness.sort(key=lambda x: x[1], reverse=True)
            max_thickness = thickness[0][1]
            thickness_loc = thickness[0][0]
        
        # 2. Maximum camber and its location
        camber = []
        for i in range(len(x_coords)//2):
            idx = i
            idx_bottom = len(x_coords) - i - 1
            if idx < len(x_coords) and idx_bottom < len(x_coords):
                c = (y_coords[idx] + y_coords[idx_bottom]) / 2
                camber.append((x_coords[idx], c))
        
        if not camber:
            # Alternative approach if first method fails
            max_camber = 0
            camber_loc = 0
            for x in np.linspace(0.1, 0.9, 50):
                if upper_x.size > 0 and lower_x.size > 0:
                    upper_y_at_x = np.interp(x, upper_x, upper_y)
                    lower_y_at_x = np.interp(x, lower_x, lower_y)
                    c = (upper_y_at_x + lower_y_at_x) / 2
                    if abs(c) > abs(max_camber):
                        max_camber = c
                        camber_loc = x
        else:
            camber.sort(key=lambda x: abs(x[1]), reverse=True)
            max_camber = camber[0][1]
            camber_loc = camber[0][0]
        
        # 3. Leading edge radius (approximation)
        # Use points near the leading edge to fit a circle
        le_points = [(x, y) for x, y in zip(x_coords, y_coords) if x < 0.05]
        if len(le_points) >= 3:
            le_points = np.array(le_points)
            le_x = le_points[:, 0]
            le_y = le_points[:, 1]
            
            # Simple approximation - distance from leading edge to nearby points
            le_radius = np.mean(np.sqrt((le_x[1:] - le_x[0])**2 + (le_y[1:] - le_y[0])**2))
        else:
            le_radius = 0.01  # Default approximation
            
        # 4. Trailing edge angle (approximation)
        te_upper = [(x, y) for x, y in zip(x_coords, y_coords) if x > 0.95 and y >= 0]
        te_lower = [(x, y) for x, y in zip(x_coords, y_coords) if x > 0.95 and y < 0]
        
        if te_upper and te_lower:
            # Sort by x coordinate
            te_upper.sort(key=lambda point: point[0])
            te_lower.sort(key=lambda point: point[0])
            
            # Get angle between last points
            if len(te_upper) > 0 and len(te_lower) > 0:
                upper_last = te_upper[-1]
                lower_last = te_lower[-1]
                upper_second = te_upper[0] if len(te_upper) == 1 else te_upper[-2]
                lower_second = te_lower[0] if len(te_lower) == 1 else te_lower[-2]
                
                # Calculate angles of both lines
                angle_upper = np.arctan2(upper_last[1] - upper_second[1], 
                                         upper_last[0] - upper_second[0])
                angle_lower = np.arctan2(lower_last[1] - lower_second[1], 
                                         lower_last[0] - lower_second[0])
                
                te_angle = np.abs(angle_upper - angle_lower) * (180/np.pi)
            else:
                te_angle = 0
        else:
            te_angle = 0
            
        # 5. Chord length (should be 1.0 for normalized airfoils)
        chord = np.max(x_coords) - np.min(x_coords)
        
        return {
            'max_thickness': max_thickness,
            'thickness_loc': thickness_loc,
            'max_camber': max_camber,
            'camber_loc': camber_loc,
            'le_radius': le_radius,
            'te_angle': te_angle,
            'chord': chord
        }
        
    except Exception as e:
        print(f"Error processing file {filepath}: {e}")
        return None

# Dictionary to store airfoil geometry data
airfoil_geometry = {}

# Process each airfoil file that appears in the results
unique_airfoils = results_df['airfoil'].unique()
print(f"Processing {len(unique_airfoils)} airfoil geometries...")

for airfoil in unique_airfoils:
    dat_file = os.path.join(AIRFOIL_DIR, f"{airfoil}.dat")
    if os.path.exists(dat_file):
        geometry = analyze_airfoil_geometry(dat_file)
        if geometry:
            airfoil_geometry[airfoil] = geometry
            print(f"Processed geometry for {airfoil}")
        else:
            print(f"Could not analyze geometry for {airfoil}")
    else:
        print(f"Airfoil file not found: {dat_file}")

print(f"Successfully analyzed {len(airfoil_geometry)} airfoil geometries")

# Create a list to hold all combined data
combined_data = []

# Combine geometry data with performance data
for airfoil in airfoil_geometry:
    # Get the airfoil's geometry data
    geometry = airfoil_geometry[airfoil]
    
    # Get performance data at each angle of attack
    airfoil_results = results_df[results_df['airfoil'] == airfoil]
    
    for _, row in airfoil_results.iterrows():
        # Create a combined entry
        entry = {
            'airfoil': airfoil,
            'alpha': row['alpha'],
            'CL': row['CL'],
            'CD': row['CD'],
            'CM': row['CM'],
            'L_D_ratio': row['CL'] / row['CD'] if row['CD'] > 0 else 0,
            'max_thickness': geometry['max_thickness'],
            'thickness_loc': geometry['thickness_loc'],
            'max_camber': geometry['max_camber'],
            'camber_loc': geometry['camber_loc'],
            'le_radius': geometry['le_radius'],
            'te_angle': geometry['te_angle'],
            'chord': geometry['chord']
        }
        combined_data.append(entry)

# Create the combined CSV file
print(f"Writing combined data to {OUTPUT_FILE}...")
with open(OUTPUT_FILE, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=combined_data[0].keys())
    writer.writeheader()
    writer.writerows(combined_data)

print(f"Combined data file created: {OUTPUT_FILE}")

# Create a summary file with best performance metrics
best_performance = {}
for airfoil in unique_airfoils:
    airfoil_data = [d for d in combined_data if d['airfoil'] == airfoil]
    if airfoil_data:
        # Find best L/D ratio
        best_ld = max(airfoil_data, key=lambda x: x['L_D_ratio'] if x['L_D_ratio'] != float('inf') else 0)
        
        # Find max CL 
        max_cl = max(airfoil_data, key=lambda x: x['CL'])
        
        # Find min CD
        min_cd = min(airfoil_data, key=lambda x: x['CD'])
        
        # Add to best performance dictionary
        if airfoil in airfoil_geometry:
            best_performance[airfoil] = {
                'max_thickness': airfoil_geometry[airfoil]['max_thickness'],
                'max_camber': airfoil_geometry[airfoil]['max_camber'],
                'best_ld': best_ld['L_D_ratio'],
                'best_ld_alpha': best_ld['alpha'],
                'max_cl': max_cl['CL'],
                'max_cl_alpha': max_cl['alpha'],
                'min_cd': min_cd['CD'],
                'min_cd_alpha': min_cd['alpha']
            }

# Write the best performance summary file
best_file = os.path.join(RESULTS_DIR, "airfoil_best_performance.csv")
print(f"Writing best performance data to {best_file}...")

with open(best_file, 'w', newline='') as f:
    fieldnames = ['airfoil', 'max_thickness', 'max_camber', 'best_ld', 
                  'best_ld_alpha', 'max_cl', 'max_cl_alpha', 'min_cd', 'min_cd_alpha']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    
    for airfoil, data in best_performance.items():
        row = {'airfoil': airfoil}
        row.update(data)
        writer.writerow(row)

print(f"Best performance summary created: {best_file}")
print("Analysis complete!")
