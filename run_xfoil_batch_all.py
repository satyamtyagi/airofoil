import os
import subprocess
import time
import re
import datetime
import sys

# CONFIG
# Path to XFOIL executable
XFOIL_PATH = "/Users/satyamtyagi/CascadeProjects/xfoil-mac/bin/xfoil"
AIRFOIL_DIR = "airfoils_uiuc"
RESULTS_DIR = "results"
ANGLES = [-5, 0, 5, 10]  # Analyze at these angles
REYNOLDS = 100000
MACH = 0.0
MAX_FILES = 999  # Set to a high number to process all files, or a small number for testing
TIMEOUT = 15  # Timeout in seconds per angle of attack (lower for faster batch processing)

# Create results dir
os.makedirs(RESULTS_DIR, exist_ok=True)

# Create a summary file
summary_file = os.path.join(RESULTS_DIR, "all_airfoils_summary.csv")
with open(summary_file, "w") as f:
    f.write("airfoil,alpha,CL,CD,CM\n")

# Get list of all DAT files
dat_files = sorted([f for f in os.listdir(AIRFOIL_DIR) if f.endswith(".dat")])
total_files = min(len(dat_files), MAX_FILES)

print(f"Found {len(dat_files)} airfoil files, will process up to {total_files}")
print(f"Angles to analyze: {ANGLES}")
print(f"Reynolds number: {REYNOLDS}, Mach: {MACH}")
print("-" * 60)

# Track statistics
successful = 0
failed = 0
timeout_count = 0
start_time = time.time()

# Process each airfoil
for idx, filename in enumerate(dat_files[:MAX_FILES], 1):
    airfoil_file = os.path.join(AIRFOIL_DIR, filename)
    airfoil_name = os.path.splitext(filename)[0]
    results_file = os.path.join(RESULTS_DIR, f"{airfoil_name}_results.txt")
    
    print(f"\n[{idx}/{total_files}] Analyzing {airfoil_name}...")
    
    # Create a results file with headers
    with open(results_file, "w") as f:
        f.write(f"# Airfoil: {airfoil_name}\n")
        f.write(f"# Reynolds: {REYNOLDS}\n")
        f.write(f"# Mach: {MACH}\n")
        f.write(f"# {'Alpha':>8}{'CL':>10}{'CD':>10}{'CDp':>10}{'CM':>10}\n")
    
    airfoil_success = False
    airfoil_results = []
    
    # Process each angle of attack separately
    for angle in ANGLES:
        sys.stdout.write(f"  Processing angle: {angle}° ... ")
        sys.stdout.flush()
        
        # Create a command file for this angle
        with open("xfoil_temp_cmd.txt", "w") as f:
            f.write("PLOP\n")       # Plot options
            f.write("G F\n")        # Graphics off
            f.write("\n")           # Exit menu
            f.write(f"LOAD {airfoil_file}\n")
            f.write("PANE\n")
            f.write("OPER\n")
            f.write(f"VISC {REYNOLDS}\n")
            f.write(f"MACH {MACH}\n")
            f.write("ITER 100\n")
            f.write(f"ALFA {angle}\n")
            f.write("QUIT\n")

        try:
            # Run XFOIL with the temp file
            result = subprocess.run(
                [XFOIL_PATH],
                stdin=open("xfoil_temp_cmd.txt", "r"),
                capture_output=True,
                text=True,
                timeout=TIMEOUT
            )
            
            output = result.stdout
            
            # Improved regex pattern to match the output format
            cl_pattern = r"a =\s+[-+]?\d*\.\d+\s+CL =\s+([-+]?\d*\.\d+)"
            cd_pattern = r"CD =\s+([-+]?\d*\.\d+)"
            cdp_pattern = r"CDp =\s+([-+]?\d*\.\d+)"
            cm_pattern = r"Cm =\s+([-+]?\d*\.\d+)"
            
            # Find all matches in case there are multiple iterations
            cl_matches = re.findall(cl_pattern, output)
            cd_matches = re.findall(cd_pattern, output)
            cdp_matches = re.findall(cdp_pattern, output)
            cm_matches = re.findall(cm_pattern, output)
            
            # Take the last match if it exists (final converged value)
            if cl_matches and cd_matches and cm_matches:
                cl = float(cl_matches[-1])
                cd = float(cd_matches[-1])
                cm = float(cm_matches[-1])
                cdp = float(cdp_matches[-1]) if cdp_matches else 0.0
                
                print(f"CL: {cl:.4f}  CD: {cd:.6f}")
                airfoil_success = True
                
                # Add to results list
                result_data = {
                    'alpha': angle,
                    'CL': cl,
                    'CD': cd,
                    'CDp': cdp,
                    'CM': cm
                }
                airfoil_results.append(result_data)
                
                # Append to results file
                with open(results_file, "a") as f:
                    f.write(f"{angle:8.2f}{cl:10.4f}{cd:10.6f}{cdp:10.6f}{cm:10.4f}\n")
                
                # Append to summary file
                with open(summary_file, "a") as f:
                    f.write(f"{airfoil_name},{angle},{cl},{cd},{cm}\n")
                    
            else:
                print(f"Failed to extract data")
        
        except subprocess.TimeoutExpired:
            print(f"Timeout")
            timeout_count += 1
        except Exception as e:
            print(f"Error: {str(e)}")
    
    # Clean up
    try:
        os.remove("xfoil_temp_cmd.txt")
    except:
        pass
    
    # Update statistics
    if airfoil_success:
        successful += 1
        
        # Print summary for this airfoil
        print(f"  Results for {airfoil_name}:")
        print(f"  {'Alpha':>8}{'CL':>10}{'CD':>10}{'L/D':>10}")
        print(f"  {'-'*38}")
        
        for r in airfoil_results:
            ld_ratio = r['CL']/r['CD'] if r['CD'] > 0 else 0
            print(f"  {r['alpha']:8.2f}{r['CL']:10.4f}{r['CD']:10.6f}{ld_ratio:10.2f}")
    else:
        failed += 1
        print(f"  ❌ No valid results for {airfoil_name}")
    
    # Progress and time estimate
    elapsed = time.time() - start_time
    avg_time_per_file = elapsed / idx
    remaining_files = total_files - idx
    estimated_remaining = avg_time_per_file * remaining_files
    
    print(f"\n  Progress: {idx}/{total_files} ({idx/total_files*100:.1f}%)")
    print(f"  Elapsed: {elapsed/60:.1f} min, Est. remaining: {estimated_remaining/60:.1f} min")

# Print final summary
end_time = time.time()
total_time = end_time - start_time

print("\n" + "="*60)
print(f"Analysis complete!")
print(f"Total time: {total_time/60:.1f} minutes")
print(f"Processed {total_files} airfoils")
print(f"Success: {successful}, Failed: {failed}, Timeouts: {timeout_count}")
print(f"Results saved to {RESULTS_DIR} directory")
print(f"Summary file: {summary_file}")
print("="*60)
