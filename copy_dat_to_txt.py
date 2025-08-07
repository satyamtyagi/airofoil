#!/usr/bin/env python3
import os
import shutil
import sys

def copy_dat_to_txt(source_dir, target_dir):
    """Copy all .dat files from source_dir to target_dir as .txt files"""
    # Ensure target directory exists
    os.makedirs(target_dir, exist_ok=True)
    
    # Count files for reporting
    total_files = 0
    copied_files = 0
    
    # List all files in source directory
    for filename in os.listdir(source_dir):
        total_files += 1
        if filename.endswith('.dat'):
            # Construct the source and destination paths
            source_path = os.path.join(source_dir, filename)
            # Replace .dat extension with .txt
            new_filename = filename[:-4] + '.txt'
            destination_path = os.path.join(target_dir, new_filename)
            
            # Copy the file
            shutil.copy2(source_path, destination_path)
            copied_files += 1
            print(f"Copied: {filename} → {new_filename}")
    
    print(f"\nProcess completed: {copied_files} files copied out of {total_files} total files.")

if __name__ == "__main__":
    # Default directories
    source_dir = "airfoils_uiuc"
    target_dir = "airfoils_uiuc_txt"
    
    # Use command line arguments if provided
    if len(sys.argv) > 2:
        source_dir = sys.argv[1]
        target_dir = sys.argv[2]
    
    print(f"Copying .dat files from '{source_dir}' to '{target_dir}' as .txt files...")
    copy_dat_to_txt(source_dir, target_dir)
