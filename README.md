# Airfoil Analysis Toolkit

A collection of Python scripts for analyzing airfoil performance using XFOIL and visualizing results.

## Overview

This toolkit provides a complete workflow for batch processing airfoil geometry files through XFOIL, extracting key aerodynamic performance metrics, and visualizing the results through both static plots and interactive dashboards.

## Scripts

- `run_xfoil_batch_all.py` - Batch process multiple airfoil .dat files using XFOIL
- `combine_airfoil_data.py` - Combine airfoil geometry data with performance results
- `visualize_simple.py` - Generate static visualizations of airfoil data
- `visualize_airfoils_fixed.py` - Interactive dashboard for exploring airfoils and performance

## Directory Structure

- `/airfoils_uiuc` - Directory containing airfoil .dat files
- `/results` - Directory where analysis results are stored

## Requirements

- Python 3.x
- XFOIL installed and accessible in your PATH (or modify the path in `run_xfoil_batch_all.py`)
- Python packages:
  - pandas
  - numpy
  - matplotlib
  - plotly
  - dash

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/your-username/airfoil-analysis.git
   cd airfoil-analysis
   ```

2. Install required Python packages:
   ```
   pip install pandas numpy matplotlib plotly dash
   ```

3. Ensure XFOIL is installed and accessible in your PATH, or update `XFOIL_PATH` in `run_xfoil_batch_all.py`

## Usage

### 1. Batch Airfoil Analysis

Run XFOIL analysis on all airfoil .dat files in the `airfoils_uiuc` directory:

```
python run_xfoil_batch_all.py
```

This will:
- Process each airfoil at -5°, 0°, 5°, and 10° angles of attack
- Extract lift coefficient (CL), drag coefficient (CD), and moment coefficient (CM)
- Save individual results files for each airfoil
- Create a summary CSV file with data from all airfoils

### 2. Combine Geometry and Performance Data

```
python combine_airfoil_data.py
```

This will:
- Extract geometric parameters from each airfoil (thickness, camber, etc.)
- Combine with performance data from XFOIL analysis
- Generate comprehensive CSV files with both geometry and performance metrics

### 3. Visualize Results

For static plots:
```
python visualize_simple.py
```

For an interactive dashboard:
```
python visualize_airfoils_fixed.py
```

The visualizations include:
- Top performing airfoils by L/D ratio
- Correlation between thickness and aerodynamic efficiency
- Comparisons of airfoil profiles
- 3D visualizations of airfoil shapes

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
