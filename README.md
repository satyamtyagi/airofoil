# Airfoil Analysis Toolkit

A collection of Python scripts for analyzing airfoil performance using XFOIL and machine learning models to predict and visualize aerodynamic results.

## Overview

This toolkit provides a complete workflow for airfoil analysis through two complementary approaches:

1. **Traditional CFD-based analysis**: Batch processing airfoil geometry files through XFOIL to extract key aerodynamic performance metrics.

2. **Machine Learning-based analysis**: Using trained neural networks to instantly predict airfoil performance and explore the design space through a low-dimensional latent representation.

## Scripts

### XFOIL-based Analysis
- `run_xfoil_batch_all.py` - Batch process multiple airfoil .dat files using XFOIL
- `combine_airfoil_data.py` - Combine airfoil geometry data with performance results
- `visualize_simple.py` - Generate static visualizations of airfoil data
- `visualize_airfoils_fixed.py` - Interactive dashboard for exploring airfoils and performance

### Machine Learning Tools
- `inspect_models.py` - Examine the structure of the PyTorch airfoil models
- `demonstrate_models.py` - CLI tool to test models on airfoils with different surrogate versions
- `compare_predictions.py` - Generate comparison visualizations between ML predictions and XFOIL
- `train_surrogate.py` - Retrain the surrogate model on your XFOIL dataset

### PARSEC Parameter Sweep
- `parsec_to_dat.py` - Convert PARSEC parameters to airfoil coordinates
- `parsec_fit.py` - Fit PARSEC parameters to existing airfoil geometries
- `generate_parsec_variations.py` - Generate airfoil variations by modifying PARSEC parameters
- `parsec_parameter_sweep.py` - Perform efficient parameter sweeps using HDF5 database and surrogate model
- `visualize_parameter_relationships.py` - Create visualizations of parameter-performance relationships
- `visualize_sweep_3d.py` - Interactive 3D visualization of top-performing airfoils from parameter sweep

## Directory Structure

- `/airfoils_uiuc` - Directory containing airfoil .dat files
- `/results` - Directory where analysis results are stored
- `/models` - Directory containing PyTorch model files
- `/demo_results` - Directory containing visualization outputs from model demonstrations
- `/parsec_results` - Directory containing PARSEC parameter statistics and fits
- `/parameter_sweep_results` - Directory containing parameter sweep results and visualizations

## Requirements

- Python 3.x
- XFOIL installed and accessible in your PATH (or modify the path in `run_xfoil_batch_all.py`)
- Python packages:
  - pandas
  - numpy
  - matplotlib
  - plotly
  - dash
  - torch (PyTorch)
  - scipy
  - tqdm

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/your-username/airfoil-analysis.git
   cd airfoil-analysis
   ```

2. Install required Python packages:
   ```
   pip install pandas numpy matplotlib plotly dash torch scipy tqdm
   ```

3. Ensure XFOIL is installed and accessible in your PATH, or update `XFOIL_PATH` in `run_xfoil_batch_all.py`

## Usage

### 1. Batch Airfoil Analysis with XFOIL

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

### 3. Visualize Results with XFOIL Data

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

### 4. Machine Learning Tools

#### Inspect the ML models:
```
python inspect_models.py
```
This shows the architecture and parameters of the encoder, decoder, and surrogate models.

#### Test the models on airfoils:
```
python demonstrate_models.py --model retrained --airfoils a18sm ag13 hs1606
```
Options:
- `--model [original|retrained]`: Choose which surrogate model to use
- `--airfoils`: List specific airfoils to analyze
- `--all`: Analyze all available airfoils

#### Compare ML predictions with XFOIL results:
```
python compare_predictions.py --airfoils a18sm ag13 hs1606
```
This generates visualizations comparing the original and retrained models against XFOIL data.

#### Retrain the surrogate model with your data:
```
python train_surrogate.py
```
This trains a new surrogate model using your airfoil geometry files and XFOIL results.

### 5. PARSEC Parameter Sweep

The PARSEC parameterization system allows for efficient exploration of the airfoil design space using a small set of meaningful parameters.

#### Run a parameter sweep with the surrogate model:
```
python parsec_parameter_sweep.py --params "rLE,Xup,Yup,Xlo,Ylo" --steps 5
```
Options:
- `--params`: Comma-separated list of PARSEC parameters to vary
- `--steps`: Number of steps for each parameter
- `--batch`: Batch size for surrogate processing
- `--output`: Output HDF5 database filename

This will:
- Generate all combinations of the specified PARSEC parameters
- Store them efficiently in an HDF5 database
- Convert parameters to coordinates in memory without creating intermediate files
- Process each airfoil through the surrogate model
- Generate visualizations of top performers

#### Visualize parameter relationships:
```
python visualize_parameter_relationships.py
```
This creates correlation matrices and performance landscapes showing how parameters affect lift, drag and efficiency.

#### Interactive 3D visualization of results:
```
python visualize_sweep_3d.py
```
This launches an interactive dashboard with 3D airfoil shapes, performance metrics, and parameter heatmaps.

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
