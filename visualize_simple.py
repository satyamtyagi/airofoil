import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Set up directories
RESULTS_DIR = "results"
PLOTS_DIR = "airfoil_plots"

# Create plots directory
os.makedirs(PLOTS_DIR, exist_ok=True)

# Load the data
print("Loading airfoil data...")
combined_data = pd.read_csv(os.path.join(RESULTS_DIR, "combined_airfoil_data.csv"))
best_performance = pd.read_csv(os.path.join(RESULTS_DIR, "airfoil_best_performance.csv"))

print(f"Loaded data for {len(best_performance)} airfoils with {len(combined_data)} total data points")

# 1. Top 10 Airfoils by L/D Ratio
print("Creating top airfoils by L/D ratio plot...")
top_ld = best_performance.sort_values('best_ld', ascending=False).head(10)
plt.figure(figsize=(12, 7))
bars = plt.bar(top_ld['airfoil'], top_ld['best_ld'])
plt.title('Top 10 Airfoils by Lift-to-Drag Ratio')
plt.xlabel('Airfoil')
plt.ylabel('Best Lift-to-Drag Ratio')
plt.xticks(rotation=45, ha='right')

# Add thickness and camber annotations
for i, bar in enumerate(bars):
    airfoil = top_ld['airfoil'].iloc[i]
    thickness = top_ld['max_thickness'].iloc[i]
    camber = top_ld['max_camber'].iloc[i]
    plt.text(bar.get_x() + bar.get_width()/2., 
             bar.get_height() + 2,
             f't:{thickness:.3f}\nc:{camber:.3f}',
             ha='center', va='bottom', rotation=0, fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'top10_ld_ratio.png'), dpi=300)
plt.close()

# 2. Thickness vs L/D Ratio (simple scatter)
print("Creating thickness vs L/D ratio plot...")
plt.figure(figsize=(10, 6))
plt.scatter(best_performance['max_thickness'], best_performance['best_ld'])

# Add labels for top performers
for i, row in best_performance.sort_values('best_ld', ascending=False).head(5).iterrows():
    plt.annotate(row['airfoil'], 
                 xy=(row['max_thickness'], row['best_ld']),
                 xytext=(5, 5),
                 textcoords='offset points',
                 fontsize=9)

plt.title('Effect of Airfoil Thickness on Best L/D Ratio')
plt.xlabel('Maximum Thickness (t/c)')
plt.ylabel('Best Lift-to-Drag Ratio')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'thickness_vs_ld_ratio.png'), dpi=300)
plt.close()

# 3. Camber vs Max CL (simple scatter)
print("Creating camber vs max CL plot...")
plt.figure(figsize=(10, 6))
plt.scatter(best_performance['max_camber'], best_performance['max_cl'])

# Add labels for top CL performers
for i, row in best_performance.sort_values('max_cl', ascending=False).head(5).iterrows():
    plt.annotate(row['airfoil'], 
                 xy=(row['max_camber'], row['max_cl']),
                 xytext=(5, 5),
                 textcoords='offset points',
                 fontsize=9)

plt.title('Effect of Airfoil Camber on Maximum Lift Coefficient')
plt.xlabel('Maximum Camber (m/c)')
plt.ylabel('Maximum Lift Coefficient (CL)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'camber_vs_max_cl.png'), dpi=300)
plt.close()

# 4. Thickness-Camber Distribution (simple scatter)
print("Creating thickness-camber distribution plot...")
plt.figure(figsize=(12, 8))
plt.scatter(best_performance['max_thickness'], 
           best_performance['max_camber'],
           s=best_performance['best_ld'])

# Add labels for top performers
for i, row in best_performance.sort_values('best_ld', ascending=False).head(10).iterrows():
    plt.annotate(row['airfoil'], 
                 xy=(row['max_thickness'], row['max_camber']),
                 xytext=(5, 5),
                 textcoords='offset points',
                 fontsize=9)

plt.title('Airfoil Thickness-Camber Distribution (Size = L/D Ratio)')
plt.xlabel('Maximum Thickness (t/c)')
plt.ylabel('Maximum Camber (m/c)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'thickness_camber_distribution.png'), dpi=300)
plt.close()

# 5. L/D ratio vs. Angle of Attack for top airfoils
print("Creating L/D vs angle of attack plot...")
plt.figure(figsize=(10, 8))

# Get top 5 airfoils by L/D ratio
top5_airfoils = best_performance.sort_values('best_ld', ascending=False).head(5)['airfoil'].tolist()

for airfoil in top5_airfoils:
    airfoil_data = combined_data[combined_data['airfoil'] == airfoil]
    airfoil_data['L_D'] = airfoil_data['CL'] / airfoil_data['CD']
    plt.plot(airfoil_data['alpha'], airfoil_data['L_D'], 
             label=f"{airfoil}", 
             marker='o')
    
plt.title('Lift-to-Drag Ratio vs. Angle of Attack for Top 5 Airfoils')
plt.xlabel('Angle of Attack (degrees)')
plt.ylabel('Lift-to-Drag Ratio')
plt.grid(True, alpha=0.3)
plt.legend(title="Airfoil", loc='best')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'ld_vs_alpha.png'), dpi=300)
plt.close()

# 6. Detailed performance curves for best airfoil
print("Creating detailed performance curves for best airfoil...")
best_airfoil = best_performance.sort_values('best_ld', ascending=False).iloc[0]['airfoil']
best_data = combined_data[combined_data['airfoil'] == best_airfoil]

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

# CL vs Alpha
ax1.plot(best_data['alpha'], best_data['CL'], 'o-', color='blue')
ax1.set_ylabel('Lift Coefficient (CL)')
ax1.set_title(f'Airfoil {best_airfoil} Performance Curves')
ax1.grid(True, alpha=0.3)

# CD vs Alpha
ax2.plot(best_data['alpha'], best_data['CD'], 'o-', color='red')
ax2.set_ylabel('Drag Coefficient (CD)')
ax2.grid(True, alpha=0.3)

# L/D vs Alpha
best_data['L_D'] = best_data['CL'] / best_data['CD']
ax3.plot(best_data['alpha'], best_data['L_D'], 'o-', color='green')
ax3.set_ylabel('Lift-to-Drag Ratio')
ax3.set_xlabel('Angle of Attack (degrees)')
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, f'{best_airfoil}_performance.png'), dpi=300)
plt.close()

print(f"All visualizations saved to {PLOTS_DIR} directory")
print("You can view the plots to analyze the relationships between airfoil geometry and performance")
