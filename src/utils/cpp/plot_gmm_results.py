import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Read the results
df = pd.read_csv("/home/moussabendjilali/test_file_5100_54440/gmm_results.csv")

# Clean the accuracy column by removing 'tensor()' wrapper and converting to float
df['accuracy'] = df['accuracy'].str.extract(r'tensor\((.*?)\)').astype(float)

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Create a figure with multiple subplots
fig = plt.figure(figsize=(20, 10))
fig.suptitle('GMM Variants Performance Analysis', fontsize=16, y=1.02)

# 1. Duration vs Iterations with trend lines
ax1 = plt.subplot(2, 2, 1)
for variant in df['variant'].unique():
    variant_data = df[df['variant'] == variant]
    # Scatter plot
    ax1.scatter(variant_data['iterations'], variant_data['duration'], 
                alpha=0.5, label=f'{variant} (points)')
    
    # Trend line
    z = np.polyfit(variant_data['iterations'], variant_data['duration'], 1)
    p = np.poly1d(z)
    ax1.plot(variant_data['iterations'], p(variant_data['iterations']), 
             label=f'{variant} (trend)', linestyle='--', linewidth=2)

ax1.set_xlabel('Iterations')
ax1.set_ylabel('Duration (seconds)')
ax1.set_title('Duration vs Iterations')
ax1.legend()
ax1.grid(True)

# 2. Accuracy vs Iterations with trend lines
ax2 = plt.subplot(2, 2, 2)
for variant in df['variant'].unique():
    variant_data = df[df['variant'] == variant]
    # Scatter plot
    ax2.scatter(variant_data['iterations'], variant_data['accuracy'], 
                alpha=0.5, label=f'{variant} (points)')
    
    # Trend line
    z = np.polyfit(variant_data['iterations'], variant_data['accuracy'], 1)
    p = np.poly1d(z)
    ax2.plot(variant_data['iterations'], p(variant_data['iterations']), 
             label=f'{variant} (trend)', linestyle='--', linewidth=2)

ax2.set_xlabel('Iterations')
ax2.set_ylabel('Accuracy')
ax2.set_title('Accuracy vs Iterations')
ax2.legend()
ax2.grid(True)

# 3. Duration vs Point Count with trend lines
ax3 = plt.subplot(2, 2, 3)
for variant in df['variant'].unique():
    variant_data = df[df['variant'] == variant]
    # Scatter plot
    ax3.scatter(variant_data['point_count'], variant_data['duration'], 
                alpha=0.5, label=f'{variant} (points)')
    
    # Trend line
    z = np.polyfit(variant_data['point_count'], variant_data['duration'], 1)
    p = np.poly1d(z)
    ax3.plot(variant_data['point_count'], p(variant_data['point_count']), 
             label=f'{variant} (trend)', linestyle='--', linewidth=2)

ax3.set_xlabel('Point Count')
ax3.set_ylabel('Duration (seconds)')
ax3.set_title('Duration vs Point Count')
ax3.legend()
ax3.grid(True)

# 4. Accuracy vs Point Count with trend lines
ax4 = plt.subplot(2, 2, 4)
for variant in df['variant'].unique():
    variant_data = df[df['variant'] == variant]
    # Scatter plot
    ax4.scatter(variant_data['point_count'], variant_data['accuracy'], 
                alpha=0.5, label=f'{variant} (points)')
    
    # Trend line
    z = np.polyfit(variant_data['point_count'], variant_data['accuracy'], 1)
    p = np.poly1d(z)
    ax4.plot(variant_data['point_count'], p(variant_data['point_count']), 
             label=f'{variant} (trend)', linestyle='--', linewidth=2)

ax4.set_xlabel('Point Count')
ax4.set_ylabel('Accuracy')
ax4.set_title('Accuracy vs Point Count')
ax4.legend()
ax4.grid(True)

# Adjust layout
plt.tight_layout()

# Save the plot
plt.savefig('/home/moussabendjilali/test_file_5100_54440/gmm_analysis.png', 
            dpi=300, bbox_inches='tight')

# Print summary statistics
print("\nSummary Statistics by Variant:")
summary = df.groupby('variant').agg({
    'accuracy': ['mean', 'std'],
    'duration': ['mean', 'std'],
    'point_count': ['mean', 'count']
}).round(4)
print(summary)

# Calculate and print efficiency metrics
print("\nPoints Processed per Second by Variant:")
df['points_per_second'] = df['point_count'] / df['duration']
efficiency = df.groupby('variant')['points_per_second'].agg(['mean', 'std']).round(2)
print(efficiency)

plt.close()

# Create additional convergence analysis plot
plt.figure(figsize=(12, 6))
for variant in df['variant'].unique():
    variant_data = df[df['variant'] == variant]
    
    # Calculate rolling average of accuracy to show convergence trend
    acc = variant_data['accuracy']
    
    plt.plot(variant_data['iterations'], acc, 
             marker='o', label=f"{variant}", linewidth=2)

plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.title('Convergence Analysis of GMM Variants')
plt.legend()
plt.grid(True)

# Save the convergence plot
plt.savefig('/home/moussabendjilali/test_file_5100_54440/gmm_convergence.png', 
            dpi=300, bbox_inches='tight')
plt.close() 