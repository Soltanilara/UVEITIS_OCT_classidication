import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV files
split = 9
test_file_path = f'split_{str(split)}/test.csv'
val_file_path = f'split_{str(split)}/val.csv'
test_data = pd.read_csv(test_file_path)
val_data = pd.read_csv(val_file_path)

# Count the label densities
test_label_counts = test_data['Label'].value_counts()
val_label_counts = val_data['Label'].value_counts()

# Ensure all categories are represented (fill missing with 0)
categories = ['negative', 'mild', 'moderate', 'severe']
test_label_counts = test_label_counts.reindex(categories, fill_value=0)
val_label_counts = val_label_counts.reindex(categories, fill_value=0)

# Calculate densities
test_density = test_label_counts / len(test_data)
val_density = val_label_counts / len(val_data)

# Plotting the histogram
x = np.arange(len(categories))  # Label positions
width = 0.35  # Width of the bars

fig, ax = plt.subplots(figsize=(12, 7))
rects1 = ax.bar(x - width/2, test_density, width, label='Test', color='steelblue')  # Blue color
rects2 = ax.bar(x + width/2, val_density, width, label='Validation', color='indianred')  # Orange color

# Add labels, title, and custom x-axis tick labels
ax.set_xlabel('Label')
ax.set_ylabel('Density')
ax.set_title('Label Densities in Test and Validation Datasets')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()

# Add gridlines
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Option to add value labels on bars
def add_labels(rects, counts, show_counts=True):
    for rect, count in zip(rects, counts):
        height = rect.get_height()
        if show_counts:
            ax.annotate(f'{int(count)}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

# Add counts as labels (optional)
add_labels(rects1, test_label_counts, show_counts=True)
add_labels(rects2, val_label_counts, show_counts=True)

plt.tight_layout()

# Save the figure
output_path = f'split_{str(split)}/val_test_label_density_histogram.png'  # Replace with your desired output path
plt.savefig(output_path)

plt.show()

print(f"Figure saved to {output_path}")
