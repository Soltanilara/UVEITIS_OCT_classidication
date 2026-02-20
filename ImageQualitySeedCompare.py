import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# List of your CSV files
csv_files = ["split_2/test_w_ImageQuality.csv", "split_9/test_w_ImageQuality.csv",
             "split_3/test_w_ImageQuality.csv", "split_10/test_w_ImageQuality.csv"]

# Metrics to plot
metrics = ["BRISQUE", "PIQE", "Entropy", "Edge Sharpness"]

# Read data from each file
dataframes = [pd.read_csv(file) for file in csv_files]

# Colors for different files
colors = sns.color_palette("tab10", len(csv_files))

# User choice: 'kde' or 'hist'
# plot_type = "kde"  # Change to "hist" for histogram overlay
plot_type = "hist"  # Change to "hist" for histogram overlay

index_to_split = {0:2,1:9,2:3,3:10}
# Generate and save plots
for metric in metrics:
    plt.figure(figsize=(10, 6))
    for i, df in enumerate(dataframes):
        if metric in df.columns:
            data = df[metric].dropna()
            if plot_type == "kde":
                # KDE plot
                sns.kdeplot(data, label=f"Split {index_to_split[i]}", color=colors[i], linewidth=2)
            elif plot_type == "hist":
                # Histogram overlay
                plt.hist(data, bins=30, alpha=0.2, label=f"Split {index_to_split[i]}", color=colors[i], density=True)
        else:
            print(f"Warning: '{metric}' not found in File {i + 1}")

    # Style the plot
    plt.title(f"{metric} Distribution Comparison for Good/Bad Splits", fontsize=16, weight='bold')
    plt.xlabel(metric, fontsize=14)
    plt.ylabel("Density", fontsize=14)
    plt.legend(title="Splits", fontsize=12, title_fontsize=14)
    plt.grid(visible=True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Save the plot
    plt.savefig(f"Split Image Quality Comparisons/{metric}_{plot_type}_overlay.png", dpi=300)
    plt.close()

print("Plots have been generated and saved.")
