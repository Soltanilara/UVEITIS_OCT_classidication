import pandas as pd
import numpy as np
import plotly.express as px
import os

# beta = '0.0'
# beta = '1000'
# # Load the CSV file (change this path to where your file is located)
# csv_path = "output_split_10/scratch_resnet50_CE_loss_batch_64_lr_0.001_10_10_1_1e-05_epochs_100_hflip_seed_0/checkpoint"
# csv_file = csv_path+"/test_features.csv"  # Specify the correct path to your CSV file
# data = pd.read_csv(csv_file)

# Set the sample size for plotting (None means no sampling, set to an integer for sampling)
sample_size = None  # Set to None if you want to visualize all points
# method = 'MDS'
# method = 'UMAP'
# method = 't-SNE'
method = 'Spectral_Embedding'

# Separate the feature vectors and labels
# X = np.load('Pretrained Features/test_features_3_10_2_9.npy')
# y_true = np.load('Pretrained Features/labels_3_10_2_9.npy')
X = np.load('Pretrained Features/test_val_features_3_10.npy')
y_true = np.load('Pretrained Features/test_val_labels_3_10.npy')
# y_prob = data['Pred_Prob_Class1'].values  # Predicted probabilities for class 1

if method=='MDS':
    from sklearn.manifold import MDS
    mds = MDS(n_components=3, random_state=42)
    X_mds = mds.fit_transform(X)
elif method=='UMAP':
    from umap import UMAP
    reducer = UMAP(n_components=3, random_state=42)
    X_mds = reducer.fit_transform(X)
elif method=='t-SNE':
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=3, random_state=42)
    X_mds = tsne.fit_transform(X)
elif method=='Spectral_Embedding':
    from sklearn.manifold import SpectralEmbedding
    spectral = SpectralEmbedding(n_components=3, random_state=42)
    X_mds = spectral.fit_transform(X)

# Create a DataFrame for the 3D coordinates and labels
df_mds = pd.DataFrame(X_mds, columns=['Dim1', 'Dim2', 'Dim3'])
df_mds['True_Label'] = y_true
# df_mds['Pred_Prob_Class1'] = y_prob

# Option to sample a subset of points for visualization
def sample_data(data, sample_size=None):
    if sample_size and sample_size < len(data):
        data = data.sample(n=sample_size, random_state=42).reset_index(drop=True)
    return data

# Sample the data only for plotting purposes
df_mds = sample_data(df_mds, sample_size=sample_size)

# # Map True_Label to symbols: cross for positive (1), circle for negative (0)
# df_mds['Symbol'] = df_mds['True_Label'].map({1: 'x', 0: 'circle'})
#
# # Plot 3D scatter plot with Plotly
# fig = px.scatter_3d(df_mds, x='Dim1', y='Dim2', z='Dim3',
#                     color='Pred_Prob_Class1',  # Color by predicted probability of class 1
#                     symbol='Symbol',  # Marker shape by true label
#                     color_continuous_scale='RdBu_r',  # Use the Reds color scale
#                     labels={'Pred_Prob_Class1': 'Positive Class Probability'},
#                     title="3D Visualization of Latent Features",
#                     opacity=0.8,  # Adjust transparency if needed
#                     )
#
# # Adjust the marker size and other visual properties
# fig.update_traces(marker=dict(size=3))  # Significantly reduce marker size
#
# # Save the plot to the same directory as the CSV file
# # output_path = os.path.dirname(csv_file)
# plot_file = os.path.join(csv_path, f"3D_visualization_{method}.html")
# fig.write_html(plot_file)

# Second Plot: Color by True Class (without changing marker shapes)
distinct_colors = [
    '#e6194b',  # Red
    '#3cb44b',  # Green
    '#ffe119',  # Yellow
    '#4363d8',  # Blue
    '#f58231',  # Orange
    '#911eb4',  # Purple
    '#46f0f0',  # Cyan
    '#f032e6',  # Magenta
    '#bcf60c',  # Lime
    '#fabebe'   # Pink
]
fig2 = px.scatter_3d(df_mds, x='Dim1', y='Dim2', z='Dim3',
                    color='True_Label',  # Color by true class (binary classification)
                    color_discrete_sequence='Plotly',  # Use the custom color palette
                    labels={'True_Label': 'True Class'},
                    title="3D MDS Visualization Colored by True Class",
                    opacity=0.4  # Adjust transparency if needed
                    )

# Adjust the marker size for the second plot (same size as the first plot)
fig2.update_traces(marker=dict(size=2))  # Reduce marker size

# Save the second plot to the same directory as the CSV file
# plot_file2 = f"Pretrained Features/3D_visualization_test_features_3_10_2_9_{method}.html"
plot_file2 = f"Pretrained Features/3D_visualization_test_val_features_3_10_{method}.html"
fig2.write_html(plot_file2)
