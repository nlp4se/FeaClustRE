import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
import joblib
import os
import seaborn as sns

# Constants for visualization
ABOVE_THRESHOLD_COLOR = '#D3D3D3'
N_PALETTE_COLORS = 50
PALETTE_NAME = 'husl'

# Helper function to add line breaks to labels
def add_line_breaks(labels):
    return [label.replace(' ', '\n') for label in labels]

# Function to plot the dendrogram
def plot_dendrogram(model, labels, color_threshold):
    # Initialize count array
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)

    # Create color palette for clusters
    palette = sns.color_palette(PALETTE_NAME, N_PALETTE_COLORS)
    cluster_colors = {}

    def get_color_for_cluster(cluster_idx):
        return palette[cluster_idx % N_PALETTE_COLORS]

    # Compute the number of samples in each node (or merged cluster)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1
                if child_idx not in cluster_colors:
                    cluster_colors[child_idx] = get_color_for_cluster(child_idx)
            else:
                current_count += counts[child_idx - n_samples]

        counts[i] = current_count

        new_cluster_idx = n_samples + i
        cluster_colors[new_cluster_idx] = get_color_for_cluster(new_cluster_idx)

    # Create linkage matrix for dendrogram
    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Update labels to have line breaks
    labels = add_line_breaks(labels=labels)

    # Plot dendrogram
    dendrogram(linkage_matrix,
               labels=labels,
               color_threshold=color_threshold,
               leaf_font_size=10,
               orientation='right',
               above_threshold_color=ABOVE_THRESHOLD_COLOR)

    # Rotate x-axis labels for readability
    plt.xticks(rotation=45, fontsize=10, ha='right')

# Function to display and save the dendrogram for a given model
def show_dendrogram(model_file):
    # Load the model information from the .pkl file
    model_info = joblib.load(model_file)

    # Extract relevant information from the model
    application_name = model_info['application_name']
    distance_threshold = model_info['distance_threshold']
    clustering_model = model_info['model']
    affinity = model_info['affinity']
    data = model_info['data_points']
    labels = model_info['labels']

    try:
        verb_weight = model_info.get('verb_weight', 'N/A')
        object_weight = model_info.get('object_weight', 'N/A')
    except KeyError as e:
        print(f"Missing key: {e}")

    if hasattr(clustering_model, 'n_clusters_'):
        n_clusters = clustering_model.n_clusters_
    else:
        n_clusters = len(set(clustering_model.labels_))

    # Define the threshold for coloring clusters in the dendrogram
    CLUSTER_COLOR_THRESHOLD = distance_threshold
    print("------------")
    print(f"Threshold: {distance_threshold}")
    print(f"Detected {n_clusters} clusters")
    print(f"Computed CLUSTER_COLOR_THRESHOLD: {CLUSTER_COLOR_THRESHOLD}")
    print("------------")

    if hasattr(clustering_model, 'children_'):
        # Determine figure size based on number of leaves (data points)
        n_leaves = len(data)
        max_figsize_width = 120
        max_figsize_height = 15
        figsize_height = min(max_figsize_width, n_leaves * 0.85)
        figsize_width = max(12, min(max_figsize_height, n_leaves * 0.85))

        # Create a folder for saving the dendrogram image
        folder_name = (f"{affinity}_{application_name}_dt-{distance_threshold}_vw-{verb_weight}_ow-{object_weight}"
                       .replace(" ", "_"))
        save_directory = os.path.join(r"C:\Users\Max\NLP4RE\Dendogram-Generator\static\png", folder_name)
        os.makedirs(save_directory, exist_ok=True)

        # Plot the dendrogram
        plt.figure(figsize=(figsize_width, figsize_height))
        plot_dendrogram(clustering_model, labels, color_threshold=CLUSTER_COLOR_THRESHOLD)

        # Set plot title and labels
        plt.title(
            f"{application_name} | {affinity} | Distance Threshold: {distance_threshold} "
            f"| Verb Weight: {verb_weight} | Object Weight: {object_weight}",
            fontsize=14)
        plt.xlabel('Features', fontsize=14)
        plt.ylabel('Distance', fontsize=14)
        plt.subplots_adjust(bottom=0.2)
        plt.tight_layout()

        # Save the dendrogram image
        dendrogram_save_path = os.path.join(save_directory, f"{application_name}_dendrogram.png")
        plt.savefig(dendrogram_save_path)
        plt.close()
        print(f"Dendrogram saved at: {dendrogram_save_path}")

    else:
        raise ValueError("The provided model is not AgglomerativeClustering.")

# Main function to process .pkl files in the specified directory
if __name__ == "__main__":
    # Directory containing the .pkl files
    pkls_directory = r"C:\Users\Max\NLP4RE\Dendogram-Generator\static\pkls"

    # Loop through all .pkl files in the directory and generate dendrograms
    for filename in os.listdir(pkls_directory):
        if filename.endswith('.pkl'):
            model_file = os.path.join(pkls_directory, filename)
            print(f"Processing: {model_file}")
            show_dendrogram(model_file)
