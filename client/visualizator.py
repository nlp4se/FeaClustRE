import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.preprocessing import LabelEncoder
import joblib
import os
import seaborn as sns

def add_line_breaks(labels):
    return [label.replace(' ', '\n') for label in labels]

def plot_dendrogram(model, labels, color_threshold):
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    cluster_colors = {}

    def get_random_color():
        return np.random.rand(3)

    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1
                if child_idx not in cluster_colors:
                    cluster_colors[child_idx] = get_random_color()
            else:
                current_count += counts[child_idx - n_samples]

        counts[i] = current_count

        new_cluster_idx = n_samples + i
        cluster_colors[new_cluster_idx] = get_random_color()

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    labels = add_line_breaks(labels=labels)

    dendrogram(linkage_matrix,
               labels=labels,
               color_threshold=color_threshold,
               leaf_font_size=10)

    plt.xticks(rotation=90, fontsize=10, ha='right')

def plot_scatter(data, labels):
    plt.figure(figsize=(10, 6))
    label_encoder = LabelEncoder()
    numeric_labels = label_encoder.fit_transform(labels)

    # Check if data is 1D or 2D
    if data.ndim == 1:
        plt.scatter(np.arange(len(data)), data, c=numeric_labels, cmap='viridis', marker='o', edgecolor='k')
        plt.xlabel('Sample Index', fontsize=12)
        plt.ylabel('Value', fontsize=12)
    elif data.ndim == 2 and data.shape[1] >= 2:
        scatter = plt.scatter(data[:, 0], data[:, 1], c=numeric_labels, cmap='viridis', marker='o', edgecolor='k')
        for cluster in np.unique(numeric_labels):
            indices = np.where(numeric_labels == cluster)[0]
            if len(indices) > 1:
                for i in range(len(indices)):
                    for j in range(i + 1, len(indices)):
                        plt.plot(
                            [data[indices[i], 0], data[indices[j], 0]],
                            [data[indices[i], 1], data[indices[j], 1]],
                            color='gray', alpha=0.5, linewidth=0.5
                        )

        plt.xlabel('Feature 1', fontsize=12)
        plt.ylabel('Feature 2', fontsize=12)

        for i, label in enumerate(labels):
            plt.text(data[i, 0], data[i, 1] - 0.02, label, fontsize=8, ha='center')

    else:
        raise ValueError("Input data must be either 1D or at least 2D with two or more features.")

    plt.title('Scatter Plot of Clusters', fontsize=14)
    plt.grid(alpha=0.75)
    plt.colorbar(scatter, label='Cluster Labels')
    plt.tight_layout()

def plot_heatmap(data):
    """Plot heatmap of the distances between clusters."""
    distance_matrix = np.linalg.norm(data[:, np.newaxis] - data, axis=2)
    plt.figure(figsize=(10, 8))
    sns.heatmap(distance_matrix, cmap='viridis', cbar=True)
    plt.title('Heatmap of Distances Between Points', fontsize=14)
    plt.xlabel('Samples', fontsize=12)
    plt.ylabel('Samples', fontsize=12)
    plt.tight_layout()

def show_dendrogram(model_file):
    model_info = joblib.load(model_file)

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

    # Dynamically determine n_clusters if available, otherwise use len of leaves
    if hasattr(clustering_model, 'n_clusters_'):
        n_clusters = clustering_model.n_clusters_
    else:
        n_clusters = len(set(clustering_model.labels_))  # Fallback if n_clusters is not available

    CLUSTER_COLOR_THRESHOLD = max(0.1, min(1.0, 0.85 - 0.01 * (n_clusters - 1)))

    print(f"Computed CLUSTER_COLOR_THRESHOLD: {CLUSTER_COLOR_THRESHOLD}")

    if hasattr(clustering_model, 'children_'):
        n_leaves = len(data)
        max_figsize_width = 120
        max_figsize_height = 15
        figsize_width = min(max_figsize_width, n_leaves * 0.85)
        figsize_height = max(12, min(max_figsize_height, n_leaves * 0.85))

        # Construct the folder name with attributes
        folder_name = (f"{application_name}_dt-{distance_threshold}_vw-{verb_weight}_ow-{object_weight}"
                       .replace(" ", "_"))
        save_directory = os.path.join(r"C:\Users\Max\NLP4RE\Dendogram-Generator\static\png", folder_name)
        os.makedirs(save_directory, exist_ok=True)

        # Dendrogram Plot
        plt.figure(figsize=(figsize_width, figsize_height))
        plot_dendrogram(clustering_model, labels, color_threshold=CLUSTER_COLOR_THRESHOLD)

        plt.title(
            f"{application_name} | {affinity} | Distance Threshold: {distance_threshold} "
            f"| Verb Weight: {verb_weight} | Object Weight: {object_weight}",
            fontsize=14)
        plt.xlabel('Features', fontsize=14)
        plt.ylabel('Distance', fontsize=14)
        plt.subplots_adjust(bottom=0.2)
        plt.tight_layout()

        dendrogram_save_path = os.path.join(save_directory, f"{application_name}_dendrogram.png")
        plt.savefig(dendrogram_save_path)
        plt.close()
        print(f"Dendrogram saved at: {dendrogram_save_path}")

        # data = np.array(data)
        # plot_scatter(data, labels)
        # scatter_save_path = os.path.join(save_directory, f"{application_name}_scatter.png")
        # plt.savefig(scatter_save_path)
        # plt.close()
        # print(f"Scatter plot saved at: {scatter_save_path}")

        # plot_heatmap(data)
        # heatmap_save_path = os.path.join(save_directory, f"{application_name}_heatmap.png")
        # plt.savefig(heatmap_save_path)
        # plt.close()
        # print(f"Heatmap saved at: {heatmap_save_path}")

    else:
        raise ValueError("The provided model is not AgglomerativeClustering.")

if __name__ == "__main__":
    pkls_directory = r"C:\Users\Max\NLP4RE\Dendogram-Generator\static\pkls"

    for filename in os.listdir(pkls_directory):
        if filename.endswith('.pkl'):
            model_file = os.path.join(pkls_directory, filename)
            print(f"Processing: {model_file}")
            show_dendrogram(model_file)
