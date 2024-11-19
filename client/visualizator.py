import joblib
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import seaborn as sns
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import pandas as pd

ABOVE_THRESHOLD_COLOR = '#D3D3D3'
N_PALETTE_COLORS = 50
PALETTE_NAME = 'husl'

model_name = "meta-llama/Llama-3.2-3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(
    'cuda' if torch.cuda.is_available() else 'cpu')

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1
)


def generate_dynamic_label(cluster_labels):
    unique_labels = list(set(cluster_labels))

    input_text = (
        "Generate a single concise label summarizing the following actions.\n\n"
        "Examples:\n"
        "Video meeting, online meeting, team video chat, conference call\n"
        "Label: Virtual Team Communication\n\n"
        "Secure chat, encrypted messaging, private message\n"
        "Label: Private Messaging\n\n"
        "Video call, group video call, secure video call, video conference\n"
        "Label: Secure Video Conferencing\n\n"
        + ", ".join(unique_labels) + "\nLabel:"
    )

    response = pipe(input_text, max_new_tokens=10, do_sample=True)
    label = response[0]['generated_text'].replace(input_text, "").strip()
    return label.split('\n')[0]


def log_clusters_at_distance_threshold(linkage_matrix, distance_threshold):
    from scipy.cluster.hierarchy import fcluster
    cluster_assignments = fcluster(linkage_matrix, t=distance_threshold, criterion='distance')
    num_clusters = len(set(cluster_assignments))
    print(f"Number of clusters at distance threshold {distance_threshold}: {num_clusters}")


def process_clusters_and_generate_dendrograms(linkage_matrix, labels, distance_threshold, application_name):
    """
    Processes each cluster individually by detecting clusters at the given distance threshold,
    saves each cluster's labels in a CSV file, and generates individual dendrograms.

    Parameters:
    - linkage_matrix: The linkage matrix used to construct the dendrogram.
    - labels: The labels of the leaf nodes.
    - distance_threshold: The distance threshold to calculate the clusters.
    - application_name: The name of the application (used for folder naming).
    """
    from scipy.cluster.hierarchy import fcluster

    # Generate cluster assignments based on the distance threshold
    cluster_assignments = fcluster(linkage_matrix, t=distance_threshold, criterion='distance')

    # Map cluster indices to their corresponding labels
    cluster_dict = {}
    for idx, cluster_id in enumerate(cluster_assignments):
        if cluster_id not in cluster_dict:
            cluster_dict[cluster_id] = []
        cluster_dict[cluster_id].append(labels[idx])

    # Create folder to save the clusters
    output_folder = f"{application_name}_clusters"
    os.makedirs(output_folder, exist_ok=True)

    # Process each cluster individually
    for cluster_id, cluster_labels in cluster_dict.items():
        print(f"Cluster {cluster_id} contains labels: {cluster_labels}")

        cluster_label = generate_dynamic_label(cluster_labels)
        print(f"Generated label for Cluster {cluster_id}: {cluster_label}")

        cluster_data = {"cluster_name": cluster_label, "feature_list": cluster_labels}
        cluster_df = pd.DataFrame(cluster_data)
        cluster_file_path = os.path.join(output_folder, f"cluster_{cluster_id}.csv")
        cluster_df.to_csv(cluster_file_path, index=False, sep=',')
        print(f"Cluster {cluster_id} saved to {cluster_file_path}")

        generate_individual_dendrogram(cluster_labels, cluster_id, application_name, cluster_label)


def generate_individual_dendrogram(cluster_labels, cluster_id, application_name, cluster_label):
    """
    Generates and saves a dendrogram for a specific cluster.

    Parameters:
    - cluster_labels: Labels of the data points in the cluster.
    - cluster_id: The ID of the cluster.
    - application_name: The name of the application (used for folder naming).
    - cluster_label: The label generated for this cluster.
    """
    if len(cluster_labels) < 2:
        print(f"Cluster {cluster_id} has less than 2 labels, skipping dendrogram generation.")
        return

    # Create dummy numerical data for the linkage function
    dummy_data = np.random.rand(len(cluster_labels), 2)  # 2D random data for visualization
    linkage_matrix = linkage(dummy_data, method='ward')

    fig, ax = plt.subplots(figsize=(10, 6))
    dendrogram(
        linkage_matrix,
        labels=cluster_labels,
        leaf_font_size=10,
        orientation='top',
        ax=ax
    )
    ax.set_title(f"Cluster {cluster_id} - {cluster_label} Dendrogram", fontsize=14)
    ax.set_xlabel("Data Points")
    ax.set_ylabel("Distance")

    output_folder = f"{application_name}_individual_dendrograms"
    os.makedirs(output_folder, exist_ok=True)
    dendrogram_path = os.path.join(output_folder, f"cluster_{cluster_id}_dendrogram.png")
    plt.savefig(dendrogram_path)
    plt.close()
    print(f"Dendrogram for Cluster {cluster_id} saved at: {dendrogram_path}")
def render_dendrogram(model_info, model, labels, color_threshold, distance_threshold):
    data = model_info['data_points']
    application_name = model_info['application_name']
    affinity = model_info['affinity']
    verb_weight = model_info.get('verb_weight', 'N/A')
    object_weight = model_info.get('object_weight', 'N/A')

    n_leaves = len(data)
    max_figsize_width = 30
    max_figsize_height = min(30, max(10, n_leaves * 0.35))

    fig, ax = plt.subplots(figsize=(max_figsize_width, max_figsize_height * 1.5))
    ax.set_title(
        f"{application_name} | {affinity} | Distance Threshold: {distance_threshold} "
        f"| Verb Weight: {verb_weight} | Object Weight: {object_weight}",
        fontsize=14
    )

    counts = np.zeros(model.children_.shape[0])
    cluster_contents = {i: [label] for i, label in enumerate(labels)}
    n_samples = len(model.labels_)
    palette = sns.color_palette("hsv", len(labels))
    cluster_colors = {}

    def get_color_for_cluster(cluster_idx):
        return palette[cluster_idx % len(palette)]

    for i, merge in enumerate(model.children_):
        merged_content = []
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1
                if child_idx not in cluster_colors:
                    cluster_colors[child_idx] = get_color_for_cluster(child_idx)
                merged_content.extend(cluster_contents[child_idx])
            else:
                current_count += counts[child_idx - n_samples]
                merged_content.extend(cluster_contents[child_idx])
        counts[i] = current_count
        new_cluster_idx = n_samples + i
        cluster_colors[new_cluster_idx] = get_color_for_cluster(new_cluster_idx)
        cluster_contents[new_cluster_idx] = merged_content

    linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)

    log_clusters_at_distance_threshold(linkage_matrix, distance_threshold)
    process_clusters_and_generate_dendrograms(linkage_matrix, labels, distance_threshold, application_name)

    dendrogram(
        linkage_matrix,
        labels=labels,
        color_threshold=color_threshold,
        leaf_font_size=10,
        orientation='right',
        distance_sort='descending',
        above_threshold_color='grey',
        ax=ax
    )
    ax.set_ylabel('Distance', fontsize=14)

    plt.tight_layout()
    plt.show()


def generate_dendogram_visualization(model_file):
    model_info = joblib.load(model_file)
    distance_threshold = 0.5
    clustering_model = model_info['model']
    labels = model_info['labels']

    if hasattr(clustering_model, 'children_'):
        save_directory = os.path.join(
            r"C:\Users\Max\NLP4RE\Dendogram-Generator\static\png",
            f"{model_info['affinity']}_{model_info['application_name']}_dt-{distance_threshold}".replace(" ", "_")
        )
        os.makedirs(save_directory, exist_ok=True)

        render_dendrogram(model_info,
                          clustering_model,
                          labels,
                          color_threshold=distance_threshold,
                          distance_threshold=distance_threshold)

        image_name = f"{model_info['affinity']}_{model_info['application_name']}_dt-{distance_threshold}_dendrogram.png".replace(" ", "_")
        dendrogram_save_path = os.path.join(save_directory, image_name)
        plt.savefig(dendrogram_save_path)
        plt.close()
        print(f"Dendrogram saved at: {dendrogram_save_path}")

    else:
        raise ValueError("The provided model is not AgglomerativeClustering.")


if __name__ == "__main__":
    pkls_directory = r"C:\Users\Max\NLP4RE\Dendogram-Generator\static\pkls"
    for filename in os.listdir(pkls_directory):
        if filename.endswith('.pkl'):
            model_file = os.path.join(pkls_directory, filename)
            print(f"Processing: {model_file}")
            generate_dendogram_visualization(model_file)
