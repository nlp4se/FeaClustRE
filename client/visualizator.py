import joblib
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import shutil
import pandas as pd
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

model_name = "meta-llama/Llama-3.2-3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(
    'cuda' if torch.cuda.is_available() else 'cpu'
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1
)


def reset_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path, exist_ok=True)


def generate_dynamic_label(cluster_labels):
    unique_labels = list(set(cluster_labels))
    input_text = (
        "Generate a single concise label summarizing the following actions:\n\n"
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


def build_hierarchical_json(linkage_matrix, labels):
    n_samples = len(labels)

    def traverse_node(node_id):
        if node_id < n_samples:
            # Leaf node
            return {"id": int(node_id), "label": labels[node_id]}
        else:
            # Internal node
            left_child = int(linkage_matrix[node_id - n_samples, 0])
            right_child = int(linkage_matrix[node_id - n_samples, 1])
            distance = float(linkage_matrix[node_id - n_samples, 2])

            return {
                "id": int(node_id),
                "distance": distance,
                "children": [traverse_node(left_child), traverse_node(right_child)],
            }

    # Start from the root node
    return traverse_node(len(linkage_matrix) + n_samples - 1)


def build_hierarchical_json_for_subcluster(linkage_matrix, cluster_labels):
    """
    Build a hierarchical JSON structure for a sub-cluster.

    Args:
        linkage_matrix (ndarray): Linkage matrix for the sub-cluster.
        cluster_labels (list): List of labels for the sub-cluster's leaf nodes.

    Returns:
        dict: Hierarchical JSON structure.
    """
    n_samples = len(cluster_labels)

    def traverse_node(node_id):
        if node_id < n_samples:
            # Leaf node
            return {"id": int(node_id), "label": cluster_labels[node_id]}
        else:
            # Internal node
            left_child = int(linkage_matrix[node_id - n_samples, 0])
            right_child = int(linkage_matrix[node_id - n_samples, 1])
            distance = float(linkage_matrix[node_id - n_samples, 2])

            return {
                "id": int(node_id),
                "distance": distance,
                "children": [traverse_node(left_child), traverse_node(right_child)],
            }

    # Start from the root node
    return traverse_node(len(linkage_matrix) + n_samples - 1)


def save_json(data, file_path):
    with open(file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)
    print(f"JSON saved at: {file_path}")


def render_dendrogram_and_process_clusters(model_info, model, labels, color_threshold, distance_threshold):
    application_name = model_info['application_name']
    affinity = model_info['affinity']
    verb_weight = model_info.get('verb_weight', 'N/A')
    object_weight = model_info.get('object_weight', 'N/A')
    static_folder = r"C:\Users\Max\NLP4RE\Dendogram-Generator\static\png"
    app_folder = os.path.join(
        static_folder,
        f"{affinity}_{application_name}_dt-{distance_threshold}_vw-{verb_weight}_ow-{object_weight}".replace(" ", "_")
    )
    reset_folder(app_folder)
    os.makedirs(app_folder, exist_ok=True)

    # Linkage matrix for the hierarchical clustering
    linkage_matrix = np.column_stack([model.children_, model.distances_, np.zeros(len(model.children_))]).astype(float)

    # Generate and save the general dendrogram
    fig, ax = plt.subplots(figsize=(30, 30))
    dendrogram_result = dendrogram(
        linkage_matrix,
        labels=labels,
        color_threshold=color_threshold,
        leaf_font_size=10,
        orientation='right',
        distance_sort='descending',
        above_threshold_color='grey',
        ax=ax
    )
    ax.set_title(
        f"{application_name} | {affinity} | Distance Threshold: {distance_threshold} "
        f"| Verb Weight: {verb_weight} | Object Weight: {object_weight}",
        fontsize=14
    )
    plt.tight_layout()
    final_dendrogram_path = os.path.join(app_folder, f"{application_name}_final_dendrogram.png")
    plt.savefig(final_dendrogram_path)
    plt.close(fig)
    print(f"Final dendrogram saved at: {final_dendrogram_path}")

    # Extract clusters based on colors
    cluster_map = {}
    for leaf, color in zip(dendrogram_result['leaves'], dendrogram_result['leaves_color_list']):
        if color == 'grey':
            continue  # Skip grey clusters
        if color not in cluster_map:
            cluster_map[color] = []
        cluster_map[color].append(labels[leaf])

    # Save hierarchical JSON for the general dendrogram
    general_json = build_hierarchical_json(linkage_matrix, labels)
    general_json_path = os.path.join(app_folder, f"{application_name}_general_hierarchy.json")
    save_json(general_json, general_json_path)

    # Process and save the individual clusters
    process_and_save_clusters(cluster_map, application_name, app_folder)

    return cluster_map


def process_and_save_clusters(cluster_map, application_name, app_folder):
    final_csv_data = []
    for cluster_id, (color, cluster_labels) in enumerate(cluster_map.items(), start=1):
        print(f"Processing Cluster {cluster_id} (Color: {color}): Labels = {cluster_labels}")

        dynamic_label = generate_dynamic_label(cluster_labels)
        print(f"Generated label for Cluster {cluster_id}: {dynamic_label}")

        cluster_label = f"Cluster_{cluster_id}_{dynamic_label.replace(' ', '_')}"
        cluster_folder = os.path.join(app_folder, cluster_label)
        os.makedirs(cluster_folder, exist_ok=True)

        cluster_data = {"Cluster Name": [dynamic_label], "Feature List": [cluster_labels]}
        cluster_df = pd.DataFrame(cluster_data)
        cluster_csv_path = os.path.join(cluster_folder, f"{cluster_label}.csv")
        cluster_df.to_csv(cluster_csv_path, index=False, sep=',')
        print(f"Cluster {cluster_id} saved to {cluster_csv_path}")

        # Generate and save hierarchical JSON for the sub-cluster
        filtered_data = np.random.rand(len(cluster_labels), 2)  # Generate dummy data for sub-cluster
        sub_linkage_matrix = linkage(filtered_data, method='ward')
        sub_json = build_hierarchical_json_for_subcluster(sub_linkage_matrix, cluster_labels)
        sub_json_path = os.path.join(cluster_folder, f"{cluster_label}_hierarchy.json")
        save_json(sub_json, sub_json_path)

        # Append to final CSV summary data
        final_csv_data.append({
            "Cluster ID": cluster_id,
            "Cluster Name": dynamic_label,
            "Feature List": cluster_labels
        })

    # Save final summary CSV
    final_csv_path = os.path.join(app_folder, f"{application_name}_clusters_summary.csv")
    final_csv_df = pd.DataFrame(final_csv_data)
    final_csv_df.to_csv(final_csv_path, index=False, sep=',')
    print(f"Final summary CSV saved at: {final_csv_path}")


def generate_dendrogram_visualization(model_file):
    model_info = joblib.load(model_file)
    distance_threshold = model_info['distance_threshold']
    clustering_model = model_info['model']
    labels = model_info['labels']

    if hasattr(clustering_model, 'children_'):
        clusters = render_dendrogram_and_process_clusters(
            model_info,
            clustering_model,
            labels,
            color_threshold=distance_threshold,
            distance_threshold=distance_threshold
        )
        return clusters
    else:
        raise ValueError("The provided model is not AgglomerativeClustering.")


if __name__ == "__main__":
    pkls_directory = r"C:\Users\Max\NLP4RE\Dendogram-Generator\static\pkls"
    for filename in os.listdir(pkls_directory):
        if filename.endswith('.pkl'):
            model_file = os.path.join(pkls_directory, filename)
            print(f"Processing: {model_file}")
            clusters = generate_dendrogram_visualization(model_file)
            print(f"Clusters for {filename}: {clusters}")
