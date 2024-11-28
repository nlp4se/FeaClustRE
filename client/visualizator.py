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
            return {"id": int(node_id), "label": labels[node_id]}
        else:
            left_child = int(linkage_matrix[node_id - n_samples, 0])
            right_child = int(linkage_matrix[node_id - n_samples, 1])
            distance = float(linkage_matrix[node_id - n_samples, 2])

            return {
                "id": int(node_id),
                "distance": distance,
                "children": [traverse_node(left_child), traverse_node(right_child)],
            }

    return traverse_node(len(linkage_matrix) + n_samples - 1)


def save_json(data, file_path):
    with open(file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)
    print(f"JSON saved at: {file_path}")


def render_dendrogram_and_process_clusters(model_info, labels, color_threshold, original_data):
    application_name = model_info['application_name']
    affinity = model_info['affinity']
    verb_weight = model_info.get('verb_weight', 'N/A')
    object_weight = model_info.get('object_weight', 'N/A')
    static_folder = r"C:\Users\Max\NLP4RE\Dendogram-Generator\static\png"
    app_folder = os.path.join(
        static_folder,
        f"{affinity}_{application_name}_dt-{color_threshold}_vw-{verb_weight}_ow-{object_weight}".replace(" ", "_")
    )
    reset_folder(app_folder)
    os.makedirs(app_folder, exist_ok=True)

    # Ensure linkage_matrix is computed based on the correct data
    linkage_matrix = linkage(original_data, method='average', metric='euclidean')

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
        f"{application_name} | {affinity} | Distance Threshold: {color_threshold} "
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
            cluster_map[color] = {'labels': [], 'distances': [], 'linkage_data': []}
        label = labels[leaf]
        cluster_map[color]['labels'].append(label)

    # Save hierarchical JSON for the general dendrogram
    general_json = build_hierarchical_json(linkage_matrix, labels)
    general_json_path = os.path.join(app_folder, f"{application_name}_general_hierarchy.json")
    save_json(general_json, general_json_path)

    # Process and save the individual clusters
    process_and_save_clusters(cluster_map, application_name, app_folder, original_data, color_threshold)

    return cluster_map


def extract_sub_linkage_matrix_from_parent(original_data, cluster_indices):

    # Filter the original data points to include only the cluster indices
    sub_data = original_data[cluster_indices]

    # Recalculate the linkage matrix for the subset of data
    sub_linkage_matrix = linkage(sub_data, method='average', metric='euclidean')

    return sub_linkage_matrix


def process_and_save_clusters(cluster_map, application_name, app_folder, original_data, color_threshold):
    final_csv_data = []

    for cluster_id, (color, cluster_data) in enumerate(cluster_map.items(), start=1):
        cluster_labels = cluster_data['labels']
        cluster_indices = [i for i, label in enumerate(cluster_labels)]

        print(f"Processing Cluster {cluster_id} (Color: {color}): Labels = {cluster_labels}")

        # Generate a dynamic label for the cluster
        dynamic_label = generate_dynamic_label(cluster_labels)
        print(f"Generated label for Cluster {cluster_id}: {dynamic_label}")

        # Create cluster folder and file paths
        cluster_label = f"Cluster_{cluster_id}_{dynamic_label.replace(' ', '_')}"
        cluster_folder = os.path.join(app_folder, cluster_label)
        os.makedirs(cluster_folder, exist_ok=True)

        # Extract sub-linkage matrix directly from the parent
        sub_linkage_matrix = extract_sub_linkage_matrix_from_parent(original_data, cluster_indices)

        # Prepare labels for the sub-cluster dendrogram
        sub_labels = [cluster_labels[i] for i in range(len(cluster_labels))]

        if len(sub_labels) != sub_linkage_matrix.shape[0] + 1:
            raise ValueError(
                f"Inconsistent dimensions: {len(sub_labels)} labels for {sub_linkage_matrix.shape[0] + 1} leaves."
            )

        # Save the dendrogram as a horizontal plot
        plt.figure(figsize=(30, 5))
        dendrogram(
            sub_linkage_matrix,
            labels=sub_labels,
            leaf_rotation=0,  # Rotate leaves for horizontal alignment
            leaf_font_size=10,
            orientation='right',
            color_threshold=color_threshold
        )
        plt.title(f"Dendrogram for Cluster {cluster_id} ({color})")
        plt.xlabel("Distance")
        plt.ylabel("Cluster Labels")
        dendrogram_path = os.path.join(cluster_folder, f"{cluster_label}_dendrogram.png")
        plt.savefig(dendrogram_path)
        plt.close()
        print(f"Dendrogram for Cluster {cluster_id} saved at: {dendrogram_path}")

        # Save hierarchical JSON for the sub-cluster
        print(f"Saving hierarchical JSON for Cluster {cluster_id}...")
        sub_json = build_hierarchical_json(sub_linkage_matrix, sub_labels)
        sub_json_path = os.path.join(cluster_folder, f"{cluster_label}_hierarchy.json")
        save_json(sub_json, sub_json_path)
        print(f"Hierarchical JSON for Cluster {cluster_id} saved at: {sub_json_path}")

        # Append cluster data to final CSV summary
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
    distance_threshold = 5
    labels = model_info['labels']
    original_data = model_info['data_points']

    clusters = render_dendrogram_and_process_clusters(
        model_info,
        labels,
        distance_threshold,
        original_data
    )
    return clusters


if __name__ == "__main__":
    pkls_directory = r"C:\Users\Max\NLP4RE\Dendogram-Generator\static\pkls"
    for filename in os.listdir(pkls_directory):
        if filename.endswith('.pkl'):
            model_file = os.path.join(pkls_directory, filename)
            print(f"Processing: {model_file}")
            clusters = generate_dendrogram_visualization(model_file)
            print(f"Clusters for {filename}: {clusters}")
