import joblib
import os
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import shutil
import pandas as pd
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import matplotlib.colors as mcolors
import numpy as np

# Define base directories
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
STAGE_3_INPUT_PATH = os.path.join(BASE_DIR, 'data', 'Stage 3 - Topic Modelling', 'input')
STAGE_3_OUTPUT_PATH = os.path.join(BASE_DIR, 'data', 'Stage 3 - Topic Modelling', 'output')

os.makedirs(STAGE_3_INPUT_PATH, exist_ok=True)
os.makedirs(STAGE_3_OUTPUT_PATH, exist_ok=True)

# Load model and tokenizer
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
    few_shot_input_text = (
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
    response = pipe(few_shot_input_text, max_new_tokens=10, do_sample=True)
    label = response[0]['generated_text'].replace(few_shot_input_text, "").strip()
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


def extract_sub_linkage_matrix_from_parent(original_data, cluster_indices):
    sub_data = original_data[cluster_indices]
    sub_linkage_matrix = linkage(sub_data, method='average', metric='euclidean')
    return sub_linkage_matrix


def generate_infinite_colors(num_colors):
    hues = np.linspace(0, 1, num_colors, endpoint=False)
    saturation = 0.9
    value = 0.9
    colors = [mcolors.hsv_to_rgb((hue, saturation, value)) for hue in hues]
    return list(set([mcolors.rgb2hex(color) for color in colors]))


def reassign_dendrogram_colors(dendrogram_result, num_colors):
    infinite_colors = generate_infinite_colors(num_colors)
    new_color_list = []
    new_leaves_color_list = []

    current_color = None
    current_cluster_id = 0
    cluster_color_map = {}

    for i, color in enumerate(dendrogram_result['color_list']):
        if color == 'grey':
            new_color_list.append('grey')
            continue

        if color != current_color:
            cluster_color_map = {}
            current_color = color
            cluster_color_map[color] = infinite_colors[current_cluster_id]
            current_cluster_id += 1
            new_color_list.append(cluster_color_map[current_color])
        else:
            new_color_list.append(cluster_color_map[current_color])

    current_color = None
    cluster_color_map = {}
    current_cluster_id = 0
    for color in dendrogram_result['leaves_color_list']:
        if color == 'grey':
            new_leaves_color_list.append('grey')
            continue

        if color != current_color:
            cluster_color_map = {}
            current_color = color
            cluster_color_map[color] = infinite_colors[current_cluster_id]
            current_cluster_id += 1

        new_leaves_color_list.append(cluster_color_map[current_color])

    dendrogram_result['color_list'] = new_color_list
    dendrogram_result['leaves_color_list'] = new_leaves_color_list

    return dendrogram_result


def process_and_save_clusters(cluster_map, application_name, app_folder, original_data, color_threshold):
    final_csv_data = []

    for cluster_id, (color, cluster_data) in enumerate(cluster_map.items(), start=1):
        cluster_labels = cluster_data['labels']
        cluster_indices = cluster_data['indices']

        print(f"Processing Cluster {cluster_id} (Color: {color}): Labels = {cluster_labels}")
        print(f"Detected {len(cluster_labels)} labels in Cluster {cluster_id}.")

        # Generate a dynamic label for the cluster
        dynamic_label = generate_dynamic_label(cluster_labels)
        print(f"Generated label for Cluster {cluster_id}: {dynamic_label}")

        # Sanitize folder name
        sanitized_label = dynamic_label.replace(" ", "_").replace("/", "_").replace("\\", "_")
        cluster_label = f"Cluster_{cluster_id}_{sanitized_label}"
        cluster_folder = os.path.join(app_folder, cluster_label)

        # Ensure the cluster folder exists
        os.makedirs(cluster_folder, exist_ok=True)

        # Process sub-dendrogram
        sub_linkage_matrix = extract_sub_linkage_matrix_from_parent(original_data, cluster_indices)
        sub_labels = [cluster_labels[i] for i in range(len(cluster_labels))]

        plt.figure(figsize=(30, 10))
        dendrogram(
            sub_linkage_matrix,
            labels=sub_labels,
            leaf_font_size=15,
            orientation='right',
            color_threshold=color_threshold
        )
        plt.title(f"Dendrogram for Cluster {cluster_id} ({color}) | label: {dynamic_label}")
        plt.xlabel("Distance")
        plt.ylabel("Cluster Labels")

        # Save dendrogram image
        sub_dendrogram_path = os.path.join(cluster_folder, f"{cluster_label}_dendrogram.png")
        plt.savefig(sub_dendrogram_path)
        plt.close()

        # Save cluster details to CSV
        cluster_csv_path = os.path.join(cluster_folder, f"{cluster_label}_features.csv")
        cluster_df = pd.DataFrame([{
            "Cluster Name": dynamic_label,
            "Feature List": ", ".join(cluster_labels)
        }])
        cluster_df.to_csv(cluster_csv_path, index=False)

        # Save hierarchical JSON
        sub_json = build_hierarchical_json(sub_linkage_matrix, sub_labels)
        sub_json_path = os.path.join(cluster_folder, f"{cluster_label}_hierarchy.json")
        save_json(sub_json, sub_json_path)

        # Append cluster summary for the final CSV
        final_csv_data.append({
            "Cluster ID": cluster_id,
            "Cluster Name": dynamic_label,
            "Feature List": ", ".join(cluster_labels)
        })

    # Save final summary CSV
    final_csv_path = os.path.join(app_folder, f"{application_name}_clusters_summary.csv")
    final_csv_df = pd.DataFrame(final_csv_data)
    final_csv_df.to_csv(final_csv_path, index=False)
    print(f"Final summary CSV saved at: {final_csv_path}")


def generate_dendrogram_visualization(dendogram_file):
    model_info = joblib.load(dendogram_file)
    distance_threshold = model_info['distance_threshold'] * 10
    labels = model_info['labels']
    original_data = model_info['data_points']

    application_name = model_info['application_name']
    affinity = model_info['affinity']
    verb_weight = model_info.get('verb_weight', 'N/A')
    object_weight = model_info.get('object_weight', 'N/A')

    app_folder = os.path.join(
        STAGE_3_OUTPUT_PATH,
        f"{affinity}_{application_name}_dt-{distance_threshold}_vw-{verb_weight}_ow-{object_weight}".replace(" ", "_")
    )
    reset_folder(app_folder)

    linkage_matrix = linkage(original_data, method='average', metric='euclidean')

    fig, ax = plt.subplots(figsize=(30, 30))
    dendrogram_result = dendrogram(
        linkage_matrix,
        labels=labels,
        color_threshold=distance_threshold,
        leaf_font_size=10,
        orientation='right',
        distance_sort='descending',
        above_threshold_color='grey',
        ax=ax,
        no_plot=True
    )

    reassign_dendrogram_colors(dendrogram_result, len(labels))
    ax.axvline(x=distance_threshold, color='red', linestyle='--', linewidth=2)
    plt.tight_layout()

    final_dendrogram_path = os.path.join(app_folder, f"{application_name}_final_dendrogram.png")
    plt.savefig(final_dendrogram_path)
    plt.close(fig)

    cluster_map = {}
    for leaf, color in zip(dendrogram_result['leaves'], dendrogram_result['leaves_color_list']):
        if color == 'grey':
            continue
        if color not in cluster_map:
            cluster_map[color] = {'labels': [], 'indices': []}
        cluster_map[color]['labels'].append(labels[leaf])
        cluster_map[color]['indices'].append(leaf)

    process_and_save_clusters(cluster_map, application_name, app_folder, original_data, distance_threshold)

    general_json = build_hierarchical_json(linkage_matrix, labels)
    general_json_path = os.path.join(app_folder, f"{application_name}_general_hierarchy.json")
    save_json(general_json, general_json_path)

    return {
        "dendrogram_path": final_dendrogram_path,
        "json_path": general_json_path,
        "clusters_summary_csv": os.path.join(app_folder, f"{application_name}_clusters_summary.csv"),
    }
