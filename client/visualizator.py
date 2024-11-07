import joblib
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
import seaborn as sns
from openai import OpenAI
from dotenv import load_dotenv
import pandas as pd

load_dotenv()
api_key = os.getenv("OPENAI_KEY")
client = OpenAI(api_key=api_key)

ABOVE_THRESHOLD_COLOR = '#D3D3D3'
N_PALETTE_COLORS = 50
PALETTE_NAME = 'husl'

def generate_dynamic_label(cluster_labels):
    input_text = ("In maximum one sentence, provide a unique concise abstract label for these actions "
                  "(do not repeat in the label the actions): ") + ", ".join(cluster_labels)

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an assistant that provides concise abstract labels."},
            {"role": "user", "content": input_text}
        ],
        max_tokens=20,
        temperature=0.5
    )

    label = response.choices[0].message
    return label.content


def render_dendrogram(model_info, model, labels, color_threshold, distance_threshold):
    data = model_info['data_points']
    application_name = model_info['application_name']
    affinity = model_info['affinity']
    verb_weight = model_info.get('verb_weight', 'N/A')
    object_weight = model_info.get('object_weight', 'N/A')

    n_leaves = len(data)
    max_figsize_width = 30
    max_figsize_height = min(30, max(10, n_leaves * 0.35))

    fig = plt.figure(figsize=(max_figsize_width, max_figsize_height * 1.5))
    ax = fig.add_subplot()
    ax.set_title(
        f"{application_name} | {affinity} | Distance Threshold: {distance_threshold} "
        f"| Verb Weight: {verb_weight} | Object Weight: {object_weight}",
        fontsize=14
    )

    counts = np.zeros(model.children_.shape[0])
    cluster_contents = {i: [label] for i, label in enumerate(labels)}
    n_samples = len(model.labels_)
    palette = sns.color_palette("hsv", len(labels))  # Example palette
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

    dendrogram_data = dendrogram(
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

    label_counter = 1
    max_labels = 5
    labeled_clusters = set()
    clusters_to_save = []

    for i, (icoord, dcoord) in enumerate(zip(dendrogram_data['icoord'], dendrogram_data['dcoord'])):
        # Calculate the midpoint for the x-coordinate and use the correct y-coordinate
        x = 0.5 * sum(icoord[1:3]) * 0.1  # Test scaling x down by a factor of 10
        y = dcoord[1] * 0.5

        # Limit the number of labels to prevent overcrowding
        if label_counter > max_labels:
            break

        cluster_idx = n_samples + i
        if cluster_idx in cluster_contents and cluster_idx not in labeled_clusters:
            # Retrieve cluster labels for current cluster
            cluster_labels = cluster_contents[cluster_idx]
            labeled_clusters.add(cluster_idx)
            # Generate a dynamic label for the cluster (assuming generate_dynamic_label function exists)
            label = generate_dynamic_label(cluster_labels)
            print(f"Label {label_counter}: '{label}' at (x={x}, y={y})")
            # Save cluster information to a list for CSV export (if needed)
            clusters_to_save.append({"cluster_name": label, "feature_list": cluster_labels})

            ax.plot(x, y, 'ro', markersize=8)  # Increased marker size for better visibility
            ax.annotate(label, (x, y), xytext=(0, -8),
                        textcoords='offset points',
                        fontsize=10,  # Larger font size for better visibility
                        va='top', ha='center')
            label_counter += 1

    # Save clusters to CSV
    save_clusters_to_csv(clusters_to_save)
    plt.tight_layout()
    ax.set_xlim(0, 100)  # Adjust the limits to fit within the visible range
    ax.set_ylim(0, 100)
    plt.show()


def save_clusters_to_csv(clusters):
    """
    Save cluster names and feature lists to a CSV file.
    """
    df = pd.DataFrame(clusters)
    df.to_csv("clusters.csv", index=False, sep=',')
    print("Cluster data saved to clusters.csv")


def generate_dendogram_visualization(model_file):
    model_info = joblib.load(model_file)
    distance_threshold = model_info['distance_threshold']
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
