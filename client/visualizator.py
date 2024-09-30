import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
import joblib
import argparse
import os

CLUSTER_COLOR_THRESHOLD = 0.08


def add_line_breaks(labels):
    return [label.replace(' ', '\n') for label in labels]


def plot_dendrogram(model, labels, **kwargs):
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)
    labels = add_line_breaks(labels=labels)

    dendrogram(linkage_matrix,
               labels=labels,
               color_threshold=CLUSTER_COLOR_THRESHOLD,
               leaf_font_size=10,
               **kwargs)

    plt.xticks(rotation=90, fontsize=10, ha='right')


def show_dendrogram(model_file):
    file = joblib.load(model_file)
    application_name = file['application_name']
    distance_threshold = file['distance_threshold']
    model = file['model']
    affinity = file['affinity']
    labels = file['labels']

    try:
        verb_weight = file['verb_weight']
    except KeyError:
        verb_weight = 'N/A'

    try:
        object_weight = file['object_weight']
    except KeyError:
        object_weight = 'N/A'

    if hasattr(model, 'children_'):
        n_leaves = len(labels)

        max_figsize_width = 30
        max_figsize_height = 15
        figsize_width = min(max_figsize_width, n_leaves * 0.5)
        figsize_height = max(12, min(max_figsize_height, n_leaves * 0.25))

        plt.figure(figsize=(figsize_width, figsize_height))

        plot_dendrogram(model, labels=labels)

        plt.title(application_name
                  + ' | ' + affinity
                  + ' | Distance Threshold: ' + str(distance_threshold)
                  + ' | Verb Weight: ' + str(verb_weight)
                  + ' | Object weight: ' + str(object_weight),
                  fontsize=14)
        plt.xlabel('Features', fontsize=14)
        plt.ylabel('Distance', fontsize=14)
        plt.subplots_adjust(bottom=0.2)
        plt.tight_layout()

        save_directory = r"C:\Users\Max\NLP4RE\Dendogram-Generator\static\png"
        base_name = os.path.splitext(os.path.basename(model_file))[0]
        save_path = os.path.join(save_directory, f"{base_name}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Dendrogram saved at: {save_path}")
    else:
        raise ValueError("The provided model is not AgglomerativeClustering.")

if __name__ == "__main__":
    pkls_directory = r"C:\Users\Max\NLP4RE\Dendogram-Generator\static\pkls"

    for filename in os.listdir(pkls_directory):
        if filename.endswith('.pkl'):
            model_file = os.path.join(pkls_directory, filename)
            print(f"Processing: {model_file}")
            show_dendrogram(model_file)