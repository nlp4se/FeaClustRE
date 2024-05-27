import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
import joblib


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
               **kwargs,
               show_leaf_counts=True,
               orientation='bottom',
               truncate_mode='level',
               p=10)
    plt.xticks(rotation=90, fontsize=10)


def show_dendrogram(model_file):
    file = joblib.load(model_file)
    model = file['model']
    affinity = file['affinity']
    labels = file['labels']
    if hasattr(model, 'children_'):
        plt.figure(figsize=(10, 6))
        plot_dendrogram(model, labels=labels)
        # TODO dynamic name
        plt.title(affinity)
        plt.xlabel('Features')
        plt.ylabel('Distance')
        plt.show()
    else:
        raise ValueError("The provided model is not AgglomerativeClustering.")


if __name__ == "__main__":
    model_file = 'static/tf_idf_cosine_agglomerative_model.pkl'
    show_dendrogram(model_file)
