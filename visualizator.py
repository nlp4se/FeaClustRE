import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
import joblib


def plot_dendrogram(model, **kwargs):
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

    dendrogram(linkage_matrix, **kwargs)


def show_dendrogram(model_file):
    model = joblib.load(model_file)

    if hasattr(model, 'children_'):
        plt.figure(figsize=(10, 6))
        plot_dendrogram(model)
        # TODO dynamic name
        plt.title('TF-IDF Cosine Distance Histogram')
        plt.xlabel('Features')
        plt.ylabel('Distance')
        plt.show()
    else:
        raise ValueError("The provided model is not AgglomerativeClustering.")


if __name__ == "__main__":
    model_file = 'static/tf_idf_cosine_agglomerative_model.pkl'
    show_dendrogram(model_file)
