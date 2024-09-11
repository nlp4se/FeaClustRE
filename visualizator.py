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
                      ** kwargs)
    '''
    for i, d, c in zip(linkage_matrix[:, 0], linkage_matrix[:, 1], linkage_matrix[:, 2]):
        x = 0
        y = c
        plt.annotate('%.2f' % c, (x, y), xytext=(550, 0),
                     textcoords='offset points', va='top', ha='center')
    '''
    plt.xticks(rotation=90, fontsize=10)


def show_dendrogram(model_file):
    file = joblib.load(model_file)
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
        plt.figure(figsize=(15, 8))
        plot_dendrogram(model, labels=labels)
        plt.title(affinity + ' | Verb Weight: ' + str(verb_weight) + ' | Object weight: ' + str(object_weight))
        plt.xlabel('Features', fontsize=12)
        plt.ylabel('Distance', fontsize=12)
        plt.xticks(rotation=90, fontsize=8)
        plt.tight_layout()
        plt.show()
    else:
        raise ValueError("The provided model is not AgglomerativeClustering.")


if __name__ == "__main__":
    model_file = 'static/pkls/viber.voip_bert_cosine_complete.pkl'
    #model_file = 'static/bert_cosine_complete.pkl'
    #model_file = 'static/paraphrase_minilm_average_cosine.pkl'
    show_dendrogram(model_file)
