import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.cluster.hierarchy import dendrogram
import joblib
from datetime import datetime
import matplotlib.animation as animation
import logging  # Add this import

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def add_line_breaks(labels):
    return [label.replace(' ', '\n') for label in labels]

def plot_dendrogram(ax, model, labels, **kwargs):
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
    dendrogram(linkage_matrix, ax=ax, labels=labels, **kwargs)
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=8)

def load_dendrograms(folder_path):
    dendrograms = []
    logger.info(f"Loading dendrograms from {folder_path}")
    for filename in os.listdir(folder_path):
        if filename.endswith('.pkl') and len(filename) == 20:  # 'YYYY-MM-DD_HH-MM.pkl'
            date_str = filename[:16]  # Extract 'YYYY-MM-DD_HH-MM'
            date = datetime.strptime(date_str, '%Y-%m-%d_%H-%M')
            file_path = os.path.join(folder_path, filename)
            logger.debug(f"Loading file: {file_path}")
            file = joblib.load(file_path)
            dendrograms.append((date, file))
    logger.info(f"Loaded {len(dendrograms)} dendrograms")
    return sorted(dendrograms, key=lambda x: x[0])

def update(val):
    idx = int(slider.val)
    logger.debug(f"Updating plot for index {idx}")
    ax.clear()
    date, file = dendrograms[idx]
    model = file['model']
    labels = file['labels']
    affinity = file['affinity']
    verb_weight = file.get('verb_weight', 'N/A')
    object_weight = file.get('object_weight', 'N/A')
    
    plot_dendrogram(ax, model, labels)
    ax.set_title(f"{date.strftime('%Y-%m-%d')} | {affinity} | Verb Weight: {verb_weight} | Object weight: {object_weight}")
    ax.set_xlabel('Features', fontsize=12)
    ax.set_ylabel('Distance', fontsize=12)
    fig.canvas.draw_idle()

def animate(frame):
    slider.set_val(frame)
    return []


parser = argparse.ArgumentParser(description='Visualize dendrograms dynamically.')
parser.add_argument('-i', '--input', type=str, required=True, help='Input folder containing dendrogram pkl files')
args = parser.parse_args()

folder_path = args.input
logger.info(f"Starting visualization for folder: {folder_path}")
dendrograms = load_dendrograms(folder_path)

fig, ax = plt.subplots(figsize=(15, 8))
plt.subplots_adjust(bottom=0.2)

slider_ax = plt.axes([0.1, 0.05, 0.8, 0.03])
slider = Slider(slider_ax, 'Date', 0, len(dendrograms) - 1, valinit=0, valstep=1)

slider.on_changed(update)

update(0)  # Plot initial dendrogram

# Create animation
anim = animation.FuncAnimation(fig, animate, frames=len(dendrograms), interval=500, blit=True, repeat=False)

plt.tight_layout()
plt.show()

# Uncomment the following lines to save the animation as a gif
# logger.info("Saving animation as gif")
# anim.save('dendrogram_timeline.gif', writer='pillow', fps=2)
# logger.info("Animation saved as dendrogram_timeline.gif")

logger.info("Visualization complete")