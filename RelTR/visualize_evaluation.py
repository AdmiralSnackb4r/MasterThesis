import json
import time
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler

# Set a global prettier style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    "axes.prop_cycle": cycler('color', [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]),
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "legend.fontsize": 12,
    "legend.title_fontsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12
})

# Plotting function
def plot_pr_curve(precisions, recalls, category='Merged Set', label=None, ax=None):
    """Plot a precision-recall curve with improved aesthetics."""
    if ax is None:
        plt.figure(figsize=(10, 8))
        ax = plt.gca()

    ax.plot(
        recalls, precisions,
        label=label,
        linewidth=2,
        marker='o',
        markersize=5,
        alpha=0.85
    )

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f'Precision-Recall Curve for {category}', pad=15)
    ax.set_xlim([0.0, 1.05])
    ax.set_ylim([0.0, 1.05])
    ax.grid(True, linestyle='--', alpha=0.6)

    return ax

# Main
ax = None
avg_precs = []
iou_thrs = []

# Start timing (optional)
start_time = time.time()

# Load JSON
with open('merged_set_eval.json', 'r') as file:
    data = json.load(file)

# Loop over thresholds
for idx, key in enumerate(data.keys()):
    avg_precs.append(data[key]['avg_prec'])
    iou_thrs.append(key)

    precisions = data[key]['precisions']
    recalls = data[key]['recalls']

    ax = plot_pr_curve(
        precisions,
        recalls,
        label=f'{float(key):.2f}',
        ax=ax
    )

# Format results
avg_precs = [round(float(ap), 4) for ap in avg_precs]
iou_thrs = [round(float(thr), 4) for thr in iou_thrs]

# Print results
print(f'\nmAP: {100 * np.mean(avg_precs):.2f}')
print('Avg Precs:', avg_precs)
print('IoU Thresholds:', iou_thrs)

# Final touches
plt.legend(loc='lower left', title='IoU Thr', frameon=True)
for xval in np.linspace(0.0, 1.0, 11):
    plt.vlines(xval, 0.0, 1.1, color='gray', alpha=0.3, linestyles='dashed')

plt.tight_layout()
plt.savefig('merged_set_eval.png', dpi=600, bbox_inches='tight')
plt.show()

# End timing (optional)
end_time = time.time()
print(f"\nPlotting and mAP calculation took {end_time - start_time:.4f} seconds.")
