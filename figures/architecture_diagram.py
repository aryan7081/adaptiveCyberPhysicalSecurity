"""
Generate architecture diagram for the report.
Run: python figures/architecture_diagram.py
Output: figures/architecture_diagram.pdf
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlim(0, 12)
ax.set_ylim(0, 8)
ax.axis("off")

# Colors
c_data = "#E8F4F8"
c_mae = "#D4EDDA"
c_ocsvm = "#FFF3CD"
c_output = "#F8D7DA"

def box(ax, xy, w, h, label, color):
    r = FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.05", facecolor=color, edgecolor="black", linewidth=1)
    ax.add_patch(r)
    ax.text(xy[0] + w/2, xy[1] + h/2, label, ha="center", va="center", fontsize=9, wrap=True)

def arrow(ax, start, end, label=""):
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(arrowstyle="->", color="black", lw=2))
    if label:
        mid = ((start[0]+end[0])/2, (start[1]+end[1])/2)
        ax.text(mid[0], mid[1], label, fontsize=7, ha="center")

# Data flow
box(ax, (1, 6), 2, 1.2, "Raw NSL-KDD\n(Benign)", c_data)
box(ax, (1, 4), 2, 1.2, "Preprocessing\n+ Features", c_data)
box(ax, (4, 5), 2.5, 1.8, "MAE Encoder\n(BERT-style mask)", c_mae)
box(ax, (4, 2.5), 2.5, 1.5, "Frozen Encoder\nEmbeddings", c_mae)
box(ax, (7.5, 4), 2.2, 1.5, "One-Class SVM\n(Decision Boundary)", c_ocsvm)
box(ax, (10, 4), 1.5, 1.2, "Anomaly\nScore", c_output)

arrow(ax, (3, 6.6), (4, 5.9), "Preprocess")
arrow(ax, (3, 4.6), (4, 5.1))
arrow(ax, (6.5, 4.75), (7.5, 4.75), "Embed")
arrow(ax, (6.5, 3.25), (7.5, 4.25))
arrow(ax, (9.7, 4.75), (10, 4.6), "Predict")

ax.text(6, 1, "Hybrid: DL representations + Probabilistic boundary", fontsize=10, ha="center")
plt.title("Adaptive CPS: MAE + One-Class SVM Architecture", fontsize=12)
plt.tight_layout()
import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
plt.savefig("figures/architecture_diagram.pdf", bbox_inches="tight")
print("Saved figures/architecture_diagram.pdf")
plt.close()
