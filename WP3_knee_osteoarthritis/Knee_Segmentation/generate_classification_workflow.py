#!/usr/bin/env python3
"""
Generates a high-level classification workflow diagram focusing on 
pre-processing, augmentation, and network backbones, avoiding library-specific jargon.
Optimized for Research Publication Best Practices (Exact bounding boxes, 
drop shadows, orthogonal routing, zero overlaps).
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import os

# ─── Configuration ────────────────────────────────────────────────────
FIG_W_CM, FIG_H_CM = 22, 7
DPI = 300
FIG_W = FIG_W_CM / 2.54
FIG_H = FIG_H_CM / 2.54

C_BG      = '#FFFFFF'
C_TEXT    = '#1A1A1A'
C_SCRIPT  = '#455A64'
C_ARROW   = '#78909C'

FONT = 'sans-serif'
MONO = 'monospace'

# Modern, muted academic palette (Colorblind-friendly & print-safe)
colors = {'data': '#F0F4F8', 'prep': '#FDF6E3', 'aug': '#FFF8E1', 'cnn': '#E3F2FD', 'out': '#F4F0F6'}
borders = {'data': '#CFD8DC', 'prep': '#E6C27A', 'aug': '#FFCC80', 'cnn': '#90CAF9', 'out': '#CE93D8'}

output_dir = 'drafts'
os.makedirs(output_dir, exist_ok=True)
output_filename = os.path.join(output_dir, 'figure_classification_workflow.png')

# ─── Helpers ──────────────────────────────────────────────────────────
def box(ax, cx, cy, w, h, title, subtext=None, tc=C_TEXT, ec='#90A4AE', bg='#FFFFFF', lw=1.5):
    """Draws a crisp, styled conceptual box with an exact width/height and drop shadow."""
    pad = 0.02
    iw = w - 2 * pad
    ih = h - 2 * pad
    x0 = cx - iw / 2
    y0 = cy - ih / 2
    
    # Drop Shadow
    ps = FancyBboxPatch((x0 + 0.005, y0 - 0.012), iw, ih, 
                        boxstyle=f'round,pad={pad},rounding_size=0.03', 
                        fc='#000000', ec='none', alpha=0.06, zorder=2)
    ax.add_patch(ps)
    
    # Main Box
    p = FancyBboxPatch((x0, y0), iw, ih, boxstyle=f'round,pad={pad},rounding_size=0.03', 
                       fc=bg, ec=ec, lw=lw, zorder=3)
    ax.add_patch(p)
    
    # Text Placement
    if subtext:
        ax.text(cx, cy + h*0.2, title, ha='center', va='center', 
                fontsize=9.5, fontweight='bold', family=FONT, color=tc, zorder=4)
        ax.text(cx, cy - h*0.1, subtext, ha='center', va='center', 
                fontsize=8, family=MONO, color=C_SCRIPT, zorder=4, linespacing=1.6)
    else:
        ax.text(cx, cy, title, ha='center', va='center', 
                fontsize=9.5, fontweight='bold', family=FONT, color=tc, zorder=4)

def arr(ax, x1, y1, x2, y2, lw=1.5):
    """Clean directional arrow from strictly defined points."""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='-|>,head_width=0.35,head_length=0.5', 
                                color=C_ARROW, lw=lw, facecolor=C_ARROW), zorder=2)

# ─── Build Figure ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=DPI)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')
fig.patch.set_facecolor(C_BG)

# 5 Blocks
w = 0.17
h = 0.6
cy = 0.5
cx = [0.1, 0.3, 0.5, 0.7, 0.9]

box(ax, cx[0], cy, w, h, "Input", "Ablated\nGrayscale\nX-Ray", bg=colors['data'], ec=borders['data'])
box(ax, cx[1], cy, w, h, "Pre-Processing", "Resize 224x224\nNormalize\nTo Tensor", bg=colors['prep'], ec=borders['prep'])
box(ax, cx[2], cy, w, h, "Augmentation", "Horiz. Flip\nRotate ±5°\nTranslate ±5%", bg=colors['aug'], ec=borders['aug'])
box(ax, cx[3], cy, w, h, "CNN Backbone", "ResNet-18 OR\nEfficientNet-B0 OR\nSwin-Tiny", bg=colors['cnn'], ec=borders['cnn'])
box(ax, cx[4], cy, w, h, "Classification", "Softmax\nProbability\n(KL0 vs KL2)", bg=colors['out'], ec=borders['out'])

# Arrows
for i in range(4):
    arr(ax, cx[i] + w/2 + 0.002, cy, cx[i+1] - w/2 - 0.002, cy)

plt.tight_layout()
plt.savefig(output_filename, bbox_inches='tight', facecolor=C_BG, edgecolor='none')
plt.close(fig)
print(f"Successfully generated: {output_filename}")