#!/usr/bin/env python3
"""
Generates a simple, publication-quality illustration positioning this thesis
in relation to Nykänen's (2022) pre-trained U-Net.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import os
import textwrap

# Connector style aligned with generate_pipeline_diagram.py
ARROW_COLOR = '#78909C'
ARROW_LW = 1.5

output_dir = 'drafts'
os.makedirs(output_dir, exist_ok=True)
output_filename = os.path.join(output_dir, 'figure_nykanen_extension.png')

# 12cm x 8cm
fig, ax = plt.subplots(figsize=(12/2.54, 8/2.54), dpi=300)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

# Colors
bg_nykanen = '#E8EEF4'  # Cool grey-blue
bg_thesis = '#E8F5E9'   # Soft green for thesis contribution
edge_color = '#90A4AE'
text_color = '#1A1A1A'

def draw_shadow_box(ax, cx, cy, w, min_h, bg_color, text, subtext, wrap_width=44):
    wrapped_title = textwrap.fill(text, width=32)
    wrapped_subtext = textwrap.fill(subtext, width=wrap_width)

    # Adaptive height so wrapped lines always remain inside the box.
    title_lines = wrapped_title.count('\n') + 1
    subtext_lines = wrapped_subtext.count('\n') + 1
    content_h = 0.11 + (title_lines * 0.04) + (subtext_lines * 0.036)
    h = max(min_h, content_h)

    # Shadow
    ax.add_patch(FancyBboxPatch((cx - w/2 + 0.015, cy - h/2 - 0.015), w, h, 
                                boxstyle='round,pad=0.02', fc='black', ec='none', alpha=0.1, zorder=1))
    # Main Box
    ax.add_patch(FancyBboxPatch((cx - w/2, cy - h/2), w, h, 
                                boxstyle='round,pad=0.02', fc=bg_color, ec=edge_color, lw=1.5, zorder=2))
    
    # Text
    ax.text(cx, cy + h*0.18, wrapped_title, ha='center', va='center', fontsize=10, fontweight='bold', color=text_color, zorder=3)
    ax.text(cx, cy - h*0.12, wrapped_subtext, ha='center', va='center', fontsize=8.5, color='#37474F', zorder=3, linespacing=1.35)

# Top Box (Nykänen)
draw_shadow_box(ax, 0.5, 0.75, 0.85, 0.25, bg_nykanen, 
                "Nykänen (2022) Pre-Trained U-Net", 
                "Grayscale X-Ray → Probability Map (Threshold 0.5) → Binary Mask (IoU = 0.93)")

# Bottom Box (Thesis)
draw_shadow_box(ax, 0.5, 0.25, 0.85, 0.25, bg_thesis, 
                "Contributions of This Thesis", 
                "Mask Expansion  →  Anatomical Ablation  →  Classification")

# Connecting Arrow
ax.annotate(
    '',
    xy=(0.5, 0.43),
    xytext=(0.5, 0.57),
    arrowprops=dict(
        arrowstyle='-|>,head_width=0.35,head_length=0.5',
        color=ARROW_COLOR,
        lw=ARROW_LW,
        facecolor=ARROW_COLOR,
    ),
    zorder=2,
)
ax.text(0.535, 0.50, 'Fixed Upstream\nComponent', ha='left', va='center', fontsize=8, fontstyle='italic', color=ARROW_COLOR)

plt.tight_layout()
plt.savefig(output_filename, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Successfully generated: {output_filename}")