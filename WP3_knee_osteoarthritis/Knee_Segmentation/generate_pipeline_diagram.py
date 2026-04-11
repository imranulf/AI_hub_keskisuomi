#!/usr/bin/env python3
"""
Figure 3.1 — Segmentation-Guided Feature Ablation Pipeline (Final)

Design principles (IEEE / academic publication best practices):
  • Grid-based layout: every element snaps to a y-row; no freehand placement.
  • All text ≥ 7 pt (IEEE minimum 8 pt for body, 6 pt absolute floor).
  • Arrows terminate at box EDGES, never inside.
  • Phase containers have ≥ 0.020 internal padding above first child.
  • No text overflows box boundaries — long names are abbreviated or boxes widened.
  • Consistent box heights within each logical row.
  • ≥ 0.025 gap between phase containers for visual breathing room.
  • Drop shadows for depth; white figure background for print fidelity.
  • 300 DPI, ~16 cm × 28 cm (taller to give each element room).
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════
FIG_W_CM, FIG_H_CM = 16.5, 28
DPI = 300
FIG_W = FIG_W_CM / 2.54
FIG_H = FIG_H_CM / 2.54

# Palette — colourblind-safe, print-safe
C = dict(
    bg      = '#FFFFFF',
    ph1     = '#E8EEF4',   # cool grey-blue
    ph2     = '#FDF5E0',   # warm cream
    ph3     = '#EEEBF3',   # soft lavender
    box     = '#FFFFFF',
    out     = '#EDF7ED',   # light green for output nodes
    text    = '#1A1A1A',
    script  = '#37474F',
    arrow   = '#546E7A',
    arch_ec = '#EF6C00',   # orange border for architectures
    final   = '#E8EAF6',
    phase_t = '#37474F',   # phase title
    note    = '#607D8B',
    accent  = '#3F51B5',   # indigo for final box
)

FONT = 'DejaVu Sans'
MONO = 'DejaVu Sans Mono'

# Standard box half-heights (for arrow endpoint calculation)
BH = 0.026   # standard box half-height
BH_L = 0.030 # large box half-height
BH_S = 0.018 # small output box half-height
ARROW_LW = 1.2  # consistent connector stroke width across the figure


# ═══════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════

def phase_bg(ax, y_bot, y_top, title, color):
    """Phase container with title ABOVE the content, inside the container."""
    h = y_top - y_bot
    p = FancyBboxPatch((0.030, y_bot), 0.940, h,
                        boxstyle='round,pad=0.010',
                        fc=color, ec='#C5CCD3', lw=0.8, alpha=0.50, zorder=0)
    ax.add_patch(p)
    # Keep phase title above arrows for readability.
    ax.text(0.055, y_top - 0.004, title, ha='left', va='top',
            fontsize=8.5, fontweight='bold', family=FONT, color=C['phase_t'],
            bbox=dict(fc=color, ec='none', pad=1.5, alpha=0.85),
            zorder=6)


def _shadow_box(ax, cx, cy, w, h, bg, ec, lw):
    """Internal: draw a rounded box with a subtle drop shadow."""
    pad = 0.008
    iw, ih = w - 2*pad, h - 2*pad
    x0, y0 = cx - iw/2, cy - ih/2
    # shadow offset
    ax.add_patch(FancyBboxPatch(
        (x0 + 0.003, y0 - 0.003), iw, ih,
        boxstyle=f'round,pad={pad}', fc='#000000', ec='none',
        alpha=0.05, zorder=2))
    # main box
    ax.add_patch(FancyBboxPatch(
        (x0, y0), iw, ih,
        boxstyle=f'round,pad={pad}', fc=bg, ec=ec, lw=lw, zorder=3))


def box(ax, cx, cy, w, h, title, sub=None, *,
        bg=None, ec='#90A4AE', lw=1.0,
        tfs=8, sfs=6.5, tc=None, sc=None):
    """Draw a labelled box. Returns (cx, cy, w, h) for arrow routing."""
    bg = bg or C['box']
    tc = tc or C['text']
    sc = sc or C['script']
    _shadow_box(ax, cx, cy, w, h, bg, ec, lw)
    if sub:
        ax.text(cx, cy + h * 0.15, title, ha='center', va='center',
                fontsize=tfs, fontweight='bold', family=FONT, color=tc, zorder=4)
        ax.text(cx, cy - h * 0.18, sub, ha='center', va='center',
                fontsize=sfs, family=MONO, color=sc, zorder=4)
    else:
        ax.text(cx, cy, title, ha='center', va='center',
                fontsize=tfs, fontweight='bold', family=FONT, color=tc, zorder=4)
    return (cx, cy, w, h)


def arr(ax, x1, y1, x2, y2, *, lw=ARROW_LW, ls='-', c=None):
    """Straight arrow from point to point."""
    c = c or C['arrow']
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(
                    arrowstyle='-|>,head_width=0.30,head_length=0.40',
                    color=c, lw=lw, ls=ls, fc=c),
                zorder=2)


def note_text(ax, cx, cy, text, fs=7, c=None, fw='normal', style='normal'):
    """Plain note text (no box)."""
    c = c or C['note']
    ax.text(cx, cy, text, ha='center', va='center',
            fontsize=fs, fontweight=fw, fontstyle=style,
            family=FONT, color=c, zorder=4)


def pill_label(ax, cx, cy, text, fs=7, tc=None, bg='#FFFFFF', ec='#C5CCD3'):
    """Text in a rounded pill/badge."""
    tc = tc or C['phase_t']
    ax.text(cx, cy, text, ha='center', va='center',
            fontsize=fs, fontweight='bold', family=FONT, color=tc,
            bbox=dict(fc=bg, ec=ec, lw=0.6, pad=3, boxstyle='round,pad=0.20'),
            zorder=5)


# ═══════════════════════════════════════════════════════════════════════
#  Y-GRID  (top-to-bottom, all y-coordinates pre-planned)
# ═══════════════════════════════════════════════════════════════════════
# Phase 1 container: 0.560 → 0.965
# Phase 2 container: 0.290 → 0.530
# Phase 3 container: 0.045 → 0.260

Y = dict(
    title   = 0.988,
    # ── Phase 1 ──  (container: 0.545 → 0.970)
    ph1_top = 0.956,
    ph1_bot = 0.545,
    inp     = 0.906,   # input boxes — shifted down for title clearance
    seg     = 0.836,   # segmentation
    abl     = 0.766,   # ablation scripts (blackout, split, sort)
    msk_out = 0.704,   # masking outputs (blackout imgs, medial, lateral)
    prep    = 0.634,   # prepare_classification_data
    cond    = 0.566,   # 4 condition labels (tightened Phase 1 bottom whitespace)
    # ── Phase 2 ──  (container: 0.280 → 0.520)
    ph2_top = 0.520,
    ph2_bot = 0.280,
    ph2_lbl = 0.480,   # "4 conditions → 3 tracks" label
    arch    = 0.420,   # architecture boxes
    notes   = 0.360,   # "Each track: ..." note (extra clearance from connector)
    notes2  = 0.338,   # "60 total runs" note
    robust  = 0.304,   # robustness validation pill
    # ── Phase 3 ──  (container: 0.042 → 0.255)
    ph3_top = 0.255,
    ph3_bot = 0.042,
    ana     = 0.200,   # analysis boxes
    ana_out = 0.135,   # analysis output boxes
    rq      = 0.092,   # RQ mapping
    final   = 0.058,   # final result box
)

# X-grid: left, centre, right columns
XL, XC, XR = 0.20, 0.50, 0.80


# ═══════════════════════════════════════════════════════════════════════
#  BUILD FIGURE
# ═══════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')
fig.patch.set_facecolor(C['bg'])

# Title
ax.text(0.50, Y['title'],
        'Figure 3.1: Segmentation-Guided Feature Ablation Pipeline',
        ha='center', va='top', fontsize=11, fontweight='bold',
        family=FONT, color=C['text'])

# ─────────────────────────────────────────────────────────────────────
#  PHASE 1 — Feature Extraction & Anatomical Ablation
# ─────────────────────────────────────────────────────────────────────
phase_bg(ax, Y['ph1_bot'], Y['ph1_top'],
         'PHASE 1: Feature Extraction & Anatomical Ablation', C['ph1'])

# Row: Inputs
box(ax, 0.32, Y['inp'], 0.38, 0.052,
    '5,428 OAI Knee Radiographs',
    'Pre-cropped 224×224 (KL0 & KL2)', sfs=6)
box(ax, 0.74, Y['inp'], 0.34, 0.052,
    'Pre-trained U-Net',
    'MODEL_unet.pth (IoU = 0.93)', sfs=6)

# Arrows: inputs → segmentation
arr(ax, 0.32, Y['inp'] - BH, 0.44, Y['seg'] + BH)
arr(ax, 0.74, Y['inp'] - BH, 0.56, Y['seg'] + BH)

# Row: Segmentation + Expansion
box(ax, XC, Y['seg'], 0.54, 0.052,
    'Segment & Expand Joint Region',
    'segment_simple.py  +  expand_mask_horizontal.py',
    ec='#5C6BC0', lw=1.4, sfs=6.0)

# Arrows: segmentation → 3 ablation branches
arr(ax, 0.35, Y['seg'] - BH, XL, Y['abl'] + BH)
arr(ax, XC,   Y['seg'] - BH, XC, Y['abl'] + BH)
arr(ax, 0.65, Y['seg'] - BH, XR, Y['abl'] + BH)

# Row: Ablation scripts
box(ax, XL, Y['abl'], 0.26, 0.048,
    'Full Blackout',
    'apply_mask_blackout.py', tfs=7.5, sfs=6.0)
box(ax, XC, Y['abl'], 0.26, 0.048,
    'Left / Right Split',
    'apply_mask_split.py', tfs=7.5, sfs=6.0)
box(ax, XR, Y['abl'], 0.26, 0.048,
    'Anatomical Sorting',
    'sort_lateral_medial.py', tfs=7.5, sfs=6.0)

# Horizontal arrow: split → sort
arr(ax, XC + 0.13, Y['abl'], XR - 0.13, Y['abl'])

# Arrows: ablation scripts → masking outputs
arr(ax, XL, Y['abl'] - 0.024, XL,   Y['msk_out'] + BH_S)
arr(ax, XR, Y['abl'] - 0.024, 0.65, Y['msk_out'] + BH_S)
arr(ax, XR, Y['abl'] - 0.024, 0.88, Y['msk_out'] + BH_S)

# Row: Masking outputs
box(ax, XL,   Y['msk_out'], 0.22, 0.036,
    'Blackout Images', tfs=7, bg=C['out'])
box(ax, 0.65, Y['msk_out'], 0.20, 0.036,
    'Medial Masked', tfs=7, bg=C['out'])
box(ax, 0.88, Y['msk_out'], 0.20, 0.036,
    'Lateral Masked', tfs=7, bg=C['out'])

# Arrows: masking outputs → prepare
arr(ax, XL,   Y['msk_out'] - BH_S, 0.38, Y['prep'] + BH)
arr(ax, 0.65, Y['msk_out'] - BH_S, XC,   Y['prep'] + BH)
arr(ax, 0.88, Y['msk_out'] - BH_S, 0.62, Y['prep'] + BH)

# Row: Dataset preparation
box(ax, XC, Y['prep'], 0.52, 0.050,
    'prepare_classification_data.py',
    'Creates train / val / test splits for PyTorch',
    tfs=7.5, sfs=6.0, ec='#78909C')

# Baseline bypass: dashed line routed OUTSIDE all boxes on far left margin
# Start from the left edge of the 5,428 OAI box, then route down the far-left margin.
bx = 0.045
input_left_edge = 0.138
input_bottom_edge = Y['inp'] - 0.018
ax.plot([input_left_edge, bx], [input_bottom_edge, input_bottom_edge],
    color=C['arrow'], lw=ARROW_LW, ls=':', zorder=1)
ax.plot([bx, bx], [input_bottom_edge, Y['prep']],
    color=C['arrow'], lw=ARROW_LW, ls=':', zorder=1)
# Keep the bypass line dotted, but render a solid arrowhead at the endpoint.
ax.plot([bx, 0.228], [Y['prep'], Y['prep']],
    color=C['arrow'], lw=ARROW_LW, ls=':', zorder=1)
arr(ax, 0.228, Y['prep'], 0.24, Y['prep'])
ax.text(bx - 0.012, (input_bottom_edge + Y['prep']) / 2,
        'Baseline (unmodified)',
    fontsize=6.0, fontweight='bold', ha='center', va='center',
    color=C['note'], family=FONT, rotation=90,
    bbox=dict(fc=C['ph1'], ec='none', pad=0.4, alpha=0.95), zorder=4)

# Arrows: prepare → conditions
arr(ax, XC, Y['prep'] - BH, XC, Y['cond'] + 0.008)

# Row: 4 condition labels
note_text(ax, XC, Y['cond'],
          'Baseline (Control)  |  Blackout  |  Medial Masked  |  Lateral Masked',
          fs=7, c=C['script'], fw='bold')

# ─────────────────────────────────────────────────────────────────────
#  PHASE 2 — Classification & Robustness
# ─────────────────────────────────────────────────────────────────────
phase_bg(ax, Y['ph2_bot'], Y['ph2_top'],
         'PHASE 2: Classification & Robustness (N = 60 Training Sequences)',
         C['ph2'])

# Arrow from phase 1 → phase 2
arr(ax, XC, Y['cond'] - 0.010, XC, Y['ph2_lbl'] + 0.012)

# Label
pill_label(ax, XC, Y['ph2_lbl'],
           '4 Experimental Conditions  →  3 Architecture Tracks',
           fs=7, tc=C['phase_t'])

# Arrows: label → 3 architecture boxes
# Single vertical stem from pill bottom, then fan out at a midpoint
stem_y = Y['ph2_lbl'] - 0.014  # just below the pill
fan_y  = (stem_y + Y['arch'] + BH_L) / 2  # midpoint for fan-out
# Vertical stem
ax.plot([XC, XC], [stem_y, fan_y], color=C['arrow'], lw=ARROW_LW, zorder=1)
# Fan-out to 3 boxes
arr(ax, XC,  fan_y, 0.22, Y['arch'] + BH_L)
arr(ax, XC,  fan_y, XC,   Y['arch'] + BH_L)
arr(ax, XC,  fan_y, 0.78, Y['arch'] + BH_L)

# Row: Architecture boxes
box(ax, 0.22, Y['arch'], 0.28, 0.058,
    'ResNet-18',
    'results_resnet/',
    ec=C['arch_ec'], lw=1.6, tfs=9, sfs=6.0)
box(ax, XC, Y['arch'], 0.28, 0.058,
    'EfficientNet-B0',
    'results_efficientnet/',
    ec=C['arch_ec'], lw=1.6, tfs=9, sfs=6.0)
box(ax, 0.78, Y['arch'], 0.28, 0.058,
    'Swin-Tiny',
    'results_swin/',
    ec=C['arch_ec'], lw=1.6, tfs=9, sfs=6.0)

# Short stubs downward from architecture boxes (stop before the notes)
stub_bot = Y['notes'] + 0.024   # stronger separation from note row
for x in [0.22, XC, 0.78]:
    ax.plot([x, x], [Y['arch'] - BH_L, stub_bot],
        color=C['arrow'], lw=ARROW_LW, zorder=1)
# Horizontal connector bar at stub bottom
ax.plot([0.22, 0.78], [stub_bot, stub_bot],
    color=C['arrow'], lw=ARROW_LW, zorder=1)

# Notes (drawn ABOVE the connector, no overlap)
note_text(ax, XC, Y['notes'],
          'Each: train → self-eval → cross-eval → robustness (5 seeds)',
          fs=6.5, style='italic')
note_text(ax, XC, Y['notes2'],
          '60 total training runs (3 architectures × 4 conditions × 5 seeds)',
          fs=7, c=C['script'], fw='bold')

# Robustness pill
pill_label(ax, XC, Y['robust'],
           'Cross-Evaluation & Robustness Validated Across 5 Random Seeds',
           fs=6.5, tc=C['phase_t'], ec='#B0BEC5')

# ─────────────────────────────────────────────────────────────────────
#  PHASE 3 — Explainability & Statistical Analysis
# ─────────────────────────────────────────────────────────────────────
phase_bg(ax, Y['ph3_bot'], Y['ph3_top'],
         'PHASE 3: Explainability & Statistical Analysis', C['ph3'])

# Arrows from robustness pill into Phase 3 analysis boxes
# Route via a center stem + fan-out below the phase title band to avoid overlap.
entry_y = Y['robust'] - 0.015
phase3_entry_top = Y['ph3_top'] + 0.004
phase3_fan_y = Y['ph3_top'] - 0.022
arr(ax, XC, entry_y, XC, phase3_entry_top)
arr(ax, XC, phase3_fan_y, 0.22, Y['ana'] + BH_L)
arr(ax, XC, phase3_fan_y, XC,   Y['ana'] + BH_L)
arr(ax, XC, phase3_fan_y, 0.78, Y['ana'] + BH_L)

# Row: Analysis boxes
box(ax, 0.22, Y['ana'], 0.28, 0.056,
    'Spatial Attention',
    'generate_gradcam*.py\n+ comparisons.py',
    tfs=7.5, sfs=6.0)
box(ax, XC, Y['ana'], 0.28, 0.056,
    'Activation Density',
    'evaluate_gradcam_\nquantitatively.py',
    tfs=7.5, sfs=6.0)
box(ax, 0.78, Y['ana'], 0.28, 0.056,
    'Hypothesis Testing',
    'perform_wilcoxon_\ntests.py',
    tfs=7.5, sfs=6.0)

# Arrows: analysis → output
arr(ax, 0.22, Y['ana'] - 0.028, 0.22, Y['ana_out'] + BH_S)
arr(ax, XC,   Y['ana'] - 0.028, XC,   Y['ana_out'] + BH_S)
arr(ax, 0.78, Y['ana'] - 0.028, 0.78, Y['ana_out'] + BH_S)

# Row: Analysis outputs
box(ax, 0.22, Y['ana_out'], 0.26, 0.035,
    'Attention Matrices', tfs=6.5, bg=C['out'])
box(ax, XC, Y['ana_out'], 0.26, 0.035,
    'Inside / Outside Ratios', tfs=6.5, bg=C['out'])
box(ax, 0.78, Y['ana_out'], 0.26, 0.035,
    'Wilcoxon p-values', tfs=6.5, bg=C['out'])

# RQ mapping
pill_label(ax, XC, Y['rq'],
           'RQ1–2: Accuracy & recall  |  RQ3: Confidence  |  RQ4: Ablation vs. Grad-CAM',
           fs=6, tc=C['script'], bg='#FFFFFF', ec='#B0BEC5')

# ─────────────────────────────────────────────────────────────────────
#  FINAL RESULT
# ─────────────────────────────────────────────────────────────────────
arr(ax, 0.22, Y['ana_out'] - BH_S, 0.38, Y['final'] + 0.020)
arr(ax, XC,   Y['ana_out'] - BH_S, XC,   Y['final'] + 0.020)
arr(ax, 0.78, Y['ana_out'] - BH_S, 0.62, Y['final'] + 0.020)

box(ax, XC, Y['final'], 0.70, 0.042,
    'Empirically Validated Thesis Results',
    'H1–H4 tested  •  3 architectures  •  5 seeds  •  16,284 images',
    bg=C['final'], ec=C['accent'], lw=1.8, tfs=8.5, sfs=6.5, sc='#283593')

# Footer
ax.text(0.50, 0.015,
        'Data: OAI Kaggle [38]  •  Segmentation: Nykänen (2022) [54]  •  '
        'Architectures: ResNet-18, EfficientNet-B0, Swin-Tiny',
    ha='center', va='center', fontsize=6, family=FONT, color=C['note'])

# ═══════════════════════════════════════════════════════════════════════
#  SAVE
# ═══════════════════════════════════════════════════════════════════════
out_dir = Path('drafts')
out_dir.mkdir(parents=True, exist_ok=True)
out = out_dir / 'figure_3_1_pipeline_diagram.png'
fig.savefig(str(out), dpi=DPI, bbox_inches='tight', facecolor=C['bg'],
            edgecolor='none', pad_inches=0.10)
plt.close(fig)
print(f'Saved: {out}')
print(f'Size: {FIG_W_CM} cm × {FIG_H_CM} cm @ {DPI} DPI')
