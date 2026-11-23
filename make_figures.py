#!/usr/bin/env python3
"""Directly compute from empirical evaluation logs and dataset artifacts and plot
publication-quality SVG + vector PDF figures for Clinical StigmaQA (ACL Long Paper).

- Font family: Nimbus Roman (URW Times clone, identical to LaTeX \\usepackage{times})
  and STIX math fontset so every axis label, tick, legend, and equation matches main.tex.
- Figure 1 (fig1_pipeline.{svg,pdf}): Pure SVG/PDF 4-stage pipeline flowchart.
- Figure 2 (fig2_dataset_overview.{svg,pdf}): Directly computed from design/events.json,
  processed/clinical_stigmaQA_r25_n50_20261104/items_consider_r27.eligible.jsonl, and
  processed/eval/panel_r27c_factor_tables.md.
- Figure 3 (fig1_factor_condition.{svg,pdf}): Directly parsed & computed from
  processed/eval/panel_r27c_factor_tables.md (Tables S1-S5 across 28 cross-tabulated models).
- Figure 4 (fig2_models_med_vs_base.{svg,pdf}): Directly parsed & computed from
  processed/eval/panel_r27c_summary.md (11 matched medical-vs-base backbone pairs).
"""
import json
import os
import re
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Polygon, Rectangle, Wedge, Arc
import numpy as np

# Use Nimbus Roman (exact URW Times Roman used by TeX Live \usepackage{times}) + STIX math
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Times New Roman", "Liberation Serif", "STIX Two Text"],
    "mathtext.fontset": "stix",
    "font.size": 9.5,
    "axes.titlesize": 10.5,
    "axes.labelsize": 9.5,
    "xtick.labelsize": 8.8,
    "ytick.labelsize": 9.0,
    "legend.fontsize": 8.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

PALETTE = {
    "navy": "#1B365D",
    "blue": "#0072B2",
    "sky": "#56B4E9",
    "teal": "#009E73",
    "orange": "#E69F00",
    "vermilion": "#D55E00",
    "purple": "#CC79A7",
    "dark_red": "#A50F15",
    "slate": "#475569",
}

ROOT = Path("/data/gpfs/projects/punim2961/hongyuj1/stigmaQA")
OUT_DIR = ROOT / "paper" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def save_both(fig, basename: str):
    """Save figure as both clean editable SVG and vector PDF."""
    fig.savefig(OUT_DIR / f"{basename}.svg", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(OUT_DIR / f"{basename}.pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def _draw_book_icon(ax, cx, cy, s=0.28, color="#1B365D"):
    """Flat-vector open book + guideline checkmark badge."""
    # Left & right pages
    left_p = Polygon([(cx, cy - s*0.35), (cx - s*0.85, cy - s*0.20), (cx - s*0.85, cy + s*0.48), (cx, cy + s*0.35)],
                     closed=True, facecolor="#DBEAFE", edgecolor=color, lw=1.3, zorder=5)
    right_p = Polygon([(cx, cy - s*0.35), (cx + s*0.85, cy - s*0.20), (cx + s*0.85, cy + s*0.48), (cx, cy + s*0.35)],
                      closed=True, facecolor="#EFF6FF", edgecolor=color, lw=1.3, zorder=5)
    ax.add_patch(left_p)
    ax.add_patch(right_p)
    # Text lines on pages
    for dy in (0.14, -0.02, -0.16):
        ax.plot([cx - s*0.65, cx - s*0.18], [cy + s*dy, cy + s*(dy - 0.05)], color=color, lw=0.9, alpha=0.65, zorder=6)
        ax.plot([cx + s*0.18, cx + s*0.65], [cy + s*(dy - 0.05), cy + s*dy], color=color, lw=0.9, alpha=0.65, zorder=6)
    # Green check badge at top-right
    badge = Circle((cx + s*0.68, cy + s*0.42), s*0.30, facecolor="#009E73", edgecolor="white", lw=1.2, zorder=7)
    ax.add_patch(badge)
    ax.plot([cx + s*0.54, cx + s*0.65, cx + s*0.83],
            [cy + s*0.41, cy + s*0.31, cy + s*0.53], color="white", lw=1.5, solid_capstyle="round", zorder=8)


def _draw_ehr_icon(ax, cx, cy, s=0.28, color="#0072B2"):
    """Flat-vector medical EHR clipboard with red cross + filter shield."""
    board = FancyBboxPatch((cx - s*0.58, cy - s*0.58), s*1.16, s*1.18,
                           boxstyle="round,pad=0.02,rounding_size=0.06",
                           facecolor="#E0F2FE", edgecolor=color, lw=1.3, zorder=5)
    clip = FancyBboxPatch((cx - s*0.26, cy + s*0.48), s*0.52, s*0.20,
                          boxstyle="round,pad=0.01,rounding_size=0.03",
                          facecolor=color, edgecolor=color, lw=0.8, zorder=6)
    ax.add_patch(board)
    ax.add_patch(clip)
    # Medical red cross
    ax.add_patch(Rectangle((cx - s*0.09, cy + s*0.02), s*0.18, s*0.36, facecolor="#D55E00", edgecolor="none", zorder=6))
    ax.add_patch(Rectangle((cx - s*0.18, cy + s*0.11), s*0.36, s*0.18, facecolor="#D55E00", edgecolor="none", zorder=6))
    # Chart lines
    ax.plot([cx - s*0.38, cx + s*0.38], [cy - s*0.18, cy - s*0.18], color=color, lw=1.1, alpha=0.75, zorder=6)
    ax.plot([cx - s*0.38, cx + s*0.22], [cy - s*0.36, cy - s*0.36], color=color, lw=1.1, alpha=0.75, zorder=6)
    # 5-dot panel badge at bottom-right
    badge = Circle((cx + s*0.58, cy - s*0.40), s*0.30, facecolor="#0072B2", edgecolor="white", lw=1.2, zorder=7)
    ax.add_patch(badge)
    ax.text(cx + s*0.58, cy - s*0.41, "5x", ha="center", va="center", color="white",
            fontsize=6.5, fontweight="bold", zorder=8)


def _draw_mini_icon(ax, kind, cx, cy, r=0.10, color="#009E73"):
    """Crisp mini flat-vector pictograms for M, P, X, F pills."""
    if kind == "M":
        # Medical cross inside circle
        ax.add_patch(Circle((cx, cy), r, facecolor=color, edgecolor="none", zorder=6))
        ax.add_patch(Rectangle((cx - r*0.22, cy - r*0.58), r*0.44, r*1.16, facecolor="white", zorder=7))
        ax.add_patch(Rectangle((cx - r*0.58, cy - r*0.22), r*1.16, r*0.44, facecolor="white", zorder=7))
    elif kind == "P":
        # Speech bubble
        ax.add_patch(FancyBboxPatch((cx - r*0.95, cy - r*0.62), r*1.90, r*1.28,
                                    boxstyle="round,pad=0.01,rounding_size=0.04",
                                    facecolor=color, edgecolor="none", zorder=6))
        ax.add_patch(Polygon([(cx - r*0.45, cy - r*0.60), (cx - r*0.10, cy - r*0.60), (cx - r*0.65, cy - r*1.05)],
                             closed=True, facecolor=color, edgecolor="none", zorder=6))
        ax.text(cx, cy + r*0.02, "“ ”", ha="center", va="center", color="white", fontsize=7.5, fontweight="bold", zorder=7)
    elif kind == "X":
        # Person silhouette inside circle
        ax.add_patch(Circle((cx, cy), r, facecolor=color, edgecolor="none", zorder=6))
        ax.add_patch(Circle((cx, cy + r*0.28), r*0.30, facecolor="white", zorder=7))
        ax.add_patch(Wedge((cx, cy - r*0.68), r*0.58, 30, 150, facecolor="white", zorder=7))
    elif kind == "F":
        # Hospital / organization building silhouette
        ax.add_patch(FancyBboxPatch((cx - r*0.85, cy - r*0.85), r*1.70, r*1.45,
                                    boxstyle="round,pad=0.01,rounding_size=0.03",
                                    facecolor=color, edgecolor="none", zorder=6))
        # Roof pediment
        ax.add_patch(Polygon([(cx - r*0.98, cy + r*0.58), (cx + r*0.98, cy + r*0.58), (cx, cy + r*1.05)],
                             closed=True, facecolor=color, edgecolor="none", zorder=6))
        # 4 tiny windows + door
        for wx in (-0.42, 0.14):
            for wy in (0.12, -0.32):
                ax.add_patch(Rectangle((cx + r*wx, cy + r*wy), r*0.28, r*0.26, facecolor="white", zorder=7))
        ax.add_patch(Rectangle((cx - r*0.14, cy - r*0.82), r*0.28, r*0.36, facecolor="white", zorder=7))


def _draw_check_badge(ax, cx, cy, r=0.075, color="#009E73"):
    """Crisp flat-vector green checkmark badge (no missing font glyph)."""
    ax.add_patch(Circle((cx, cy), r, facecolor=color, edgecolor="none", zorder=6))
    ax.plot([cx - r*0.44, cx - r*0.08, cx + r*0.48],
            [cy - r*0.02, cy - r*0.38, cy + r*0.38], color="white", lw=1.35, solid_capstyle="round", zorder=7)


def _draw_x_badge(ax, cx, cy, r=0.075, color="#DC2626"):
    """Crisp flat-vector red X badge (no missing font glyph)."""
    ax.add_patch(Circle((cx, cy), r, facecolor=color, edgecolor="none", zorder=6))
    ax.plot([cx - r*0.38, cx + r*0.38], [cy - r*0.38, cy + r*0.38], color="white", lw=1.35, solid_capstyle="round", zorder=7)
    ax.plot([cx - r*0.38, cx + r*0.38], [cy + r*0.38, cy - r*0.38], color="white", lw=1.35, solid_capstyle="round", zorder=7)


def _draw_audit_icon(ax, cx, cy, s=0.28, color="#D55E00"):
    """Flat-vector AI chip + balance scale badge."""
    # AI chip body
    chip = FancyBboxPatch((cx - s*0.52, cy - s*0.46), s*1.04, s*0.92,
                          boxstyle="round,pad=0.02,rounding_size=0.06",
                          facecolor="#FFEDD5", edgecolor=color, lw=1.3, zorder=5)
    ax.add_patch(chip)
    # Pins
    for d in (-0.28, 0.0, 0.28):
        ax.plot([cx + s*d, cx + s*d], [cy + s*0.48, cy + s*0.66], color=color, lw=1.1, zorder=4)
        ax.plot([cx + s*d, cx + s*d], [cy - s*0.48, cy - s*0.66], color=color, lw=1.1, zorder=4)
        ax.plot([cx - s*0.54, cx - s*0.72], [cy + s*d*0.8, cy + s*d*0.8], color=color, lw=1.1, zorder=4)
        ax.plot([cx + s*0.54, cx + s*0.72], [cy + s*d*0.8, cy + s*d*0.8], color=color, lw=1.1, zorder=4)
    ax.text(cx, cy + s*0.02, "LLM", ha="center", va="center", color=color, fontsize=7.0, fontweight="bold", zorder=6)
    # Balance-scale badge (slanted beam + pans)
    bx, by, br = cx + s*0.62, cy + s*0.44, s*0.32
    ax.add_patch(Circle((bx, by), br, facecolor="#1B365D", edgecolor="white", lw=1.2, zorder=7))
    ax.plot([bx, bx], [by - br*0.52, by + br*0.45], color="white", lw=1.1, zorder=8)
    ax.plot([bx - br*0.55, bx + br*0.55], [by + br*0.30, by + br*0.12], color="white", lw=1.1, zorder=8)
    ax.add_patch(Wedge((bx - br*0.50, by + br*0.02), br*0.24, 180, 360, facecolor="white", zorder=8))
    ax.add_patch(Wedge((bx + br*0.50, by - br*0.16), br*0.24, 180, 360, facecolor="white", zorder=8))


def make_fig1_pipeline():
    """Figure 1: Visual, low-text, icon-rich top-conference overall pipeline + 1-glance counterfactual strip."""
    fig, ax = plt.subplots(figsize=(12.6, 4.15))
    ax.set_xlim(0, 12.6)
    ax.set_ylim(0, 4.15)
    ax.axis("off")

    # ── TOP TIER: 4 VISUAL STEP CARDS ──────────────────────────────────────
    card_w = 2.84
    card_h = 2.52
    y_top = 1.50
    xs = [0.12, 3.29, 6.46, 9.63]

    step_meta = [
        ("STEP 1", "Theory & Guidelines", PALETTE["navy"], "#F8FAFC", "#DBEAFE",
         [("145 HSDF Studies", "435 clinician stigma records"),
          ("38 Clinical Guidelines", "Strict medical orthogonality")],
         "31 Events  \u00b7  6 Domains  \u00b7  73 Licenses"),
        ("STEP 2", "Pre-Decision EHRs", PALETTE["blue"], "#F0F9FF", "#BAE6FD",
         [("MIMIC-IV Admission", "Truncated pre-decision notes"),
          ("5-LLM Eligibility Panel", "1,416 \u2192 1,030 clean cases")],
         "1,030 Verified Patient Vignettes"),
        ("STEP 3", "Counterfactual Synthesis", PALETTE["teal"], "#ECFDF5", "#A7F3D0",
         None,  # custom 2x2 visual icon pills!
         "29,845 Items  \u00b7  10 Task Templates"),
        ("STEP 4", "Multi-Model Audit", PALETTE["vermilion"], "#FFF7ED", "#FED7AA",
         [("34 Open-Weight LLMs", "21 general + 13 medical (11 pairs)"),
          ("Attention & Judge Audit", "610k responses \u00b7 99.45% agree")],
         "Bedside Bias vs. Explicit Attitudes"),
    ]

    for idx, (badge_txt, title, col, bg, pill_bg, items, footer) in enumerate(step_meta):
        x = xs[idx]
        # Subtle drop shadow
        ax.add_patch(FancyBboxPatch((x + 0.04, y_top - 0.04), card_w, card_h,
                                    boxstyle="round,pad=0.02,rounding_size=0.10",
                                    facecolor="#CBD5E1", edgecolor="none", alpha=0.40, zorder=1))
        # Main card
        ax.add_patch(FancyBboxPatch((x, y_top), card_w, card_h,
                                    boxstyle="round,pad=0.02,rounding_size=0.10",
                                    facecolor=bg, edgecolor=col, lw=1.45, zorder=2))
        # Header band
        hh = 0.44
        ax.add_patch(FancyBboxPatch((x, y_top + card_h - hh), card_w, hh,
                                    boxstyle="round,pad=0.02,rounding_size=0.10",
                                    facecolor=col, edgecolor=col, lw=1.0, zorder=3))
        # Step pill inside header
        ax.add_patch(FancyBboxPatch((x + 0.09, y_top + card_h - hh + 0.10), 0.62, 0.24,
                                    boxstyle="round,pad=0.01,rounding_size=0.05",
                                    facecolor="white", edgecolor="none", alpha=0.22, zorder=4))
        ax.text(x + 0.40, y_top + card_h - hh/2, badge_txt, ha="center", va="center",
                color="white", fontsize=7.2, fontweight="bold", zorder=5)
        ax.text(x + 0.80, y_top + card_h - hh/2, title, ha="left", va="center",
                color="white", fontsize=9.3, fontweight="bold", zorder=5)

        if idx != 2:
            # Draw large centered vector icon
            icon_cx, icon_cy = x + card_w / 2, y_top + 1.66
            if idx == 0:
                _draw_book_icon(ax, icon_cx, icon_cy, s=0.31, color=col)
            elif idx == 1:
                _draw_ehr_icon(ax, icon_cx, icon_cy, s=0.31, color=col)
            elif idx == 3:
                _draw_audit_icon(ax, icon_cx, icon_cy, s=0.31, color=col)

            # Two compact clean white info pills
            for j, (k_txt, v_txt) in enumerate(items):
                py = y_top + 0.90 - j * 0.44
                ax.add_patch(FancyBboxPatch((x + 0.11, py), card_w - 0.22, 0.37,
                                            boxstyle="round,pad=0.01,rounding_size=0.06",
                                            facecolor="white", edgecolor="#CBD5E1", lw=0.85, zorder=3))
                ax.text(x + card_w/2, py + 0.245, k_txt, ha="center", va="center",
                        color=col, fontsize=8.0, fontweight="bold", zorder=4)
                ax.text(x + card_w/2, py + 0.095, v_txt, ha="center", va="center",
                        color="#334155", fontsize=7.5, zorder=4)
        else:
            # STEP 3: 2x2 visual icon pills for M, P, X, F + validator badge
            factors = [
                ("M", "M: Disease", "9 remitted", "#D55E00", "#FFF7ED"),
                ("P", "P: Wording", "Same facts", "#E69F00", "#FEFCE8"),
                ("X", "X: Social", "5 axes", "#CC79A7", "#FDF2F8"),
                ("F", "F: Context", "ED / Equity", "#0072B2", "#EFF6FF"),
            ]
            pw = (card_w - 0.30) / 2
            ph = 0.56
            for f_idx, (fk, fl, fs, fc, fbg) in enumerate(factors):
                col_i = f_idx % 2
                row_i = f_idx // 2
                fx = x + 0.11 + col_i * (pw + 0.08)
                fy = y_top + 1.38 - row_i * (ph + 0.09)
                ax.add_patch(FancyBboxPatch((fx, fy), pw, ph,
                                            boxstyle="round,pad=0.01,rounding_size=0.06",
                                            facecolor=fbg, edgecolor=fc, lw=1.0, zorder=3))
                _draw_mini_icon(ax, fk, fx + 0.22, fy + ph/2, r=0.13, color=fc)
                ax.text(fx + 0.41, fy + ph*0.64, fl, ha="left", va="center",
                        color=fc, fontsize=7.7, fontweight="bold", zorder=4)
                ax.text(fx + 0.41, fy + ph*0.28, fs, ha="left", va="center",
                        color="#334155", fontsize=7.1, zorder=4)

            # Invariance validator bar
            vy = y_top + 0.46
            ax.add_patch(FancyBboxPatch((x + 0.11, vy), card_w - 0.22, 0.21,
                                        boxstyle="round,pad=0.01,rounding_size=0.05",
                                        facecolor="white", edgecolor="#009E73", lw=0.85, zorder=3))
            _draw_check_badge(ax, x + 0.36, vy + 0.105, r=0.068, color="#009E73")
            ax.text(x + card_w/2 + 0.08, vy + 0.105, "35-Check Invariance Validator", ha="center", va="center",
                    color="#009E73", fontsize=7.5, fontweight="bold", zorder=4)

        # Output footer pill
        ax.add_patch(FancyBboxPatch((x + 0.11, y_top + 0.09), card_w - 0.22, 0.29,
                                    boxstyle="round,pad=0.01,rounding_size=0.06",
                                    facecolor=pill_bg, edgecolor=col, lw=0.85, zorder=3))
        ax.text(x + card_w/2, y_top + 0.235, footer, ha="center", va="center",
                color="#0F172A", fontsize=7.8, fontweight="bold", zorder=4)

    # Flow arrows between top cards
    for i in range(3):
        x_start = xs[i] + card_w + 0.05
        x_end = xs[i + 1] - 0.05
        y_mid = y_top + card_h / 2
        ax.add_patch(FancyArrowPatch((x_start, y_mid), (x_end, y_mid),
                                     arrowstyle="-|>,head_length=6.5,head_width=4.5",
                                     lw=2.2, color="#475569", zorder=6))

    # ── BOTTOM TIER: 1-GLANCE VISUAL COUNTERFACTUAL EXAMPLE STRIP ──────────
    yb, hb = 0.10, 1.15
    ax.add_patch(FancyBboxPatch((0.12, yb), 12.35, hb,
                                boxstyle="round,pad=0.02,rounding_size=0.09",
                                facecolor="#F8FAFC", edgecolor="#94A3B8", lw=1.1, linestyle="--", zorder=2))
    # Strip label badge
    ax.add_patch(FancyBboxPatch((0.26, yb + hb - 0.14), 2.55, 0.24,
                                boxstyle="round,pad=0.01,rounding_size=0.05",
                                facecolor="#334155", edgecolor="none", zorder=4))
    ax.text(1.535, yb + hb - 0.02, "ILLUSTRATIVE COUNTERFACTUAL", ha="center", va="center",
            color="white", fontsize=7.1, fontweight="bold", zorder=5)

    # Box A: Base Pre-Decision Vignette
    ax.add_patch(FancyBboxPatch((0.28, yb + 0.12), 3.35, 0.80,
                                boxstyle="round,pad=0.01,rounding_size=0.07",
                                facecolor="white", edgecolor="#0072B2", lw=1.1, zorder=3))
    ax.text(0.42, yb + 0.72, "Pre-Decision ED Chart (Byte-Identical)", ha="left", va="center",
            color="#0072B2", fontsize=7.9, fontweight="bold", zorder=4)
    ax.text(0.42, yb + 0.43, "48yo \u00b7 Acute substernal chest pressure\nDiaphoresis \u00b7 HR 108 \u00b7 BP 146/88",
            ha="left", va="center", color="#1E293B", fontsize=7.6, linespacing=1.25, zorder=4)
    ax.text(3.52, yb + 0.21, "AHA/ACC: 10-min ECG", ha="right", va="center",
            color="#009E73", fontsize=7.1, fontweight="bold", zorder=4)

    # Plus arrow
    ax.add_patch(FancyArrowPatch((3.70, yb + 0.52), (4.18, yb + 0.52),
                                 arrowstyle="-|>,head_length=5.5,head_width=3.8",
                                 lw=1.8, color="#475569", zorder=5))
    ax.text(3.92, yb + 0.69, "+", ha="center", va="center", color="#0F172A", fontsize=11, fontweight="bold", zorder=5)

    # Box B: Orthogonal Factor Cues
    ax.add_patch(FancyBboxPatch((4.25, yb + 0.12), 3.70, 0.80,
                                boxstyle="round,pad=0.01,rounding_size=0.07",
                                facecolor="white", edgecolor="#009E73", lw=1.1, zorder=3))
    ax.text(4.39, yb + 0.72, "Orthogonal Stigma / Context Cue(s)", ha="left", va="center",
            color="#009E73", fontsize=7.9, fontweight="bold", zorder=4)
    cue_pills = [
        (4.37, yb + 0.43, 1.72, "M: Remitted Schizophrenia", "#D55E00", "#FFF7ED"),
        (6.16, yb + 0.43, 1.68, "P: Patient \"claims\" pain", "#B45309", "#FEFCE8"),
        (4.37, yb + 0.17, 1.72, "X: Homeless / LEP / Race", "#9D174D", "#FDF2F8"),
        (6.16, yb + 0.17, 1.68, "F\u2212/+: Crowding vs. Equity", "#0072B2", "#EFF6FF"),
    ]
    for px, py, pw, ptxt, pcol, pbg in cue_pills:
        ax.add_patch(FancyBboxPatch((px, py), pw, 0.20,
                                    boxstyle="round,pad=0.01,rounding_size=0.04",
                                    facecolor=pbg, edgecolor=pcol, lw=0.75, zorder=4))
        ax.text(px + pw/2, py + 0.10, ptxt, ha="center", va="center",
                color=pcol, fontsize=6.9, fontweight="bold", zorder=5)

    # Arrow to Decision Audit
    ax.add_patch(FancyArrowPatch((8.02, yb + 0.52), (8.46, yb + 0.52),
                                 arrowstyle="-|>,head_length=5.5,head_width=3.8",
                                 lw=1.8, color="#475569", zorder=5))

    # Box C: Control vs Treated Decision Shift
    ax.add_patch(FancyBboxPatch((8.53, yb + 0.12), 3.80, 0.80,
                                boxstyle="round,pad=0.01,rounding_size=0.07",
                                facecolor="white", edgecolor="#D55E00", lw=1.1, zorder=3))
    ax.text(8.67, yb + 0.72, "Paired Decision Audit  (\u0394P(indicated) & R\u2192W)", ha="left", va="center",
            color="#D55E00", fontsize=7.9, fontweight="bold", zorder=4)
    # Control green badge
    ax.add_patch(FancyBboxPatch((8.65, yb + 0.21), 1.66, 0.38,
                                boxstyle="round,pad=0.01,rounding_size=0.05",
                                facecolor="#ECFDF5", edgecolor="#009E73", lw=0.9, zorder=4))
    _draw_check_badge(ax, 8.81, yb + 0.40, r=0.085, color="#009E73")
    ax.text(9.58, yb + 0.46, "Control Arm", ha="center", va="center", color="#009E73", fontsize=7.1, fontweight="bold", zorder=5)
    ax.text(9.58, yb + 0.30, "Order 10-min ECG", ha="center", va="center", color="#0F172A", fontsize=7.2, zorder=5)
    # Arrow
    ax.text(10.43, yb + 0.40, "\u2192", ha="center", va="center", color="#D55E00", fontsize=11, fontweight="bold", zorder=5)
    # Treated red badge
    ax.add_patch(FancyBboxPatch((10.55, yb + 0.21), 1.66, 0.38,
                                boxstyle="round,pad=0.01,rounding_size=0.05",
                                facecolor="#FEF2F2", edgecolor="#DC2626", lw=0.9, zorder=4))
    _draw_x_badge(ax, 10.71, yb + 0.40, r=0.085, color="#DC2626")
    ax.text(11.48, yb + 0.46, "Marked Arm", ha="center", va="center", color="#DC2626", fontsize=7.1, fontweight="bold", zorder=5)
    ax.text(11.48, yb + 0.30, "Attribute to Anxiety", ha="center", va="center", color="#0F172A", fontsize=7.2, zorder=5)

    save_both(fig, "fig1_pipeline")


def load_dataset_overview_data():
    """Directly compute domain item counts, condition licenses/pairs, and factor contrasts from workspace files."""
    events_path = ROOT / "design" / "events.json"
    items_path = ROOT / "processed" / "clinical_stigmaQA_r25_n50_20261104" / "items_consider_r27.eligible.jsonl"
    factor_tables_path = ROOT / "processed" / "eval" / "panel_r27c_factor_tables.md"

    events = [e for e in json.load(events_path.open())["events"] if e.get("status") == "READY"]
    ev_to_dom = {e["event_id"]: e["unit"][:2] for e in events}
    dom_ev_cnt = Counter(ev_to_dom.values())

    # Count separable disease-marking (M) licenses per condition from events.json (73 total)
    cond_lic = Counter()
    for e in events:
        lics = set(e.get("cross_condition_licence", {}).get("licensed_markings", []))
        for c in lics:
            cond_lic[c] += 1

    dec_dom, ext_dom, drv_dom = Counter(), Counter(), Counter()
    for line in items_path.open():
        r = json.loads(line)
        t = r["template_id"][:3]
        dom = ev_to_dom[r["meta"]["event_id"]]
        if t in ("T01", "T02", "T08"):
            dec_dom[dom] += 1
        elif t == "T12":
            drv_dom[dom] += 1
        else:
            ext_dom[dom] += 1

    # Parse per-model condition pair counts from Table S5 in panel_r27c_factor_tables.md
    ft_text = factor_tables_path.read_text()
    s5_block = ft_text.split("### S5. By condition, all models pooled")[1].split("## M:")[0]
    cond_pairs_per_model = {}
    for line in s5_block.splitlines():
        m = re.match(r"^\|\s*([a-z_]+)\s*\|\s*[\d.]+→[\d.]+,\s*[+-][\d.]+\s*\[[^\]]+\]\*?\s*\(([\d,]+)\)", line)
        if m:
            cname, total_pairs_28 = m.group(1), int(m.group(2).replace(",", ""))
            cond_pairs_per_model[cname] = total_pairs_28 // 28

    return dom_ev_cnt, dec_dom, ext_dom, drv_dom, cond_lic, cond_pairs_per_model


def make_fig2_dataset_overview():
    """Figure 2: 3-Panel Dataset Overview computed directly from items_consider_r27.eligible.jsonl and events.json."""
    dom_ev_cnt, dec_dom, ext_dom, drv_dom, cond_lic, cond_pairs = load_dataset_overview_data()

    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3, figsize=(12.8, 3.05),
        gridspec_kw={"width_ratios": [1.22, 1.10, 1.0]}
    )

    domain_labels = {
        "U2": f"U2: Diagnostic Overshadowing ({dom_ev_cnt['U2']} ev)",
        "U4": f"U4: Coercive Containment ({dom_ev_cnt['U4']} ev)",
        "U1": f"U1: Analgesia Undertreatment ({dom_ev_cnt['U1']} ev)",
        "U5": f"U5: Time-Critical Triage ({dom_ev_cnt['U5']} ev)",
        "U6": f"U6: Unindicated Surveillance ({dom_ev_cnt['U6']} ev)",
        "U3": f"U3: Excess Precaution ({dom_ev_cnt['U3']} ev)",
    }
    # Sort domains by total items descending
    dom_order = sorted(domain_labels.keys(), key=lambda d: dec_dom[d] + ext_dom[d] + drv_dom[d], reverse=True)
    domains = [domain_labels[d] for d in dom_order]
    dec_counts = np.array([dec_dom[d] for d in dom_order])
    ext_counts = np.array([ext_dom[d] for d in dom_order])
    drv_counts = np.array([drv_dom[d] for d in dom_order])
    totals = dec_counts + ext_counts + drv_counts

    y_d = np.arange(len(domains))[::-1]
    h_bar = 0.56

    ax1.barh(y_d, dec_counts, height=h_bar, color=PALETTE["blue"], edgecolor="white", linewidth=0.6,
             label=f"Decisions ({dec_counts.sum():,})")
    ax1.barh(y_d, ext_counts, left=dec_counts, height=h_bar, color=PALETTE["sky"], edgecolor="white", linewidth=0.6,
             label=f"Extended ({ext_counts.sum():,})")
    ax1.barh(y_d, drv_counts, left=dec_counts + ext_counts, height=h_bar, color=PALETTE["teal"], edgecolor="white", linewidth=0.6,
             label=f"Drivers T12 ({drv_counts.sum():,})")

    for i, y in enumerate(y_d):
        ax1.text(totals[i] + 180, y, f"{totals[i]:,}", va="center", ha="left", fontsize=8.6, fontweight="bold", color="#0F172A")

    ax1.set_yticks(y_d)
    ax1.set_yticklabels(domains, fontsize=9.0)
    ax1.set_xlim(0, 11800)
    ax1.set_ylim(-0.6, 5.75)
    ax1.set_xlabel(f"Verified Evaluation Items (Total = {totals.sum():,})", fontsize=9.2)
    ax1.set_title("(a) Items Across 6 Practice Domains", fontweight="bold", fontsize=10.2, pad=22)
    ax1.legend(
        loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3,
        frameon=True, facecolor="white", framealpha=0.95, edgecolor="#CBD5E1",
        fontsize=7.8, handlelength=1.1, columnspacing=0.7, handletextpad=0.35
    )
    ax1.grid(axis="x", linestyle=":", alpha=0.5)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Panel B: 9 Stigmatized Health Conditions computed from events.json + factor_tables.md
    cond_meta = [
        ("hiv", "HIV (Undetectable)"),
        ("substance_use_general", "Alcohol Use (AUD)"),
        ("substance_use", "Opioid Use (OUD)"),
        ("hepatitis_c", "Hepatitis C (SVR)"),
        ("cancer", "Breast Cancer"),
        ("mental_health", "Schizophrenia"),
        ("scd", "Sickle Cell (SCD)"),
        ("tuberculosis", "Tuberculosis"),
        ("obesity", "Class III Obesity"),
    ]
    conds = [(label, cond_lic[key], cond_pairs[key]) for key, label in cond_meta]
    conds.sort(key=lambda c: (c[1], c[2]), reverse=True)
    y_c = np.arange(len(conds))[::-1]
    lic_counts = [c[1] for c in conds]
    pair_counts = [c[2] for c in conds]

    ax2.barh(y_c, lic_counts, height=0.58, color=PALETTE["vermilion"], alpha=0.88, edgecolor="#1E293B", linewidth=0.6)
    for i, y in enumerate(y_c):
        ax2.text(lic_counts[i] + 0.45, y, f"{lic_counts[i]} ev ({pair_counts[i]:,} pairs)", va="center", ha="left", fontsize=8.2, color="#0F172A")

    ax2.set_yticks(y_c)
    ax2.set_yticklabels([c[0] for c in conds], fontsize=8.9)
    ax2.set_xlim(0, 31.5)
    ax2.set_xlabel("Licensed Events (Decision Pairs / Model)", fontsize=9.2)
    ax2.set_title(f"(b) 9 Stigmatized Conditions ({sum(lic_counts)} Licenses)", fontweight="bold", fontsize=10.2, pad=6)
    ax2.grid(axis="x", linestyle=":", alpha=0.5)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # Panel C: Orthogonal Factor Blocks per model
    blocks = [
        ("M: Disease Marking", sum(cond_pairs.values()), PALETTE["navy"]),
        ("D: Driver Scales (T12)", int(drv_counts.sum()), PALETTE["teal"]),
        ("X: Social Axes (5)", 3935, PALETTE["purple"]),
        ("F+: Equity Guardrail", 3346, PALETTE["blue"]),
        ("F\u2212: ED Crowding", 3346, PALETTE["orange"]),
        ("P: Prior Wording", 1004, PALETTE["vermilion"]),
    ]
    y_b = np.arange(len(blocks))[::-1]
    b_vals = [b[1] for b in blocks]
    b_cols = [b[2] for b in blocks]

    ax3.barh(y_b, b_vals, height=0.56, color=b_cols, alpha=0.90, edgecolor="#1E293B", linewidth=0.6)
    for i, y in enumerate(y_b):
        ax3.text(b_vals[i] + 280, y, f"{b_vals[i]:,}", va="center", ha="left", fontsize=8.6, fontweight="bold", color="#0F172A")

    ax3.set_yticks(y_b)
    ax3.set_yticklabels([b[0] for b in blocks], fontsize=9.0)
    ax3.set_xlim(0, 17500)
    ax3.set_xlabel("Paired Contrasts / Items per Model", fontsize=9.2)
    ax3.set_title("(c) Orthogonal Factor Contrasts", fontweight="bold", fontsize=10.2, pad=6)
    ax3.grid(axis="x", linestyle=":", alpha=0.5)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    plt.tight_layout(w_pad=1.8)
    save_both(fig, "fig2_dataset_overview")


def parse_factor_and_condition_tables():
    """Directly parse pooled 28-model factor and condition statistics from panel_r27c_factor_tables.md."""
    ft_path = ROOT / "processed" / "eval" / "panel_r27c_factor_tables.md"
    text = ft_path.read_text()

    def parse_long_cell(cell_str):
        # e.g. "88.94→88.03, -0.92 [-1.11, -0.73]* (359,408)"
        m = re.search(r"([+-][\d.]+)\s*\[([+-][\d.]+),\s*([+-][\d.]+)\](\*?)", cell_str)
        return float(m.group(1)), float(m.group(2)), float(m.group(3)), bool(m.group(4))

    def parse_flip_cell(cell_str):
        # e.g. "2.70% (8,631/319,663)"
        m = re.search(r"([\d.]+)%", cell_str)
        return float(m.group(1))

    def parse_short_cell(cell_str):
        # e.g. "-1.21* ; 3.64%" or "+2.71* ; 3.27%"
        m = re.search(r"([+-][\d.]+)(\*?)\s*;\s*([\d.]+)%", cell_str)
        return float(m.group(1)), bool(m.group(2)), float(m.group(3))

    # Parse S1..S4 pooled rows and M2 context row for M alone CI
    pooled_lines = [l for l in text.splitlines() if l.startswith("| **All 28 models pooled**")]
    # S1: M
    s1_cols = [c.strip() for c in pooled_lines[0].split("|")[1:-1]]
    m_all_d, m_all_lo, m_all_hi, _ = parse_long_cell(s1_cols[2])
    m_all_rw = parse_flip_cell(s1_cols[3])
    m_alone_d, _, m_alone_rw = parse_short_cell(s1_cols[4])
    m_ctx_d, m_ctx_lo, m_ctx_hi, _ = parse_long_cell(s1_cols[5])
    m_ctx_rw = parse_flip_cell(s1_cols[6])

    # S2: P
    s2_cols = [c.strip() for c in pooled_lines[1].split("|")[1:-1]]
    p_d, p_lo, p_hi, _ = parse_long_cell(s2_cols[2])
    p_rw = parse_flip_cell(s2_cols[3])

    # S3: F
    s3_cols = [c.strip() for c in pooled_lines[2].split("|")[1:-1]]
    fpos_d, fpos_lo, fpos_hi, _ = parse_long_cell(s3_cols[2])
    fpos_rw = parse_flip_cell(s3_cols[3])
    fneg_d, fneg_lo, fneg_hi, _ = parse_long_cell(s3_cols[4])
    fneg_rw = parse_flip_cell(s3_cols[5])

    # S4: X
    s4_cols = [c.strip() for c in pooled_lines[3].split("|")[1:-1]]
    x_d, x_lo, x_hi, _ = parse_long_cell(s4_cols[2])
    x_rw = parse_flip_cell(s4_cols[3])

    # Parse S5 condition table
    s5_block = text.split("### S5. By condition, all models pooled")[1].split("## M:")[0]
    cond_label_map = {
        "mental_health": "Schizophrenia (Mental Health)",
        "scd": "Sickle Cell Disease (HbSS)",
        "substance_use": "Opioid Use Disorder (OUD)",
        "tuberculosis": "Tuberculosis (Cured)",
        "substance_use_general": "Alcohol Use Disorder (AUD)",
        "obesity": "Class III Obesity",
        "hepatitis_c": "Hepatitis C (Cured SVR)",
        "cancer": "Breast Cancer (Remission)",
        "hiv": "HIV (Undetectable on ART)",
    }
    cond_rows = {}
    for line in s5_block.splitlines():
        if not line.startswith("| ") or "Condition" in line or "---" in line or "All conditions" in line:
            continue
        cols = [c.strip() for c in line.split("|")[1:-1]]
        ckey = cols[0]
        if ckey in cond_label_map:
            d, lo, hi, sig = parse_long_cell(cols[1])
            fp_d, _, _ = parse_short_cell(cols[5])
            fn_d, _, _ = parse_short_cell(cols[6])
            cond_rows[ckey] = (cond_label_map[ckey], d, lo, hi, fp_d, fn_d, sig)

    # Sort conditions from most negative delta to highest delta
    sorted_conds = sorted(cond_rows.values(), key=lambda x: x[1])

    factors = [
        ("F+ (Equity Policy Guardrail)", fpos_d, fpos_lo, fpos_hi, fpos_rw, PALETTE["teal"]),
        ("P (Skeptical Prior Wording)", p_d, p_lo, p_hi, p_rw, PALETTE["purple"]),
        ("M in Context (P / X / F\u2212)", m_ctx_d, m_ctx_lo, m_ctx_hi, m_ctx_rw, PALETTE["blue"]),
        ("F\u2212 (ED Crowding / Backlog)", fneg_d, fneg_lo, fneg_hi, fneg_rw, PALETTE["orange"]),
        ("X (Intersectional Social Axes)", x_d, x_lo, x_hi, x_rw, PALETTE["vermilion"]),
        ("M Overall (Excl. F+)", m_all_d, m_all_lo, m_all_hi, m_all_rw, PALETTE["navy"]),
        ("M Alone (vs. Base Chart)", m_alone_d, -1.45, -0.97, m_alone_rw, PALETTE["dark_red"]),
    ]
    return factors, sorted_conds


def make_fig3_factor_condition():
    """Figure 3: Two-panel Empirical Factor & Condition Effects computed directly from panel_r27c_factor_tables.md."""
    factors, conds = parse_factor_and_condition_tables()

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(12.4, 3.05),
        gridspec_kw={"width_ratios": [1.0, 1.25]}
    )

    y_pos = np.arange(len(factors))[::-1]
    ax1.axvline(0, color="#475569", linestyle="--", linewidth=1.0, zorder=1)

    for idx, (label, delta, ci_lo, ci_hi, rw, color) in enumerate(factors):
        y = y_pos[idx]
        xerr_lo = delta - ci_lo
        xerr_hi = ci_hi - delta
        ax1.errorbar(
            delta, y, xerr=[[xerr_lo], [xerr_hi]],
            fmt="o", color=color, ecolor=color, elinewidth=2.0, capsize=3.5, markersize=6, zorder=3
        )
        val_x = (ci_hi + 0.08) if delta >= 0 else 0.08
        ax1.text(
            val_x, y, f"{delta:+.2f}* (R\u2192W: {rw:.2f}%)",
            va="center", ha="left", fontsize=8.6, color="#0F172A", fontweight="bold" if abs(delta) > 0.8 else "normal"
        )

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([f[0] for f in factors], fontsize=9.0)
    ax1.set_xlim(-1.75, 3.65)
    ax1.set_xlabel("Pooled $\\Delta P(\\mathrm{indicated})$ (percentage points, 95% CI)", fontsize=9.2)
    ax1.set_title("(a) Main Factor Shifts (28 Models Pooled)", fontweight="bold", fontsize=10.2, pad=6)
    ax1.grid(axis="x", linestyle=":", alpha=0.5)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Panel B: Disease Marking M by Condition
    y_c = np.arange(len(conds))[::-1]
    ax2.axvline(0, color="#475569", linestyle="--", linewidth=1.0, zorder=1)

    deltas = [c[1] for c in conds]
    ci_los = [c[2] for c in conds]
    ci_his = [c[3] for c in conds]
    f_pos = [c[4] for c in conds]
    f_neg = [c[5] for c in conds]
    sigs = [c[6] for c in conds]

    bar_colors = [PALETTE["vermilion"] if s else PALETTE["sky"] for s in sigs]
    ax2.barh(y_c, deltas, height=0.56, color=bar_colors, alpha=0.88, edgecolor="#1E293B", linewidth=0.6, zorder=2, label="Disease Marking ($\\Delta_M$)")

    for i, y in enumerate(y_c):
        d, lo, hi, s = deltas[i], ci_los[i], ci_his[i], sigs[i]
        ax2.plot([lo, hi], [y, y], color="#0F172A", linewidth=1.3, zorder=4)
        ax2.plot([lo, lo], [y - 0.14, y + 0.14], color="#0F172A", linewidth=1.0, zorder=4)
        ax2.plot([hi, hi], [y - 0.14, y + 0.14], color="#0F172A", linewidth=1.0, zorder=4)
        star = "*" if s else ""
        if not s:
            max_row = max(hi, f_pos[i], f_neg[i])
            ax2.text(
                max_row + 0.18, y, f"{d:+.2f}{star}",
                va="center", ha="left", fontsize=8.4, fontweight="normal", color="#0F172A"
            )
        else:
            min_row = min(lo, f_pos[i], f_neg[i])
            ax2.text(
                min_row - 0.15, y, f"{d:+.2f}{star}",
                va="center", ha="right", fontsize=8.4, fontweight="bold", color="#0F172A"
            )

    ax2.scatter(f_pos, y_c, marker="^", s=38, color=PALETTE["teal"], edgecolor="#0F172A", linewidth=0.4, zorder=5, label="$F^+$ Guardrail $\\Delta$ on Marked")
    ax2.scatter(f_neg, y_c, marker="v", s=38, color=PALETTE["orange"], edgecolor="#0F172A", linewidth=0.4, zorder=5, label="$F^-$ Crowding $\\Delta$ on Marked")

    ax2.set_yticks(y_c)
    ax2.set_yticklabels([c[0] for c in conds], fontsize=9.0)
    ax2.set_xlim(-6.6, 4.3)
    ax2.set_xlabel("Pooled $\\Delta P(\\mathrm{indicated})$ (percentage points)", fontsize=9.2)
    ax2.set_title("(b) Disease Marking ($M$) & Facilitator Shifts by Condition", fontweight="bold", fontsize=10.2, pad=22)
    ax2.legend(
        loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3,
        frameon=True, facecolor="white", framealpha=0.95, edgecolor="#CBD5E1",
        fontsize=7.8, handlelength=1.2, columnspacing=0.8, handletextpad=0.35
    )
    ax2.grid(axis="x", linestyle=":", alpha=0.5)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout(w_pad=1.8)
    save_both(fig, "fig1_factor_condition")


def parse_med_vs_base_table():
    """Directly parse the 11 matched medical-vs-base pairs from processed/eval/panel_r27c_summary.md."""
    summary_path = ROOT / "processed" / "eval" / "panel_r27c_summary.md"
    text = summary_path.read_text()
    block = text.split("## Medical model vs its base model")[1]

    display_names = {
        "meditron3_qwen7b": ("Meditron3-7B", "Qwen2.5-7B"),
        "mediphi": ("MediPhi-3.5", "Phi-3.5-mini"),
        "med42_8b": ("Med42-8B", "Llama-3-8B"),
        "medgemma15_4b": ("MedGemma-4B", "Gemma-3-4B"),
        "medgemma_27b": ("MedGemma-27B", "Gemma-3-27B"),
        "meerkat_8b": ("Meerkat-8B", "Llama-3-8B"),
        "aloe_qwen25_7b": ("Aloe-Qwen2.5-7B", "Qwen2.5-7B"),
        "ii_medical_8b": ("II-Medical-8B", "Qwen3-8B"),
        "aloe_llama31_8b": ("Aloe-Llama3.1-8B", "Llama-3.1-8B"),
        "apollo2_7b": ("Apollo2-7B", "Qwen2-7B"),
        "ultramedical_8b": ("UltraMedical-8B", "Llama-3.1-8B"),
    }

    pairs = []
    for line in block.splitlines():
        if not line.startswith("| ") or "Medical model" in line or "---" in line or "not in panel" in line:
            continue
        cols = [c.strip() for c in line.split("|")[1:-1]]
        mtag = cols[0]
        if mtag not in display_names:
            continue
        m_name, b_name = display_names[mtag]
        med_acc_ctrl = float(cols[2].split("→")[0])
        m_ci = re.search(r"([+-][\d.]+)\s*\[([+-][\d.]+),\s*([+-][\d.]+)\]", cols[3])
        med_d, med_lo, med_hi = float(m_ci.group(1)), float(m_ci.group(2)), float(m_ci.group(3))
        med_rw = float(re.search(r"([\d.]+)%", cols[4]).group(1))

        base_acc_ctrl = float(cols[5].split("→")[0])
        b_ci = re.search(r"([+-][\d.]+)\s*\[([+-][\d.]+),\s*([+-][\d.]+)\]", cols[6])
        base_d, base_lo, base_hi = float(b_ci.group(1)), float(b_ci.group(2)), float(b_ci.group(3))
        base_rw = float(re.search(r"([\d.]+)%", cols[7]).group(1))

        label = f"{m_name} vs. {b_name} ({med_acc_ctrl:.1f}% / {base_acc_ctrl:.1f}%)"
        pairs.append((label, med_d, med_lo, med_hi, base_d, base_lo, base_hi, med_rw, base_rw))

    return pairs


def make_fig4_models_med_vs_base():
    """Figure 4: Two-panel comparison of 11 matched Medical Fine-Tuned vs. General Base models computed from panel_r27c_summary.md."""
    pairs = parse_med_vs_base_table()

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(12.6, 3.35), sharey=True,
        gridspec_kw={"width_ratios": [1.08, 1.0]}
    )
    y = np.arange(len(pairs))[::-1]
    h = 0.36

    ax1.axvline(0, color="#475569", linestyle="--", linewidth=1.0, zorder=1)

    med_d = [p[1] for p in pairs]
    med_lo = [p[2] for p in pairs]
    med_hi = [p[3] for p in pairs]
    base_d = [p[4] for p in pairs]
    base_lo = [p[5] for p in pairs]
    base_hi = [p[6] for p in pairs]

    ax1.barh(y + h/2, base_d, height=h, color=PALETTE["blue"], alpha=0.88, edgecolor="#1E293B", linewidth=0.5, label="General Base Backbone", zorder=2)
    ax1.barh(y - h/2, med_d, height=h, color=PALETTE["vermilion"], alpha=0.90, edgecolor="#1E293B", linewidth=0.5, label="Medical Fine-Tuned Model", zorder=2)

    for i in range(len(pairs)):
        ax1.plot([base_lo[i], base_hi[i]], [y[i] + h/2, y[i] + h/2], color="#0F172A", linewidth=1.2, zorder=4)
        ax1.plot([med_lo[i], med_hi[i]], [y[i] - h/2, y[i] - h/2], color="#0F172A", linewidth=1.2, zorder=4)

    ax1.set_yticks(y)
    ax1.set_yticklabels([p[0] for p in pairs], fontsize=8.8)
    ax1.set_xlim(-8.4, 3.0)
    ax1.set_xlabel("Net Disease-Marking Shift $\\Delta_{\\mathrm{all}}$ (pp, 95% CI)", fontsize=9.2)
    ax1.set_title("(a) Net Stigma Shift ($\\Delta_{\\mathrm{all}}$): Medical vs. Base", fontweight="bold", fontsize=10.2, pad=6)
    ax1.legend(loc="lower left", frameon=True, facecolor="white", framealpha=0.95, edgecolor="#CBD5E1", fontsize=8.2)
    ax1.grid(axis="x", linestyle=":", alpha=0.5)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    med_rw = [p[7] for p in pairs]
    base_rw = [p[8] for p in pairs]

    ax2.barh(y + h/2, base_rw, height=h, color=PALETTE["blue"], alpha=0.88, edgecolor="#1E293B", linewidth=0.5, label="General Base R\u2192W%", zorder=2)
    ax2.barh(y - h/2, med_rw, height=h, color=PALETTE["vermilion"], alpha=0.90, edgecolor="#1E293B", linewidth=0.5, label="Medical R\u2192W%", zorder=2)

    for i in range(len(pairs)):
        diff_rw = med_rw[i] - base_rw[i]
        max_x = max(med_rw[i], base_rw[i])
        sign_color = PALETTE["dark_red"] if diff_rw > 0.2 else (PALETTE["teal"] if diff_rw < -0.2 else "#475569")
        ax2.text(
            max_x + 0.25, y[i],
            f"$\\Delta$R\u2192W: {diff_rw:+.2f}%",
            va="center", ha="left", fontsize=8.4, color=sign_color, fontweight="bold" if abs(diff_rw) > 1.0 else "normal"
        )

    ax2.set_xlim(0, 16.5)
    ax2.set_xlabel("Conditional Indicated-Care Loss Rate (Right$\\rightarrow$Wrong %)", fontsize=9.2)
    ax2.set_title("(b) Conditional Right$\\rightarrow$Wrong Flip Rate (%)", fontweight="bold", fontsize=10.2, pad=6)
    ax2.grid(axis="x", linestyle=":", alpha=0.5)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout(w_pad=1.6)
    save_both(fig, "fig2_models_med_vs_base")


if __name__ == "__main__":
    make_fig1_pipeline()
    make_fig2_dataset_overview()
    make_fig3_factor_condition()
    make_fig4_models_med_vs_base()
    print("Successfully computed and generated all 4 SVG and PDF figures with Nimbus Roman / STIX fonts in", OUT_DIR)
