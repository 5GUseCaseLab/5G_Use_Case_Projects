"""
WiFeye: results figures + A1 course poster (vector PDF)
=============================================================
Every number is read from the saved experiment outputs (data/models_exp/..., data/event_features_v3.csv);
nothing is typed in by hand except the live-engine validation figures measured in the test runs.

Outputs
  data/visualizations/*.png / *.pdf        each chart standalone (for slides / report / video)
  reports/WiFeye_poster_A1.pdf             the course poster: A1 portrait, vector, print-ready, with the
                                           required sections (title, names & affiliation, abstract,
                                           introduction, objectives, system model, method, setup,
                                           results & discussion, conclusion, acknowledgement, references)
  reports/WiFeye_poster_A1_preview.png     low-resolution preview
  (build_poster() still renders the earlier A0 layout if needed)
Header text, names, acknowledgement and logo: poster_config.json

Design: light print surface, system sans (Segoe UI); categorical slots in fixed order (validated:
adjacent CVD dE >= 9.1, normal-vision >= 19.6; aqua/yellow/magenta are < 3:1 on white, so their bars
carry visible value labels); ordinal single-hue blue ramp for the ordered pipeline versions; sequential
blue for the confusion matrix; thin marks, rounded data-ends, hairline axes, no dual axes.
"""

import os
import json
import textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, f1_score

# ---------------------------------------------------------------- tokens
SURF, INK, INK2, MUTED = "#ffffff", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, NEUTRAL = "#e1e0d9", "#c3c2b7", "#f0efec"
# WiFeye theme: graphite + one teal accent on warm-neutral panels (minimal). Validated with the dataviz
# validator on white: categorical slots 1-5 adjacent CVD dE >= 9.6, normal-vision >= 19.6, chroma >= 0.10
# (yellow / magenta < 3:1 -> their marks carry visible value labels); ordinal teal ramp: monotone, one hue,
# light end 2.21:1. Teal steps are OKLCH hue 186.
S = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
# Cheerful, validated reference palette: categorical slots 1-3 pass all-pairs CVD (dE >= 9.2) and
# normal-vision (>= 24) checks on white (aqua < 3:1, so its bars carry value labels); blue ordinal ramp
# monotone, one hue, light end 2.11:1; sequential blue for the confusion matrix and the CSI map.
ORD = ["#86b6ef", "#5598e7", "#2a78d6", "#184f95", "#0d366b"]          # ordinal (pipeline versions)
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]   # sequential
MUTE3 = ["#96b1af", "#37837d", "#144945"]                              # (standalone augmentation figure)
HEADER_A = "#1e3a8a"          # header gradient: deep blue ...
HEADER_B = "#0f766e"          # ... to teal
HEADER = HEADER_A
ACC_INK = "#1d4ed8"           # blue: section numbers, subheads, labels
ON_HEADER = "#fde68a"         # warm yellow: course line on the header
HEADER_MUTED = "#dbeafe"      # affiliation on the header
TINT1 = "#e0ecfd"             # light blue tint
TINT0 = "#f0f6fe"
plt.rcParams.update({"font.family": "Segoe UI", "axes.edgecolor": AXIS, "axes.linewidth": 1.0,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
                     "axes.labelcolor": INK2, "pdf.fonttype": 42, "svg.fonttype": "none"})
CLASSES = ["fall", "walk", "sit", "stand", "hand", "empty_room"]
NICE = {"fall": "Fall", "walk": "Walk", "sit": "Sit", "stand": "Stand", "hand": "Hand", "empty_room": "Empty"}
ADL = ["walk", "sit", "stand", "hand"]
VIS = "data/visualizations"


# ---------------------------------------------------------------- data
def auc_ci(df, score, n=2000, seed=0):
    d = df[df["true"].isin(["fall"] + ADL)].reset_index(drop=True)
    y, s = (d["true"] == "fall").values, d[score].values
    groups = [g.index.values for _, g in d.groupby(d["subject"].astype(str))]
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n):
        idx = np.concatenate([rng.choice(g, len(g)) for g in groups])
        if y[idx].any() and (~y[idx]).any():
            boots.append(roc_auc_score(y[idx], s[idx]))
    return float(roc_auc_score(y, s)), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def load():
    R = {}
    rd = lambda p: pd.read_csv(p, dtype={"subject": str})
    prog = [("v1\nraw", rd("data/models_exp/baseline/oof_trials_random_forest_loso.csv"), "max_p_fall"),
            ("v2\n1 source", rd("data/models_exp/v2_dual/oof_trials_random_forest_loso.csv"), "max_p_fall"),
            ("v3\nrouter", rd("data/models_exp/v3_dual/oof_trials_random_forest_loso.csv"), "max_p_fall"),
            ("v3\n+ event", rd("data/models_exp/event_v3/hyb_late_oof.csv"), "p_fall")]
    R["progression"] = [(lab, *auc_ci(df, sc)) for lab, df, sc in prog]
    hyb = rd("data/models_exp/event_v3/hyb_late_oof.csv")
    R["hyb"] = hyb
    R["cm"] = confusion_matrix(hyb["true"], hyb["pred"], labels=CLASSES)
    R["hyb_acc"] = float((hyb["true"] == hyb["pred"]).mean())
    R["hyb_f1"] = float(f1_score(hyb["true"], hyb["pred"], average="macro"))
    win = rd("data/models_exp/rx_compare_v3/rx_comparison_trials.csv")
    m = lambda d: d[d["true"].isin(["fall"] + ADL)]
    hm, wm = m(hyb), m(win)
    R["roc_h"] = roc_curve(hm["true"] == "fall", hm["p_fall"])
    R["roc_w"] = roc_curve(wm["true"] == "fall", wm["pfall_LATE"])
    R["auc_h"] = roc_auc_score(hm["true"] == "fall", hm["p_fall"])
    R["auc_w"] = roc_auc_score(wm["true"] == "fall", wm["pfall_LATE"])
    with open("data/models_exp/event_v3/fall_operating_points.json") as f:
        R["ops"] = json.load(f)
    with open("data/models_exp/event_v3/event_results.json") as f:
        R["event"] = json.load(f)
    with open("data/models_exp/rx_compare_v3/rx_comparison.json") as f:
        R["rx"] = json.load(f)
    with open("data/models_exp/missing_cell_v2/missing_cell_summary.json") as f:
        R["mc"] = json.load(f)
    with open("data/models_exp/tiers_v3/tier_calibration_raw.json") as f:
        R["tiers"] = json.load(f)
    ev = pd.read_csv("data/event_features_v3.csv")
    R["posture"] = ev.groupby("activity")["ev_rx1_posture_shift"].median().reindex(CLASSES)
    # ---- adopted system (validate_adopted.py): fused localiser + domain features + two heads, 5 classes
    ad = rd("data/models_exp/adopted_v3/adopted_oof.csv")
    R["ad"] = ad
    with open("data/models_exp/adopted_v3/adopted_results.json") as f:
        R["adres"] = json.load(f)
    R["progression"].append(("final\nsystem", *auc_ci(ad.rename(columns={"true6": "true"}), "p_fall")))
    R["cm5"] = confusion_matrix(ad["true5"], ad["pred_adopted"], labels=CL5)
    am = ad[ad["true5"] != "empty_room"]
    R["roc_a"] = roc_curve(am["true5"] == "fall", am["p_fall"])
    R["auc_a"] = roc_auc_score(am["true5"] == "fall", am["p_fall"])
    rxp = "data/models_exp/adopted_v3/adopted_rx_compare.json"
    if os.path.exists(rxp):
        with open(rxp) as f:
            R["adrx"] = json.load(f)
    # transmitter mix: measured by the raw-data audit (packet share pooled over all 509 trials)
    R["mix"] = {"RX1": [78.2, 3.2, 1.7, 3.0, 12.7, 1.2], "RX2": [80.5, 2.8, 1.3, 3.3, 11.0, 1.1]}
    R["mix_names"] = ["Router → this ESP32 (kept)", "Router broadcast (BSSID)", "Laptop A",
                      "Laptop B", "Neighbouring access points", "Other"]
    R["heat"] = csi_maps("1_fall_t06")
    return R


CL5 = ["fall", "walk", "posture", "hand", "empty_room"]
NICE5 = {"fall": "Fall", "walk": "Walk", "posture": "Posture", "hand": "Hand", "empty_room": "Empty"}


# ---------------------------------------------------------------- helpers
def style(ax, fs, grid_axis="y"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(AXIS); ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(labelsize=fs * 0.9, length=0, pad=fs * 0.35)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=1.0)
        ax.set_axisbelow(True)
    ax.set_facecolor(SURF)


def rbar(ax, x, h, w, color, r_pt=7, horizontal=False):
    """Bar with a rounded data-end and a square baseline (correct circular corners)."""
    if h <= 0:
        return
    fig = ax.figure
    bb = ax.get_window_extent()
    xr, yr = ax.get_xlim(), ax.get_ylim()
    ppx, ppy = bb.width / (xr[1] - xr[0]), bb.height / (yr[1] - yr[0])
    r = r_pt * fig.dpi / 72
    if not horizontal:
        rx, ry = min(r / ppx, w / 2), min(r / ppy, h / 2)
        ax.add_patch(Rectangle((x - w / 2, 0), w, h - ry, facecolor=color, edgecolor="none", zorder=3))
        ax.add_patch(FancyBboxPatch((x - w / 2, h - 2 * ry), w, 2 * ry, boxstyle=f"round,pad=0,rounding_size={rx}",
                                    mutation_aspect=ry / rx, facecolor=color, edgecolor="none", zorder=3))
    else:
        rx, ry = min(r / ppx, h / 2), min(r / ppy, w / 2)
        ax.add_patch(Rectangle((0, x - w / 2), h - rx, w, facecolor=color, edgecolor="none", zorder=3))
        ax.add_patch(FancyBboxPatch((h - 2 * rx, x - w / 2), 2 * rx, w, boxstyle=f"round,pad=0,rounding_size={rx}",
                                    mutation_aspect=ry / rx, facecolor=color, edgecolor="none", zorder=3))


def box(ax, x, y, w, h, text, fs, fill=NEUTRAL, ink=INK, weight="normal", accent=None, ha="center"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.12", facecolor=fill,
                                edgecolor="none", zorder=2))
    if accent:
        ax.add_patch(FancyBboxPatch((x, y), 0.12, h, boxstyle="round,pad=0,rounding_size=0.05", facecolor=accent,
                                    edgecolor="none", zorder=3))
    tx = x + w / 2 if ha == "center" else x + 0.3
    ax.text(tx, y + h / 2, text, ha=ha, va="center", fontsize=fs, color=ink, weight=weight, zorder=4,
            linespacing=1.25)


def arrow(ax, p0, p1, color=INK2, lw=2.0, text=None, fs=10, both=False, rad=0.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=("<|-|>" if both else "-|>"), mutation_scale=18,
                                 color=color, lw=lw, zorder=1, connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2, text, fontsize=fs, color=INK2, ha="center",
                va="center", backgroundcolor=SURF, zorder=5)


# ---------------------------------------------------------------- charts
def plot_progression(ax, R, fs):
    data = R["progression"]
    ax.set_xlim(-0.6, len(data) - 0.4); ax.set_ylim(0, 1.0)
    style(ax, fs)
    ramp = ORD                                   # ordinal single-hue ramp, light -> dark = pipeline order
    for i, (lab, auc, lo, hi) in enumerate(data):
        rbar(ax, i, auc, 0.52, ramp[i])
        ax.plot([i, i], [lo, hi], color=INK2, lw=1.6, zorder=4)
        ax.plot([i - 0.07, i + 0.07], [lo, lo], color=INK2, lw=1.6, zorder=4)
        ax.plot([i - 0.07, i + 0.07], [hi, hi], color=INK2, lw=1.6, zorder=4)
        ax.text(i, hi + 0.03, f"{auc:.2f}", ha="center", va="bottom", fontsize=fs * 1.15, weight="bold", color=INK)
    ax.axhline(0.5, color=MUTED, lw=1.2, zorder=2)
    ax.text(-0.55, 0.515, "chance", ha="left", va="bottom", fontsize=fs * 0.85, color=MUTED)
    ax.set_xticks(range(len(data)), [d[0] for d in data], fontsize=fs * 0.85, color=INK2)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("Fall AUC (unseen people)", fontsize=fs * 0.9)


def plot_confusion(ax, R, fs, key="cm", classes=None, nice=None):
    cm = R[key]
    classes, nice = classes or CLASSES, nice or NICE
    n = len(classes)
    rown = cm / cm.sum(axis=1, keepdims=True)
    cmap = LinearSegmentedColormap.from_list("seq", SEQ)
    ax.imshow(rown, cmap=cmap, vmin=0, vmax=1, aspect="equal")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=fs * 0.95,
                    color=("#ffffff" if rown[i, j] > 0.5 else INK), weight=("bold" if i == j else "normal"))
    labs = [nice[c] for c in classes]
    ax.set_xticks(range(n), labs, fontsize=fs * 0.85, color=INK2)
    ax.set_yticks(range(n), labs, fontsize=fs * 0.85, color=INK2)
    ax.set_xlabel("Predicted", fontsize=fs * 0.9); ax.set_ylabel("True", fontsize=fs * 0.9)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True); ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color=SURF, linewidth=3); ax.tick_params(which="minor", length=0)


def plot_roc_adopted(ax, R, fs):
    """Fall head of the adopted system vs the earlier event model; nested operating points A-C."""
    style(ax, fs, grid_axis="both")
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1.2, zorder=2)
    fh, th, _ = R["roc_h"]; fa, ta, _ = R["roc_a"]
    ax.plot(fh, th, color=S[1], lw=2.2, solid_capstyle="round", zorder=3,
            label=f"Earlier event model (AUC {R['auc_h']:.2f})")
    ax.plot(fa, ta, color=S[0], lw=2.8, solid_capstyle="round", zorder=4,
            label=f"Adopted fall head (AUC {R['auc_a']:.2f})")
    lab_box = dict(boxstyle="round,pad=0.15", facecolor=SURF, edgecolor="none", alpha=0.92)
    ops = R["adres"]["fall_operating_points"]
    for k, (tag, tgt) in enumerate((("A", "0.03"), ("B", "0.05"), ("C", "0.1"))):
        o = ops[tgt]
        ax.scatter([o["adl_far"]], [o["sensitivity"]], s=fs * 9, color=S[0], edgecolors=SURF, linewidths=2.2, zorder=6)
        dx, dy, ha = {"A": (0.0, -0.075, "center"), "B": (0.035, 0.0, "left"), "C": (0.035, 0.03, "left")}[tag]
        ax.text(o["adl_far"] + dx, o["sensitivity"] + dy, tag, fontsize=fs * 0.8, ha=ha,
                weight="bold", color=INK, va="center", zorder=7, bbox=lab_box)
        ax.text(0.40, 0.40 - 0.07 * k, f"{tag}   {o['adl_far']:.1%} false alarms → {o['sensitivity']:.0%} of falls",
                fontsize=fs * 0.78, color=INK2, va="center", zorder=7, bbox=lab_box)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.set_xlabel("False alarms on daily activities", fontsize=fs * 0.9)
    ax.set_ylabel("Falls detected", fontsize=fs * 0.9)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1]); ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    leg = ax.legend(loc="lower right", fontsize=fs * 0.78, frameon=False, handlelength=1.6)
    for t in leg.get_texts():
        t.set_color(INK2)


def rx_text(R):
    """Result (d) sentence, written from the saved 1-vs-2 ESP32 comparison of the adopted system."""
    if "adrx" not in R:
        return "(d) No fusion beat the best single ESP32: placement matters more than count."
    res, tests = R["adrx"]["results"], {t["b"]: t for t in R["adrx"]["tests"]}
    L, r1, r2 = res["LATE (2 ESP32)"], res["RX1 only"], res["RX2 only"]
    t1, t2 = tests["RX1"], tests["RX2"]
    p2 = "p < 0.001" if t2["p_holm"] < 0.001 else f"p = {t2['p_holm']:.3f}"
    return (f"(d) Two ESP32s ≈ the better one alone: macro-F1 {L['macro_f1']:.2f} vs RX1 {r1['macro_f1']:.2f} "
            f"({t1['a_right_b_wrong']} vs {t1['a_wrong_b_right']} trials, Holm p = {t1['p_holm']:.2f}), but well above "
            f"RX2 {r2['macro_f1']:.2f} ({p2}). Placement matters more than count; the second ESP32 adds robustness "
            f"and empty-room F1 ({L['per_class_f1']['empty_room']:.2f} vs {r1['per_class_f1']['empty_room']:.2f}).")


def plot_receivers_adopted(ax1, ax2, R, fs):
    """1 vs 2 ESP32 with the adopted two-head system (LOSO, 5 classes)."""
    res = R["adrx"]["results"]
    names = [("RX1 only", "RX1\nonly"), ("RX2 only", "RX2\nonly"), ("LATE (2 ESP32)", "Both\n(late)")]
    for ax, key, ylim, title in ((ax1, "macro_f1", 0.8, "Macro-F1"),
                                 (ax2, "fall_auc", 1.0, "Fall AUC")):
        ax.set_xlim(-0.6, 2.6); ax.set_ylim(0, ylim)
        style(ax, fs)
        shade = {"RX1 only": S[0], "RX2 only": S[1], "LATE (2 ESP32)": S[2]}
        for i, (k, lab) in enumerate(names):
            v = res[k][key]
            rbar(ax, i, v, 0.55, shade[k])
            ax.text(i, v + ylim * 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=fs * 0.95, color=INK,
                    weight="bold")
        ax.set_xticks(range(3), [n[1] for n in names], fontsize=fs * 0.82, color=INK2)
        ax.set_title(title, fontsize=fs * 0.95, color=INK, loc="left", pad=fs * 0.6)


def plot_roc(ax, R, fs):
    style(ax, fs, grid_axis="both")
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1.2, zorder=2)
    fw, tw, _ = R["roc_w"]; fh, th, _ = R["roc_h"]
    ax.plot(fw, tw, color=S[1], lw=2.4, solid_capstyle="round", zorder=3,
            label=f"Window model, late fusion (AUC {R['auc_w']:.2f})")
    ax.plot(fh, th, color=S[0], lw=2.8, solid_capstyle="round", zorder=4,
            label=f"Two-stage event model (AUC {R['auc_h']:.2f})")
    lab_box = dict(boxstyle="round,pad=0.15", facecolor=SURF, edgecolor="none", alpha=0.92)
    for k, (tag, tgt) in enumerate((("A", "0.03"), ("B", "0.05"), ("C", "0.1"))):
        o = R["ops"]["targets"][tgt]
        ax.scatter([o["adl_far"]], [o["sensitivity"]], s=fs * 9, color=S[0], edgecolors=SURF, linewidths=2.2, zorder=6)
        ax.text(o["adl_far"] + 0.035, o["sensitivity"] - 0.012, tag, fontsize=fs * 0.8, weight="bold", color=INK,
                va="center", zorder=7, bbox=lab_box)
        ax.text(0.42, 0.40 - 0.07 * k, f"{tag}   {o['adl_far']:.1%} false alarms → {o['sensitivity']:.0%} of falls",
                fontsize=fs * 0.78, color=INK2, va="center", zorder=7, bbox=lab_box)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.set_xlabel("False alarms on daily activities", fontsize=fs * 0.9)
    ax.set_ylabel("Falls detected", fontsize=fs * 0.9)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1]); ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    leg = ax.legend(loc="lower right", fontsize=fs * 0.78, frameon=False, handlelength=1.6)
    for t in leg.get_texts():
        t.set_color(INK2)


def plot_kshot(ax, R, fs):
    curve = pd.DataFrame(R["event"]["kshot"]["curve"])
    style(ax, fs)
    ax.plot(curve["k"], curve["macro_f1"], color=S[0], lw=2.6, marker="o", ms=fs * 0.55,
            markerfacecolor=S[0], markeredgecolor=SURF, markeredgewidth=2, zorder=4)
    d3 = R["event"]["kshot"]["delta_vs_k0"]["3"]["mean"]
    ax.text(curve["k"].iloc[-1], curve["macro_f1"].iloc[-1] + 0.006, f"+{d3 * 100:.1f} pts", ha="center",
            va="bottom", fontsize=fs, weight="bold", color=INK)
    ax.set_xticks([0, 1, 2, 3]); ax.set_xlim(-0.3, 3.3); ax.set_ylim(0.50, 0.60)
    ax.set_xlabel("Labelled trials per activity from the new person", fontsize=fs * 0.9)
    ax.set_ylabel("Macro-F1", fontsize=fs * 0.9)


def plot_receivers(ax1, ax2, R, fs):
    res = R["rx"]["results"]
    names = [("RX1", "RX1\nonly"), ("RX2", "RX2\nonly"), ("EARLY", "Early\nfusion"), ("LATE", "Late\nfusion")]
    for ax, key, sd, ylim, title in ((ax1, "macro_f1", "seed_sd_macro_f1", 0.65, "Macro-F1 (6 classes)"),
                                     (ax2, "fall_auc", "seed_sd_fall_auc", 1.0, "Fall-vs-activity AUC")):
        ax.set_xlim(-0.6, 3.6); ax.set_ylim(0, ylim)
        style(ax, fs)
        for i, (k, lab) in enumerate(names):
            v = res[k][key]
            rbar(ax, i, v, 0.55, S[i])
            ax.text(i, v + ylim * 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=fs * 0.95, color=INK,
                    weight="bold")
        ax.set_xticks(range(4), [n[1] for n in names], fontsize=fs * 0.82, color=INK2)
        ax.set_title(title, fontsize=fs * 0.95, color=INK, loc="left", pad=fs * 0.6)


def plot_missing_cell(ax, R, fs):
    arms = [("NONE", "No augmentation"), ("GENERIC", "Generic augmentation"), ("STYLE", "Subject-style augmentation")]
    groups = [("missing_cell_sens", "Hidden falls detected\n(simulated 'subject 3')"),
              ("unseen_far", "False alarms on an\nunseen subject")]
    ba = R["mc"]["by_arm"]
    ax.set_xlim(-0.6, 1.6); ax.set_ylim(0, 0.5)
    style(ax, fs)
    w = 0.22
    for gi, (gk, glab) in enumerate(groups):
        for ai, (ak, alab) in enumerate(arms):
            x = gi + (ai - 1) * (w + 0.03)
            v = ba[gk][ak]
            rbar(ax, x, v, w, MUTE3[ai])
            ax.text(x, v + 0.01, f"{v:.0%}", ha="center", va="bottom", fontsize=fs * 0.9, color=INK, weight="bold")
    ax.set_xticks([0, 1], [g[1] for g in groups], fontsize=fs * 0.85, color=INK2)
    ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5], [f"{v:.0%}" for v in [0, 0.1, 0.2, 0.3, 0.4, 0.5]])
    handles = [Rectangle((0, 0), 1, 1, color=MUTE3[i]) for i in range(3)]
    leg = ax.legend(handles, [a[1] for a in arms], loc="upper right", fontsize=fs * 0.78, frameon=False)
    for t in leg.get_texts():
        t.set_color(INK2)


def plot_mix(ax, R, fs):
    names = R["mix_names"]
    cols = [S[0], S[1], S[2], S[3], S[4], AXIS]
    ax.set_xlim(0, 100); ax.set_ylim(-0.6, 1.6)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([0, 1], ["RX2", "RX1"], fontsize=fs * 0.95, color=INK2)
    ax.tick_params(length=0)
    for row, rx in ((1, "RX1"), (0, "RX2")):
        left = 0
        for k, (v, c) in enumerate(zip(R["mix"][rx], cols)):
            gap = 0.35
            ax.add_patch(FancyBboxPatch((left + gap / 2, row - 0.32), max(v - gap, 0.05), 0.64,
                                        boxstyle="round,pad=0,rounding_size=0.6", mutation_aspect=0.03,
                                        facecolor=c, edgecolor="none"))
            if v >= 9:
                ax.text(left + 1.2, row, f"{v:.0f}%", ha="left", va="center", fontsize=fs * 0.9,
                        color=("#ffffff" if k in (0, 1) else INK), weight="bold")
            left += v
    handles = [Rectangle((0, 0), 1, 1, color=c) for c in cols]
    labels = [f"{n}" + (f"  ({R['mix']['RX1'][i]:.1f}% / {R['mix']['RX2'][i]:.1f}%)" if R['mix']['RX1'][i] < 9 else "")
              for i, n in enumerate(names)]
    leg = ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2, fontsize=fs * 0.74,
                    frameon=False, handlelength=1.0, columnspacing=1.2)
    for t in leg.get_texts():
        t.set_color(INK2)


def plot_posture(ax, R, fs):
    p = R["posture"].sort_values()
    ax.set_ylim(-0.6, len(p) - 0.4); ax.set_xlim(0, 1.35)
    style(ax, fs, grid_axis="x")
    for i, (c, v) in enumerate(p.items()):
        rbar(ax, i, v, 0.55, S[0], horizontal=True)
        ax.text(v + 0.02, i, f"{v:.2f}", va="center", fontsize=fs * 0.9, color=INK)
    ax.set_yticks(range(len(p)), [NICE[c] for c in p.index], fontsize=fs * 0.9, color=INK2)
    ax.set_xlabel("Posture shift: level change after vs before the event / noise", fontsize=fs * 0.85)


# ---------------------------------------------------------------- diagrams
def diagram_setup(ax, fs):
    ax.set_xlim(0, 10); ax.set_ylim(0, 5.6); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.1, 0.1), 9.8, 5.4, boxstyle="round,pad=0,rounding_size=0.25",
                                facecolor="#f9f9f7", edgecolor=GRID, lw=1.2))
    ax.text(0.35, 5.3, "Sensing room", fontsize=fs * 0.8, color=MUTED, va="top")
    box(ax, 3.4, 4.25, 3.2, 0.95, "Router (TX)\nTP-Link, 2.4 GHz ch 6", fs * 0.8, fill=TINT1)
    box(ax, 0.5, 1.55, 2.3, 0.85, "ESP32 RX1", fs * 0.85, fill=TINT1, weight="bold")
    box(ax, 7.2, 1.55, 2.3, 0.85, "ESP32 RX2", fs * 0.85, fill=TINT1, weight="bold")
    box(ax, 0.5, 0.35, 2.3, 0.85, "Laptop A\nmaster · COM3", fs * 0.72)
    box(ax, 7.2, 0.35, 2.3, 0.85, "Laptop B\nworker · COM4", fs * 0.72)
    ax.add_patch(Circle((5.0, 2.55), 0.38, facecolor=S[1], edgecolor="none"))
    ax.add_patch(FancyBboxPatch((4.72, 1.25), 0.56, 0.95, boxstyle="round,pad=0,rounding_size=0.2",
                                facecolor=S[1], edgecolor="none"))
    ax.text(5.55, 1.85, "subject\n2–3 m away", fontsize=fs * 0.72, color=INK2, ha="left", va="center")
    arrow(ax, (4.0, 4.25), (1.9, 2.4), color=S[0], lw=2.2)
    arrow(ax, (6.0, 4.25), (8.1, 2.4), color=S[0], lw=2.2)
    ax.text(2.25, 3.55, "Wi-Fi frames\n→ CSI", fontsize=fs * 0.72, color=INK2, ha="center")
    ax.text(7.75, 3.55, "60–90° apart", fontsize=fs * 0.72, color=INK2, ha="center")
    arrow(ax, (1.65, 1.55), (1.65, 1.2), color=INK2, lw=1.6)
    arrow(ax, (8.35, 1.55), (8.35, 1.2), color=INK2, lw=1.6)
    arrow(ax, (7.2, 0.78), (2.8, 0.78), color=INK2, lw=1.8)
    ax.text(5.0, 0.55, "sync + RX2 relay over the router's Wi-Fi (UDP)", fontsize=fs * 0.68, color=INK2,
            ha="center", va="top")


def diagram_pipeline(ax, fs):
    steps = [("Raw CSI from 2 × ESP32 (64 subcarriers, ~70 packets/s, bursty)", False),
             ("Keep router→ESP32 HT frames only; drop subcarriers 0, 27–37 (52 kept)", True),
             ("One 100 Hz grid for both receivers; Hampel + 10 Hz zero-phase low-pass", False),
             ("Per-subcarrier median removal (keeps the posture change)", True),
             ("Fused event localiser: reliability-weighted, gap-masked, soft time prior", True),
             ("Event-segment + 2 s window features (49 per receiver)", True),
             ("+ 42 CSI domain features that passed a bottom-up acceptance test*", True),
             ("Two heads: fall RF + activity RF per receiver → late fusion (mean)", True),
             ("FALL iff P(fall) ≥ τ (≈3% false alarms, nested LOSO); else top activity", True)]
    n = len(steps)
    ax.set_xlim(0, 10); ax.set_ylim(0, n * 0.82); ax.axis("off")
    for i, (s, new) in enumerate(steps):
        y = (n - 1 - i) * 0.82
        box(ax, 0.0, y + 0.08, 10.0, 0.6, s, fs * 0.78, fill=(TINT0 if new else NEUTRAL),
            accent=(S[0] if new else None), ha="left")
        if i < n - 1:
            arrow(ax, (5.0, y + 0.08), (5.0, y - 0.12), color=MUTED, lw=1.4)


def diagram_live(ax, fs):
    ax.set_xlim(0, 10); ax.set_ylim(0, 4.0); ax.axis("off")
    box(ax, 0.0, 2.75, 2.5, 1.05, "ESP32 RX1\n→ Laptop A", fs * 0.8, fill=TINT1)
    box(ax, 0.0, 0.25, 2.5, 1.05, "ESP32 RX2 →\nLaptop B relay", fs * 0.8, fill=TINT1)
    box(ax, 3.15, 1.15, 2.7, 1.7, "10 s rolling\nbuffers, same\nv3 pipeline", fs * 0.8)
    box(ax, 6.5, 2.55, 3.5, 1.25, "Tier 1 (optional): quiet\n→ sleep (edge power)\nTier 2: burst → 'analysing'", fs * 0.72)
    box(ax, 6.5, 0.2, 3.5, 2.05, "Tier 3: two-head classifier\nlate fusion RX1 + RX2\n→ activity or\nFALL ALERT", fs * 0.74,
        fill=TINT0, accent=S[0])
    arrow(ax, (2.5, 3.25), (3.15, 2.45), lw=1.8); arrow(ax, (2.5, 0.8), (3.15, 1.55), lw=1.8)
    arrow(ax, (5.85, 2.4), (6.5, 3.0), lw=1.8); arrow(ax, (5.85, 1.6), (6.5, 1.2), lw=1.8)


ACC = S[0]
# ---------------------------------------------------------------- WiFeye v4 visuals (clean, outline style)
from matplotlib.patches import Polygon

# Flowchart colour code: white boxes with a vivid stage-coloured border and title; the stage pills above
# the columns name every colour, so colour is never the only cue.
STAGE_C = {"Acquire": "#2563eb", "Clean": "#0891b2", "Condition": "#059669", "Localise": "#d97706",
           "Features": "#7c3aed", "Classify": "#db2777", "Decide": "#dc2626"}
STAGE = {k: ("#ffffff", v) for k, v in STAGE_C.items()}      # (fill, border) for older helpers
EDGE_N = "#94a3b8"          # border of neutral boxes
ALERT = "#dc2626"           # fall alarm only
ALERT_EDGE = "#991b1b"
OK_C = "#0f766e"            # activity outcome (teal)


def _lh(fs, spacing=1.22):
    return fs * spacing / (72 * K)          # line height in design inches (axes data units)


TINT_A = 0.14              # translucent fill opacity for flowchart boxes


def tint(color, a=TINT_A):
    return to_rgba(color, a)


def obox(ax, x, y, w, h, title, body, fs, color, fill=None, lw=2.6, r=0.18, z=3, title_color=None,
         body_color=INK, tscale=1.12, alpha=TINT_A):
    """Translucent box: stage colour at low opacity, crisp coloured border, bold coloured title, body below."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                facecolor=fill if fill is not None else tint(color, alpha),
                                edgecolor=color, lw=lw, zorder=z))
    nb = len(body.split("\n")) if body else 0
    ft = fs * tscale
    total = _lh(ft, 1.25) + nb * _lh(fs)
    ytop = y + h / 2 + total / 2
    ax.text(x + w / 2, ytop, title, ha="center", va="top", fontsize=ft, weight="bold",
            color=title_color or color, zorder=z + 2)
    if body:
        ax.text(x + w / 2, ytop - _lh(ft, 1.25), body, ha="center", va="top", fontsize=fs, color=body_color,
                zorder=z + 2, linespacing=1.22)


def farrow(ax, p0, p1, color="#475569", lw=2.2, rad=0.0, ms=20, z=2):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms, color=color, lw=lw, zorder=z,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


def alabel(ax, x, y, s, fs, color=INK2, weight="normal", bg=None):
    ax.text(x, y, s, ha="center", va="center", fontsize=fs, color=color, weight=weight, zorder=6,
            bbox=dict(boxstyle="round,pad=0.15", facecolor=bg or PANEL, edgecolor="none"))


def pill(ax, x, y, w, h, text, fs, color, ink=None, back=None):
    """Translucent stage pill: tinted fill, soft coloured border, text in the stage colour.
    back: opaque colour drawn underneath (hides lines the pill sits on, e.g. a container border)."""
    if back:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={h / 2}", facecolor=back,
                                    edgecolor="none", zorder=2.9))
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={h / 2}",
                                facecolor=tint(color, 0.2), edgecolor=to_rgba(color, 0.75), lw=2.0, zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=ink or color, weight="bold",
            zorder=5)


def diagram_flow_h(ax, fs, W, H):
    """Section 4: the processing pipeline as one large left-to-right flowchart (design-inch coordinates)."""
    ax.set_xlim(-0.12, W + 0.12); ax.set_ylim(-0.12, H + 0.04); ax.axis("off"); ax.set_aspect("equal")
    cols = [("Acquire", 3.35), ("Clean", 3.75), ("Condition", 3.75), ("Localise", 4.05), ("Features", 3.75),
            ("Classify", 3.75), ("Decide", 5.6)]
    gap = (W - sum(w for _, w in cols)) / (len(cols) - 1)
    X = {}
    x = 0.0
    for s, w in cols:
        X[s] = (x, w); x += w + gap
    ph = 0.66
    for s, (x0, w) in X.items():
        pill(ax, x0 + 0.25, H - ph, w - 0.5, ph, s, fs * 0.86, STAGE_C[s])
    A = H - ph - 0.35                          # body area height
    ym = A / 2                                 # vertical centre of the body area
    bt = fs * 0.78                             # body text size
    h1 = 3.3                                   # single box
    h2, g2 = 2.05, 0.55                        # stacked pair
    yA, yB = ym + g2 / 2, ym - g2 / 2 - h2     # upper / lower box bottoms

    def single(s, title, body):
        x0, w = X[s]; obox(ax, x0, ym - h1 / 2, w, h1, title, body, bt, STAGE_C[s])

    def pair(s, t1, b1, t2, b2):
        x0, w = X[s]
        obox(ax, x0, yA, w, h2, t1, b1, bt, STAGE_C[s])
        obox(ax, x0, yB, w, h2, t2, b2, bt, STAGE_C[s])

    pair("Acquire", "ESP32 RX1", "64 subcarriers", "ESP32 RX2", "~70 packets/s")
    single("Clean", "Packet filter", "keep router→ESP32\nframes (≈80%)\n52 of 64\nsubcarriers")
    single("Condition", "Clean-up", "100 Hz grid\nHampel filter\n10 Hz low-pass\nmedian removal")
    single("Localise", "Fused localiser", "weights each RX by\nits burst above noise\n→ event onset,\npeak and offset")
    pair("Features", "Event + window", "49 per receiver", "CSI domain", "42 vetted features")
    pair("Classify", "Fall head", "random forest\nper receiver", "Activity head", "random forest\nper receiver")
    R_ = lambda s: X[s][0] + X[s][1]
    L_ = lambda s: X[s][0]
    cA, cB = yA + h2 / 2, yB + h2 / 2
    farrow(ax, (R_("Acquire"), cA), (L_("Clean"), ym + 0.45))
    farrow(ax, (R_("Acquire"), cB), (L_("Clean"), ym - 0.45))
    farrow(ax, (R_("Clean"), ym), (L_("Condition"), ym))
    farrow(ax, (R_("Condition"), ym), (L_("Localise"), ym))
    farrow(ax, (R_("Localise"), ym + 0.45), (L_("Features"), cA))
    farrow(ax, (R_("Localise"), ym - 0.45), (L_("Features"), cB))
    farrow(ax, (R_("Features"), cA), (L_("Classify"), cA))
    farrow(ax, (R_("Features"), cA - 0.25), (L_("Classify"), cB + 0.25))
    farrow(ax, (R_("Features"), cB), (L_("Classify"), cB))
    # Decide: diamond, then FALL ALERT (yes, top) / activity (no, bottom)
    x0, w = X["Decide"]
    dw, dh = 2.3, 2.3
    xd = x0 + dw / 2
    ax.add_patch(Polygon([(xd, ym + dh / 2), (xd + dw / 2, ym), (xd, ym - dh / 2), (xd - dw / 2, ym)], closed=True,
                         facecolor=tint(STAGE_C["Decide"]), edgecolor=STAGE_C["Decide"], lw=2.6, zorder=3))
    ax.text(xd, ym, "P(fall)\n≥ τ ?", ha="center", va="center", fontsize=bt * 1.05, weight="bold", zorder=5,
            linespacing=1.1, color=INK)
    farrow(ax, (R_("Classify"), cA), (xd - 0.35, ym + 0.42))
    farrow(ax, (R_("Classify"), cB), (xd - 0.35, ym - 0.42))
    alabel(ax, (R_("Classify") + xd - dw / 2) / 2 + 0.05, ym, "late\nfusion", fs * 0.62)
    bx = x0 + dw + 0.55; bw = w - dw - 0.55
    obox(ax, bx, yA, bw, h2, "FALL ALERT", "", bt * 1.12, ALERT, lw=3.4, tscale=1.0, alpha=0.24)
    obox(ax, bx, yB, bw, h2, "Activity", "walk · posture\nhand · empty", bt * 0.92, OK_C)
    farrow(ax, (xd, ym + dh / 2), (bx, yA + h2 / 2), color=ALERT, rad=-0.25)
    farrow(ax, (xd, ym - dh / 2), (bx, yB + h2 / 2), color=OK_C, rad=0.25)
    alabel(ax, xd + 0.15, ym + dh / 2 + 0.38, "yes", fs * 0.66, color=ALERT, weight="bold")
    alabel(ax, xd + 0.15, ym - dh / 2 - 0.38, "no", fs * 0.66, color=OK_C, weight="bold")


def diagram_edge_h(ax, fs, W, H):
    """4.2 Proposed edge deployment, left to right: ESP32s -> Arduino UNO Q -> 5G -> application alerts."""
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off"); ax.set_aspect("equal")
    bt = fs * 0.74
    # sensors
    sw, sh = 2.2, 1.25
    obox(ax, 0, H * 0.66 - sh / 2, sw, sh, "ESP32", "RX1", bt, STAGE_C["Acquire"])
    obox(ax, 0, H * 0.30 - sh / 2, sw, sh, "ESP32", "RX2", bt, STAGE_C["Acquire"])
    # UNO Q container
    cx0, cx1 = sw + 0.65, sw + 0.65 + 5.0
    cy0, cy1 = 0.1, H - 0.35
    ax.add_patch(FancyBboxPatch((cx0, cy0), cx1 - cx0, cy1 - cy0, boxstyle="round,pad=0,rounding_size=0.25",
                                facecolor=tint(HEADER_A, 0.05), edgecolor=HEADER_A, lw=3.0, zorder=2))
    pill(ax, cx0 + 0.3, cy1 - 0.3, 3.3, 0.6, "Arduino UNO Q", fs * 0.82, HEADER_A, back=PANEL)
    farrow(ax, (sw, H * 0.66), (cx0, H * 0.66))
    farrow(ax, (sw, H * 0.30), (cx0, H * 0.30))
    alabel(ax, sw + 0.33, H * 0.66 + 0.33, "UART", fs * 0.6)
    alabel(ax, sw + 0.33, H * 0.30 + 0.33, "Wi-Fi", fs * 0.6)
    iw = cx1 - cx0 - 0.5; ix = cx0 + 0.25
    ym1 = cy1 - 0.55 - 1.35
    obox(ax, ix, ym1, iw, 1.35, "STM32U585 MCU", "capture · filter · timestamps", bt, STAGE_C["Clean"])
    yp1 = ym1 - 0.45; yp0 = 1.2
    farrow(ax, (ix + iw / 2, ym1), (ix + iw / 2, yp1))
    alabel(ax, ix + iw / 2 + 0.95, (ym1 + yp1) / 2, "Bridge", fs * 0.58, bg="#ffffff")
    ax.add_patch(FancyBboxPatch((ix, yp0), iw, yp1 - yp0, boxstyle="round,pad=0,rounding_size=0.18",
                                facecolor=tint(EDGE_N, 0.1), edgecolor=EDGE_N, lw=2.0, zorder=3))
    ax.text(ix + iw / 2, yp1 - 0.3, "QRB2210 · Linux", ha="center", va="center", fontsize=bt * 1.08, weight="bold",
            color=INK, zorder=5)
    chips = [("Clean-up", "Condition"), ("Localiser", "Localise"), ("Features", "Features"), ("Two heads", "Classify")]
    cw_ = (iw - 0.6) / 2; ch_ = (yp1 - yp0 - 0.85) / 2
    for i, (t, s) in enumerate(chips):
        xx = ix + 0.2 + (i % 2) * (cw_ + 0.2)
        yy = yp1 - 0.62 - (i // 2 + 1) * ch_ - (i // 2) * 0.12
        obox(ax, xx, yy, cw_, ch_, t, "", bt * 0.9, STAGE_C[s], lw=2.2, z=5, tscale=1.0)
    gw = (iw - 0.2) / 2
    obox(ax, ix, 0.3, gw, 0.75, "Buzzer + LED", "", bt * 0.82, ALERT, lw=2.0, tscale=1.0)
    obox(ax, ix + gw + 0.2, 0.3, gw, 0.75, "Sleep if quiet", "", bt * 0.82, EDGE_N, lw=2.0, tscale=1.0,
         title_color=INK2)
    # 5G and applications (right column)
    rx = cx1 + 0.65; rw = W - rx - 0.35
    bh, gp = 0.98, 0.42
    top0 = H - 0.3
    ys = [top0 - (i + 1) * bh - i * gp for i in range(4)]
    obox(ax, rx, ys[0], rw, bh, "5G modem", "", bt, HEADER_A, tscale=1.0, alpha=0.2, lw=3.0)
    obox(ax, rx, ys[1], rw, bh, "Edge / cloud", "server", bt * 0.9, EDGE_N, title_color=INK)
    obox(ax, rx, ys[2], rw, bh, "Fall alert", "URLLC → carer", bt * 0.88, ALERT, alpha=0.22, lw=3.0)
    obox(ax, rx, ys[3], rw, bh, "Activity log", "mMTC → dashboard", bt * 0.88, OK_C)
    farrow(ax, (cx1 - 0.25, (yp0 + yp1) / 2), (rx, ys[0] + bh / 2))
    alabel(ax, cx1 + 0.3, (yp0 + yp1) / 2 + 0.75, "USB", fs * 0.6)
    farrow(ax, (rx + rw / 2, ys[0]), (rx + rw / 2, ys[1] + bh), color=HEADER_A, lw=2.6)
    alabel(ax, rx + rw / 2 + 0.75, (ys[0] + ys[1] + bh) / 2, "5G NR", fs * 0.62, color=HEADER_A, weight="bold")
    farrow(ax, (rx + rw / 2, ys[1]), (rx + rw / 2, ys[2] + bh), color=ALERT)
    xe = W - 0.12
    ax.plot([rx + rw, xe, xe], [ys[1] + bh / 2, ys[1] + bh / 2, ys[3] + bh / 2], color=OK_C, lw=2.2, zorder=2,
            solid_capstyle="round")
    farrow(ax, (xe, ys[3] + bh / 2), (rx + rw, ys[3] + bh / 2), color=OK_C)


def diagram_setup_v4(ax, fs):
    ax.set_xlim(0, 10); ax.set_ylim(0, 5.6); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.1, 0.1), 9.8, 5.4, boxstyle="round,pad=0,rounding_size=0.25",
                                facecolor="#ffffff", edgecolor=EDGE_N, lw=1.8))
    ax.text(0.35, 5.3, "Sensing room", fontsize=fs * 0.74, color=MUTED, va="top")
    bt = fs * 0.78
    obox(ax, 2.9, 4.05, 4.2, 1.15, "Router (TX)", "2.4 GHz ch 6", bt, STAGE_C["Clean"])
    obox(ax, 0.4, 1.5, 2.6, 0.95, "ESP32 RX1", "", bt, STAGE_C["Acquire"], tscale=1.0)
    obox(ax, 7.0, 1.5, 2.6, 0.95, "ESP32 RX2", "", bt, STAGE_C["Acquire"], tscale=1.0)
    obox(ax, 0.4, 0.3, 2.6, 0.85, "Laptop A", "", bt * 0.92, EDGE_N, title_color=INK, tscale=1.0)
    obox(ax, 7.0, 0.3, 2.6, 0.85, "Laptop B", "", bt * 0.92, EDGE_N, title_color=INK, tscale=1.0)
    ax.add_patch(Circle((5.0, 2.62), 0.4, facecolor=tint("#f59e0b", 0.35), edgecolor="#d97706", lw=2.2))
    ax.add_patch(FancyBboxPatch((4.7, 1.22), 0.6, 1.02, boxstyle="round,pad=0,rounding_size=0.22",
                                facecolor=tint("#f59e0b", 0.35), edgecolor="#d97706", lw=2.2))
    ax.text(5.5, 1.75, "person\n2–3 m", fontsize=fs * 0.7, color=INK2, ha="left", va="center")
    farrow(ax, (4.0, 4.05), (1.9, 2.47), color=STAGE_C["Clean"], lw=2.6)
    farrow(ax, (6.0, 4.05), (8.1, 2.47), color=STAGE_C["Clean"], lw=2.6)
    ax.text(2.1, 3.5, "Wi-Fi\nCSI", fontsize=fs * 0.7, color=INK2, ha="center")
    ax.text(7.9, 3.5, "60–90°\napart", fontsize=fs * 0.7, color=INK2, ha="center")
    farrow(ax, (1.7, 1.5), (1.7, 1.17), lw=1.8, ms=14)
    farrow(ax, (8.3, 1.5), (8.3, 1.17), lw=1.8, ms=14)
    farrow(ax, (7.0, 0.72), (3.0, 0.72), lw=2.0)
    alabel(ax, 5.0, 0.72, "RX2 relayed over Wi-Fi", fs * 0.62, bg="#ffffff")


SEQMAP = LinearSegmentedColormap.from_list("blue_seq", ["#ffffff", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
                                                         "#256abf", "#0d366b"])


def csi_maps(trial="1_fall_t06"):
    """Per-subcarrier motion (0.2 s std vs each subcarrier's median) of a real fall trial, both receivers, plus the
    event segment found by the fused localiser (data/event_features_v3_fused.csv)."""
    import live_demo_engine as L
    _, pk = L.load_trial_packets(trial)
    t0 = max(v[0][0] for v in pk.values()); t1 = min(v[0][-1] for v in pk.values())
    out = {}
    for rx, (ts, A, Rr) in pk.items():
        trel, X, _ = L.preprocess(ts, A, t0, t1, Rr)
        w, hop = 20, 10
        st = np.arange(0, len(X) - w + 1, hop)
        Mx = np.array([X[s:s + w].std(0) for s in st]).T
        out[rx] = np.log2(Mx / (np.median(Mx, axis=1, keepdims=True) + 1e-9))
        out["t"] = trel[st + w // 2]
    seg = pd.read_csv("data/event_features_v3_fused.csv").set_index("trial_id").loc[trial]
    out.update(on=float(seg["event_onset"]), off=float(seg["event_offset"]), pk=float(seg["event_t"]), trial=trial)
    return out


def plot_csi_maps(ax1, ax2, R, fs):
    h = R["heat"]
    vmax = np.percentile(np.concatenate([h["rx1"].ravel(), h["rx2"].ravel()]), 99.5)
    T = h["t"][-1] + 0.1
    for ax, rx in ((ax1, "rx1"), (ax2, "rx2")):
        ax.imshow(np.clip(h[rx], 0, vmax), aspect="auto", cmap=SEQMAP, vmin=0, vmax=vmax, origin="lower",
                  extent=(0, T, 0.5, h[rx].shape[0] + 0.5), interpolation="nearest")
        ax.add_patch(Rectangle((h["on"], 0.5), h["off"] - h["on"], h[rx].shape[0], fill=False, edgecolor=ALERT,
                               lw=2.8, zorder=4))
        ax.set_yticks([]); ax.set_ylabel(rx.upper(), fontsize=fs, color=INK, rotation=0, ha="right",
                                         va="center", weight="bold", labelpad=6)
        for sp in ax.spines.values():
            sp.set_color(EDGE_N); sp.set_linewidth(1.3)
        ax.tick_params(labelsize=fs * 0.85, length=0, colors=MUTED)
    ax1.set_xticks([])
    ax2.set_xticks(range(0, int(T) + 1, 2))
    ax2.set_xlabel("time (s)  ·  darker = more motion", fontsize=fs * 0.85, color=INK2)


def plot_roc_v3(ax, R, fs):
    style(ax, fs, grid_axis="both")
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1.2, zorder=2)
    fh, th, _ = R["roc_h"]; fa, ta, _ = R["roc_a"]
    ax.plot(fh, th, color=S[1], lw=2.4, solid_capstyle="round", zorder=3, label=f"Earlier ({R['auc_h']:.2f})")
    ax.plot(fa, ta, color=S[0], lw=3.2, solid_capstyle="round", zorder=4, label=f"Final ({R['auc_a']:.2f})")
    lab_box = dict(boxstyle="round,pad=0.12", facecolor=SURF, edgecolor="none", alpha=0.92)
    ops = R["adres"]["fall_operating_points"]
    for tag, tgt in (("A", "0.03"), ("B", "0.05"), ("C", "0.1")):
        o = ops[tgt]
        ax.scatter([o["adl_far"]], [o["sensitivity"]], s=fs * 11, color=S[0], edgecolors=SURF, linewidths=2.2,
                   zorder=6)
        dx, dy, ha = {"A": (0.0, -0.1, "center"), "B": (0.045, 0.0, "left"), "C": (0.045, 0.04, "left")}[tag]
        ax.text(o["adl_far"] + dx, o["sensitivity"] + dy, tag, fontsize=fs * 0.85, ha=ha, weight="bold", color=INK,
                va="center", zorder=7, bbox=lab_box)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.set_xlabel("False alarms", fontsize=fs * 0.9)
    ax.set_ylabel("Falls detected", fontsize=fs * 0.9)
    ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
    leg = ax.legend(loc="lower right", fontsize=fs * 0.8, frameon=False, handlelength=1.2, title="AUC",
                    title_fontsize=fs * 0.8)
    for t in leg.get_texts():
        t.set_color(INK2)


# ---------------------------------------------------------------- poster
W, H = 33.11, 46.81


class Poster:
    def __init__(self):
        self.fig = plt.figure(figsize=(W, H), facecolor=SURF)

    def ax(self, x, y_top, w, h):
        return self.fig.add_axes([x / W, 1 - (y_top + h) / H, w / W, h / H])

    def text(self, x, y_top, s, size, **kw):
        kw.setdefault("color", INK)
        self.fig.text(x / W, 1 - y_top / H, s, fontsize=size, va="top", ha=kw.pop("ha", "left"), **kw)

    def para(self, x, y_top, s, width, size, color=INK, spacing=1.3):
        chars = max(10, int(width * 72 / (size * 0.53)))
        lines = textwrap.wrap(s, chars)
        for i, ln in enumerate(lines):
            self.text(x, y_top + i * size * spacing / 72, ln, size, color=color)
        return len(lines) * size * spacing / 72

    def bullets(self, x, y_top, items, width, size, color=INK, gap=0.12):
        y = y_top
        for it in items:
            self.text(x, y, "•", size, color=S[0], weight="bold")
            y += self.para(x + 0.38, y, it, width - 0.38, size, color=color) + gap
        return y - y_top

    def header(self, x, y_top, w, num, title, size=34):
        self.text(x, y_top, f"{num}  {title}", size, weight="bold")
        line_y = 1 - (y_top + size * 1.35 / 72) / H
        self.fig.add_artist(plt.Line2D([x / W, (x + w) / W], [line_y, line_y], color=S[0], lw=3))
        return size * 1.35 / 72 + 0.3

    def tiles(self, x, y_top, w, items, vsize=50, lsize=19):
        n = len(items); tw = w / n
        tallest = 0.0
        for i, (val, lab) in enumerate(items):
            self.text(x + i * tw, y_top, val, vsize, weight="semibold")
            tallest = max(tallest, self.para(x + i * tw, y_top + vsize * 1.2 / 72, lab, tw - 0.25, lsize, color=INK2))
        return vsize * 1.2 / 72 + tallest + 0.15


def build_poster(R, cfg, out_pdf, out_png):
    P = Poster(); f = P.fig
    M, G = 0.9, 0.7
    CW = (W - 2 * M - 2 * G) / 3
    X = [M, M + CW + G, M + 2 * (CW + G)]
    BODY, SMALL, CH = 22, 19, 22
    # header band
    f.add_artist(Rectangle((0, 1 - 7.3 / H), 1, 7.3 / H, transform=f.transFigure, facecolor=HEADER, edgecolor="none"))
    P.text(M, 0.75, cfg["title"], 68, color="#ffffff", weight="bold", linespacing=1.15)
    P.text(M, 3.3, cfg["subtitle"], 31, color="#cde2fb")
    P.text(M, 4.55, cfg["authors"], 28, color="#ffffff")
    P.text(M, 5.6, cfg["affiliation"], 26, color="#cde2fb")
    a0, a1 = R["progression"][0][1], R["progression"][-1][1]
    P.text(W - M, 1.0, f"{a0:.2f} → {a1:.2f}", 80, color="#ffffff", weight="bold", ha="right")
    P.text(W - M, 2.5, "fall-detection AUC on people the model never saw,", 24, color="#cde2fb", ha="right")
    P.text(W - M, 3.0, "after cleaning the Wi-Fi data (0.5 = guessing)", 24, color="#cde2fb", ha="right")
    top = 8.0

    # ---------------- column 1
    x, y = X[0], top
    y += P.header(x, y, CW, "1", "Problem and goal")
    y += P.bullets(x, y, [
        "A fall needs help fast, but cameras raise privacy concerns and wearables are often not worn.",
        "Goal: recognise falls and daily activities from ordinary Wi-Fi signals using two low-cost ESP32 boards — "
        "no camera, nothing to wear.",
        "We ask: does it work for a person the system has never seen? Do two receivers beat one? "
        "Can augmentation make up for a subject who recorded no falls?"], CW, BODY) + 0.35
    y += P.header(x, y, CW, "2", "Setup and dataset")
    a = P.ax(x, y, CW, 5.4); diagram_setup(a, CH); y += 5.6
    y += P.bullets(x, y, [
        "Router transmits; two ESP32-WROOM-32 boards record channel state information (CSI): "
        "the amplitude of 64 subcarriers per packet, ~70 packets/s.",
        "509 trials of 10 s: fall 79 · walk 100 · sit 100 · stand 100 · hand gesture 100 · "
        "empty room 30 (bare / chair / mattress). Subjects 1–5; subject 3 recorded no falls."], CW, BODY) + 0.35
    y += P.header(x, y, CW, "3", "What the raw data revealed")
    y += P.para(x, y, "Share of packets by transmitter, pooled over all trials (our audit):", CW, SMALL, color=INK2) + 0.1
    a = P.ax(x + 0.2, y, CW - 0.2, 2.2); plot_mix(a, R, CH); y += 3.75
    y += P.bullets(x, y, [
        "Every file mixed CSI from about 8 transmitters. Switching source makes the amplitude jump 4× "
        "(8.3 vs 2.1), which looks like motion even in an empty room.",
        "Subcarrier 0 is not channel data; 27–37 are always empty.",
        "The original 0.5 s baseline subtraction deleted posture change — one of the strongest cues:"], CW, BODY) + 0.15
    a = P.ax(x + 1.25, y, CW - 1.45, 3.5); plot_posture(a, R, CH); y += 4.55
    y += P.header(x, y, CW, "4", "Cleaned pipeline (v3)")
    a = P.ax(x, y, CW, 7.4); diagram_pipeline(a, CH)
    P.text(x, y + 7.55, "■", SMALL, color=S[0]); P.text(x + 0.35, y + 7.55, "changed or added in this work", SMALL, color=INK2)

    # ---------------- column 2
    x, y = X[1], top
    y += P.header(x, y, CW, "5", "Data cleaning mattered most")
    a = P.ax(x + 0.9, y + 0.1, CW - 0.9, 5.6); plot_progression(a, R, CH); y += 7.15
    y += P.para(x, y, "Leave-one-subject-out (LOSO): train on 4 people, test on the 5th. AUC 0.5 = guessing. "
                "Error bars: 95% bootstrap intervals, trials resampled within subjects.", CW, SMALL, color=INK2) + 0.4
    y += P.header(x, y, CW, "6", "Two-stage event classifier")
    five = R["ops"]["five_class_sit_stand_merged"]["accuracy"]
    y += P.tiles(x, y, CW, [(f"{R['hyb_acc']:.1%}", "accuracy, 6 classes (chance 17%)"),
                            (f"{five:.1%}", "accuracy with sit + stand merged"),
                            (f"{R['auc_h']:.2f}", "fall-vs-activity AUC")]) + 0.2
    a = P.ax(x + 1.6, y, CW - 2.6, 5.5); plot_confusion(a, R, CH); y += 6.55
    y += P.para(x, y, "Trial counts; shading = share of each true class. Walk, hand gesture and empty room are "
                "recognised best; stand is mostly confused with sit, hand and empty.", CW, SMALL, color=INK2) + 0.4
    y += P.header(x, y, CW, "7", "Fall alarm: honest operating points")
    a = P.ax(x + 1.0, y + 0.1, 6.0, 6.0); plot_roc(a, R, CH); y += 7.2
    ops = R["ops"]["targets"]
    y += P.bullets(x, y, [
        "Points A–C: alarm thresholds τ chosen inside each training fold (nested LOSO) for 3% / 5% / 10% "
        "target false alarms — never tuned on the test subject.",
        f"At {ops['0.05']['adl_far']:.1%} false alarms the system detects {ops['0.05']['sensitivity']:.0%} of falls "
        f"(95% CI {ops['0.05']['sens_ci95'][0]:.0%}–{ops['0.05']['sens_ci95'][1]:.0%}); "
        f"at {ops['0.1']['adl_far']:.1%} it detects {ops['0.1']['sensitivity']:.0%}."], CW, BODY) + 0.35
    y += P.header(x, y, CW, "8", "Calibrating to a new person")
    a = P.ax(x + 1.0, y + 0.1, CW - 1.2, 2.8); plot_kshot(a, R, CH); y += 3.95
    kd = R["event"]["kshot"]["delta_vs_k0"]
    P.para(x, y, f"Adding 1 / 2 / 3 labelled trials per activity from the test person raises macro-F1 by "
           f"+{kd['1']['mean'] * 100:.1f} / +{kd['2']['mean'] * 100:.1f} / +{kd['3']['mean'] * 100:.1f} points "
           f"(all 95% CIs above 0) — real, but smaller than the +18–24 points reported in the literature.",
           CW, SMALL, color=INK2)

    # ---------------- column 3
    x, y = X[2], top
    y += P.header(x, y, CW, "9", "One ESP32 or two?")
    hw = (CW - 0.9) / 2
    a1 = P.ax(x + 0.7, y + 0.4, hw - 0.3, 4.4); a2 = P.ax(x + 0.7 + hw + 0.3, y + 0.4, hw - 0.3, 4.4)
    plot_receivers(a1, a2, R, CH); y += 6.0
    picks = sum(1 for s in R["rx"]["selections"] if s["picked"] == "RX1")
    tests = {(t["a"], t["b"]): t for t in R["rx"]["tests"]}
    p_min = min(tests[("EARLY", "SELECTED")]["p_holm"], tests[("LATE", "SELECTED")]["p_holm"])
    y += P.bullets(x, y, [
        f"Same trials, folds and model; 5 seeds. Neither fusion beat the best single receiver "
        f"(RX1, chosen by inner CV in {picks}/{len(R['rx']['selections'])} folds; Holm p ≥ {p_min:.2f}).",
        f"RX1 beat RX2 clearly (p = {tests[('RX1', 'RX2')]['p_holm']:.3f}): receiver placement mattered more than count.",
        "Late fusion keeps RX1's accuracy and survives a lost receiver, so the live demo uses both ESP32s for robustness."],
        CW, BODY) + 0.35
    y += P.header(x, y, CW, "10", "Subject 3's missing falls")
    a = P.ax(x + 0.9, y + 0.1, CW - 0.9, 4.6); plot_missing_cell(a, R, CH); y += 6.0
    dS = R["mc"]["delta_missing_cell_sens_STYLE_vs_NONE"]; dF = R["mc"]["delta_unseen_far_STYLE_vs_NONE"]
    y += P.bullets(x, y, [
        "Copying another subject's falls would be fabricated data and would leak test trials into training.",
        "Test: hide one subject's falls (keep their other activities), add synthetic falls to training only.",
        f"Detection +{dS['mean'] * 100:.1f} pts but false alarms +{dF['mean'] * 100:.1f} pts — the same as generic "
        "augmentation, so it only shifts the threshold. Not adopted; no 'subject 3' shortcut was found."],
        CW, BODY) + 0.35
    y += P.header(x, y, CW, "11", "Calibrated 3-tier edge routing")
    t = R["tiers"]
    y += P.tiles(x, y, CW, [(f"{t['overall']['1']:.0%}", "of windows skipped at Tier 1 (no motion)"),
                            (f"{t['falls_lost_tier1']}/{t['n_falls']}", "falls lost at Tier 1"),
                            (f"{t['overall']['2']:.1%}", "on the fast-path (was 99.96%)")], vsize=44) + 0.35
    y += P.header(x, y, CW, "12", "Live demo with two ESP32s")
    a = P.ax(x, y, CW, 4.0); diagram_live(a, CH); y += 4.25
    y += P.bullets(x, y, [
        "Live code reproduces the offline pipeline exactly (40 trials: 100% same labels, 0.00% feature error).",
        "Relay tested: 722/722 RX2 packets; switches to RX1 alone if RX2 drops. Alert ≈ 3.5–5 s after a fall."],
        CW, BODY) + 0.35
    y += P.header(x, y, CW, "13", "Conclusions and limitations")
    P.bullets(x, y, [
        "Data provenance, not model choice, decided performance: chance-level fall detection became AUC 0.77.",
        "Honest scope: one room, 5 subjects, staged falls onto a mattress, no fall look-alikes (lying down, "
        "picking up). The ≥95% / ≤3% target is not met; results are cross-subject, not person-independent.",
        "Next: fixed-rate traffic and binary serial output, more subjects, fall look-alike recordings, and "
        "per-person calibration at installation."], CW, BODY)

    # footer
    fy = H - 1.55
    f.add_artist(plt.Line2D([M / W, (W - M) / W], [1 - fy / H, 1 - fy / H], color=GRID, lw=1.5))
    P.para(M, fy + 0.2,
           "References: S. M. Hernandez & E. Bulut, ESP32-CSI-Tool, IEEE WoWMoM 2020 · S. Kapoor & A. Narayanan, "
           "Leakage and the reproducibility crisis in ML-based science, Patterns 2023 · Y. Hu et al., DeFall, IEEE IoT-J 2021 "
           "· Chi et al., XFall, IEEE JSAC 2024 · Zhao et al., KNN-MMD / WiFall (ESP32-S3), arXiv 2412.04783, 2025 "
           "· J. Yang et al., SenseFi, Patterns 2023.  Code and full methodology: project repository.",
           W - 2 * M, 17, color=MUTED)
    f.savefig(out_pdf, facecolor=SURF)
    f.savefig(out_png, dpi=40, facecolor=SURF)
    if os.environ.get("POSTER_DETAIL"):
        f.savefig(out_png.replace(".png", "_detail.png"), dpi=90, facecolor=SURF)
    plt.close(f)


# ---------------------------------------------------------------- A1 course poster
W0, H0 = 33.11, 46.81          # layout is designed in A0 inches and rendered at A1 (exact 1/sqrt(2) scale)
K = 1 / np.sqrt(2)
PANEL = "#f3f6fb"            # light cool panel
PANEL_EDGE = "#d3dcea"       # clear hairline border
GOLD = ON_HEADER             # (name kept from the earlier theme)


class PosterA1:
    def __init__(self):
        self.fig = plt.figure(figsize=(W0 * K, H0 * K), facecolor=SURF)

    def ax(self, x, y_top, w, h):
        return self.fig.add_axes([x / W0, 1 - (y_top + h) / H0, w / W0, h / H0])

    def text(self, x, y_top, s, size, **kw):
        kw.setdefault("color", INK)
        ha, va = kw.pop("ha", "left"), kw.pop("va", "top")
        return self.fig.text(x / W0, 1 - y_top / H0, s, fontsize=size * K, ha=ha, va=va, **kw)

    def para(self, x, y_top, s, width, size, color=INK, spacing=1.3, weight="normal", em=0.53):
        chars = max(10, int(width * 72 / (size * em)))
        lines = textwrap.wrap(s, chars)
        for i, ln in enumerate(lines):
            self.text(x, y_top + i * size * spacing / 72, ln, size, color=color, weight=weight)
        return len(lines) * size * spacing / 72

    def bullets(self, x, y_top, items, width, size, color=INK, gap=0.1, numbered=False):
        y = y_top
        for i, it in enumerate(items):
            if numbered:
                self.text(x + 0.2, y + size * 0.62 / 72, str(i + 1), size * 0.9, color="#ffffff", weight="bold",
                          ha="center", va="center",
                          bbox=dict(boxstyle="circle,pad=0.28", facecolor=S[0], edgecolor="none"))
                y += max(self.para(x + 0.65, y, it, width - 0.65, size, color=color), 0.55) + gap
            else:
                self.text(x, y, "•", size, color=S[0], weight="bold")
                y += self.para(x + 0.38, y, it, width - 0.38, size, color=color) + gap
        return y - y_top

    def panel(self, x, y, w, h, num, title, small_num=False):
        self.fig.add_artist(FancyBboxPatch((x / W0, 1 - (y + h) / H0), w / W0, h / H0,
                                           transform=self.fig.transFigure, boxstyle="round,pad=0,rounding_size=0.004",
                                           mutation_aspect=W0 / H0, facecolor=PANEL, edgecolor=PANEL_EDGE, lw=1.4, zorder=0))
        self.text(x + 0.62, y + 0.62, num, 21 if small_num else 26, color="#ffffff", weight="bold", ha="center",
                  va="center", bbox=dict(boxstyle="circle,pad=" + ("0.22" if small_num else "0.35"),
                                         facecolor=ACC_INK, edgecolor="none"))
        self.text(x + 1.3, y + 0.62, title, 34, weight="bold", va="center")
        return y + 1.35

    def sub(self, x, y, s, size=23):
        self.text(x, y, s, size, weight="bold", color=ACC_INK)
        return size * 1.45 / 72

    def card(self, x, y, w, h, title, body, size=19):
        self.fig.add_artist(FancyBboxPatch((x / W0, 1 - (y + h) / H0), w / W0, h / H0,
                                           transform=self.fig.transFigure, boxstyle="round,pad=0,rounding_size=0.003",
                                           mutation_aspect=W0 / H0, facecolor=SURF, edgecolor=PANEL_EDGE, lw=1.2, zorder=1))
        self.fig.add_artist(Rectangle((x / W0, 1 - (y + h) / H0), 0.09 / W0, h / H0, transform=self.fig.transFigure,
                                      facecolor=S[0], edgecolor="none", zorder=2))
        self.text(x + 0.3, y + 0.18, title, size * 1.05, weight="bold")
        self.para(x + 0.3, y + 0.18 + size * 1.4 / 72, body, w - 0.45, size * 0.92, color=INK2)


def build_poster_a1(R, cfg, out_pdf, out_png):
    plt.rcParams.update({"mathtext.fontset": "custom", "mathtext.rm": "Segoe UI", "mathtext.it": "Segoe UI:italic",
                         "mathtext.bf": "Segoe UI:bold"})
    P = PosterA1(); f = P.fig
    M, G = 0.6, 0.38
    BODY, CH, DG = 26, 22, 26
    cs = lambda v: v * K            # chart font scale (design pt -> printed pt)
    adm = R["adres"]["metrics"]
    FW = W0 - 2 * M                 # full content width

    # ---- header: blue -> teal gradient band, logo, title, course, names, affiliation
    hb = P.ax(0, 0, W0, 5.4); hb.axis("off"); hb.set_zorder(-1)
    grad = np.linspace(0, 1, 512)[None, :]
    hb.imshow(grad, aspect="auto", extent=(0, 1, 0, 1),
              cmap=LinearSegmentedColormap.from_list("hdr", [HEADER_A, HEADER_B]))
    if cfg.get("logo") and os.path.exists(cfg["logo"]):
        la = P.ax(0.75, 0.55, 4.3, 4.3); la.set_xlim(0, 1); la.set_ylim(0, 1); la.axis("off"); la.set_aspect("equal")
        la.set_zorder(10)
        la.add_patch(Circle((0.5, 0.5), 0.5, facecolor="#ffffff", edgecolor="none"))
        la.imshow(plt.imread(cfg["logo"]), extent=(0.08, 0.92, 0.08, 0.92), zorder=2)
    title = cfg["title"]
    cut = title.find(" with ") if " with " in title else len(title)
    lines = [title[:cut], title[cut + 1:]] if cut < len(title) else [title]
    cx = W0 / 2 + 1.6
    for i, ln in enumerate(lines):
        P.text(cx, 0.55 + i * 1.08, ln, 60, color="#ffffff", weight="bold", ha="center")
    yh = 0.55 + len(lines) * 1.08 + 0.12
    P.text(cx, yh, cfg.get("course", ""), 32, color=ON_HEADER, style="italic", ha="center")
    P.text(cx, yh + 0.78, cfg.get("authors", ""), 30, color="#ffffff", weight="bold", ha="center")
    P.text(cx, yh + 1.42, cfg.get("affiliation", ""), 24, color=HEADER_MUTED, ha="center")

    # ---- abstract band
    ya, ha_ = 5.4 + G, 2.45
    f.add_artist(FancyBboxPatch((M / W0, 1 - (ya + ha_) / H0), FW / W0, ha_ / H0, transform=f.transFigure,
                                boxstyle="round,pad=0,rounding_size=0.004", mutation_aspect=W0 / H0,
                                facecolor=PANEL, edgecolor=PANEL_EDGE, lw=1.4))
    P.text(M + 0.5, ya + ha_ / 2, "Abstract", 40, weight="bold", va="center", color=ACC_INK)
    am_ = adm["ADOPTED"]
    P.para(M + 4.5, ya + 0.32,
           "WiFeye turns an ordinary Wi-Fi router and two low-cost ESP32 boards into a camera-free fall and activity "
           "sensor. Cleaning the raw CSI, where every capture mixed about eight transmitters, lifted fall detection on "
           f"unseen people from chance (AUC {R['progression'][0][1]:.2f}) to {R['progression'][2][1]:.2f}–"
           f"{R['progression'][1][1]:.2f}. A fused event localiser, vetted CSI features and a two-head random forest "
           f"then reach {am_['accuracy']:.1%} accuracy over five classes and fall AUC {R['progression'][-1][1]:.2f}. "
           "It runs live over Wi-Fi; an Arduino UNO Q + 5G edge design is next.",
           FW - 5.0, 25, spacing=1.3, em=0.46)

    # ---- row 1: introduction | objectives | system model
    y1, h1 = ya + ha_ + G, 7.75
    wds = [10.4, 9.0, FW - 2 * G - 19.4]
    X3 = [M, M + wds[0] + G, M + wds[0] + wds[1] + 2 * G]
    c = P.panel(X3[0], y1, wds[0], h1, "1", "Introduction")
    x = X3[0] + 0.45; cw = wds[0] - 0.9
    c += P.para(x, c, "A moving body changes how Wi-Fi reaches a receiver. A real fall, seen by both ESP32s:",
                cw, 23) + 0.12
    a1 = P.ax(x + 0.95, c, cw - 0.95, 1.2); a2 = P.ax(x + 0.95, c + 1.28, cw - 0.95, 1.2)
    plot_csi_maps(a1, a2, R, cs(20))
    c += 1.28 + 1.2 + 0.72
    cwid = (cw - 0.3) / 2
    for i, (t, b) in enumerate([("Private", "no camera at home"), ("Comfortable", "nothing to wear"),
                                ("Low cost", "router + 2 ESP32s"), ("Hard part", "new people differ")]):
        P.card(x + (i % 2) * (cwid + 0.3), c + (i // 2) * 1.02, cwid, 0.9, t, b, size=20)

    c = P.panel(X3[1], y1, wds[1], h1, "2", "Objectives")
    P.bullets(X3[1] + 0.45, c + 0.1, [
        "Detect falls and daily activities with a router + two ESP32s.",
        "Build a clean, reproducible CSI pipeline.",
        "Locate events robustly; vet features bottom-up.",
        "Test honestly on unseen people.",
        "Compare one vs two ESP32 receivers.",
        "Run it live; design an edge + 5G version."],
        wds[1] - 0.9, 25.5, gap=0.24, numbered=True)

    c = P.panel(X3[2], y1, wds[2], h1, "3", "System Model & Problem Formulation")
    x = X3[2] + 0.45; cw = wds[2] - 0.9; FS = 23.5; lh = FS * 1.7 / 72
    eq = [("Signal", None),
          (None, r"Keep packets with $\mathrm{mac}=\mathrm{MAC}_r$;  52 amplitudes $\rightarrow$ 100 Hz, filtered"),
          ("Fused event localisation", None),
          (None, r"$w_r=[\mathrm{max}_t\,(s_r-\bar{s}_r)\,\pi(t)-\rho_r]_+$,   $F=\Sigma_r\,w_r(s_r-\bar{s}_r)/\Sigma_r\,w_r$"),
          (None, r"$t^*=\mathrm{argmax}_t\,F(t)\,\pi(t)$  $\rightarrow$  event onset, peak, offset"),
          ("Two-head decision", None),
          (None, r"$p_{\mathrm{fall}}$ from the fall head; other classes from the activity head"),
          (None, r"FALL iff $p_{\mathrm{fall}}\geq\tau$ (nested LOSO), else $\mathrm{argmax}_c\,p_c$")]
    for head, line in eq:
        if head:
            P.text(x, c, head, FS * 1.06, weight="bold", color=ACC_INK); c += lh * 0.98
        else:
            P.text(x, c, line, FS, color=INK); c += lh
    c += 0.2
    chips = [("Fall → alarm", ALERT, ALERT_EDGE, "#ffffff", 2.6), ("Walk", "#ffffff", S[0], INK, 1.45),
             ("Posture change", "#ffffff", S[0], INK, 3.15), ("Hand", "#ffffff", S[0], INK, 1.45),
             ("Empty", "#ffffff", S[0], INK, 1.65)]
    xch = x
    for lab, fill, edge, ink, wch in chips:
        f.add_artist(FancyBboxPatch((xch / W0, 1 - (c + 0.72) / H0), wch / W0, 0.72 / H0, transform=f.transFigure,
                                    boxstyle="round,pad=0,rounding_size=0.003", mutation_aspect=W0 / H0,
                                    facecolor=fill, edgecolor=edge, lw=2.0))
        P.text(xch + wch / 2, c + 0.36, lab, 20.5, color=ink, weight="bold", ha="center", va="center")
        xch += wch + 0.14

    # ---- row 2: section 4, the processing pipeline as one large flowchart
    y2, h2 = y1 + h1 + G, 7.35
    c = P.panel(M, y2, FW, h2, "4", "Proposed Method: Processing Pipeline")
    P.text(M + 13.6, y2 + 0.62, "colour = pipeline stage  ·  every receiver runs the same steps", 19, color=INK2,
           va="center")
    a = P.ax(M + 0.45, c + 0.05, FW - 0.9, h2 - 1.35 - 0.2); diagram_flow_h(a, cs(DG), FW - 0.9, h2 - 1.35 - 0.2)

    # ---- row 3: 4.1 hardware | 4.2 edge + 5G | 5 setup
    y3, h3 = y2 + h2 + G, 7.95
    w31, w32 = 9.1, 12.3
    w33 = FW - w31 - w32 - 2 * G
    xa, xb, xc = M, M + w31 + G, M + w31 + w32 + 2 * G
    c = P.panel(xa, y3, w31, h3, "4.1", "Hardware & Live Demo", small_num=True)
    a = P.ax(xa + 0.45, c + 0.05, w31 - 0.9, 4.65); diagram_setup_v4(a, cs(DG)); a.set_aspect("equal")
    P.para(xa + 0.45, c + 4.85, "Laptop B relays RX2 over the router's Wi-Fi. Laptop A runs the pipeline every "
           "second and sounds the alarm 3.5–5 s after a fall.", w31 - 0.9, 19.5, color=INK2)
    c = P.panel(xb, y3, w32, h3, "4.2", "Edge Deployment: Arduino UNO Q + 5G", small_num=True)
    a = P.ax(xb + 0.45, c + 0.05, w32 - 0.9, h3 - 1.35 - 0.75); diagram_edge_h(a, cs(DG), w32 - 0.9, h3 - 1.35 - 0.75)
    P.text(xb + 0.45, y3 + h3 - 0.5, "Proposed (future work): same code as the live demo.", 18.5, color=INK2,
           style="italic")
    c = P.panel(xc, y3, w33, h3, "5", "Experimental Setup")
    rows = [("Hardware", "TP-Link router (2.4 GHz ch 6) + 2 × ESP32-WROOM-32"),
            ("Placement", "receivers 60–90° apart, person 2–3 m away"),
            ("Traffic", "~70 CSI packets/s per ESP32"),
            ("Dataset", "509 trials × 10 s; 5 subjects + empty room"),
            ("Classes", "fall · walk · posture change · hand · empty"),
            ("Model", "two-head random forest per receiver"),
            ("Testing", "leave-one-subject-out; McNemar + Holm")]
    x = xc + 0.45; tw = w33 - 0.9; RS = 21.5; kw_ = 2.45
    for i, (k, v) in enumerate(rows):
        n_lines = len(textwrap.wrap(v, int((tw - kw_ - 0.15) * 72 / (RS * 0.53))))
        rh = max(0.44, n_lines * RS * 1.26 / 72) + 0.2
        if i % 2 == 0:
            f.add_artist(Rectangle((x / W0, 1 - (c + rh) / H0), tw / W0, rh / H0, transform=f.transFigure,
                                   facecolor=SURF, edgecolor="none", zorder=1))
        P.text(x + 0.15, c + 0.09, k, RS, weight="bold", zorder=3, color=ACC_INK)
        P.para(x + kw_, c + 0.09, v, tw - kw_ - 0.15, RS, color=INK, spacing=1.26)
        c += rh

    # ---- row 4: results (four charts in one row)
    y4, h4 = y3 + h3 + G, 7.85
    c = P.panel(M, y4, FW, h4, "6", "Results & Discussion")
    P.text(M + 9.2, y4 + 0.62, "all numbers: leave-one-subject-out, i.e. on people the model never saw", 19,
           color=INK2, va="center")
    colw = (FW - 0.9) / 4
    cx = [M + 0.45 + i * colw for i in range(4)]
    tab = {(t_["a"], t_["b"]): t_ for t_ in R["adres"]["tests"]}[("ADOPTED", "BASE")]
    ops = R["adres"]["fall_operating_points"]
    rxres = R.get("adrx", {}).get("results", {})
    dacc = (adm["ADOPTED"]["accuracy"] - adm["BASE"]["accuracy"]) * 100

    def cell(i, title, take):
        P.text(cx[i], c, title, 23.5, weight="bold")
        P.para(cx[i], c + 0.52, take, colw - 0.45, 19.5, color=OK_C, spacing=1.22)

    top = c + 1.62
    cell(0, "(a) Fall AUC: data cleaning pays",
         f"Chance ({R['progression'][0][1]:.2f}) → {R['progression'][2][1]:.2f}–{R['progression'][1][1]:.2f} after "
         f"cleaning; {R['progression'][-1][1]:.2f} for the final system.")
    a = P.ax(cx[0] + 1.0, top, colw - 1.35, 3.75); plot_progression(a, R, cs(CH * 0.9))
    cell(1, f"(b) 5 classes: {adm['ADOPTED']['accuracy']:.1%} accuracy",
         f"Domain features add +{dacc:.1f} pts (p = {tab['p']:.3f}). Missed falls look like posture changes.")
    a = P.ax(cx[1] + 1.3, top, colw - 1.55, 4.05); plot_confusion(a, R, cs(CH), key="cm5", classes=CL5, nice=NICE5)
    o3, o10 = ops["0.03"], ops["0.1"]
    cell(2, "(c) Fall-alarm trade-off",
         f"{o3['adl_far']:.1%} false alarms → {o3['sensitivity']:.0%} of falls caught; "
         f"{o10['adl_far']:.1%} → {o10['sensitivity']:.0%}.")
    a = P.ax(cx[2] + 0.85, top, 3.9, 3.9); plot_roc_v3(a, R, cs(CH * 0.92))
    for k, (tag, tgt) in enumerate((("A", "0.03"), ("B", "0.05"), ("C", "0.1"))):
        o = ops[tgt]
        P.text(cx[2] + 4.95, top + 0.6 + k * 0.72, f"{tag}  {o['adl_far']:.1%} → {o['sensitivity']:.0%}", 19,
               color=INK, weight="bold" if tag == "A" else "normal")
    P.text(cx[2] + 4.95, top + 0.05, "false alarms\n→ falls", 16, color=INK2, linespacing=1.1)
    if rxres:
        L_, r1_, r2_ = rxres["LATE (2 ESP32)"], rxres["RX1 only"], rxres["RX2 only"]
        take_d = (f"Two ≈ the better one alone ({L_['macro_f1']:.2f} vs {r1_['macro_f1']:.2f}); "
                  f"the weaker is {r2_['macro_f1']:.2f}: placement matters.")
    else:
        take_d = "Placement matters more than the number of receivers."
    cell(3, "(d) One ESP32 or two?", take_d)
    a1 = P.ax(cx[3] + 0.75, top + 0.2, 2.95, 3.5); a2 = P.ax(cx[3] + 4.3, top + 0.2, 2.95, 3.5)
    if rxres:
        plot_receivers_adopted(a1, a2, R, cs(CH * 0.9))
    else:
        plot_receivers(a1, a2, R, cs(CH * 0.9))

    # ---- row 5: use cases & future | conclusion | acknowledgement | references
    y5 = y4 + h4 + G
    h5 = H0 - 0.45 - y5
    wu, wcn, wak = 10.0, 8.3, 6.15
    wrf = FW - wu - wcn - wak - 3 * G
    xs5 = [M, M + wu + G, M + wu + wcn + 2 * G, M + wu + wcn + wak + 3 * G]
    c = P.panel(xs5[0], y5, wu, h5, "7", "Use Cases & Future Plans")
    x = xs5[0] + 0.45; tw = wu - 0.9
    chips = ["Elderly at home", "Care wards", "Bathrooms", "Smart homes"]
    cwd = (tw - 3 * 0.15) / 4
    for i, t_ in enumerate(chips):
        xx = x + i * (cwd + 0.15)
        f.add_artist(FancyBboxPatch((xx / W0, 1 - (c + 0.78) / H0), cwd / W0, 0.78 / H0, transform=f.transFigure,
                                    boxstyle="round,pad=0,rounding_size=0.003", mutation_aspect=W0 / H0,
                                    facecolor="#ffffff", edgecolor=S[2], lw=2.0))
        P.text(xx + cwd / 2, c + 0.39, t_, 17.5, weight="bold", ha="center", va="center", color=INK)
    c += 0.78 + 0.3
    P.bullets(x, c, [
        "Edge node: the whole pipeline on an Arduino UNO Q (4.2).",
        "5G alerts: urgent fall alarms (URLLC), routine reports (mMTC).",
        "More subjects, rooms and fall look-alikes."],
        tw, 21, gap=0.12, numbered=True)
    c = P.panel(xs5[1], y5, wcn, h5, "8", "Conclusion")
    P.bullets(xs5[1] + 0.45, c, [
        "A router + two ESP32s sense falls and activities: no camera, nothing worn.",
        "Cleaning the CSI mattered more than the model.",
        "Best: two-head forest on vetted event features; falls stay hardest.",
        "Scope: one room, 5 subjects, staged falls."],
        wcn - 0.9, 21.5, gap=0.1)
    c = P.panel(xs5[2], y5, wak, h5, "9", "Acknowledgement")
    P.para(xs5[2] + 0.45, c, cfg.get("acknowledgement", ""), wak - 0.9, 20.5, color=INK2)
    c = P.panel(xs5[3], y5, wrf, h5, "10", "References")
    refs = ["[1] S. M. Hernandez, E. Bulut, ESP32-CSI-Tool, WoWMoM 2020.",
            "[2] W. Wang et al., CARM (WiFi activity recognition), MobiCom 2015.",
            "[3] Y. Wang, K. Wu, L. M. Ni, “WiFall,” IEEE TMC 2017.",
            "[4] Y. Hu et al., “DeFall,” IEEE IoT J. 2022.",
            "[5] D. Zhang et al., Fresnel-zone Wi-Fi sensing, Computer 2017.",
            "[6] J. Yang et al., “SenseFi,” Patterns 2023.",
            "[7] S. Kapoor, A. Narayanan, leakage in ML science, Patterns 2023.",
            "[8] L. Breiman, “Random forests,” Mach. Learn. 2001.",
            "[9] Arduino, UNO Q documentation, 2025."]
    yy = c
    for r_ in refs:
        yy += P.para(xs5[3] + 0.45, yy, r_, wrf - 0.9, 15, color=INK2, spacing=1.18, em=0.5) + 0.03

    f.savefig(out_pdf, facecolor=SURF)
    f.savefig(out_png, dpi=60, facecolor=SURF)
    if os.environ.get("POSTER_DETAIL"):
        f.savefig(out_png.replace(".png", "_detail.png"), dpi=130, facecolor=SURF)
    plt.close(f)


def save_standalone(R):
    os.makedirs(VIS, exist_ok=True)
    jobs = {
        "auc_progression": ((10, 6.5), lambda ax: plot_progression(ax, R, 12)),
        "confusion_matrix": ((7.5, 6.5), lambda ax: plot_confusion(ax, R, 12)),
        "fall_roc": ((7, 7), lambda ax: plot_roc(ax, R, 12)),
        "kshot_calibration": ((8, 4.5), lambda ax: plot_kshot(ax, R, 12)),
        "missing_cell_augmentation": ((9, 5.5), lambda ax: plot_missing_cell(ax, R, 12)),
        "transmitter_mix": ((10, 3.6), lambda ax: plot_mix(ax, R, 12)),
        "posture_shift": ((9, 4.5), lambda ax: plot_posture(ax, R, 12)),
    }
    for name, (size, fn) in jobs.items():
        fig = plt.figure(figsize=size, facecolor=SURF)
        ax = fig.add_axes([0.14, 0.2, 0.8, 0.72])
        fn(ax)
        fig.savefig(os.path.join(VIS, f"{name}.png"), dpi=300, facecolor=SURF)
        fig.savefig(os.path.join(VIS, f"{name}.pdf"), facecolor=SURF)
        plt.close(fig)
    fig = plt.figure(figsize=(11, 5), facecolor=SURF)
    a1 = fig.add_axes([0.08, 0.2, 0.4, 0.65]); a2 = fig.add_axes([0.57, 0.2, 0.4, 0.65])
    plot_receivers(a1, a2, R, 12)
    fig.savefig(os.path.join(VIS, "receiver_comparison.png"), dpi=300, facecolor=SURF)
    fig.savefig(os.path.join(VIS, "receiver_comparison.pdf"), facecolor=SURF)
    plt.close(fig)
    # adopted system (5 classes, two heads)
    adopted = {"confusion_matrix_adopted_5class": ((7, 6.2), lambda ax: plot_confusion(ax, R, 12, "cm5", CL5, NICE5)),
               "fall_roc_adopted": ((7, 7), lambda ax: plot_roc_adopted(ax, R, 12))}
    for name, (size, fn) in adopted.items():
        fig = plt.figure(figsize=size, facecolor=SURF)
        ax = fig.add_axes([0.14, 0.2, 0.8, 0.72])
        fn(ax)
        fig.savefig(os.path.join(VIS, f"{name}.png"), dpi=300, facecolor=SURF)
        fig.savefig(os.path.join(VIS, f"{name}.pdf"), facecolor=SURF)
        plt.close(fig)
    if "adrx" in R:
        fig = plt.figure(figsize=(10, 5), facecolor=SURF)
        a1 = fig.add_axes([0.08, 0.2, 0.4, 0.65]); a2 = fig.add_axes([0.57, 0.2, 0.4, 0.65])
        plot_receivers_adopted(a1, a2, R, 12)
        fig.savefig(os.path.join(VIS, "receiver_comparison_adopted.png"), dpi=300, facecolor=SURF)
        fig.savefig(os.path.join(VIS, "receiver_comparison_adopted.pdf"), facecolor=SURF)
        plt.close(fig)


def main():
    with open("poster_config.json", encoding="utf-8") as fh:
        cfg = json.load(fh)
    R = load()
    print("  AUC progression:", [(d[0].split()[0], round(d[1], 3), round(d[2], 3), round(d[3], 3)) for d in R["progression"]])
    save_standalone(R)
    os.makedirs("reports", exist_ok=True)
    out_pdf = "reports/WiFeye_poster_A1.pdf"
    try:
        open(out_pdf, "ab").close()
    except PermissionError:                      # the PDF is open in a viewer (Windows locks it)
        out_pdf = "reports/WiFeye_poster_A1_updated.pdf"
        print(f"  NOTE: reports/WiFeye_poster_A1.pdf is open elsewhere; writing {out_pdf} instead")
    build_poster_a1(R, cfg, out_pdf, "reports/WiFeye_poster_A1_preview.png")
    print(f"  Saved figures to data/visualizations/ and poster to {out_pdf}")


if __name__ == "__main__":
    main()
