"""
visualize_masks.py
===================
Utilities to visualise the boolean connectivity masks produced by
`build_masks()` in `bio_masks.py`. Saves three stacked heatmaps showing
Input->Spine, Spine->Dendrite and Dendrite->Soma connectivity.

Usage (from repo root):
    python3 -c "from visualize_masks import plot_masks; plot_masks(128,4,128)"

Or run as a CLI:
    python3 visualize_masks.py --n_spines 128 --n_dendrites 4 --n_soma 128

The script saves PNG(s) into `rANN/vis_examples/` by default.
"""

import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bio_masks import build_masks


def _figure_style():
    plt.style.use("seaborn-v0_8-muted")


def _input_feature_ranges(n_time: int = 8, n_freq: int = 32, n_itd: int = 32, n_ild: int = 32):
    """Return feature index ranges for the default input layout used by `build_masks()`.

    The input is ordered as frequency features first, then ITD, then ILD.
    """
    freq_end = n_time * n_freq
    itd_end = freq_end + n_time * n_itd
    ild_end = itd_end + n_time * n_ild
    return {
        "freq": (0, freq_end),
        "itd": (freq_end, itd_end),
        "ild": (itd_end, ild_end),
        "input_dim": ild_end,
    }


def _set_sparse_integer_ticks(ax, length: int, axis: str = "x", max_ticks: int = 8):
    """Set sparse integer ticks for an image axis."""
    if length <= 0:
        return
    step = max(1, int(np.ceil(length / max_ticks)))
    ticks = np.arange(0, length, step)
    if ticks.size == 0 or ticks[-1] != length - 1:
        ticks = np.append(ticks, length - 1)
    if axis == "x":
        ax.set_xticks(ticks)
    else:
        ax.set_yticks(ticks)


def plot_masks(n_spines: int, n_dendrites_per_soma: int, n_soma: int,
               outdir: Optional[str] = "rANN/vis_examples",
               save: bool = True,
               dpi: int = 150,
               itd_spine_frac: float = 0.5,
               overlap: float = 0.0,
               dendrite_rule: str = "topographic",
               channel_dend_split: bool = True,
               file_suffix: Optional[str] = None,
               mode: str = "constrained") -> str:
    """Build masks and plot three heatmaps. Returns path of saved file.

    Parameters
    ----------
    n_dendrites_per_soma : each soma has this many dendrites

    The produced figure contains three stacked subplots (input->spine,
    spine->dendrite, dendrite->soma). If `save` is True the PNG is written
    into `outdir` and the path returned; otherwise the matplotlib figure
    object is returned as an empty string.
    """
    _figure_style()
    masks = build_masks(
        n_spines=n_spines,
        n_dendrites_per_soma=n_dendrites_per_soma,
        n_soma=n_soma,
        itd_spine_frac=itd_spine_frac,
        overlap=overlap,
        dendrite_rule=dendrite_rule,
        channel_dend_split=channel_dend_split,
    )
    total_dendrites = n_dendrites_per_soma * n_soma

    if mode == "constrained":
        m1 = masks["input_spine"]         # shape: (input_dim, n_spines)
        m2 = masks["spine_dendrite"]     # shape: (n_spines, total_dendrites)
        m3 = masks["dendrite_soma"]      # shape: (total_dendrites, n_soma)  [per-soma block diagonal]

    summary = masks["summary"]

    # Render an explicit all-to-all heatmap when requested (baseline)
    if mode == "alltoall":
        input_dim = summary["config"]["input_dim"]
        m1 = np.ones((input_dim, n_spines), dtype=np.float32)
        m2 = np.ones((n_spines, total_dendrites), dtype=np.float32)
        m3 = np.ones((total_dendrites, n_soma), dtype=np.float32)
        # update summary diagnostics for display
        summary = dict(summary)
        summary["active_weights"] = dict(
            input_spine=int(m1.sum()),
            spine_dendrite=int(m2.sum()),
            dendrite_soma=int(m3.sum()),
            total=int(m1.sum() + m2.sum() + m3.sum()),
            ch1_input_spine=summary.get("active_weights", {}).get("ch1_input_spine", 0),
            ch2_input_spine=summary.get("active_weights", {}).get("ch2_input_spine", 0),
        )
        summary["all2all_weights"] = dict(total=int(input_dim * n_spines + n_spines * total_dendrites + total_dendrites * n_soma))
        summary["density"] = dict(
            input_spine=round(summary["active_weights"]["input_spine"] / (summary["config"]["input_dim"] * n_spines), 4),
            spine_dendrite=round(summary["active_weights"]["spine_dendrite"] / (n_spines * total_dendrites), 4),
            dendrite_soma=round(summary["active_weights"]["dendrite_soma"] / (total_dendrites * n_soma), 4),
            overall=round((summary["active_weights"]["input_spine"] + summary["active_weights"]["spine_dendrite"] + summary["active_weights"]["dendrite_soma"]) /
                           summary["all2all_weights"]["total"], 4),
        )

    channels = summary.get("channels", {})
    ch1 = channels.get("ITD", {}).get("spines", (0, 0))
    ch2 = channels.get("ILD", {}).get("spines", (0, 0))
    feature_ranges = _input_feature_ranges()

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), constrained_layout=True)

    # Input -> Spine
    ax = axes[0]
    im = ax.imshow(m1.T, aspect="auto", cmap="viridis")
    ax.set_ylabel("Spine index")
    ax.set_xlabel("Input feature index")
    ax.set_title(f"Input → Spine  (input_dim={summary['config']['input_dim']}, spines={n_spines})")
    # Mark the input feature partitions used by build_masks().
    spans = [
        ("Frequency", "#5dade2", feature_ranges["freq"]),
        ("ITD", "#58d68d", feature_ranges["itd"]),
        ("ILD", "#f5b041", feature_ranges["ild"]),
    ]
    y_top = -0.8
    y_text = -0.3
    for label, color, (x0, x1) in spans:
        ax.axvline(x0, color=color, lw=1.2, alpha=0.75)
        ax.text((x0 + x1) / 2, y_text, label, ha="center", va="bottom",
                fontsize=9, fontweight="bold", color=color,
                transform=ax.transData)
    ax.axvline(feature_ranges["ild"][1], color="#f5b041", lw=1.2, alpha=0.75)
    ax.text(feature_ranges["freq"][0] + 2, y_top, "input features", ha="left",
            va="bottom", fontsize=8, color="black", transform=ax.transData)

    # vertical line to mark the ITD/ILD spine boundary
    ch1_start, ch1_end = ch1
    if ch1_end > ch1_start:
        ax.hlines([ch1_end - 0.5], xmin=0, xmax=m1.shape[0] - 1, colors="white", linewidth=1)
    _set_sparse_integer_ticks(ax, m1.shape[0], axis="x")
    _set_sparse_integer_ticks(ax, m1.shape[1], axis="y")

    # Spine -> Dendrite
    ax = axes[1]
    im = ax.imshow(m2.T, aspect="auto", cmap="viridis")
    ax.set_ylabel("Dendrite index")
    ax.set_xlabel("Spine index")
    ax.set_title(f"Spine → Dendrite  (spines={n_spines}, total_dendrites={total_dendrites})")
    _set_sparse_integer_ticks(ax, m2.shape[0], axis="x")
    _set_sparse_integer_ticks(ax, m2.shape[1], axis="y")
    fig.colorbar(im, ax=ax, orientation="vertical", fraction=0.03)

    # Dendrite -> Soma
    ax = axes[2]
    im = ax.imshow(m3.T, aspect="auto", cmap="viridis")
    ax.set_ylabel("Soma index")
    ax.set_xlabel("Dendrite index")
    ax.set_title(f"Dendrite → Soma  (total_dendrites={total_dendrites}, soma={n_soma}) [per-soma bands]")
    # Mark per-soma boundaries
    _set_sparse_integer_ticks(ax, m3.shape[0], axis="x")
    _set_sparse_integer_ticks(ax, m3.shape[1], axis="y")
    # Add vertical lines to mark per-soma band boundaries
    for soma_idx in range(1, n_soma):
        ax.axvline(soma_idx * n_dendrites_per_soma - 0.5, color="white", linewidth=0.5, alpha=0.3)
    fig.colorbar(im, ax=ax, orientation="vertical", fraction=0.03)

    if save:
        os.makedirs(outdir, exist_ok=True)
        # Filename: masks_s<n_spines>_dps<n_dendrites_per_soma>_so<n_soma>.png
        suffix = f"_{file_suffix}" if file_suffix else ""
        fname = f"masks_{mode}_s{n_spines}_dps{n_dendrites_per_soma}_so{n_soma}{suffix}.png"
        fpath = os.path.join(outdir, fname)
        fig.savefig(fpath, dpi=dpi)
        plt.close(fig)
        print(f"Saved mask figure → {fpath}")
        return fpath
    else:
        plt.show()
        return ""


def _parse_args_and_run():
    import argparse

    p = argparse.ArgumentParser(description="Visualise bio_masks connectivity")
    p.add_argument("--n_spines", type=int, default=128)
    p.add_argument("--n_dendrites_per_soma", type=int, default=4)
    p.add_argument("--n_soma", type=int, default=128)
    p.add_argument("--outdir", type=str, default="rANN/vis_examples")
    p.add_argument("--dpi", type=int, default=150)
    args = p.parse_args()

    plot_masks(args.n_spines, args.n_dendrites_per_soma, args.n_soma,
               outdir=args.outdir, dpi=args.dpi, save=True)


if __name__ == "__main__":
    _parse_args_and_run()


# ---------------------------------------------------------------------------
# Architecture diagram
# ---------------------------------------------------------------------------

def plot_architectures(n_spines: int, n_dendrites_per_soma: int, n_soma: int,
                      outdir: Optional[str] = "rANN/vis_examples",
                      save: bool = True,
                      dpi: int = 150,
                      itd_spine_frac: float = 0.5,
                      overlap: float = 0.0,
                      dendrite_rule: str = "topographic",
                      channel_dend_split: bool = True,
                      file_suffix: Optional[str] = None,
                      mode: str = "constrained") -> str:
    """Draw a schematic wiring diagram for the two-channel barn owl mask.

    Channel 1 (ITD pathway — NL -> ICc lateral shell) is drawn in blue.
    Channel 2 (ILD pathway — NA -> VLVp -> ICc core) is drawn in coral.
    The all-to-all output convergence is drawn in teal.

    Parameters
    ----------
    n_dendrites_per_soma : each soma has this many dendrites

    Returns the saved file path if save=True, otherwise empty string.
    """
    _figure_style()
    masks = build_masks(
        n_spines=n_spines,
        n_dendrites_per_soma=n_dendrites_per_soma,
        n_soma=n_soma,
        itd_spine_frac=itd_spine_frac,
        overlap=overlap,
        dendrite_rule=dendrite_rule,
        channel_dend_split=channel_dend_split,
    )
    summary = masks["summary"]
    cs       = summary["channel_sizes"]
    cfg      = summary["config"]
    ch_split = summary["topology"]["channel_dend_split"]

    n_itd_sp = cs["itd_spines"];  n_ild_sp = cs["ild_spines"]
    n_itd_de = cs["itd_dends"];   n_ild_de = cs["ild_dends"]
    n_itd_so = cs["itd_soma"];    n_ild_so = cs["ild_soma"]

    total_dendrites = cfg["total_dendrites"]
    a1 = summary["active_weights"]["input_spine"]
    a2 = summary["active_weights"]["spine_dendrite"]
    a3 = summary["active_weights"]["dendrite_soma"]
    aa = summary["all2all_weights"]["total"]
    reduction = summary["weight_reduction_pct"]

    alltoall = (mode == "alltoall")
    # If requested, render the all-to-all baseline wiring (no masks):
    if alltoall:
        # compute full all-to-all active counts per stage
        a1 = int(cfg["input_dim"] * n_spines)
        a2 = int(n_spines * total_dendrites)
        a3 = int(total_dendrites * n_soma)
        reduction = 0.0

    BLUE  = "#378ADD"; BLUE_L  = "#E6F1FB"
    CORAL = "#D85A30"; CORAL_L = "#FAECE7"
    TEAL  = "#1D9E75"; TEAL_L  = "#E1F5EE"
    GRAY  = "#888780"; GRAY_L  = "#F1EFE8"

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    def _box(cx, cy, w, h, label, sub, edge, face):
        from matplotlib.patches import FancyBboxPatch
        rect = FancyBboxPatch(
            (cx - w / 2, cy - h / 2), w, h,
            boxstyle="round,pad=0.06", linewidth=0.9,
            edgecolor=edge, facecolor=face,
        )
        ax.add_patch(rect)
        ax.text(cx, cy + 0.14, label, ha="center", va="center",
                fontsize=9, fontweight="bold", color=edge)
        if sub:
            ax.text(cx, cy - 0.22, sub, ha="center", va="center",
                    fontsize=7, color=edge, alpha=0.85)

    def _arrow(x1, y1, x2, y2, color, label="", side="right"):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-|>", color=color,
                            lw=1.2, mutation_scale=10),
        )
        if label:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            dx = 0.16 if side == "right" else -0.16
            ax.text(mx + dx, my, label, fontsize=7, color=color,
                    ha="left" if side == "right" else "right", va="center")

    # Input
    _box(5, 9.2, 5.5, 0.72, "Input",
         f"{cfg['input_dim']} features  ·  freq [0-255]  ·  ITD [256-511]  ·  ILD [512-767]",
         GRAY, GRAY_L)

    # Spine layer — two channels (or single all-to-all block when requested)
    if alltoall:
        _box(5.0, 7.4, 6.2, 0.72,
             f"Spines  ({n_spines})",
             "all-to-all inputs", TEAL, TEAL_L)
        _arrow(5, 8.86, 5, 7.78, TEAL, "all-to-all", "right")
    else:
        _box(2.5, 7.4, 3.2, 0.72,
             f"Ch1  ITD spines  ({n_itd_sp})",
             "freq (full broadcast) + ITD (topographic)", BLUE, BLUE_L)
        _box(7.5, 7.4, 3.0, 0.72,
             f"Ch2  ILD spines  ({n_ild_sp})",
             "ILD topographic only — no frequency", CORAL, CORAL_L)

        _arrow(3.8, 8.86, 2.8, 7.78, BLUE,  "freq (all)", "left")
        _arrow(4.5, 8.86, 3.2, 7.78, BLUE,  "ITD topo",   "right")
        _arrow(6.5, 8.86, 7.2, 7.78, CORAL, "ILD topo",   "left")

    # Dendrite layer
    if alltoall:
        _box(5, 5.6, 6.0, 0.72,
             f"Dendrites  ({total_dendrites})",
             "all-to-all connections", TEAL, TEAL_L)
        _arrow(5, 7.05, 5, 5.96, TEAL)
    else:
        if ch_split:
            _box(2.5, 5.6, 2.8, 0.72,
                 f"Ch1  dendrites  ({n_itd_de})",
                 f"{n_dendrites_per_soma} per soma, topographic", BLUE, BLUE_L)
            _box(7.5, 5.6, 2.8, 0.72,
                 f"Ch2  dendrites  ({n_ild_de})",
                 f"{n_dendrites_per_soma} per soma, topographic", CORAL, CORAL_L)
            _arrow(2.5, 7.05, 2.5, 5.96, BLUE)
            _arrow(7.5, 7.05, 7.5, 5.96, CORAL)
            ax.text(5, 6.32, "channel boundary preserved",
                    ha="center", fontsize=7, color=GRAY, style="italic")
            ax.annotate("", xy=(3.9, 6.32), xytext=(3.1, 6.32),
                        arrowprops=dict(arrowstyle="->", color=GRAY, lw=0.7))
            ax.annotate("", xy=(6.1, 6.32), xytext=(6.9, 6.32),
                        arrowprops=dict(arrowstyle="->", color=GRAY, lw=0.7))
        else:
            _box(5, 5.6, 3.8, 0.72,
                 f"Dendrites  ({total_dendrites})",
                 f"{n_dendrites_per_soma} per soma — merged", GRAY, GRAY_L)
            _arrow(2.5, 7.05, 4.0, 5.96, GRAY)
            _arrow(7.5, 7.05, 6.0, 5.96, GRAY)

    # Soma layer — block-diagonal: each soma receives only its own dendrite band
    if alltoall:
        _box(5, 3.8, 6.0, 0.72,
             f"Soma  ({n_soma})",
             "all-to-all readout", TEAL, TEAL_L)
        _arrow(5, 5.25, 5, 4.17, TEAL)
    else:
        if ch_split:
            _box(2.5, 3.8, 2.8, 0.72,
                 f"Ch1  soma  ({n_itd_so})",
                 "block-diagonal from Ch1 dendrites", BLUE, BLUE_L)
            _box(7.5, 3.8, 2.8, 0.72,
                 f"Ch2  soma  ({n_ild_so})",
                 "block-diagonal from Ch2 dendrites", CORAL, CORAL_L)
            _arrow(2.5, 5.25, 2.5, 4.17, BLUE)
            _arrow(7.5, 5.25, 7.5, 4.17, CORAL)
        else:
            _box(5, 3.8, 3.5, 0.72,
                 f"Soma  ({n_soma})",
                 "block-diagonal from merged dendrites", GRAY, GRAY_L)
            _arrow(5, 5.25, 5, 4.17, GRAY)

    # Output — all-to-all
    _box(5, 2.0, 4.2, 0.72, "Output  (all-to-all)",
         "azimuth (deg)  ·  elevation (deg)", TEAL, TEAL_L)
    if ch_split:
        _arrow(2.5, 3.45, 4.0, 2.38, TEAL, "all-to-all", "left")
        _arrow(7.5, 3.45, 6.0, 2.38, TEAL, "all-to-all", "right")
    else:
        _arrow(5, 3.45, 5, 2.38, TEAL, "all-to-all", "right")

    # Stats footer
    ax.text(
        5, 0.88,
        (f"Active weights:  M1={a1:,}  M2={a2:,}  M3={a3:,}  "
         f"total={a1+a2+a3:,}  /  all-to-all={aa:,}  ->  {reduction}% reduction"),
        ha="center", fontsize=8, color=GRAY,
    )

    ax.set_title(
        (f"Architecture — two-channel barn owl mask\n"
         f"spines={n_spines}  ·  dendrites_per_soma={n_dendrites_per_soma}  ·  "
         f"soma={n_soma}  ·  total_dendrites={total_dendrites}  ·  "
         f"channel split={'on' if ch_split else 'off'}"),
        fontsize=10, pad=8,
    )

    if save:
        os.makedirs(outdir, exist_ok=True)
        suffix = f"_{file_suffix}" if file_suffix else ""
        fname = f"arch_{mode}_s{n_spines}_dps{n_dendrites_per_soma}_so{n_soma}{suffix}.png"
        fpath = os.path.join(outdir, fname)
        fig.savefig(fpath, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved architecture figure -> {fpath}")
        return fpath
    else:
        plt.show()
        return ""



def plot_all(n_spines: int, n_dendrites_per_soma: int, n_soma: int,
             outdir: Optional[str] = "rANN/vis_examples",
             save: bool = True,
             dpi: int = 150,
             itd_spine_frac: float = 0.5,
             overlap: float = 0.0,
             dendrite_rule: str = "topographic",
             channel_dend_split: bool = True,
             file_suffix: Optional[str] = None,
             mode: str = "constrained"):
    """Run plot_masks() and plot_architectures() together.

    Returns (masks_path, architecture_path).
    """
    p1 = plot_masks(n_spines, n_dendrites_per_soma, n_soma,
                  outdir=outdir, save=save, dpi=dpi,
                  itd_spine_frac=itd_spine_frac,
                  overlap=overlap,
                  dendrite_rule=dendrite_rule,
                  channel_dend_split=channel_dend_split,
                  file_suffix=file_suffix,
                  mode=mode)
    p2 = plot_architectures(n_spines, n_dendrites_per_soma, n_soma,
                      outdir=outdir, save=save, dpi=dpi,
                      itd_spine_frac=itd_spine_frac,
                      overlap=overlap,
                      dendrite_rule=dendrite_rule,
                      channel_dend_split=channel_dend_split,
                      file_suffix=file_suffix,
                      mode=mode)
    return p1, p2


# ---------------------------------------------------------------------------
# Mask verification
# ---------------------------------------------------------------------------

def _mask_fingerprint(masks: dict) -> dict:
    """Compute a deterministic fingerprint from a masks dict.

    Uses active weight counts and per-layer densities — cheap to compute,
    unique to a given (size × topology) configuration.
    """
    s = masks["summary"]
    return dict(
        # Layer sizes
        input_dim          = s["config"]["input_dim"],
        n_spines           = s["config"]["n_spines"],
        n_dendrites_per_soma = s["config"]["n_dendrites_per_soma"],
        total_dendrites    = s["config"]["total_dendrites"],
        n_soma             = s["config"]["n_soma"],
        # Active weight counts per stage
        active_m1          = s["active_weights"]["input_spine"],
        active_m2          = s["active_weights"]["spine_dendrite"],
        active_m3          = s["active_weights"]["dendrite_soma"],
        active_total       = s["active_weights"]["total"],
        ch1_active         = s["active_weights"]["ch1_input_spine"],
        ch2_active         = s["active_weights"]["ch2_input_spine"],
        # Density per stage
        density_m1         = s["density"]["input_spine"],
        density_m2         = s["density"]["spine_dendrite"],
        density_m3         = s["density"]["dendrite_soma"],
        # Topology params
        itd_spine_frac     = s["topology"]["itd_spine_frac"],
        overlap            = s["topology"]["overlap"],
        dendrite_rule      = s["topology"]["dendrite_rule"],
        channel_dend_split = s["topology"]["channel_dend_split"],
        # Channel split sizes
        itd_spines         = s["channel_sizes"]["itd_spines"],
        ild_spines         = s["channel_sizes"]["ild_spines"],
        itd_dends          = s["channel_sizes"]["itd_dends"],
        ild_dends          = s["channel_sizes"]["ild_dends"],
        itd_soma           = s["channel_sizes"]["itd_soma"],
        ild_soma           = s["channel_sizes"]["ild_soma"],
    )


def verify_figure_masks(
    n_spines: int,
    n_dendrites_per_soma: int,
    n_soma: int,
    run_dir: str,
    itd_spine_frac: float = 0.5,
    overlap: float = 0.0,
    dendrite_rule: str = "topographic",
    mode: str = "constrained",
    channel_dend_split: bool = True,
) -> dict:
    """Verify that the mask figures in run_dir match the given config.

    Rebuilds the mask from scratch using the supplied parameters, computes a
    fingerprint, and checks it against the fingerprint stored in config.json.
    Also confirms that both expected PNG files are present on disk.

    Parameters
    ----------
    run_dir : path to one sweep run directory (must contain config.json)

    Returns
    -------
    dict with keys:
        ok             : True if all checks passed
        checks         : dict of individual check results
        fingerprint    : the freshly-computed fingerprint
        stored         : the fingerprint loaded from config.json (if present)
        missing_files  : list of expected files that are absent
        mismatches     : list of fingerprint fields that differ
    """
    result = dict(ok=True, checks={}, fingerprint={}, stored={},
                  missing_files=[], mismatches=[])

    # 1. Rebuild mask and compute fingerprint
    fresh_masks = build_masks(
        n_spines=n_spines,
        n_dendrites_per_soma=n_dendrites_per_soma,
        n_soma=n_soma,
        itd_spine_frac=itd_spine_frac,
        overlap=overlap,
        dendrite_rule=dendrite_rule,
        channel_dend_split=channel_dend_split,
    )
    fp = _mask_fingerprint(fresh_masks)
    result["fingerprint"] = fp

    # 2. Check PNG files exist
    import os
    suffix = (
        f"if{_frac_name(itd_spine_frac)}"
        f"_ov{_overlap_name(overlap)}"
        f"_dr{dendrite_rule}"
        f"_cd{'split' if channel_dend_split else 'merged'}"
    )
    expected_files = [
        f"masks_{mode}_s{n_spines}_dps{n_dendrites_per_soma}_so{n_soma}_{suffix}.png",
        f"arch_s{n_spines}_dps{n_dendrites_per_soma}_so{n_soma}_{suffix}.png",
        "config.json",
        "history.json",
        "model.keras",
    ]
    for fname in expected_files:
        fpath = os.path.join(run_dir, fname)
        present = os.path.exists(fpath)
        result["checks"][f"file:{fname}"] = present
        if not present:
            result["missing_files"].append(fname)
            result["ok"] = False

    # 3. Compare fingerprint against config.json stored mask_summary
    cfg_path = os.path.join(run_dir, "config.json")
    if os.path.exists(cfg_path):
        import json
        with open(cfg_path) as f:
            cfg = json.load(f)
        stored_summary = cfg.get("mask_summary", {})
        stored_fp = _fingerprint_from_stored_summary(stored_summary)
        result["stored"] = stored_fp

        for key, fresh_val in fp.items():
            stored_val = stored_fp.get(key)
            if stored_val is None:
                continue  # not stored — skip
            match = abs(float(fresh_val) - float(stored_val)) < 1e-6 \
                if isinstance(fresh_val, float) else fresh_val == stored_val
            result["checks"][f"fp:{key}"] = match
            if not match:
                result["mismatches"].append(
                    dict(field=key, expected=fresh_val, stored=stored_val)
                )
                result["ok"] = False

    # 4. Biological rule checks on fresh masks
    bio_ok, bio_msgs = _biological_rule_checks(fresh_masks)
    result["checks"]["biological_rules"] = bio_ok
    result["bio_messages"] = bio_msgs
    if not bio_ok:
        result["ok"] = False

    return result


def _frac_name(itd_spine_frac: float) -> str:
    """Map float frac back to catalogue name for filename reconstruction."""
    from bio_masks import ITD_SPINE_FRAC_CONFIGS
    for name, val in ITD_SPINE_FRAC_CONFIGS.items():
        if abs(val - itd_spine_frac) < 1e-6:
            return name
    return str(itd_spine_frac)


def _overlap_name(overlap: float) -> str:
    from bio_masks import OVERLAP_CONFIGS
    for name, val in OVERLAP_CONFIGS.items():
        if abs(val - overlap) < 1e-6:
            return name
    return str(overlap)


def _fingerprint_from_stored_summary(stored: dict) -> dict:
    """Extract fingerprint fields from a stored mask_summary dict."""
    out = {}
    cfg = stored.get("config", {})
    out["n_spines"]             = cfg.get("n_spines")
    out["n_dendrites_per_soma"] = cfg.get("n_dendrites_per_soma")
    out["total_dendrites"]      = cfg.get("total_dendrites")
    out["n_soma"]               = cfg.get("n_soma")
    out["input_dim"]            = cfg.get("input_dim")

    aw = stored.get("active_weights", {})
    out["active_m1"]    = aw.get("input_spine")
    out["active_m2"]    = aw.get("spine_dendrite")
    out["active_m3"]    = aw.get("dendrite_soma")
    out["active_total"] = aw.get("total")
    out["ch1_active"]   = aw.get("ch1_input_spine")
    out["ch2_active"]   = aw.get("ch2_input_spine")

    dn = stored.get("density", {})
    out["density_m1"] = dn.get("input_spine")
    out["density_m2"] = dn.get("spine_dendrite")
    out["density_m3"] = dn.get("dendrite_soma")

    topo = stored.get("topology", {})
    out["itd_spine_frac"]     = topo.get("itd_spine_frac")
    out["overlap"]            = topo.get("overlap")
    out["dendrite_rule"]      = topo.get("dendrite_rule")
    out["channel_dend_split"] = topo.get("channel_dend_split")

    cs = stored.get("channel_sizes", {})
    out["itd_spines"] = cs.get("itd_spines")
    out["ild_spines"] = cs.get("ild_spines")
    out["itd_dends"]  = cs.get("itd_dends")
    out["ild_dends"]  = cs.get("ild_dends")
    out["itd_soma"]   = cs.get("itd_soma")
    out["ild_soma"]   = cs.get("ild_soma")
    return out


def _biological_rule_checks(masks: dict) -> tuple[bool, list[str]]:
    """Run the seven biological rule assertions and return (all_pass, messages)."""
    m1 = masks["input_spine"]
    m2 = masks["spine_dendrite"]
    m3 = masks["dendrite_soma"]
    s  = masks["summary"]
    ch = s["channels"]
    cs = s["channel_sizes"]

    ch1_s, ch1_e = ch["ITD"]["spines"]
    ch2_s, ch2_e = ch["ILD"]["spines"]
    cd1_s, cd1_e = ch["ITD"]["dendrites"]
    cd2_s, cd2_e = ch["ILD"]["dendrites"]

    n_freq_inputs = 8 * 32  # 256

    msgs  = []
    all_ok = True

    def check(cond, pass_msg, fail_msg):
        nonlocal all_ok
        if cond:
            msgs.append(f"PASS  {pass_msg}")
        else:
            msgs.append(f"FAIL  {fail_msg}")
            all_ok = False

    check(
        m1[:n_freq_inputs, ch1_s:ch1_e].min() == 1.0,
        f"All {cs['itd_spines']} ITD spines receive full frequency broadcast",
        "Some ITD spines missing frequency broadcast",
    )
    check(
        m1[:n_freq_inputs, ch2_s:ch2_e].sum() == 0,
        f"All {cs['ild_spines']} ILD spines receive no frequency input",
        "ILD spines incorrectly receiving frequency input",
    )
    check(
        m1[n_freq_inputs: n_freq_inputs * 2, ch2_s:ch2_e].sum() == 0,
        "ITD inputs do not reach ILD channel spines",
        "ITD inputs leaking into ILD channel",
    )
    check(
        m1[n_freq_inputs * 2:, ch1_s:ch1_e].sum() == 0,
        "ILD inputs do not reach ITD channel spines",
        "ILD inputs leaking into ITD channel",
    )
    if s["topology"]["channel_dend_split"]:
        check(
            m2[ch1_s:ch1_e, cd2_s:cd2_e].sum() == 0
            and m2[ch2_s:ch2_e, cd1_s:cd1_e].sum() == 0,
            "Spine->dendrite connections respect channel boundaries",
            "Channel boundary violation in spine->dendrite mask",
        )
    check(
        (m2.sum(axis=1) == 1).all(),
        "Each spine connects to exactly 1 dendrite",
        f"Spines with != 1 dendrite connection: {int((m2.sum(axis=1) != 1).sum())}",
    )
    n_dps = s["config"]["n_dendrites_per_soma"]
    check(
        (m3.sum(axis=1) == 1).all(),
        "Each dendrite projects to exactly 1 soma (block-diagonal M3)",
        f"Dendrites with != 1 soma connection: {int((m3.sum(axis=1) != 1).sum())}",
    )
    check(
        (m3.sum(axis=0) == n_dps).all(),
        f"Each soma receives from exactly {n_dps} dendrites (n_dendrites_per_soma)",
        f"Soma units with != {n_dps} dendrite inputs: {int((m3.sum(axis=0) != n_dps).sum())}",
    )

    return all_ok, msgs


def print_verification(result: dict):
    """Pretty-print the output of verify_figure_masks()."""
    status = "OK" if result["ok"] else "FAILED"
    print(f"\n{'='*55}")
    print(f"  Mask verification: {status}")
    print(f"{'='*55}")

    print("\n── Biological rules ──")
    for msg in result.get("bio_messages", []):
        print(f"  {msg}")

    if result["missing_files"]:
        print("\n── Missing files ──")
        for f in result["missing_files"]:
            print(f"  MISSING  {f}")

    if result["mismatches"]:
        print("\n── Fingerprint mismatches ──")
        for m in result["mismatches"]:
            print(f"  {m['field']:<25}  expected={m['expected']}  stored={m['stored']}")

    if result["ok"]:
        fp = result["fingerprint"]
        print(f"\n── Fingerprint (fresh) ──")
        print(f"  active M1={fp['active_m1']:,}  M2={fp['active_m2']:,}  "
              f"M3={fp['active_m3']:,}  total={fp['active_total']:,}")
        print(f"  density M1={fp['density_m1']:.3f}  M2={fp['density_m2']:.3f}  "
              f"M3={fp['density_m3']:.3f}")
        print(f"  ITD spines={fp['itd_spines']}  ILD spines={fp['ild_spines']}")
        print(f"  ch_split={fp['channel_dend_split']}  rule={fp['dendrite_rule']}")
    print()


def verify_sweep(
    sweep_dir: str,
    all_masks: Optional[dict] = None,
    verbose: bool = True,
) -> 'pd.DataFrame':
    """Verify all run directories in a sweep_results/runs/ folder.

    Scans every run directory, reads its config.json to recover the mask
    parameters, rebuilds the mask, and runs verify_figure_masks().

    Parameters
    ----------
    sweep_dir  : path to the 'runs/' subdirectory of a sweep
    all_masks  : optional pre-built mask dict from sweep_masks() —
                 if supplied, fingerprints are compared against it directly
                 rather than rebuilt from config.json

    Returns
    -------
    pandas DataFrame with one row per run, columns:
        run_name, ok, n_missing_files, n_mismatches, bio_ok, details
    """
    import os, json
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required for verify_sweep()")

    rows = []
    run_names = sorted(
        d for d in os.listdir(sweep_dir)
        if os.path.isdir(os.path.join(sweep_dir, d))
    )

    for run_name in run_names:
        run_dir  = os.path.join(sweep_dir, run_name)
        cfg_path = os.path.join(run_dir, "config.json")

        if not os.path.exists(cfg_path):
            rows.append(dict(run_name=run_name, ok=False,
                             n_missing_files=1, n_mismatches=0,
                             bio_ok=False, details="missing config.json"))
            continue

        with open(cfg_path) as f:
            cfg = json.load(f)

        ms = cfg.get("mask_summary", {})
        topo = ms.get("topology", {})

        result = verify_figure_masks(
            n_spines             = cfg["n_spines"],
            n_dendrites_per_soma = cfg["n_dendrites_per_soma"],
            n_soma               = cfg["n_soma"],
            run_dir              = run_dir,
            itd_spine_frac       = topo.get("itd_spine_frac", 0.5),
            overlap              = topo.get("overlap", 0.0),
            dendrite_rule        = topo.get("dendrite_rule", "topographic"),
            channel_dend_split   = topo.get("channel_dend_split", True),
            mode                = cfg["mode"],
        )

        if verbose and not result["ok"]:
            print_verification(result)

        rows.append(dict(
            run_name        = run_name,
            ok              = result["ok"],
            n_missing_files = len(result["missing_files"]),
            n_mismatches    = len(result["mismatches"]),
            bio_ok          = result["checks"].get("biological_rules", False),
            details         = "; ".join(
                [f"missing: {f}" for f in result["missing_files"]] +
                [f"mismatch: {m['field']}" for m in result["mismatches"]]
            ) or "all checks passed",
        ))

    df = pd.DataFrame(rows)
    n_ok  = df["ok"].sum() if len(df) else 0
    n_tot = len(df)
    print(f"\nSweep verification: {n_ok}/{n_tot} runs OK")
    if not df["ok"].all() and len(df):
        print("Failed runs:")
        for _, row in df[~df["ok"]].iterrows():
            print(f"  {row['run_name']}: {row['details']}")
    return df



