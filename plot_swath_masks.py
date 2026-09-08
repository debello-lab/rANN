"""
plot_swath_masks.py
===================
Reads a config swath (dict-of-dicts or list-of-dicts) and plots the
connectivity masks for every run in a single combined figure.

Supports both storage formats:
  • .npz  (new format from sweep_config.py — fast binary)
  • .json (old mask_config.json format — lists converted to arrays)

Usage
-----
    from plot_swath_masks import plot_swath_masks

    # From a swath dict (cell 43 format)
    plot_swath_masks(swath_dict)

    # From a list of configs (build_sweep_configs format)
    configs = build_sweep_configs(...)
    plot_swath_masks(configs)

    # Just one panel type
    plot_swath_masks(configs, panels=["input_spine"])

    # Save without showing
    plot_swath_masks(configs, save_path="sweep_results/swath_masks.png", show=False)
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Union

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# Colour palette (matches visualize_masks.py)
# ---------------------------------------------------------------------------
BLUE   = "#378ADD"
CORAL  = "#D85A30"
TEAL   = "#1D9E75"
GRAY   = "#888780"
LGRAY  = "#F1EFE8"

PANEL_LABELS = {
    "input_spine":    "Input → Spines",
    "spine_dendrite": "Spines → Dendrites",
    "dendrite_soma":  "Dendrites → Soma",
}


# ---------------------------------------------------------------------------
# Mask loading — handles both .npz and .json formats
# ---------------------------------------------------------------------------

def _load_masks_from_config(cfg: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the three mask arrays from whatever path is stored in cfg.

    Tries, in order:
      1. cfg["mask_npz_path"]  → np.load (.npz, new format)
      2. cfg["mask_config_path"] → json.load, convert lists to arrays
    """
    # ── .npz path (sweep_config.py format) ───────────────────────────────────
    npz_path = cfg.get("mask_npz_path")
    if npz_path and os.path.exists(npz_path):
        data = np.load(npz_path)
        return (
            data["input_spine"].astype(np.float32),
            data["spine_dendrite"].astype(np.float32),
            data["dendrite_soma"].astype(np.float32),
        )

    # ── JSON path (generate_mask_and_mask_config format) ─────────────────────
    json_path = cfg.get("mask_config_path")
    if json_path and os.path.exists(json_path):
        with open(json_path) as f:
            mc = json.load(f)
        return (
            np.array(mc["input_spine"],    dtype=np.float32),
            np.array(mc["spine_dendrite"], dtype=np.float32),
            np.array(mc["dendrite_soma"],  dtype=np.float32),
        )

    raise FileNotFoundError(
        f"No loadable mask found for run '{cfg.get('run_name', '?')}'. "
        f"Checked mask_npz_path={npz_path!r} and mask_config_path={json_path!r}."
    )


def _iter_configs(swath: Union[dict, list]) -> list[dict]:
    """
    Normalise swath to a flat list of config dicts regardless of input format:
      • dict-of-dicts  {"run_1": {...}, "run_2": {...}}  → list of values
      • list-of-dicts  [{...}, {...}]                    → same list
      • tuple-of-dicts ({...}, {...})                    → unwrapped to list
        (guards against the common trailing-comma mistake:
         swath_dict = {...},  ← creates a 1-tuple, not a dict)
    """
    # Unwrap a 1-tuple caused by a trailing comma: swath_dict = {...},
    if isinstance(swath, tuple):
        if len(swath) == 1 and isinstance(swath[0], (dict, list)):
            swath = swath[0]
        else:
            raise TypeError(
                f"swath is a tuple with {len(swath)} element(s). "
                "If you wrote `swath_dict = {{...}},` with a trailing comma "
                "that creates a tuple — remove the trailing comma."
            )

    if isinstance(swath, dict):
        # Validate that the values are dicts (not strings from key iteration)
        values = list(swath.values())
        if values and not isinstance(values[0], dict):
            raise TypeError(
                f"swath dict values should be config dicts, "
                f"got {type(values[0]).__name__}. "
                "Check that you passed the swath dict itself, "
                "not a key or a nested value."
            )
        return values

    # list or other sequence
    items = list(swath)
    if items and not isinstance(items[0], dict):
        raise TypeError(
            f"swath list items should be config dicts, "
            f"got {type(items[0]).__name__}."
        )
    return items


def _config_label(cfg: dict) -> str:
    """Build a short multi-line label for a config."""
    mode = "constrained" if cfg.get("constrained", True) else "all-to-all"
    ns   = cfg.get("n_spines", "?")
    ndps = cfg.get("n_dendrites_per_soma", "?")
    nso  = cfg.get("n_soma", "?")
    ifn  = cfg.get("itd_spine_frac_name") or cfg.get("itd_spine_frac") or "—"
    ovn  = cfg.get("overlap_name") or cfg.get("overlap") or "—"
    dr   = cfg.get("dendrite_rule") or "—"
    name = cfg.get("run_name") or cfg.get("tag") or "run"
    return (
        f"{name}\n"
        f"[{mode}]\n"
        f"s={ns}  d={ndps}ps  so={nso}\n"
        f"frac={ifn}  ov={ovn}  dr={dr}"
    )


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def plot_swath_masks(
    swath: Union[dict, list],
    panels: Optional[List[str]] = None,
    cmap: str = "viridis",
    save_path: Optional[str] = None,
    show: bool = True,
    dpi: int = 150,
    max_cols: int = 6,
    fig_width_per_col: float = 3.2,
    fig_height_per_row: float = 2.8,
) -> plt.Figure:
    """
    Plot connectivity masks for every run in a config swath.

    Each run gets one column; each panel (mask stage) gets one row.
    Constrained runs are titled in blue, all-to-all in coral.

    Parameters
    ----------
    swath         : dict-of-dicts (cell 43 format) or list-of-dicts
                    (build_sweep_configs format). Each entry must have
                    either 'mask_npz_path' or 'mask_config_path'.
    panels        : which mask stages to show. Defaults to all three:
                    ["input_spine", "spine_dendrite", "dendrite_soma"]
    cmap          : matplotlib colormap for the heatmaps
    save_path     : if given, save the figure here
    show          : if True, call plt.show()
    dpi           : resolution for saved figure
    max_cols      : wrap into multiple rows of figures if more configs
    fig_width_per_col  : inches per config column
    fig_height_per_row : inches per mask-stage row

    Returns
    -------
    matplotlib Figure
    """
    configs = _iter_configs(swath)
    if not configs:
        raise ValueError("swath is empty — nothing to plot.")

    if panels is None:
        panels = ["input_spine", "spine_dendrite", "dendrite_soma"]
    invalid = [p for p in panels if p not in PANEL_LABELS]
    if invalid:
        raise ValueError(
            f"Unknown panels: {invalid}. "
            f"Valid options: {list(PANEL_LABELS.keys())}"
        )

    # ── Deduplicate all-to-all configs ───────────────────────────────────────
    # All-to-all masks for the same (n_spines, n_dendrites_per_soma, n_soma)
    # are always identical fully-dense matrices — plot each unique size once.
    # Constrained configs are always kept because their masks differ by topology.
    seen_alltoall_sizes: set = set()
    deduped: list = []
    for cfg in configs:
        if not cfg.get("constrained", True):
            size_key = (
                cfg.get("n_spines"),
                cfg.get("n_dendrites_per_soma"),
                cfg.get("n_soma"),
            )
            if size_key in seen_alltoall_sizes:
                continue   # skip — already have one all-to-all for this size
            seen_alltoall_sizes.add(size_key)
        deduped.append(cfg)

    n_skipped = len(configs) - len(deduped)
    if n_skipped:
        print(f"  plot_swath_masks: skipped {n_skipped} duplicate all-to-all "
              f"config(s) (same size, always identical mask).")
    configs = deduped

    n_configs = len(configs)
    n_panels  = len(panels)

    # ── Load all masks upfront (fast with .npz) ───────────────────────────────
    masks_per_run = []
    labels        = []
    load_errors   = []

    for cfg in configs:
        try:
            m1, m2, m3 = _load_masks_from_config(cfg)
            masks_per_run.append({"input_spine": m1,
                                   "spine_dendrite": m2,
                                   "dendrite_soma": m3})
            labels.append(_config_label(cfg))
        except FileNotFoundError as e:
            load_errors.append(str(e))
            masks_per_run.append(None)
            labels.append(_config_label(cfg) + "\n[MASK NOT FOUND]")

    if load_errors:
        print(f"Warning: {len(load_errors)} mask(s) could not be loaded:")
        for err in load_errors:
            print(f"  {err}")

    # ── Layout: if too many configs, split into pages ─────────────────────────
    # For simplicity we put everything in one figure and let the user scroll.
    # max_cols controls how wide before wrapping column groups.
    n_col_groups = (n_configs + max_cols - 1) // max_cols  # ceil division
    rows_total   = n_panels * n_col_groups

    fig_w = fig_width_per_col  * min(n_configs, max_cols)
    fig_h = fig_height_per_row * rows_total

    fig = plt.figure(figsize=(fig_w, fig_h))

    # Title
    mode_counts = {
        "constrained": sum(1 for c in configs if c.get("constrained", True)),
        "all-to-all":  sum(1 for c in configs if not c.get("constrained", True)),
    }
    fig.suptitle(
        f"Swath mask overview  —  {n_configs} configs  "
        f"({mode_counts['constrained']} constrained, {mode_counts['all-to-all']} all-to-all)",
        fontsize=11, fontweight="normal", y=1.01,
    )

    # ── Draw each config in its own column group ──────────────────────────────
    for group_idx in range(n_col_groups):
        group_start = group_idx * max_cols
        group_cfgs  = configs[group_start: group_start + max_cols]
        group_masks = masks_per_run[group_start: group_start + max_cols]
        group_lbls  = labels[group_start: group_start + max_cols]
        n_in_group  = len(group_cfgs)

        # Build a GridSpec block for this group
        gs = gridspec.GridSpec(
            n_panels, n_in_group,
            figure=fig,
            top=1.0 - group_idx * (n_panels / rows_total) - 0.01 / rows_total,
            bottom=1.0 - (group_idx + 1) * (n_panels / rows_total) + 0.02 / rows_total,
            hspace=0.55,
            wspace=0.35,
        )

        for col_idx, (cfg, masks, lbl) in enumerate(
            zip(group_cfgs, group_masks, group_lbls)
        ):
            constrained = cfg.get("constrained", True)
            title_color = BLUE if constrained else CORAL

            for row_idx, panel in enumerate(panels):
                ax = fig.add_subplot(gs[row_idx, col_idx])

                if masks is None:
                    # Mask not found — show placeholder
                    ax.set_facecolor("#f5f5f5")
                    ax.text(0.5, 0.5, "not found",
                            ha="center", va="center", fontsize=8,
                            color=GRAY, transform=ax.transAxes)
                    ax.set_xticks([]); ax.set_yticks([])
                else:
                    mat = masks[panel]   # shape depends on stage
                    im  = ax.imshow(
                        mat.T, aspect="auto", cmap=cmap,
                        interpolation="nearest", origin="upper",
                        vmin=0, vmax=1,
                    )

                    # Density annotation bottom-left
                    density = mat.mean()
                    active  = int(mat.sum())
                    ax.text(
                        0.02, 0.02,
                        f"density={density:.3f}\nactive={active:,}",
                        transform=ax.transAxes, fontsize=6,
                        color="white" if density > 0.4 else "black",
                        va="bottom", ha="left",
                        bbox=dict(boxstyle="round,pad=0.15",
                                  fc="black", alpha=0.35, lw=0),
                    )

                    # Channel boundary line for input_spine
                    if panel == "input_spine" and constrained:
                        n_spines = cfg.get("n_spines", mat.shape[1])
                        itd_frac = cfg.get("itd_spine_frac_name")
                        if itd_frac is not None:
                            # Try to draw the channel split line
                            try:
                                from bio_masks import ITD_SPINE_FRAC_CONFIGS
                                frac = ITD_SPINE_FRAC_CONFIGS.get(itd_frac, 0.5)
                                split = int(round(frac * n_spines))
                                ax.axhline(split - 0.5, color="white",
                                           lw=0.8, ls="--", alpha=0.7)
                                ax.text(mat.shape[0] * 0.02, split - 1,
                                        "Ch1/Ch2", fontsize=5,
                                        color="white", va="bottom")
                            except ImportError:
                                pass

                    ax.tick_params(labelsize=6)
                    _set_sparse_ticks(ax, mat.shape[0], axis="x")
                    _set_sparse_ticks(ax, mat.shape[1], axis="y")

                # Row label (left-most column only)
                if col_idx == 0:
                    ax.set_ylabel(PANEL_LABELS[panel], fontsize=8)
                else:
                    ax.set_ylabel("")

                # Column title (top panel only)
                if row_idx == 0:
                    ax.set_title(lbl, fontsize=7, color=title_color,
                                 pad=4, loc="center")

                # X label (bottom panel only)
                if row_idx == n_panels - 1:
                    ax.set_xlabel("pre-synaptic index", fontsize=6)

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_patches = [
        mpatches.Patch(color=BLUE,  label="constrained run"),
        mpatches.Patch(color=CORAL, label="all-to-all run"),
    ]
    fig.legend(
        handles=legend_patches, loc="lower center",
        ncol=2, fontsize=8, frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )

    fig.tight_layout(rect=[0, 0.03, 1, 0.99])

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved swath mask figure → {save_path}")

    if show:
        plt.show()

    return fig


# ---------------------------------------------------------------------------
# Companion: architecture overview across swath
# ---------------------------------------------------------------------------

def plot_swath_summary_table(
    swath: Union[dict, list],
    save_path: Optional[str] = None,
    show: bool = True,
    dpi: int = 150,
) -> plt.Figure:
    """
    Print a compact visual summary table of all configs in the swath:
    one row per config, columns for key parameters and mask densities.

    Useful as a quick sanity check before running a long sweep.
    """
    configs = _iter_configs(swath)
    rows = []

    for cfg in configs:
        constrained = cfg.get("constrained", True)
        row = {
            "run":            cfg.get("run_name") or cfg.get("tag") or "?",
            "mode":           "C" if constrained else "A",
            "spines":         cfg.get("n_spines", "?"),
            "d/soma":         cfg.get("n_dendrites_per_soma", "?"),
            "soma":           cfg.get("n_soma", "?"),
            "frac":           cfg.get("itd_spine_frac_name") or "—",
            "overlap":        cfg.get("overlap_name") or "—",
            "drule":          cfg.get("dendrite_rule") or "—",
            "ch_split":       cfg.get("channel_dend_split_name") or "—",
            "epochs":         cfg.get("epochs", "?"),
        }
        # Try to add density info
        try:
            m1, m2, m3 = _load_masks_from_config(cfg)
            row["density_m1"] = f"{m1.mean():.3f}"
            row["density_m2"] = f"{m2.mean():.3f}"
            row["density_m3"] = f"{m3.mean():.3f}"
            row["active_W"]   = f"{int(m1.sum()+m2.sum()+m3.sum()):,}"
        except FileNotFoundError:
            row["density_m1"] = row["density_m2"] = row["density_m3"] = "N/A"
            row["active_W"]   = "N/A"
        rows.append(row)

    col_keys = ["run", "mode", "spines", "d/soma", "soma",
                "frac", "overlap", "drule", "ch_split",
                "density_m1", "density_m2", "density_m3", "active_W", "epochs"]
    col_data  = [[str(r[k]) for r in rows] for k in col_keys]
    col_widths = [max(len(k), max(len(v) for v in vals))
                  for k, vals in zip(col_keys, col_data)]

    fig_w = min(max(sum(col_widths) * 0.13, 10), 28)
    fig_h = max(len(rows) * 0.38 + 1.2, 3)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    table = ax.table(
        cellText  = [[str(r[k]) for k in col_keys] for r in rows],
        colLabels = col_keys,
        cellLoc   = "center",
        loc       = "center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.auto_set_column_width(list(range(len(col_keys))))

    # Colour rows by mode
    for i, row in enumerate(rows):
        colour = "#E6F1FB" if row["mode"] == "C" else "#FAECE7"
        for j in range(len(col_keys)):
            table[i + 1, j].set_facecolor(colour)

    fig.suptitle(
        f"Swath config summary  —  {len(rows)} runs  "
        f"(C=constrained  A=all-to-all)",
        fontsize=10,
    )

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved summary table → {save_path}")
    if show:
        plt.show()

    return fig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _set_sparse_ticks(ax, length: int, axis: str = "x", max_ticks: int = 5):
    if length <= 0:
        return
    step = max(1, int(np.ceil(length / max_ticks)))
    ticks = list(range(0, length, step))
    if ticks and ticks[-1] != length - 1:
        ticks.append(length - 1)
    if axis == "x":
        ax.set_xticks(ticks)
    else:
        ax.set_yticks(ticks)


# ---------------------------------------------------------------------------
# CLI / quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Minimal smoke test with synthetic masks (no bio_masks dependency needed)
    print("Running smoke test with synthetic masks…")

    _fake_swath = {
        "run_1": {
            "run_name": "constrained_s64_d4ps_so32",
            "constrained": True,
            "n_spines": 64, "n_dendrites_per_soma": 4, "n_soma": 32,
            "itd_spine_frac_name": "equal_split",
            "overlap_name": "strict",
            "dendrite_rule": "topographic",
            "channel_dend_split_name": "split",
            "epochs": 500,
            # Inline arrays for testing — normally these come from a file
            "_test_masks": (
                np.random.randint(0, 2, (768, 64)).astype(np.float32),
                np.eye(64, 128).astype(np.float32),
                np.random.randint(0, 2, (128, 32)).astype(np.float32),
            ),
        },
        "run_2": {
            "run_name": "alltoall_s64_d4ps_so32",
            "constrained": False,
            "n_spines": 64, "n_dendrites_per_soma": 4, "n_soma": 32,
            "itd_spine_frac_name": None,
            "epochs": 500,
            "_test_masks": (
                np.ones((768, 64), dtype=np.float32),
                np.ones((64, 128), dtype=np.float32),
                np.ones((128, 32), dtype=np.float32),
            ),
        },
    }

    # Patch loader for the smoke test
    _orig_load = _load_masks_from_config
    def _test_load(cfg):
        if "_test_masks" in cfg:
            return cfg["_test_masks"]
        return _orig_load(cfg)

    import plot_swath_masks as _self
    _self._load_masks_from_config = _test_load

    plot_swath_masks(_fake_swath, show=False,
                     save_path="/tmp/smoke_swath_masks.png")
    plot_swath_summary_table(_fake_swath, show=False,
                             save_path="/tmp/smoke_swath_table.png")
    print("Smoke test complete — figures saved to /tmp/")