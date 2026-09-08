"""
sweep_config.py
===============
Builds per-run configuration dictionaries and pre-generates masks for the
rANN parameter sweep.

Design
------
Numpy arrays cannot live in a JSON-serialisable dict, so masks are stored on
disk as .npz (binary, typed, compressed) files. Each run's config dict holds
only a string path to that file.  When the sweep loop needs the arrays it
calls load_mask(path) — one line, instant load.

Layout
------
<output_root>/
    <tag>/
        config.json        ← run metadata (JSON-safe scalars only)
        mask.npz           ← binary mask arrays (float32, compressed)
        [generated after training]
        history.json
        model.keras
        trained_weights_<tag>.png

Usage
-----
    # 1. Build one config
    cfg = generate_config(
        output_root = "sweep_results/runs",
        mode        = "constrained",
        n_spines    = 128,
        n_dendrites_per_soma = 4,
        n_soma      = 128,
        itd_spine_frac_name      = "equal_split",
        overlap_name             = "strict",
        dendrite_rule            = "topographic",
        channel_dend_split_name  = "split",
    )

    # 2. Build a sweep list automatically
    configs = build_sweep_configs(
        output_root          = "sweep_results/runs",
        spine_sizes          = [128, 256],
        dendrite_per_soma_sizes = [4, 8],
        soma_sizes           = [128],
        itd_spine_frac_names = ["equal_split"],
        overlap_names        = ["strict", "50pct"],
        dendrite_rules       = ["topographic"],
        channel_dend_split_names = ["split"],
        include_baseline     = True,
    )

    # 3. Load masks inside the sweep loop
    for cfg in configs:
        m1, m2, m3, summary = load_mask(cfg["mask_npz_path"])
        # train...
"""

from __future__ import annotations

import json
import os
import time
from itertools import product
from typing import List, Optional
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf

# ---------------------------------------------------------------------------
# Bio-mask imports — replaces the stub np.zeros calls
# ---------------------------------------------------------------------------
from bio_masks import (
    build_masks,
    ITD_SPINE_FRAC_CONFIGS,
    OVERLAP_CONFIGS,
    CHANNEL_DEND_SPLIT_CONFIGS,
)

from rANN_model import rANN, PerTargetMAECallback, TargetMAE


# ---------------------------------------------------------------------------
# Mask I/O  (the key piece — replaces JSON serialisation of arrays)
# ---------------------------------------------------------------------------

def save_mask(masks: dict, path: str) -> str:
    """
    Save mask arrays to a compressed .npz file.

    Parameters
    ----------
    masks : dict returned by build_masks()
    path  : full file path, e.g. "runs/constrained_s128/mask.npz"

    Returns
    -------
    Absolute path of the saved file.
    """
    np.savez_compressed(
        path,
        input_spine    = masks["input_spine"],
        spine_dendrite = masks["spine_dendrite"],
        dendrite_soma  = masks["dendrite_soma"],
    )
    # np.savez appends .npz if not present
    actual_path = path if path.endswith(".npz") else path + ".npz"
    return os.path.abspath(actual_path)


def load_mask(npz_path: str):
    """
    Load mask arrays from a .npz file.

    Returns
    -------
    (input_spine, spine_dendrite, dendrite_soma)  — all float32 np.ndarrays
    """
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Mask file not found: {npz_path}")
    data = np.load(npz_path)
    return (
        data["input_spine"].astype(np.float32),
        data["spine_dendrite"].astype(np.float32),
        data["dendrite_soma"].astype(np.float32),
    )


# ---------------------------------------------------------------------------
# Single config builder
# ---------------------------------------------------------------------------

def generate_config(
    output_root:             str,
    mode:                    str,           # "constrained" | "alltoall"
    n_spines:                int,
    n_dendrites_per_soma:    int,
    n_soma:                  int,
    spine_activation:         str = "relu",
    dendrite_activation:      str = "relu",
    soma_activation:          str = "relu",
    # topology (ignored for alltoall — set to None)
    itd_spine_frac_name:     Optional[str] = None,
    overlap_name:            Optional[str] = None,
    dendrite_rule:           Optional[str] = None,
    channel_dend_split_name: Optional[str] = None,
    # training hyperparams
    epochs:                  int   = 500,
    batch_size:              int   = 32,
    lr:                      float = 1e-3,
    l2_reg:                  float = 1e-4,
    patience:                Optional[int] = None,
    # dataset dimensions (must match make_data.py defaults)
    n_time: int = 8, n_freq: int = 32, n_itd: int = 32, n_ild: int = 32,
    input_dim: int = 8*32*3,
    output_dim: int = 2,
    # options
    overwrite:               bool = False,
) -> dict:
    """
    Build one run directory, generate (or reuse) the mask, and return a
    JSON-safe config dict.

    The dict contains only scalars and strings — no numpy arrays.
    The mask lives in mask.npz; its path is stored under "mask_npz_path".

    Returns
    -------
    config dict (also written to config.json in the run directory)
    """
    if mode not in ("constrained", "alltoall"):
        raise ValueError(f"mode must be 'constrained' or 'alltoall', got {mode!r}")

    total_dendrites = n_dendrites_per_soma * n_soma

    if input_dim != n_time * (n_freq + n_itd + n_ild):
        raise ValueError(f"input_dim must be n_time*(n_freq+n_itd+n_ild), got {input_dim}")
    
    # ── Build tag ─────────────────────────────────────────────────────────────
    if mode == "constrained":
        if any(v is None for v in [itd_spine_frac_name, overlap_name,
                                    dendrite_rule, channel_dend_split_name]):
            raise ValueError("All topology params required for constrained mode.")
        topo_tag = (
            f"_if{itd_spine_frac_name}_ov{overlap_name}"
            f"_dr{dendrite_rule}_cd{channel_dend_split_name}"
        )
    else:
        topo_tag = ""

    tag     = f"{mode}_s{n_spines}_d{n_dendrites_per_soma}ps_so{n_soma}{topo_tag}"
    run_dir = os.path.join(output_root, tag)
    os.makedirs(run_dir, exist_ok=True)

    # ── Generate or reuse mask ────────────────────────────────────────────────
    npz_path    = os.path.join(run_dir, "mask.npz")
    meta_path   = os.path.join(run_dir, "mask_meta.json")
    cfg_path    = os.path.join(run_dir, "config.json")

    if not os.path.exists(npz_path) or overwrite:
        if mode == "constrained":
            masks = build_masks(
                n_spines             = n_spines,
                n_dendrites_per_soma = n_dendrites_per_soma,
                n_soma               = n_soma,
                itd_spine_frac       = ITD_SPINE_FRAC_CONFIGS[itd_spine_frac_name],
                overlap              = OVERLAP_CONFIGS[overlap_name],
                dendrite_rule        = dendrite_rule,
                channel_dend_split   = CHANNEL_DEND_SPLIT_CONFIGS[channel_dend_split_name],
                n_time=n_time, n_freq=n_freq, n_itd=n_itd, n_ild=n_ild,
            )
            summary = masks["summary"]
        else:
            # alltoall: fully dense masks
            masks = _alltoall_masks(input_dim, n_spines, total_dendrites, n_soma)
            summary = _alltoall_summary(
                input_dim, n_spines, n_dendrites_per_soma, total_dendrites, n_soma
            )

        save_mask(masks, npz_path)

        # Save summary separately as JSON (small scalars — no arrays)
        _safe_summary = _make_json_safe(summary)
        with open(meta_path, "w") as f:
            json.dump(_safe_summary, f, indent=2)

        print(f"  Saved mask  → {npz_path}")
    else:
        with open(meta_path) as f:
            summary = json.load(f)
        print(f"  Reused mask → {npz_path}")

    # ── Build config dict (JSON-safe only) ───────────────────────────────────
    config = {
        # Identity
        "run_name":     tag,
        "tag":          tag,
        "run_dir":      run_dir,
        "mode":         mode,
        "constrained":  mode == "constrained",
        # Mask location — the only reference to the arrays
        "mask_npz_path": os.path.abspath(npz_path),
        "mask_meta_path": os.path.abspath(meta_path),
        # Architecture sizes (redundant with summary but handy at top-level)
        "input_dim":            input_dim,
        "output_dim":           output_dim,
        "n_spines":             n_spines,
        "n_dendrites_per_soma": n_dendrites_per_soma,
        "total_dendrites":      total_dendrites,
        "n_soma":               n_soma,
        'spine_activation':        spine_activation,
        'dendrite_activation':     dendrite_activation,
        'soma_activation':         soma_activation,
        # Topology
        "itd_spine_frac_name":     itd_spine_frac_name,
        "overlap_name":            overlap_name,
        "dendrite_rule":           dendrite_rule,
        "channel_dend_split_name": channel_dend_split_name,
        # Training
        "epochs":     epochs,
        "batch_size": batch_size,
        "lr":         lr,
        "l2_reg":     l2_reg,
        "patience":   patience,
        # Populated after training
        "n_epochs_trained": 0,
        "final_val_loss":   None,
        "final_mae_az":     None,
        "final_mae_el":     None,
        "wall_time_s":      None,
    }

    with open(cfg_path, "w") as f:
        json.dump(config, f, indent=2)

    return config


# ---------------------------------------------------------------------------
# Sweep config builder
# ---------------------------------------------------------------------------

def build_sweep_configs(
    output_root:             str,
    spine_sizes:             List[int]  = (128,),
    spine_activations:         List[str]  = ("relu",),
    dendrite_activations:     List[str]  = ("relu",),
    soma_activations:         List[str]  = ("relu",),
    dendrite_per_soma_sizes: List[int]  = (4,),
    soma_sizes:              List[int]  = (128,),
    itd_spine_frac_names:    List[str]  = ("equal_split",),
    overlap_names:           List[str]  = ("strict",),
    dendrite_rules:          List[str]  = ("topographic",),
    channel_dend_split_names:List[str]  = ("split",),
    include_baseline:        bool       = True,
    input_dim:               int        = 8*(32+32+32),  # n_time*(n_freq+n_itd+n_ild)
    output_dim:              int        = 2,
    epochs:                  int        = 500,
    batch_size:              int        = 32,
    lr:                      float      = 1e-3,
    l2_reg:                  float      = 1e-4,
    n_time: int = 8, n_freq: int = 32, n_itd: int = 32, n_ild: int = 32,
    overwrite:               bool       = False,
) -> list[dict]:
    """
    Generate a config dict for every combination in the sweep grid.

    Returns a flat list of config dicts ready to pass to _run_sweep.
    Each entry is JSON-safe; masks are pre-built and saved to disk.

    The list alternates constrained / alltoall for easy paired comparison:
        [constrained_A, alltoall_A, constrained_B, alltoall_B, ...]
    """
    configs = []

    size_combos = list(product(spine_sizes, dendrite_per_soma_sizes, soma_sizes))
    topo_combos = list(product(
        itd_spine_frac_names, overlap_names, dendrite_rules, channel_dend_split_names
    ))
    activation_combos = list(product(spine_activations, dendrite_activations, soma_activations))

    n_total = (len(topo_combos) + 1) * len(size_combos) * len(activation_combos)

    print(f"Building {n_total} configs "
          f"({len(size_combos)} size × {len(topo_combos)} topology × {len(activation_combos)} activation"
          f"{' + ' + str(len(size_combos) * len(activation_combos)) + ' (+ baseline)' if include_baseline else ''})…")

    seen_activations = []

    for ns, ndps, nso in size_combos:
        for ifn, ovn, dr, cdn in topo_combos:
            for sa, da, soa in activation_combos:

                shared = dict(
                    output_root=output_root,
                    spine_activation=sa,
                    dendrite_activation=da,
                    soma_activation=soa,
                    n_spines=ns,
                    n_dendrites_per_soma=ndps,
                    n_soma=nso,
                    epochs=epochs,
                    batch_size=batch_size,
                    lr=lr,
                    l2_reg=l2_reg,
                    n_time=n_time, n_freq=n_freq, n_itd=n_itd, n_ild=n_ild,
                    input_dim=input_dim,
                    output_dim=output_dim,
                    overwrite=overwrite,
                )

                # Constrained run
                cfg_c = generate_config(
                    mode="constrained",
                    itd_spine_frac_name=ifn,
                    overlap_name=ovn,
                    dendrite_rule=dr,
                    channel_dend_split_name=cdn,
                    **shared,
                )
                configs.append(cfg_c)

                # All-to-all baseline (same size, no topology)
                if include_baseline:
                    # Skip duplicate activations for all-to-all baseline
                    if (sa, da, soa) in seen_activations:
                        continue
                    seen_activations.append((sa, da, soa))
                    cfg_a = generate_config(
                        mode="alltoall",
                        **shared,
                    )
                    configs.append(cfg_a)

    print(f"Done — {len(configs)} configs ready.")
    return configs


# ---------------------------------------------------------------------------
# Sweep runner (replaces _run_sweep)
# ---------------------------------------------------------------------------

def run_sweep_from_configs(
    configs:        list[dict],
    training_data:   tuple[np.ndarray, np.ndarray],
    testing_data:    tuple[np.ndarray, np.ndarray],
    plot_history:   bool = False,
    plot_weights:   bool = False,
    output_root:   str = "sweep_results",
    save_csv:      bool = True,
) -> "pd.DataFrame":
    """
    Run training for every config in the list.

    Inside the loop, masks are loaded on demand from each config's
    mask_npz_path — never stored in memory all at once.

    Parameters
    ----------
    configs       : list of dicts from build_sweep_configs()
    training_data : (x_tr, y_tr)
    testing_data  : (x_te, y_te)
    plot_history  : if True, save a loss/MAE curve PNG per run
    plot_weights  : if True, save a trained-weight heatmap PNG per run
    patience      : early-stopping patience passed to _run_one
    output_root   : root directory; sweep_summary.csv is written here
    save_csv      : if True, write/update sweep_summary.csv after every run

    Returns
    -------
    pandas DataFrame with one row per run.
    """
    os.makedirs(output_root, exist_ok=True)
    csv_path = os.path.join(output_root, "sweep_summary.csv")
    rows = []

    for i, cfg in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}]  {cfg['tag']}")

        # ── Mask summary (small JSON — scalars only) ──────────────────────────
        meta_path = cfg.get("mask_meta_path")
        if meta_path and os.path.exists(meta_path):
            with open(meta_path) as f:
                cfg["mask_summary"] = json.load(f)

        # ── Topology figures (idempotent) ─────────────────────────────────────
        _generate_topology_figures(cfg, output_root)

        # ── Train or resume ───────────────────────────────────────────────────
        cfg = _load_or_train_model(cfg, training_data, testing_data, plot_history=plot_history, plot_weights=plot_weights)
        rows.append(cfg.copy())

        if save_csv:
            pd.DataFrame(rows).to_csv(csv_path, index=False)

    df = pd.DataFrame(rows)
    if save_csv:
        df.to_csv(csv_path, index=False)
        print(f"\nSweep complete → {csv_path}")
    return df

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _generate_topology_figures(cfg: dict, output_root: str) -> None:
    """Generate mask + architecture figures into cfg['run_dir'], update cfg in place."""
    try:
        from visualize_masks import plot_masks, plot_architectures
    except ImportError:
        return

    mode    = cfg["mode"]
    run_dir = cfg["run_dir"]
    ns, ndps, nso = cfg["n_spines"], cfg["n_dendrites_per_soma"], cfg["n_soma"]
    suffix  = cfg.get("file_suffix") or _build_file_suffix(cfg)

    masks_png = os.path.join(run_dir, f"masks_{mode}_s{ns}_dps{ndps}_so{nso}_{suffix}.png")
    arch_png  = os.path.join(run_dir, f"arch_{mode}_s{ns}_dps{ndps}_so{nso}_{suffix}.png")

    # Skip if both already exist
    if os.path.exists(masks_png) and os.path.exists(arch_png):
        cfg.setdefault("topology_png", os.path.relpath(masks_png, output_root))
        cfg.setdefault("arch_png",     os.path.relpath(arch_png,  output_root))
        return

    constrained = cfg["constrained"]
    kwargs = dict(
        outdir=run_dir, save=True, mode=mode, file_suffix=suffix,
        itd_spine_frac = ITD_SPINE_FRAC_CONFIGS.get(cfg.get("itd_spine_frac_name")) if constrained else None,
        overlap        = OVERLAP_CONFIGS.get(cfg.get("overlap_name"))                if constrained else None,
        dendrite_rule  = cfg.get("dendrite_rule")                                    if constrained else None,
        channel_dend_split = CHANNEL_DEND_SPLIT_CONFIGS.get(cfg.get("channel_dend_split_name")) if constrained else None,
    )

    try:
        plot_masks(ns, ndps, nso, **kwargs)
        plot_architectures(ns, ndps, nso, **kwargs)
        cfg["topology_png"] = os.path.relpath(masks_png, output_root)
        cfg["arch_png"]     = os.path.relpath(arch_png,  output_root)
    except Exception as e:
        import traceback
        print(f"  ERROR generating figures: {e}")
        traceback.print_exc()


def _build_file_suffix(cfg: dict) -> str:
    """Reconstruct the file suffix from cfg topology fields."""
    if not cfg.get("constrained"):
        return "alltoall"
    return (
        f"if{cfg['itd_spine_frac_name']}"
        f"_ov{cfg['overlap_name']}"
        f"_dr{cfg['dendrite_rule']}"
        f"_cd{cfg['channel_dend_split_name']}"
    )

def _load_or_train_model(cfg, training_data, testing_data,
    plot_history, plot_weights
):
    """Train or load a model, return full row dict for the summary."""
    run_dir = cfg['run_dir']
    tag     = cfg['tag']
    
    if os.path.exists(os.path.join(run_dir, "model.keras")):
        x_te, y_te = testing_data

        # Load saved model and metrics
        print(f"  Skipping (already exists) — loading metrics from config.json")
        with open(os.path.join(run_dir, "config.json")) as f:
            saved_cfg = json.load(f)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            y_pred = tf.keras.models.load_model(run_dir + "/model.keras")(x_te).numpy()

        abs_err = np.abs(y_pred - y_te)
        val_losses = cfg['history']["val_loss"]
        best_val = min(val_losses)
        threshold = val_losses[0] - 0.9 * (val_losses[0] - best_val)
        epoch_90 = next((i + 1 for i, v in enumerate(val_losses) if v <= threshold), len(val_losses))

        if plot_history:
            _plot_training_history(cfg['history'], run_name=tag, run_dir=run_dir, show=False)

        ms = cfg.get("mask_summary")

        cfg.update({
            "trainable_params": saved_cfg.get("trainable_params"),
            "final_train_loss": cfg['history']["loss"][-1],
            "final_val_loss":   best_val,
            "final_mae_az":     float(np.mean(abs_err[:, 0])),
            "final_mae_el":     float(np.mean(abs_err[:, 1])),
            "epoch_90pct":      epoch_90,
            "wall_time_s":      None,
            "active_weights":   ms["active_weights"]["total"]   if ms else None,
            "all2all_weights":  ms["all2all_weights"]["total"]  if ms else None,
            "weight_reduction": ms["weight_reduction_pct"]      if ms else None,
            "density_overall":  ms["density"]["overall"]        if ms else None,
        })
        
        return cfg
    else:
        # Train new model
        return _run_one(cfg, training_data, testing_data, plot_history=plot_history, plot_weights=plot_weights)

def _run_one(cfg, training_data, testing_data, plot_history=False, plot_weights=False) -> dict:
    """
    Run one training session for a single config dict.
    Returns a dict of results (JSON-safe).
    """
    tag = cfg["tag"]

    # Load masks on demand
    m1, m2, m3 = load_mask(cfg["mask_npz_path"])
    spine_mask    = m1 if cfg["constrained"] else None
    dendrite_mask = m2 if cfg["constrained"] else None
    soma_mask     = m3 if cfg["constrained"] else None

    # ── Build model ───────────────────────────────────────────────────────────
    model = rANN(
        input_dim    = cfg['input_dim'],
        output_dim   = cfg['output_dim'],
        n_spines     = cfg['n_spines'],
        n_dendrites_per_soma = cfg['n_dendrites_per_soma'],
        n_soma       = cfg['n_soma'],
        spine_mask    = spine_mask,
        dendrite_mask = dendrite_mask,
        soma_mask     = soma_mask,
        l2_reg        = cfg['l2_reg'],
    )
    model.compile(
        optimizer = 'adam', #tf.keras.optimizers.Adam(learning_rate=lr),
        loss      = tf.keras.losses.MeanSquaredError(),
        metrics   = [
            tf.keras.metrics.MeanAbsoluteError(name="mae"),
            tf.keras.metrics.Accuracy(name="accuracy"),
            TargetMAE(0, name="mae_az"),
            TargetMAE(1, name="mae_el"),
        ],
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    t_start = time.perf_counter()

    # Early stopping: stop if val_loss plateaus (prevent overfitting) and restore best weights at the end
    if cfg['patience'] is not None and cfg['patience'] > 0:
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=cfg['patience'],           # stop if no improvement for x epochs
            restore_best_weights=True,
            verbose=0,
        )   

    x_tr, y_tr = training_data
    x_te, y_te = testing_data
    
    ptcb = PerTargetMAECallback(x_te, y_te, x_train=x_tr, y_train=y_tr)

    # Convert to TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices(training_data).batch(cfg['batch_size'])
    test_dataset = tf.data.Dataset.from_tensor_slices(testing_data).batch(cfg['batch_size'])

    if x_tr.shape[1] != cfg['input_dim']:
        raise ValueError(f"Input dimension mismatch: expected {cfg['input_dim']}, got {x_tr.shape[1]}")
    if y_tr.shape[1] != cfg['output_dim']:
        raise ValueError(f"Output dimension mismatch: expected {cfg['output_dim']}, got {y_tr.shape[1]}")

    callbacks = [ptcb]
    if cfg.get('patience') and cfg['patience'] > 0:
        callbacks.append(early_stop)

    hist = model.fit(
        train_dataset, 
        validation_data=test_dataset, 
        epochs=cfg['epochs'], 
        batch_size=cfg['batch_size'],
        verbose=0,
        callbacks=callbacks
        )

    wall_time = time.perf_counter() - t_start

    # ── Per-target MAE ────────────────────────────────────────────────────────
    y_pred   = model.predict(x_te, verbose=0)
    abs_err  = np.abs(y_pred - y_te)
    mae_az   = float(np.mean(abs_err[:, 0]))
    mae_el   = float(np.mean(abs_err[:, 1]))


    # Epoch at which val_loss first reached 90% of its final improvement
    val_losses = hist.history["val_loss"]
    best_val   = min(val_losses)
    threshold  = val_losses[0] - 0.9 * (val_losses[0] - best_val)
    epoch_90   = next((i + 1 for i, v in enumerate(val_losses) if v <= threshold), cfg['epochs'])
    convergence_efficiency = epoch_90 / (best_val + 1e-8)

    print(
        f"  {tag:<40}  "
        f"val_loss={best_val:.3f}  "
        f"mae_az={mae_az:.2f}°  mae_el={mae_el:.2f}°  "
        f"epoch_90={epoch_90:>4}  "
        f"time={wall_time:.1f}s"
        )

    # ── Save model ─────────────────────────────────────────────────────────
    model.save(os.path.join(cfg['run_dir'], "model.keras"))
    
    # ── Save history ─────────────────────────────────────────────────────────
    if "val_mae_az" in hist.history and "val_mae_el" in hist.history:
        az = hist.history["val_mae_az"]
        el = hist.history["val_mae_el"]
        hist.history["val_mae"] = [(a + e) / 2.0 for a, e in zip(az, el)]

    if plot_history:
        _plot_training_history(hist.history, run_name=tag, run_dir=cfg['run_dir'], show=True)

    if plot_weights:
        _plot_weights(model, run_name=tag, run_dir=cfg['run_dir'], show=True)

    # ── Update config on disk with results ──────────────────────────────
    cfg['final_train_loss'] = float(hist.history["loss"][-1])
    cfg['final_val_loss']  = best_val
    cfg['final_mae_az']    = mae_az
    cfg['final_mae_el']    = mae_el
    cfg['epoch_90pct']     = epoch_90
    cfg['convergence_efficiency'] = convergence_efficiency
    cfg['wall_time_s']     = round(wall_time, 2)
    cfg['history']         = hist.history
    with open(os.path.join(cfg["run_dir"], "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    return cfg

def _plot_training_history(history: dict, run_name: str = None, run_dir: str = None, show: bool = True, accuracy: bool = False):
    """Plot the same loss and validation per-target MAE curves shown in the notebook training cell."""
    loss_values = history.get("loss", [])
    if not loss_values:
        return None

    epochs = np.arange(1, len(loss_values) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, loss_values, label="Train Loss", color="tab:blue")
    if accuracy:
        axes[0].plot(epochs, history.get("accuracy", []), label="Train Accuracy", color="tab:blue", linestyle="--")
    if "val_loss" in history:
        axes[0].plot(epochs, history["val_loss"], label="Val Loss", color="tab:orange")
    if accuracy and "val_accuracy" in history:
        axes[0].plot(epochs, history["val_accuracy"], label="Val Accuracy", color="tab:orange", linestyle="--")
    axes[0].set_title("Loss/Accuracy vs Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss/Accuracy")
    axes[0].set_xscale("log")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    if "val_mae_az" in history:
        axes[1].plot(epochs, history["val_mae_az"], color="tab:orange", linestyle="--", label="Val MAE Azimuth")
    if "val_mae_el" in history:
        axes[1].plot(epochs, history["val_mae_el"], color="tab:orange", label="Val MAE Elevation")

    # Also plot training per-target MAE (dashed lines) when available
    if "mae_az" in history:
        axes[1].plot(epochs, history["mae_az"], color="tab:blue", linestyle="--", label="Train MAE Azimuth")
    if "mae_el" in history:
        axes[1].plot(epochs, history["mae_el"], color="tab:blue", label="Train MAE Elevation")

    axes[1].set_title("Validation Per-Target MAE")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAE (deg)")
    axes[1].set_xscale("log")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    if run_name:
        fig.suptitle(run_name)
    plt.tight_layout()

    if run_dir is not None:
        plot_path = os.path.join(run_dir, "training_history.png")
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"  Saved training plot -> {plot_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig

def _plot_weights(model = None, run_name: str = None, run_dir: str = None, show: bool = True):
    """Visualize the learned weights of the model's spine, dendrite, and soma layers."""
    
    layers_to_plot = [model.spine_layer, model.dendrite_layer, model.soma_layer]

    fig, axes = plt.subplots(2, len(layers_to_plot), figsize=(5 * len(layers_to_plot), 10))

    for col, layer in enumerate(layers_to_plot):

        weights = layer.w.numpy() 

        # Row 0: Heatmap of weights
        ax_img = axes[0, col]
        im = ax_img.imshow(weights, aspect="auto", cmap="viridis")
        ax_img.set_title(f"{layer.name} weights")
        ax_img.set_xlabel("Output units")
        ax_img.set_ylabel("Input units")
        fig.colorbar(im, ax=ax_img, fraction=0.046, pad=0.04)

        # Row 1: weight distribution
        ax_dist = axes[1, col]
        flat = weights.flatten()
        active = flat[np.abs(flat) > 1e-8] # exclude masked zeros

        ax_dist.hist(active, bins=50, alpha=0.7, color="blue", edgecolor="black")
        ax_dist.axvline(0, color="#D85A30", lw=1.0, ls="--", label="zero")
        ax_dist.axvline(active.mean(), color="#1D9E75", lw=1.2, ls="-",
                        label=f"μ={active.mean():.3f}")
        ax_dist.set_title(f"{layer.name} weight distribution")
        ax_dist.set_xlabel("Weight value")
        ax_dist.set_ylabel("Frequency")
        ax_dist.grid(alpha=0.3)
        ax_dist.legend()

        # Stats annotation
        ax_dist.text(0.97, 0.95,
                     f"σ={active.std():.3f}\n"
                     f"max={active.max():.3f}\n"
                     f"non-zero: {len(active):,}/{len(flat):,}",
                     transform=ax_dist.transAxes, fontsize=7,
                     va="top", ha="right",
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7, lw=0))

        # Row labels on left edge
        axes[0, 0].set_ylabel("Post-synaptic index\n(heatmap)", fontsize=8)
        axes[1, 0].set_ylabel("Count\n(distribution)", fontsize=8)

    if run_name:
        fig.suptitle(f"Trained weights — {run_name}", fontsize=11, y=1.01)

    plt.tight_layout()

    if run_dir is not None:
        plot_path = os.path.join(run_dir, "final_weights.png")
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"  Saved training plot -> {plot_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig

def _alltoall_masks(input_dim, n_spines, total_dendrites, n_soma):
    """Return fully dense (all-ones) masks for the all-to-all baseline."""
    return {
        "input_spine":    np.ones((input_dim,       n_spines),       dtype=np.float32),
        "spine_dendrite": np.ones((n_spines,         total_dendrites), dtype=np.float32),
        "dendrite_soma":  np.ones((total_dendrites,  n_soma),          dtype=np.float32),
    }


def _alltoall_summary(input_dim, n_spines, n_dps, total_dendrites, n_soma):
    aa = input_dim * n_spines + n_spines * total_dendrites + total_dendrites * n_soma
    return {
        "config": {
            "input_dim": input_dim, "n_spines": n_spines,
            "n_dendrites_per_soma": n_dps,
            "total_dendrites": total_dendrites, "n_soma": n_soma,
        },
        "topology": {
            "itd_spine_frac": None, "overlap": None,
            "dendrite_rule": None, "channel_dend_split": None,
        },
        "active_weights": {"total": aa},
        "all2all_weights": {"total": aa},
        "density": {"overall": 1.0},
        "weight_reduction_pct": 0.0,
    }


def _make_json_safe(obj):
    """Recursively convert numpy scalars to Python natives for JSON."""
    if isinstance(obj, dict):
        return {k: _make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_json_safe(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        # Should not reach here — arrays go to .npz
        raise TypeError(f"numpy array found in summary — move it to .npz: {obj.shape}")
    return obj


def load_config(run_dir: str) -> dict:
    """Load a config.json from a run directory."""
    with open(os.path.join(run_dir, "config.json")) as f:
        return json.load(f)


def load_sweep_configs(output_root: str) -> list[dict]:
    """
    Reconstruct a config list by scanning all run directories.
    Useful for resuming a sweep or post-hoc analysis.
    """
    configs = []
    if not os.path.isdir(output_root):
        return configs
    for name in sorted(os.listdir(output_root)):
        cfg_path = os.path.join(output_root, name, "config.json")
        if os.path.exists(cfg_path):
            configs.append(load_config(os.path.join(output_root, name)))
    return configs
