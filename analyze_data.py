"""Helpers for analyzing sweep_summary.csv and plotting history data from individual runs.

The notebook can import these functions to keep plotting logic reusable.
"""

from __future__ import annotations

import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_SUMMARY_CSV = "/home/mrsarti/rANN/sweep_results/sweep_summary.csv"


def load_sweep_summary(summary_csv: str = DEFAULT_SUMMARY_CSV) -> pd.DataFrame:
    """Load the sweep summary CSV and validate that it exists."""
    if not os.path.exists(summary_csv):
        raise FileNotFoundError(f"Summary CSV not found: {summary_csv}")
    return pd.read_csv(summary_csv)


def _resolve_metric_col(df: pd.DataFrame, metric_col: str | None) -> str:
    if metric_col is not None:
        if metric_col not in df.columns:
            raise KeyError(f"'{metric_col}' does not exist in df. Available columns: {list(df.columns)}")
        return metric_col
    raise KeyError(f"'Please specify a valid metric column. Available columns: {list(df.columns)}")


def _make_ann_label(row: pd.Series) -> str:
    parts = [
        #row.get("run_name", "run"),
        f"if={row.get('itd_frac_name', 'na')}",
        f"ov={row.get('overlap_name', 'na')}",
        f"dr={row.get('dendrite_rule', 'na')}",
        f"cd={row.get('channel_dend_split_name', 'na')}",
        f"s={row.get('n_spines', 'na')}",
        f"dps={row.get('n_dendrites_per_soma', 'na')}",
        f"so={row.get('n_soma', 'na')}",
    ]
    return " | ".join(map(str, parts))


def plot_epoch_90pct_ranking(
    summary: str | pd.DataFrame = DEFAULT_SUMMARY_CSV,
    metric_col: str | None = None,
    figsize: tuple[float, float] | None = None,
    show: bool = True,
):
    """Plot all ANN types as a sorted horizontal bar chart."""
    df = load_sweep_summary(summary) if isinstance(summary, str) else summary.copy()
    metric_col = _resolve_metric_col(df, metric_col)

    plot_df = df.dropna(subset=[metric_col]).copy()
    if plot_df.empty:
        raise ValueError(f"No non-null values found in {metric_col}.")

    plot_df["ann_type"] = plot_df.apply(_make_ann_label, axis=1)
    plot_df = plot_df.sort_values(metric_col, ascending=True).reset_index(drop=True)

    if figsize is None:
        figsize = (14, max(6, 0.28 * len(plot_df)))

    fig, ax = plt.subplots(figsize=figsize)
    colors = np.where(plot_df["constrained"].astype(bool), "tab:blue", "tab:orange")
    ax.barh(plot_df["ann_type"], plot_df[metric_col], color=colors, alpha=0.9)
    ax.invert_yaxis()
    ax.set_xlabel(metric_col)
    ax.set_ylabel("ANN type")
    ax.set_title(f"Sorted {metric_col} across all ANN types")
    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()

    if show:
        plt.show()
    return fig, ax, plot_df

def plot_by_metrics(
        summary: str | pd.DataFrame = DEFAULT_SUMMARY_CSV,
        metric_x_col: str | None = None,
        metric_y_col: str | None = None,
        figsize: tuple[float, float] | None = None,
        show: bool = True,
):
    """Bidirectional bar graph of constrained vs alltoall metric y plotted along metric x."""
    df = load_sweep_summary(summary) if isinstance(summary, str) else summary.copy()
    metric_x_col = _resolve_metric_col(df, metric_x_col)
    metric_y_col = _resolve_metric_col(df, metric_y_col)

    comparison_df = df.dropna(subset=[metric_x_col, metric_y_col]).copy()
    if comparison_df.empty:
        raise ValueError(f"No non-null values found for {metric_x_col} and {metric_y_col}.")
    
    comparison_df["ann_type"] = comparison_df.apply(_make_ann_label, axis=1)
    comparison_df = comparison_df.sort_values(metric_x_col, ascending=True).reset_index(drop=True)

    if figsize is None:
        figsize = (14, max(6, 0.28 * len(comparison_df)))

    fig, ax = plt.subplots(figsize=figsize)

    colors = np.where(comparison_df["constrained"].astype(bool), "tab:blue", "tab:orange")
    ax.barh(comparison_df["ann_type"], comparison_df[metric_y_col], color=colors, alpha=0.9)
    ax.barh(comparison_df["ann_type"], -comparison_df[metric_x_col], color="tab:gray", alpha=0.5, left=comparison_df[metric_y_col])
    
    ax.axvline(0, color="black", linewidth=0.8)
    
    ax.set_xlabel(metric_y_col)
    ax.set_ylabel("ANN type")
    ax.set_title(f"{metric_y_col} vs {metric_x_col} across all ANN types")

    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()

    if show:
        plt.show()
    return fig, ax, comparison_df

    


def plot_epoch_90pct_comparison(
    summary: str | pd.DataFrame = DEFAULT_SUMMARY_CSV,
    metric_col: str | None = None,
    figsize: tuple[float, float] | None = None,
    show: bool = True,
):
    """Plot constrained vs alltoall values grouped by matching config."""
    df = load_sweep_summary(summary) if isinstance(summary, str) else summary.copy()
    metric_col = _resolve_metric_col(df, metric_col)

    comparison_df = df.dropna(subset=[metric_col]).copy()
    config_cols = [
        col
        for col in [
            "itd_frac_name",
            "overlap_name",
            "dendrite_rule",
            "channel_dend_split_name",
            "n_spines",
            "n_dendrites_per_soma",
            "n_soma",
        ]
        if col in comparison_df.columns
    ]

    if config_cols:
        comparison_df["config_key"] = comparison_df[config_cols].astype(str).agg(" | ".join, axis=1)
    else:
        comparison_df["config_key"] = comparison_df["run_name"]

    counts = comparison_df.groupby("config_key")["constrained"].nunique()
    valid_configs = counts[counts == 2].index
    comparison_df = comparison_df[comparison_df["config_key"].isin(valid_configs)].copy()
    comparison_df = comparison_df.sort_values(["config_key", "constrained"], ascending=[True, False])

    if comparison_df.empty:
        raise ValueError("No paired constrained/alltoall configs available for the grouped plot.")

    pivot = comparison_df.pivot_table(
        index="config_key",
        columns="constrained",
        values=metric_col,
        aggfunc="mean",
    ).sort_index()

    labels = list(pivot.index)
    x = np.arange(len(labels))
    width = 0.38

    if figsize is None:
        figsize = (max(14, 0.7 * len(labels)), 7)

    fig, ax = plt.subplots(figsize=figsize)
    if False in pivot.columns:
        ax.bar(x - width / 2, pivot[False].values, width, label="alltoall", color="tab:orange", alpha=0.9)
    if True in pivot.columns:
        ax.bar(x + width / 2, pivot[True].values, width, label="constrained", color="tab:blue", alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel(metric_col)
    ax.set_title(f"{metric_col} by config: constrained vs alltoall")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    plt.tight_layout()

    if show:
        plt.show()
    return fig, ax, pivot


def plot_val_loss_ranking(
    summary: str | pd.DataFrame = DEFAULT_SUMMARY_CSV,
    figsize: tuple[float, float] | None = None,
    show: bool = True,
):
    """Plot all ANN types ranked by final validation loss (lower is better)."""
    df = load_sweep_summary(summary) if isinstance(summary, str) else summary.copy()

    plot_df = df.dropna(subset=["final_val_loss"]).copy()
    if plot_df.empty:
        raise ValueError("No non-null values found in final_val_loss.")

    plot_df["ann_type"] = plot_df.apply(_make_ann_label, axis=1)
    plot_df = plot_df.sort_values("final_val_loss", ascending=True).reset_index(drop=True)

    if figsize is None:
        figsize = (14, max(6, 0.28 * len(plot_df)))

    fig, ax = plt.subplots(figsize=figsize)
    colors = np.where(plot_df["constrained"].astype(bool), "tab:blue", "tab:orange")
    ax.barh(plot_df["ann_type"], plot_df["final_val_loss"], color=colors, alpha=0.9)
    ax.invert_yaxis()
    ax.set_xlabel("Final Validation Loss (MSE)")
    ax.set_ylabel("ANN type")
    ax.set_title("Validation Loss Ranking (lower is better)")
    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()

    if show:
        plt.show()
    return fig, ax, plot_df


def plot_val_loss_comparison(
    summary: str | pd.DataFrame = DEFAULT_SUMMARY_CSV,
    figsize: tuple[float, float] | None = None,
    show: bool = True,
):
    """Plot constrained vs alltoall validation loss grouped by config."""
    df = load_sweep_summary(summary) if isinstance(summary, str) else summary.copy()

    comparison_df = df.dropna(subset=["final_val_loss"]).copy()
    config_cols = [
        col
        for col in [
            "itd_frac_name",
            "overlap_name",
            "dendrite_rule",
            "channel_dend_split_name",
            "n_spines",
            "n_dendrites_per_soma",
            "n_soma",
        ]
        if col in comparison_df.columns
    ]

    if config_cols:
        comparison_df["config_key"] = comparison_df[config_cols].astype(str).agg(" | ".join, axis=1)
    else:
        comparison_df["config_key"] = comparison_df["run_name"]

    counts = comparison_df.groupby("config_key")["constrained"].nunique()
    valid_configs = counts[counts == 2].index
    comparison_df = comparison_df[comparison_df["config_key"].isin(valid_configs)].copy()
    comparison_df = comparison_df.sort_values(["config_key", "constrained"], ascending=[True, False])

    if comparison_df.empty:
        raise ValueError("No paired constrained/alltoall configs available for comparison.")

    pivot = comparison_df.pivot_table(
        index="config_key",
        columns="constrained",
        values="final_val_loss",
        aggfunc="mean",
    ).sort_index()

    labels = list(pivot.index)
    x = np.arange(len(labels))
    width = 0.38

    if figsize is None:
        figsize = (max(14, 0.7 * len(labels)), 7)

    fig, ax = plt.subplots(figsize=figsize)
    if False in pivot.columns:
        ax.bar(x - width / 2, pivot[False].values, width, label="alltoall", color="tab:orange", alpha=0.9)
    if True in pivot.columns:
        ax.bar(x + width / 2, pivot[True].values, width, label="constrained", color="tab:blue", alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Final Validation Loss (MSE)")
    ax.set_title("Validation Loss Comparison: constrained vs alltoall (lower is better)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    plt.tight_layout()

    if show:
        plt.show()
    return fig, ax, pivot


def plot_mae_ranking(
    summary: str | pd.DataFrame = DEFAULT_SUMMARY_CSV,
    target: str = "combined",
    figsize: tuple[float, float] | None = None,
    show: bool = True,
):
    """Plot ANN types ranked by per-target MAE (lower is better).
    
    Parameters
    ----------
    target : 'az' (azimuth), 'el' (elevation), or 'combined' (mean of both)
    """
    df = load_sweep_summary(summary) if isinstance(summary, str) else summary.copy()

    if target == "az":
        metric_col = "final_mae_az"
    elif target == "el":
        metric_col = "final_mae_el"
    elif target == "combined":
        metric_col = "final_mae_az"  # Use as primary sort, will also show el
        if "final_mae_az" in df.columns and "final_mae_el" in df.columns:
            df["combined_mae"] = (df["final_mae_az"] + df["final_mae_el"]) / 2.0
            metric_col = "combined_mae"
    else:
        raise ValueError(f"target must be 'az', 'el', or 'combined', got {target}")

    plot_df = df.dropna(subset=[metric_col]).copy()
    if plot_df.empty:
        raise ValueError(f"No non-null values found in {metric_col}.")

    plot_df["ann_type"] = plot_df.apply(_make_ann_label, axis=1)
    plot_df = plot_df.sort_values(metric_col, ascending=True).reset_index(drop=True)

    if figsize is None:
        figsize = (14, max(6, 0.28 * len(plot_df)))

    fig, ax = plt.subplots(figsize=figsize)
    colors = np.where(plot_df["constrained"].astype(bool), "tab:blue", "tab:orange")
    ax.barh(plot_df["ann_type"], plot_df[metric_col], color=colors, alpha=0.9)
    ax.invert_yaxis()
    ax.set_xlabel("Final MAE (degrees)")
    ax.set_ylabel("ANN type")
    ax.set_title(f"Per-Target MAE Ranking ({target}, lower is better)")
    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()

    if show:
        plt.show()
    return fig, ax, plot_df


def plot_mae_comparison(
    summary: str | pd.DataFrame = DEFAULT_SUMMARY_CSV,
    target: str = "combined",
    figsize: tuple[float, float] | None = None,
    show: bool = True,
):
    """Plot constrained vs alltoall per-target MAE grouped by config.
    
    Parameters
    ----------
    target : 'az' (azimuth), 'el' (elevation), or 'combined' (mean of both)
    """
    df = load_sweep_summary(summary) if isinstance(summary, str) else summary.copy()

    if target == "az":
        metric_col = "final_mae_az"
    elif target == "el":
        metric_col = "final_mae_el"
    elif target == "combined":
        if "final_mae_az" in df.columns and "final_mae_el" in df.columns:
            df["combined_mae"] = (df["final_mae_az"] + df["final_mae_el"]) / 2.0
            metric_col = "combined_mae"
        else:
            metric_col = "final_mae_az"
    else:
        raise ValueError(f"target must be 'az', 'el', or 'combined', got {target}")

    comparison_df = df.dropna(subset=[metric_col]).copy()
    config_cols = [
        col
        for col in [
            "itd_frac_name",
            "overlap_name",
            "dendrite_rule",
            "channel_dend_split_name",
            "n_spines",
            "n_dendrites_per_soma",
            "n_soma",
        ]
        if col in comparison_df.columns
    ]

    if config_cols:
        comparison_df["config_key"] = comparison_df[config_cols].astype(str).agg(" | ".join, axis=1)
    else:
        comparison_df["config_key"] = comparison_df["run_name"]

    counts = comparison_df.groupby("config_key")["constrained"].nunique()
    valid_configs = counts[counts == 2].index
    comparison_df = comparison_df[comparison_df["config_key"].isin(valid_configs)].copy()
    comparison_df = comparison_df.sort_values(["config_key", "constrained"], ascending=[True, False])

    if comparison_df.empty:
        raise ValueError("No paired constrained/alltoall configs available for comparison.")

    pivot = comparison_df.pivot_table(
        index="config_key",
        columns="constrained",
        values=metric_col,
        aggfunc="mean",
    ).sort_index()

    labels = list(pivot.index)
    x = np.arange(len(labels))
    width = 0.38

    if figsize is None:
        figsize = (max(14, 0.7 * len(labels)), 7)

    fig, ax = plt.subplots(figsize=figsize)
    if False in pivot.columns:
        ax.bar(x - width / 2, pivot[False].values, width, label="alltoall", color="tab:orange", alpha=0.9)
    if True in pivot.columns:
        ax.bar(x + width / 2, pivot[True].values, width, label="constrained", color="tab:blue", alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Final MAE (degrees)")
    ax.set_title(f"Per-Target MAE Comparison ({target}): constrained vs alltoall (lower is better)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    plt.tight_layout()

    if show:
        plt.show()
    return fig, ax, pivot
