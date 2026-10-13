#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from matplotlib.gridspec import GridSpec


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot two UMAPs from saved ESM2 embeddings (CSBJ style)."
    )
    parser.add_argument(
        "--embedding-dir",
        default="train_data/embeddings",
        help="Directory containing *_embeddings.npy and *_metadata.csv files.",
    )
    parser.add_argument(
        "--out-dir",
        default="train_data/embeddings",
        help="Directory to save UMAP figures and coordinate CSVs.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=30000,
        help="Maximum points for each UMAP plot (for speed). Use <=0 for all.",
    )
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--min-dist", type=float, default=0.1)
    parser.add_argument("--metric", default="cosine")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="Raster output DPI (default 600 for journal).",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=6.0,
        help="Scatter marker size.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.55,
        help="Scatter alpha.",
    )
    parser.add_argument(
        "--base-fontsize",
        type=float,
        default=11.0,
        help="Base font size for axis labels / ticks.",
    )
    return parser.parse_args()


def _normalize_is_enzyme_label(value):
    if pd.isna(value):
        return "unknown"
    text = str(value).strip().lower()
    if text in {"1", "true", "enzyme", "yes"}:
        return "enzyme"
    if text in {"0", "false", "non-enzyme", "nonenzyme", "no"}:
        return "non-enzyme"
    return str(value)


def stratified_subsample_indices(metadata, label_col, max_points, random_state):
    if max_points <= 0 or len(metadata) <= max_points:
        return np.arange(len(metadata))

    labels = metadata[label_col].fillna("__NA__")
    class_counts = labels.value_counts()
    total = len(metadata)

    ideal = (class_counts / total) * max_points
    base = np.floor(ideal).astype(int)
    allocations = base.copy()
    allocations[allocations < 1] = 1
    allocations = np.minimum(allocations, class_counts)

    used = int(allocations.sum())
    if used > max_points:
        over = used - max_points
        for cls in class_counts.sort_values(ascending=False).index:
            if over <= 0:
                break
            can_remove = allocations[cls] - 1
            if can_remove <= 0:
                continue
            remove_now = min(can_remove, over)
            allocations[cls] -= remove_now
            over -= remove_now
    elif used < max_points:
        remain = max_points - used
        frac = (ideal - np.floor(ideal)).sort_values(ascending=False)
        for cls in frac.index:
            if remain <= 0:
                break
            available = class_counts[cls] - allocations[cls]
            if available <= 0:
                continue
            add_now = min(available, remain)
            allocations[cls] += add_now
            remain -= add_now

    sampled_idx = []
    for cls, n_take in allocations.items():
        cls_df = metadata[labels == cls]
        sampled = cls_df.sample(n=int(n_take), random_state=random_state, replace=False)
        sampled_idx.extend(sampled.index.tolist())

    sampled_idx = np.array(sorted(sampled_idx))
    return sampled_idx


def maybe_subsample(embeddings, metadata, max_points, random_state, label_col=None):
    if max_points <= 0 or len(metadata) <= max_points:
        return embeddings, metadata

    if label_col is None:
        sampled_idx = (
            metadata.sample(n=max_points, random_state=random_state, replace=False).index
        )
        sampled_idx = np.sort(sampled_idx.to_numpy())
    else:
        sampled_idx = stratified_subsample_indices(
            metadata, label_col, max_points, random_state
        )
    return embeddings[sampled_idx], metadata.iloc[sampled_idx].reset_index(drop=True)


def run_umap(embeddings, n_neighbors, min_dist, metric, random_state):
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    return reducer.fit_transform(embeddings)


def style_axes(ax, base_fontsize: float):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="both", labelsize=base_fontsize - 0.5, width=0.6, length=3.5)
    ax.xaxis.label.set_size(base_fontsize)
    ax.yaxis.label.set_size(base_fontsize)
    ax.title.set_size(base_fontsize + 1.0)
    ax.title.set_fontweight("bold")


def panel_label(ax, label: str, fontsize: float):
    ax.text(
        -0.08,
        1.05,
        label,
        transform=ax.transAxes,
        fontsize=fontsize + 2,
        fontweight="bold",
        fontfamily="sans-serif",
        va="bottom",
        ha="left",
    )


def save_figure(fig, stem: str, out_dir: str, dpi: int):
    png_path = os.path.join(out_dir, f"{stem}.png")
    pdf_path = os.path.join(out_dir, f"{stem}.pdf")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


def extract_first_digit(ec_number):
    if pd.isna(ec_number):
        return None
    text = str(ec_number)
    match = re.search(r"\b([1-7])\.", text)
    if not match:
        return None
    return int(match.group(1))


def prepare_enzyme_vs_non_enzyme(args):
    emb_path = os.path.join(args.embedding_dir, "enzyme_embeddings.npy")
    meta_path = os.path.join(args.embedding_dir, "enzyme_metadata.csv")
    if not os.path.exists(emb_path) or not os.path.exists(meta_path):
        raise FileNotFoundError(
            "Missing enzyme files. Expected enzyme_embeddings.npy and enzyme_metadata.csv"
        )

    emb = np.load(emb_path)
    meta = pd.read_csv(meta_path)
    if "is_enzyme" not in meta.columns:
        raise ValueError("enzyme_metadata.csv must contain 'is_enzyme' column.")
    if len(emb) != len(meta):
        raise ValueError(
            f"Length mismatch: embeddings={len(emb)} metadata={len(meta)} for enzyme files."
        )

    meta = meta.copy()
    meta["plot_label"] = meta["is_enzyme"].map(_normalize_is_enzyme_label)
    emb, meta = maybe_subsample(
        emb,
        meta,
        args.max_points,
        args.random_state,
        label_col="plot_label",
    )
    coords = run_umap(
        emb, args.n_neighbors, args.min_dist, args.metric, args.random_state
    )
    return coords, meta


def prepare_ec_first_digit(args):
    emb_path = os.path.join(args.embedding_dir, "ec_embeddings.npy")
    meta_path = os.path.join(args.embedding_dir, "ec_metadata.csv")
    if not os.path.exists(emb_path) or not os.path.exists(meta_path):
        raise FileNotFoundError(
            "Missing EC files. Expected ec_embeddings.npy and ec_metadata.csv"
        )

    emb = np.load(emb_path)
    meta = pd.read_csv(meta_path)
    if "ec_number" not in meta.columns:
        raise ValueError("ec_metadata.csv must contain 'ec_number' column.")
    if len(emb) != len(meta):
        raise ValueError(
            f"Length mismatch: embeddings={len(emb)} metadata={len(meta)} for EC files."
        )

    meta = meta.copy()
    meta["ec_first_digit"] = meta["ec_number"].apply(extract_first_digit)
    valid = meta["ec_first_digit"].isin([1, 2, 3, 4, 5, 6, 7])
    meta = meta[valid].reset_index(drop=True)
    emb = emb[valid.to_numpy()]

    if len(meta) == 0:
        raise ValueError("No valid EC first-digit classes (1..7) found in ec_metadata.csv.")

    emb, meta = maybe_subsample(
        emb,
        meta,
        args.max_points,
        args.random_state,
        label_col="ec_first_digit",
    )
    coords = run_umap(
        emb, args.n_neighbors, args.min_dist, args.metric, args.random_state
    )
    return coords, meta


def draw_enzyme_panel(ax, coords, meta, args, show_panel_letter: str | None = "A"):
    color_map = {"enzyme": "#0072B2", "non-enzyme": "#E69F00"}
    labels = meta["plot_label"]
    # Draw non-enzyme first so enzyme sits on top a bit more clearly
    order = ["non-enzyme", "enzyme"]
    present = [lab for lab in order if (labels == lab).any()]
    present += [lab for lab in sorted(labels.unique()) if lab not in present]

    for label in present:
        mask = labels == label
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=args.point_size,
            alpha=args.alpha,
            c=color_map.get(label, "#999999"),
            label=f"{label} (n={int(mask.sum()):,})",
            linewidths=0,
            rasterized=True,
        )

    ax.set_title("UMAP: Enzyme vs Non-enzyme")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(
        loc="best",
        fontsize=args.base_fontsize - 1.5,
        frameon=False,
        markerscale=2.2,
        handletextpad=0.4,
    )
    style_axes(ax, args.base_fontsize)
    if show_panel_letter:
        panel_label(ax, show_panel_letter, args.base_fontsize)


def draw_ec_panel(ax, coords, meta, args, show_panel_letter: str | None = "B"):
    class_colors = {
        1: "#0072B2",
        2: "#E69F00",
        3: "#009E73",
        4: "#D55E00",
        5: "#CC79A7",
        6: "#8C564B",
        7: "#56B4E9",
    }
    for cls in [1, 2, 3, 4, 5, 6, 7]:
        mask = meta["ec_first_digit"] == cls
        if mask.sum() == 0:
            continue
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=args.point_size,
            alpha=args.alpha,
            c=class_colors[cls],
            label=f"class {cls} (n={int(mask.sum()):,})",
            linewidths=0,
            rasterized=True,
        )

    ax.set_title("UMAP: Enzymes by EC First Digit (1–7)")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(
        loc="best",
        fontsize=args.base_fontsize - 2.0,
        frameon=False,
        markerscale=2.0,
        handletextpad=0.35,
        labelspacing=0.35,
    )
    style_axes(ax, args.base_fontsize)
    if show_panel_letter:
        panel_label(ax, show_panel_letter, args.base_fontsize)


def save_coords_csv(path, coords, label_col, labels):
    df = pd.DataFrame(
        {
            "umap1": coords[:, 0],
            "umap2": coords[:, 1],
            label_col: labels,
        }
    )
    df.to_csv(path, index=False)
    print(f"Saved: {path}")


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("Computing enzyme vs non-enzyme UMAP...", flush=True)
    coords_a, meta_a = prepare_enzyme_vs_non_enzyme(args)
    save_coords_csv(
        os.path.join(args.out_dir, "enzyme_vs_non_enzyme_umap_coords.csv"),
        coords_a,
        "label",
        meta_a["plot_label"].to_numpy(),
    )

    print("Computing EC first-digit UMAP...", flush=True)
    coords_b, meta_b = prepare_ec_first_digit(args)
    save_coords_csv(
        os.path.join(args.out_dir, "ec_first_digit_umap_coords.csv"),
        coords_b,
        "ec_class",
        meta_b["ec_first_digit"].astype(int).to_numpy(),
    )

    # Individual panels
    fig_a, ax_a = plt.subplots(figsize=(7.2, 5.6))
    draw_enzyme_panel(ax_a, coords_a, meta_a, args, show_panel_letter=None)
    fig_a.tight_layout()
    save_figure(fig_a, "enzyme_vs_non_enzyme_umap", args.out_dir, args.dpi)

    fig_b, ax_b = plt.subplots(figsize=(7.2, 5.6))
    draw_ec_panel(ax_b, coords_b, meta_b, args, show_panel_letter=None)
    fig_b.tight_layout()
    save_figure(fig_b, "ec_first_digit_umap", args.out_dir, args.dpi)

    # Combined A|B manuscript figure (double-column ~180 mm)
    fig = plt.figure(figsize=(12.6, 5.4))
    gs = GridSpec(1, 2, figure=fig, wspace=0.22)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    draw_enzyme_panel(axA, coords_a, meta_a, args, show_panel_letter="A")
    draw_ec_panel(axB, coords_b, meta_b, args, show_panel_letter="B")
    fig.tight_layout()
    save_figure(fig, "Fig_umap_AB_manuscript", args.out_dir, args.dpi)

    print("Done.")


if __name__ == "__main__":
    main()
