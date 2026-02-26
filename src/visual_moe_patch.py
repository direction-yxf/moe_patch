#!/usr/bin/env python3
"""
Quick utility to visualize MoE load ratios recorded by the asynchronous monitor.

Usage:
    # Single experiment
    python visual_moe_patch.py /path/to/logs/moe_monitor --iter 1200 --out heatmap_iter1200.png
    python visual_moe_patch.py /path/to/logs/moe_monitor --layers 6 12 18  # Plot specific layers by index
    python visual_moe_patch.py /path/to/logs/moe_monitor --layers layer_6 layer_12  # Plot by layer name

    # Compare multiple experiments
    python visual_moe_patch.py \
      --exp exp1:/path/to/logs/moe_monitor_exp1 \
      --exp exp2:/path/to/logs/moe_monitor_exp2 \
      --iter 1200 --layers 6 12 --out comparison.png
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Plot MoE load ratios为热力图，可合并多个 rank。')
    parser.add_argument(
        'log_dir',
        type=Path,
        nargs='?',
        default=None,
        help='MoE monitor log directory containing *.jsonl files. (Single experiment mode)',
    )
    parser.add_argument(
        '--exp',
        type=str,
        action='append',
        default=None,
        help='Specify multiple experiments for comparison. Format: name:/path/to/logs/moe_monitor. Can be repeated.',
    )
    parser.add_argument('--iter', type=int, default=None, help='Filter by iteration (default: all iterations).')
    parser.add_argument(
        '--out', type=Path, default=Path('moe_heatmap.png'), help='Output image path (default: moe_heatmap.png).')
    parser.add_argument(
        '--layers',
        type=str,
        nargs='+',
        default=None,
        help='Filter by layer names or indices (e.g., --layers layer_6 layer_12 or --layers 6 12). If not specified, plot all layers.',
    )
    return parser.parse_args()


def load_records(paths: List[Path], target_iter: int | None) -> Dict[str, Dict[int, Dict[str, np.ndarray]]]:
    # records[layer][iteration] = {'actual': ndarray, 'theoretical': ndarray}
    records: Dict[str, Dict[int, Dict[str, np.ndarray]]] = defaultdict(dict)
    min_iter: int | None = None
    max_iter: int | None = None
    nearest_lower: int | None = None
    nearest_upper: int | None = None
    for path in paths:
        with path.open('r', encoding='utf-8') as f:
            for line in f:
                try:
                    row = json.loads(line)
                except Exception:
                    # Be robust to truncated/partial lines (e.g. interrupted training).
                    continue
                iteration = row.get('iteration')
                if iteration is None:
                    continue
                try:
                    iteration = int(iteration)
                except Exception:
                    continue
                if min_iter is None or iteration < min_iter:
                    min_iter = iteration
                if max_iter is None or iteration > max_iter:
                    max_iter = iteration
                if target_iter is not None:
                    if iteration <= target_iter and (nearest_lower is None or iteration > nearest_lower):
                        nearest_lower = iteration
                    if iteration >= target_iter and (nearest_upper is None or iteration < nearest_upper):
                        nearest_upper = iteration
                if target_iter is not None and iteration != target_iter:
                    continue
                actual = row.get('actual_assignments')
                if actual is None:
                    continue
                if isinstance(actual, dict):
                    # Explicitly reject the {expert_id: count} format to avoid silently mis-parsing.
                    raise RuntimeError('actual_assignments expects a list/array; dict format is no longer supported.')
                actual_arr = np.asarray(actual, dtype=np.float32)
                layer_name = row.get('layer')
                if not layer_name:
                    continue
                num_experts_val = row.get('num_experts')
                try:
                    num_experts = int(num_experts_val)
                except Exception:
                    num_experts = None
                if num_experts is None or num_experts <= 0:
                    num_experts = actual_arr.shape[0]
                if num_experts <= 0:
                    continue
                if actual_arr.shape[0] != num_experts:
                    # Prefer the observed assignment vector length.
                    num_experts = actual_arr.shape[0]
                try:
                    tokens = float(row.get('tokens'))
                except Exception:
                    continue
                try:
                    top_k = float(row.get('top_k'))
                except Exception:
                    continue
                try:
                    theoretical_val = tokens * top_k / float(num_experts)
                except Exception:
                    continue
                try:
                    theoretical_arr = np.full_like(actual_arr, float(theoretical_val))
                except Exception:
                    continue
                layer_dict = records.setdefault(layer_name, {})
                slot = layer_dict.setdefault(
                    iteration,
                    {
                        'actual': np.zeros_like(actual_arr, dtype=np.float32),
                        'theoretical': np.zeros_like(actual_arr, dtype=np.float32)
                    })
                slot['actual'] += actual_arr
                slot['theoretical'] += theoretical_arr

    if not records:
        if target_iter is None:
            raise RuntimeError('No valid records found. Ensure JSONL files exist and contain valid records.')
        if min_iter is None or max_iter is None:
            raise RuntimeError('No valid records found. Ensure JSONL files exist and contain valid records.')
        raise RuntimeError(
            f'No records found for --iter {target_iter}. '
            f'Available iteration range: [{min_iter}, {max_iter}]. '
            f'Nearest: lower={nearest_lower}, upper={nearest_upper}.'
        )
    return records


def _layer_sort_key(name: str):
    m = re.search(r'(\d+)$', name)
    if m:
        return (0, int(m.group(1)))
    return (1, name)


def _discover_rank_logs(log_dir: Path) -> List[Path]:
    paths = sorted(log_dir.glob('*.jsonl'))
    if not paths:
        raise RuntimeError(f'No *.jsonl files found under: {log_dir}')
    return paths


def build_matrix(records: Dict[str, Dict[int, Dict[str, np.ndarray]]], layer_filters: List[str] | None = None) -> Tuple[
    List[str], np.ndarray]:
    layer_names = sorted(records.keys(), key=_layer_sort_key)

    # Filter layers if specified
    if layer_filters is not None:
        filtered_layers = []
        available_layers = set(layer_names)

        for filter_spec in layer_filters:
            # Try to match by exact name first
            if filter_spec in available_layers:
                filtered_layers.append(filter_spec)
            else:
                # Try to match by layer index (e.g., "6" matches "layer_6")
                matched = False
                for layer_name in layer_names:
                    m = re.search(r'(\d+)$', layer_name)
                    if m and m.group(1) == filter_spec:
                        filtered_layers.append(layer_name)
                        matched = True
                        break

                if not matched:
                    raise RuntimeError(
                        f'Layer "{filter_spec}" not found. Available layers: {", ".join(sorted(layer_names, key=_layer_sort_key))}'
                    )

        if not filtered_layers:
            raise RuntimeError('No layers matched the specified filters.')

        layer_names = sorted(filtered_layers, key=_layer_sort_key)

    num_experts = next(iter(next(iter(records.values())).values()))['actual'].shape[0]
    matrix = np.zeros((len(layer_names), num_experts), dtype=np.float32)
    eps = 1e-8
    for idx, layer in enumerate(layer_names):
        iter_dict = records[layer]
        # Weighted mean across all records: sum(actual) / sum(theoretical).
        total_actual = np.zeros(num_experts, dtype=np.float32)
        total_theoretical = np.zeros(num_experts, dtype=np.float32)
        for stats in iter_dict.values():
            total_actual += stats['actual']
            total_theoretical += stats['theoretical']
        matrix[idx] = total_actual / np.maximum(total_theoretical, eps)
    return layer_names, matrix


def _auto_figsize(num_experts: int, num_layers: int) -> Tuple[float, float]:
    # Show all experts on x-axis; scale figure to keep it readable.
    fig_w = max(12.0, 0.32 * float(num_experts) + 6.0)
    fig_h = max(4.0, 0.38 * float(num_layers) + 2.0)
    return fig_w, fig_h


def _auto_xtick_fontsize(num_experts: int) -> int:
    if num_experts <= 32:
        return 9
    if num_experts <= 64:
        return 7
    if num_experts <= 128:
        return 6
    if num_experts <= 256:
        return 5
    return 4


def _diverging_norm(matrix: np.ndarray) -> tuple[TwoSlopeNorm, tuple[float, float]]:
    finite = matrix[np.isfinite(matrix)]
    if finite.size == 0:
        return TwoSlopeNorm(vcenter=1.0, vmin=0.7, vmax=1.3), (0.7, 1.3)
    lo, hi = np.percentile(finite, [2.5, 97.5])
    span = max(abs(lo - 1.0), abs(hi - 1.0))
    if not np.isfinite(span) or span < 1e-6:
        span = 0.3
    vmin = 1.0 - span
    vmax = 1.0 + span
    return TwoSlopeNorm(vcenter=1.0, vmin=vmin, vmax=vmax), (vmin, vmax)


def _shrink_axis_height(ax: plt.Axes, cbar: plt.Axes | None, factor: float = 0.82) -> None:
    """Visually compress rows without changing the overall figure height."""
    factor = max(0.1, min(1.0, factor))
    bbox = ax.get_position()
    new_h = bbox.height * factor
    delta = (bbox.height - new_h) * 0.5
    ax.set_position([bbox.x0, bbox.y0 + delta, bbox.width, new_h])
    if cbar is not None:
        cbox = cbar.get_position()
        cbar.set_position([cbox.x0, cbox.y0 + delta, cbox.width, new_h])


def plot_heatmap(layer_names: List[str], matrix: np.ndarray, output: Path, iteration: int | None,
                 title_suffix: str | None = None) -> None:
    fig_w, fig_h = _auto_figsize(matrix.shape[1], len(layer_names))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    norm, (vmin, vmax) = _diverging_norm(matrix)
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto', norm=norm)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Actual / Theoretical load (assignments)')
    cbar.ax.yaxis.set_major_formatter('{x:.2f}')
    ax.set_yticks(range(len(layer_names)))
    ax.set_yticklabels(layer_names, fontsize=9)
    x_fontsize = _auto_xtick_fontsize(matrix.shape[1])
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_xticklabels([f'E{i}' for i in range(matrix.shape[1])], ha='center', fontsize=x_fontsize)
    ax.set_xlabel('Experts')
    ax.set_ylabel('Layers')
    title = 'MoE Load Balancing Heatmap'
    if title_suffix:
        title = f'{title} - {title_suffix}'
    if iteration is not None:
        title += f' (iter={iteration})'
    ax.set_title(title)
    # Light grid to visually separate cells.
    ax.set_xticks(np.arange(-0.5, matrix.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, matrix.shape[0], 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=0.6, alpha=0.9)
    ax.tick_params(which='minor', bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            text_color = '#1c1c1c' if abs(val - 1.0) < 0.2 else 'white'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=text_color, fontsize=8.5)
    fig.tight_layout()
    _shrink_axis_height(ax, cbar.ax if cbar else None, factor=0.82)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_multi_experiment_heatmap(
        experiments: Dict[str, Tuple[List[str], np.ndarray]],
        output: Path,
        iteration: int | None
) -> None:
    """
    Plot multiple experiments vertically stacked for comparison.
    experiments: dict mapping experiment_name -> (layer_names, matrix)
    """
    n_exps = len(experiments)
    if n_exps == 0:
        raise RuntimeError('No experiments to plot.')

    # Get first experiment to determine matrix shape
    first_exp_name = next(iter(experiments.keys()))
    layer_names, first_matrix = experiments[first_exp_name]
    num_experts = first_matrix.shape[1]
    num_layers = len(layer_names)

    # Calculate figure size - vertically stacked
    single_w, single_h = _auto_figsize(num_experts, num_layers)
    fig_w = single_w
    fig_h = single_h * n_exps + 2.0  # Add some spacing

    fig, axes = plt.subplots(n_exps, 1, figsize=(fig_w, fig_h))
    if n_exps == 1:
        axes = [axes]

    # Calculate a common normalization across all experiments for fair comparison
    all_matrices = [matrix for _, matrix in experiments.values()]
    combined = np.concatenate([m.flatten() for m in all_matrices])
    norm, (vmin, vmax) = _diverging_norm(combined)

    x_fontsize = _auto_xtick_fontsize(num_experts)

    for idx, (exp_name, (layers, matrix)) in enumerate(experiments.items()):
        ax = axes[idx]
        im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto', norm=norm)

        # Show y-axis labels on all subplots
        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels(layers, fontsize=9)
        ax.set_ylabel('Layers')

        # Show x-axis labels on all subplots
        ax.set_xticks(range(matrix.shape[1]))
        ax.set_xticklabels([f'E{i}' for i in range(matrix.shape[1])], ha='center', fontsize=x_fontsize)
        ax.set_xlabel('Experts')

        title = exp_name
        if iteration is not None:
            title += f' (iter={iteration})'
        ax.set_title(title, fontsize=10, pad=10)

        # Grid
        ax.set_xticks(np.arange(-0.5, matrix.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, matrix.shape[0], 1), minor=True)
        ax.grid(which='minor', color='white', linewidth=0.6, alpha=0.9)
        ax.tick_params(which='minor', bottom=False, left=False)

        for spine in ax.spines.values():
            spine.set_visible(False)

        # Add text annotations
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                val = matrix[i, j]
                text_color = '#1c1c1c' if abs(val - 1.0) < 0.2 else 'white'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=text_color, fontsize=7.5)

    # Add a shared colorbar on the right side
    # Adjust spacing: top margin closer to title, bottom margin larger
    fig.subplots_adjust(top=0.95, bottom=0.05, right=0.88, hspace=0.4)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Actual / Theoretical load (assignments)', fontsize=10)
    cbar.ax.yaxis.set_major_formatter('{x:.2f}')

    fig.suptitle('MoE Load Balancing Comparison', fontsize=14)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    args = parse_args()

    # Determine mode: single experiment or multi-experiment comparison
    if args.exp is not None:
        # Multi-experiment mode
        experiments = {}
        for exp_spec in args.exp:
            if ':' not in exp_spec:
                raise RuntimeError(f'Invalid experiment specification: {exp_spec}. Expected format: name:/path/to/logs')
            exp_name, exp_path = exp_spec.split(':', 1)
            exp_path = Path(exp_path)
            paths = _discover_rank_logs(exp_path)
            records = load_records(paths, args.iter)
            layers, matrix = build_matrix(records, args.layers)
            experiments[exp_name] = (layers, matrix)

        plot_multi_experiment_heatmap(experiments, args.out, args.iter)
        print(f'Multi-experiment comparison heatmap saved to {args.out}')
    else:
        # Single experiment mode
        if args.log_dir is None:
            raise RuntimeError('Either provide a log_dir positional argument or use --exp for multi-experiment mode.')
        paths = _discover_rank_logs(args.log_dir)
        records = load_records(paths, args.iter)
        layers, matrix = build_matrix(records, args.layers)
        plot_heatmap(layers, matrix, args.out, args.iter)
        print(f'Heatmap saved to {args.out}')


if __name__ == '__main__':
    main()
