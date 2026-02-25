#!/usr/bin/env python3
"""MoE 负载均衡分析工具"""

import json
import re
import numpy as np
import matplotlib.pyplot as plt


def extract_layer_index(layer_name):
    """提取层索引: 'model.layers.0.mlp.experts' -> 0"""
    match = re.search(r'layers\.(\d+)', str(layer_name))
    return int(match.group(1)) if match else layer_name


def compute_balance_metrics(expert_counts):
    """计算负载均衡指标"""
    results = {'per_layer': {}, 'overall': {}}
    all_cv = []
    
    for layer_name, layer_counts in expert_counts.items():
        layer_idx = extract_layer_index(layer_name)
        n = max(int(k) for k in layer_counts.keys()) + 1
        counts = np.array([layer_counts.get(str(i), 0) for i in range(n)], dtype=float)
        
        if counts.sum() == 0:
            continue
        
        mean = counts.mean()
        cv = counts.std() / mean if mean > 0 else 0
        
        results['per_layer'][layer_idx] = {
            'cv': round(cv, 4),
            'total_tokens': int(counts.sum()),
        }
        all_cv.append(cv)
    
    if all_cv:
        results['overall'] = {
            'avg_cv': round(np.mean(all_cv), 4),
            'num_layers': len(all_cv),
        }
    
    return results


def plot_layer_load(expert_counts, layer_idx, output_path=None):
    """绘制负载分布热力图（单行）"""
    # 查找层
    layer_key = None
    for key in expert_counts.keys():
        if extract_layer_index(key) == layer_idx:
            layer_key = key
            break
    
    if not layer_key:
        print(f"❌ 层 {layer_idx} 不存在")
        return
    
    # 获取数据
    layer_counts = expert_counts[layer_key]
    n = max(int(k) for k in layer_counts.keys()) + 1
    counts = np.array([layer_counts.get(str(i), 0) for i in range(n)], dtype=float)
    
    # 计算相对负载
    mean = counts.mean()
    relative_load = counts / mean if mean > 0 else counts
    cv = counts.std() / mean if mean > 0 else 0
    
    # 单行显示
    grid = relative_load.reshape(1, -1)
    
    # 绘图（使用黄-橙-红配色）
    fig, ax = plt.subplots(figsize=(20, 2))
    im = ax.imshow(grid, cmap='YlOrRd', aspect='auto', vmin=0)
    
    # 添加 expert ID 标注（显示在格子上方）
    for i in range(n):
        ax.text(i, -0.5, str(i), ha='center', va='bottom', fontsize=6)
    
    ax.set_title(f'Layer {layer_idx} Expert Load (CV={cv:.4f}, N={n})', fontsize=12, pad=20)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # 颜色条
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.2, fraction=0.05)
    cbar.set_label('Relative Expert Load', fontsize=10)
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"📊 已保存: {output_path}")
    else:
        plt.show()
    plt.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--stats-file', required=True)
    parser.add_argument('--output')
    parser.add_argument('--plot-layer', type=int)
    parser.add_argument('--plot-output')
    args = parser.parse_args()
    
    with open(args.stats_file) as f:
        data = json.load(f)
    counts = data.get('counts', data)
    
    if args.plot_layer is not None:
        plot_layer_load(counts, args.plot_layer, args.plot_output)
    else:
        metrics = compute_balance_metrics(counts)
        print(f"📊 balance_score: {metrics['overall'].get('balance_score', 'N/A')}")
        print(f"   avg_cv: {metrics['overall'].get('avg_cv', 'N/A')}")
        print(f"   num_layers: {metrics['overall'].get('num_layers', 'N/A')}")
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(metrics, f, indent=2)
            print(f"💾 已保存: {args.output}")
