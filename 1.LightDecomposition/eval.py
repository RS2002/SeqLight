import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from env import LightingEnv
from models import SeqLight
from light_mix import compute_mixed_lighting


# ------------------------------ 已有的距离度量函数 ------------------------------
def circular_l1_distance(p, q, bins=360):
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p /= p.sum() + 1e-8
    q /= q.sum() + 1e-8
    min_dist = np.inf
    for shift in range(bins):
        q_shifted = np.roll(q, shift)
        dist = np.sum(np.abs(p - q_shifted)) / bins
        if dist < min_dist:
            min_dist = dist
    return min_dist

def linear_l1_distance(p, q):
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p /= p.sum() + 1e-8
    q /= q.sum() + 1e-8
    return np.sum(np.abs(p - q)) / len(p)


# ------------------------------ 新增评估指标函数（考虑循环） ------------------------------
def circular_wasserstein_distance(p, q, bins=360):
    """一维 Wasserstein 距离（累积分布差的积分），通过循环移位取最小"""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p /= p.sum() + 1e-8
    q /= q.sum() + 1e-8
    # 计算累积分布
    cdf_p = np.cumsum(p)
    cdf_q = np.cumsum(q)
    min_dist = np.inf
    for shift in range(bins):
        cdf_q_shifted = np.roll(cdf_q, shift)
        # 需要处理移位导致的累积分布不连续？但直接减可能引入误差，但作为近似可接受
        # 更精确的应该对概率质量移位，但累积分布移位相当于分布整体平移，Wasserstein 距离应等于原始距离 + 移位值？
        # 简单起见，我们沿用循环移位概率分布的思想，对概率分布移位再计算 Wasserstein
        q_shifted = np.roll(q, shift)
        # 重新计算累积分布
        cdf_q_shifted = np.cumsum(q_shifted)
        dist = np.sum(np.abs(cdf_p - cdf_q_shifted)) / bins  # 积分近似
        if dist < min_dist:
            min_dist = dist
    return min_dist

def circular_js_divergence(p, q, bins=360):
    """Jensen-Shannon 散度，通过循环移位取最小"""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p /= p.sum() + 1e-8
    q /= q.sum() + 1e-8
    min_js = np.inf
    for shift in range(bins):
        q_shifted = np.roll(q, shift)
        m = 0.5 * (p + q_shifted)
        # 计算 KL(p||m) 和 KL(q_shifted||m)
        kl1 = np.sum(p * np.log((p + 1e-12) / (m + 1e-12)))
        kl2 = np.sum(q_shifted * np.log((q_shifted + 1e-12) / (m + 1e-12)))
        js = 0.5 * (kl1 + kl2)
        if js < min_js:
            min_js = js
    return min_js

def circular_kl_divergence(p, q, bins=360):
    """KL 散度，通过循环移位取最小（注意不对称）"""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p /= p.sum() + 1e-8
    q /= q.sum() + 1e-8
    min_kl = np.inf
    for shift in range(bins):
        q_shifted = np.roll(q, shift)
        kl = np.sum(p * np.log((p + 1e-12) / (q_shifted + 1e-12)))
        if kl < min_kl:
            min_kl = kl
    return min_kl

def circular_peak_discrepancy(p, q, bins=360):
    """主峰位置差异（角度差），考虑循环取最小"""
    peak_p = np.argmax(p)
    peak_q = np.argmax(q)
    diff = abs(peak_p - peak_q)
    return min(diff, bins - diff) * (360.0 / bins)  # 转换为度数

def bhattacharyya_distance(p, q):
    """Bhattacharyya 距离（线性），无需循环"""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p /= p.sum() + 1e-8
    q /= q.sum() + 1e-8
    bc = np.sum(np.sqrt(p * q))
    return -np.log(bc + 1e-12)

def circular_bhattacharyya_distance(p, q, bins=360):
    """Bhattacharyya 距离，通过循环移位取最小"""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p /= p.sum() + 1e-8
    q /= q.sum() + 1e-8
    min_dist = np.inf
    for shift in range(bins):
        q_shifted = np.roll(q, shift)
        bc = np.sum(np.sqrt(p * q_shifted))
        dist = -np.log(bc + 1e-12)
        if dist < min_dist:
            min_dist = dist
    return min_dist

def cosine_similarity(p, q):
    """余弦相似度（线性）"""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p /= p.sum() + 1e-8
    q /= q.sum() + 1e-8
    dot = np.sum(p * q)
    norm_p = np.sqrt(np.sum(p**2))
    norm_q = np.sqrt(np.sum(q**2))
    return dot / (norm_p * norm_q + 1e-12)

def circular_cosine_similarity(p, q, bins=360):
    """余弦相似度，通过循环移位取最大"""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p /= p.sum() + 1e-8
    q /= q.sum() + 1e-8
    max_sim = -1.0
    for shift in range(bins):
        q_shifted = np.roll(q, shift)
        dot = np.sum(p * q_shifted)
        norm_p = np.sqrt(np.sum(p**2))
        norm_q = np.sqrt(np.sum(q_shifted**2))
        sim = dot / (norm_p * norm_q + 1e-12)
        if sim > max_sim:
            max_sim = sim
    return max_sim


# ------------------------------ 灯光可视化函数 ------------------------------
def visualize_individual_and_mixed_lights(
    positions: np.ndarray,
    hues: np.ndarray,          # 0~360
    values: np.ndarray,        # 建议 0~1
    grid_size=(120, 160),
    decay_model="gaussian",
    sigma=0.20,
    ncols=3,                   # 每行放几张图（建议 3~5）
    show_mixed_hue=True,       # 是否显示混合后的色相图
    figsize=(12, 8),
    dpi=100,
    title_prefix=""
):
    # ... 函数体保持不变（同原代码）...
    N = len(hues)
    if N == 0:
        print("No lights provided.")
        return

    values_norm = np.asarray(values, dtype=float)
    if values_norm.max() > 1.5:
        values_norm /= 255.03
    values_norm = np.clip(values_norm, 0, 1)

    mixed = compute_mixed_lighting(
        positions=positions,
        hues=hues,
        values=values,
        grid_size=grid_size,
        decay_model=decay_model,
        sigma=sigma
    )

    value_map_mixed = mixed['value_map']
    hue_map_mixed   = mixed['mixed_hue_map']

    h, w = grid_size
    yy, xx = np.mgrid[0:h, 0:w]
    grid_yx = np.stack([xx, yy], axis=-1).astype(float)

    pos_norm = positions.copy()
    if pos_norm.max() > 2.0:
        pos_norm[:, 0] /= w
        pos_norm[:, 1] /= h

    diag = np.sqrt(w**2 + h**2)
    sigma_pix = sigma * diag

    n_plots = N + 1 + (1 if show_mixed_hue else 0)
    nrows = (n_plots + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=dpi, squeeze=False)
    axes = axes.ravel()
    plot_idx = 0

    for i in range(N):
        dy = grid_yx[..., 1] - pos_norm[i, 1] * h
        dx = grid_yx[..., 0] - pos_norm[i, 0] * w
        dist2 = dx*dx + dy*dy

        if decay_model == "gaussian":
            weight = np.exp(-dist2 / (2 * sigma_pix**2))
        elif decay_model == "none":
            weight = np.ones((h, w))
        elif decay_model == "inverse_square":
            dist = np.sqrt(dist2 + 1e-8)
            weight = 1.0 / (dist ** 2)
        else:
            weight = np.zeros((h, w))

        weight = np.clip(weight / (weight.max() + 1e-8), 0, 1)

        hsv = np.zeros((h, w, 3))
        hsv[..., 0] = hues[i] / 360.0
        hsv[..., 1] = 0.85
        hsv[..., 2] = weight * values_norm[i]

        rgb = hsv_to_rgb(hsv)

        ax = axes[plot_idx]
        ax.imshow(rgb)
        ax.scatter(pos_norm[i, 0] * w, pos_norm[i, 1] * h,
                   c='white', s=80, edgecolor='black', zorder=10)
        ax.set_title(f"{title_prefix}Light {i+1}\nHue={hues[i]:.0f}°  Val={values_norm[i]:.2f}")
        ax.axis('off')
        plot_idx += 1

    ax = axes[plot_idx]
    im = ax.imshow(value_map_mixed, cmap='hot', vmin=0, vmax=value_map_mixed.max())
    ax.scatter(positions[:, 0] * w, positions[:, 1] * h,
               c='white', s=80, edgecolor='black', zorder=10)
    ax.set_title(f"{title_prefix}Mixed Brightness")
    ax.axis('off')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Intensity')
    plot_idx += 1

    if show_mixed_hue:
        ax = axes[plot_idx]
        hsv_mixed = np.zeros((h, w, 3))
        hsv_mixed[..., 0] = hue_map_mixed / 360.0
        hsv_mixed[..., 1] = 1.0
        hsv_mixed[..., 2] = value_map_mixed
        rgb_mixed = hsv_to_rgb(hsv_mixed)
        ax.imshow(rgb_mixed)
        ax.scatter(positions[:, 0] * w, positions[:, 1] * h,
                   c='white', s=80, edgecolor='black', zorder=10)
        ax.set_title(f"{title_prefix}Mixed Approximate Color")
        ax.axis('off')
        plot_idx += 1

    for i in range(plot_idx, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


def visualize_target_as_color_block(target_hue, target_value, figsize=(3,3)):
    # ... 函数体保持不变 ...
    peak_hue_bin = np.argmax(target_hue)
    peak_hue = peak_hue_bin
    peak_value_bin = np.argmax(target_value)
    peak_value = (peak_value_bin + 0.5) / 100.0

    hsv = np.array([[[peak_hue/360.0, 1.0, peak_value]]], dtype=np.float32)
    rgb = hsv_to_rgb(hsv)[0,0]

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow([[rgb]], extent=[0,1,0,1])
    ax.set_title(f"Target Distribution\nPeak Hue={peak_hue}°, Peak Value={peak_value:.2f}")
    ax.axis('off')
    plt.show()


# ------------------------------ 缩放搜索函数（不变） ------------------------------
def find_best_scale(final_hues, final_values, target_hue, target_value, env,
                    scale_min=0.5, scale_max=2.0, num_steps=100, criterion='l1'):
    best_scale = 1.0
    best_value = float('inf')
    scales = np.linspace(scale_min, scale_max, num_steps)

    if criterion == 'mean':
        bin_centers = np.arange(100) * 0.01 + 0.005
        target_mean = np.sum(target_value * bin_centers)

    for s in scales:
        scaled_values = np.clip(final_values * s, 0, 1)
        result = compute_mixed_lighting(
            positions=env.positions_raw,
            hues=final_hues,
            values=scaled_values,
            grid_size=(env.grid_h, env.grid_w),
            decay_model=env.decay_model,
            sigma=env.sigma,
            value_power=env.value_power,
            eps=env.eps
        )
        if criterion == 'l1':
            scaled_value_hist = result['value_histogram']
            dist = linear_l1_distance(scaled_value_hist, target_value)
        elif criterion == 'mean':
            scaled_mean = np.sum(result['value_histogram'] * bin_centers)
            dist = abs(scaled_mean - target_mean)
        else:
            raise ValueError(f"Unknown criterion: {criterion}")

        if dist < best_value:
            best_value = dist
            best_scale = s

    return best_scale, best_value


# ------------------------------ 评估主函数（增加多指标） ------------------------------
def evaluate(env, policy, num_episodes, device, deterministic=True, t=1.0,
             plot=False, mode=None,
             scale_values=False, scale_range=(0.5, 2.0), scale_criterion='l1',
             extra_metrics=False):
    """
    在环境中运行策略，评估目标分布与最终分布的差异。
    如果 extra_metrics 为 True，则计算多种额外指标。
    """
    policy.eval()

    # 存储原始指标
    metrics_raw = {
        'hue_l1': [], 'value_l1': [],
        'hue_wasserstein': [], 'value_wasserstein': [],
        'hue_js': [], 'value_js': [],
        'hue_kl': [], 'value_kl': [],
        'hue_peak': [], 'value_peak': [],
        'hue_bhattacharyya': [], 'value_bhattacharyya': [],
        'hue_cosine': [], 'value_cosine': [],
    }

    # 如果启用缩放，存储缩放后指标
    metrics_scaled = None
    best_scales = [] if scale_values else None
    if scale_values:
        metrics_scaled = {k: [] for k in metrics_raw.keys()}

    for ep in range(num_episodes):
        state = env.reset(mode=mode)
        done = False
        while not done:
            state_tensor = {}
            for k, v in state.items():
                if isinstance(v, np.ndarray):
                    state_tensor[k] = torch.from_numpy(v).unsqueeze(0).to(device)
                else:
                    state_tensor[k] = torch.tensor([v]).to(device)

            with torch.no_grad():
                _, hue_action, val_action, _, _ = policy(state_tensor, deterministic=deterministic, t=t)
                action = np.array([hue_action.cpu().numpy()[0], val_action.cpu().numpy()[0]])

            next_state, _, done, info = env.step(action)
            state = next_state

        final_hue_raw = info['hue_hist_after']
        final_value_raw = info['value_hist_after']
        target_hue = info['target_hue_hist']
        target_value = info['target_value_hist']

        # ---- 计算原始指标 ----
        # 基础 L1
        hue_l1 = circular_l1_distance(final_hue_raw, target_hue)
        value_l1 = linear_l1_distance(final_value_raw, target_value)
        metrics_raw['hue_l1'].append(hue_l1)
        metrics_raw['value_l1'].append(value_l1)

        if extra_metrics:
            # Hue 指标（循环）
            metrics_raw['hue_wasserstein'].append(circular_wasserstein_distance(final_hue_raw, target_hue))
            metrics_raw['hue_js'].append(circular_js_divergence(final_hue_raw, target_hue))
            metrics_raw['hue_kl'].append(circular_kl_divergence(final_hue_raw, target_hue))
            metrics_raw['hue_peak'].append(circular_peak_discrepancy(final_hue_raw, target_hue))
            metrics_raw['hue_bhattacharyya'].append(circular_bhattacharyya_distance(final_hue_raw, target_hue))
            metrics_raw['hue_cosine'].append(circular_cosine_similarity(final_hue_raw, target_hue))

            # Value 指标（线性）
            metrics_raw['value_wasserstein'].append(circular_wasserstein_distance(final_value_raw, target_value, bins=100))
            metrics_raw['value_js'].append(circular_js_divergence(final_value_raw, target_value, bins=100))
            metrics_raw['value_kl'].append(circular_kl_divergence(final_value_raw, target_value, bins=100))
            metrics_raw['value_peak'].append(circular_peak_discrepancy(final_value_raw, target_value, bins=100))
            metrics_raw['value_bhattacharyya'].append(circular_bhattacharyya_distance(final_value_raw, target_value, bins=100))
            metrics_raw['value_cosine'].append(circular_cosine_similarity(final_value_raw, target_value, bins=100))

        msg = f"Episode {ep+1}: Hue L1 = {hue_l1:.4f}, Value L1 = {value_l1:.4f}"

        # ---- 缩放处理（如果启用） ----
        if scale_values:
            best_scale, _ = find_best_scale(
                env.hues.copy(), env.values.copy(),
                target_hue, target_value, env,
                scale_min=scale_range[0], scale_max=scale_range[1],
                criterion=scale_criterion
            )
            best_scales.append(best_scale)

            scaled_values = np.clip(env.values.copy() * best_scale, 0, 1)
            scaled_result = compute_mixed_lighting(
                positions=env.positions_raw,
                hues=env.hues,
                values=scaled_values,
                grid_size=(env.grid_h, env.grid_w),
                decay_model=env.decay_model,
                sigma=env.sigma,
                value_power=env.value_power,
                eps=env.eps
            )
            final_hue_scaled = scaled_result['hue_histogram']
            final_value_scaled = scaled_result['value_histogram']

            # 缩放后的基础 L1
            hue_l1_scaled = circular_l1_distance(final_hue_scaled, target_hue)
            value_l1_scaled = linear_l1_distance(final_value_scaled, target_value)
            metrics_scaled['hue_l1'].append(hue_l1_scaled)
            metrics_scaled['value_l1'].append(value_l1_scaled)

            if extra_metrics:
                metrics_scaled['hue_wasserstein'].append(circular_wasserstein_distance(final_hue_scaled, target_hue))
                metrics_scaled['hue_js'].append(circular_js_divergence(final_hue_scaled, target_hue))
                metrics_scaled['hue_kl'].append(circular_kl_divergence(final_hue_scaled, target_hue))
                metrics_scaled['hue_peak'].append(circular_peak_discrepancy(final_hue_scaled, target_hue))
                metrics_scaled['hue_bhattacharyya'].append(circular_bhattacharyya_distance(final_hue_scaled, target_hue))
                metrics_scaled['hue_cosine'].append(circular_cosine_similarity(final_hue_scaled, target_hue))

                metrics_scaled['value_wasserstein'].append(circular_wasserstein_distance(final_value_scaled, target_value, bins=100))
                metrics_scaled['value_js'].append(circular_js_divergence(final_value_scaled, target_value, bins=100))
                metrics_scaled['value_kl'].append(circular_kl_divergence(final_value_scaled, target_value, bins=100))
                metrics_scaled['value_peak'].append(circular_peak_discrepancy(final_value_scaled, target_value, bins=100))
                metrics_scaled['value_bhattacharyya'].append(circular_bhattacharyya_distance(final_value_scaled, target_value, bins=100))
                metrics_scaled['value_cosine'].append(circular_cosine_similarity(final_value_scaled, target_value, bins=100))

            msg += f", Best scale = {best_scale:.3f}, Scaled Value L1 = {value_l1_scaled:.4f}"

            plot_hue = final_hue_scaled
            plot_value = final_value_scaled
            plot_values = scaled_values
        else:
            plot_hue = final_hue_raw
            plot_value = final_value_raw
            plot_values = env.values

        print(msg)

        if plot:
            # 绘图部分（略作修改，使用 plot_hue/plot_value）
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            axes[0].bar(np.arange(360), target_hue, width=1, alpha=0.7, label='Target')
            axes[0].bar(np.arange(360), plot_hue, width=1, alpha=0.7, label='Final')
            title_hue = f'Hue Distribution (L1={hue_l1:.4f})'
            if scale_values:
                title_hue += f' [Scaled L1={hue_l1_scaled:.4f}]'
            axes[0].set_title(title_hue)
            axes[0].set_xlabel('Hue bin')
            axes[0].set_ylabel('Probability')
            axes[0].legend()

            axes[1].bar(np.arange(100), target_value, width=1, alpha=0.7, label='Target')
            axes[1].bar(np.arange(100), plot_value, width=1, alpha=0.7, label='Final')
            title_val = f'Value Distribution (L1={value_l1:.4f})'
            if scale_values:
                title_val += f' [Scaled L1={value_l1_scaled:.4f}]'
            axes[1].set_title(title_val)
            axes[1].set_xlabel('Value bin')
            axes[1].set_ylabel('Probability')
            axes[1].legend()
            plt.tight_layout()
            plt.show()

            visualize_target_as_color_block(target_hue, target_value)

            if hasattr(env, 'ground_truth_hues') and env.ground_truth_hues is not None:
                visualize_individual_and_mixed_lights(
                    positions=env.positions_raw,
                    hues=env.ground_truth_hues,
                    values=env.ground_truth_values,
                    grid_size=(env.grid_h, env.grid_w),
                    decay_model=env.decay_model,
                    sigma=env.sigma,
                    ncols=3,
                    show_mixed_hue=True,
                    title_prefix=f"Episode {ep+1} Ground Truth - "
                )

            visualize_individual_and_mixed_lights(
                positions=env.positions_raw,
                hues=env.hues,
                values=plot_values,
                grid_size=(env.grid_h, env.grid_w),
                decay_model=env.decay_model,
                sigma=env.sigma,
                ncols=3,
                show_mixed_hue=True,
                title_prefix=f"Episode {ep+1} Policy{' (Scaled)' if scale_values else ''} - "
            )

    # ---- 输出统计结果 ----
    print("\n" + "="*60)
    print(f"Evaluation over {num_episodes} episodes")
    print("="*60)

    def print_metrics(prefix, metrics_dict):
        for key in sorted(metrics_dict.keys()):
            values = metrics_dict[key]
            if values:
                mean_val = np.mean(values)
                std_val = np.std(values)
                print(f"{prefix} {key}: mean = {mean_val:.6f}, std = {std_val:.6f}")

    print_metrics("Raw", metrics_raw)

    if scale_values and metrics_scaled:
        print("\n--- After scaling ---")
        print_metrics("Scaled", metrics_scaled)
        print(f"Average scale factor: {np.mean(best_scales):.4f} ± {np.std(best_scales):.4f}")

    return metrics_raw, metrics_scaled, best_scales


# ------------------------------ 主程序 ------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate trained policy for lighting control")
    # 环境参数
    parser.add_argument('--min_lights', type=int, default=8)
    parser.add_argument('--max_lights', type=int, default=8)
    parser.add_argument('--simple_layout', action='store_true', default=True)
    parser.add_argument('--max_n_peaks', type=int, default=3)
    parser.add_argument('--max_hue_similarity', type=int, default=3)

    # 模型参数
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--model_path', type=str, default="./grpo_latest.pth", help='Path to trained model weights')

    # 评估参数
    parser.add_argument('--num_episodes', type=int, default=5, help='Number of episodes to evaluate')
    parser.add_argument('--deterministic', action='store_true', default=True, help='Use deterministic actions (mean)')
    parser.add_argument('--t', type=float, default=1, help='Temperature')
    parser.add_argument('--mode', type=int, default=0)
    parser.add_argument('--plot', action='store_true', default=True, help='Plot distribution comparisons for each episode')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='Disable CUDA')

    # 值缩放选项
    parser.add_argument('--scale_values', action='store_true', default=True, help='Enable global scaling of final values')
    parser.add_argument('--scale_min', type=float, default=0.01, help='Minimum scaling factor to search')
    parser.add_argument('--scale_max', type=float, default=10.0, help='Maximum scaling factor to search')
    parser.add_argument('--scale_criterion', type=str, default='mean', choices=['l1', 'mean'],
                        help='Criterion for optimal scaling: l1 (minimize L1 distance) or mean (match mean)')

    # 额外指标选项
    parser.add_argument('--extra_metrics', action='store_true', default=True,
                        help='Compute additional metrics (Wasserstein, JS, KL, peak, Bhattacharyya, cosine)')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")

    # 初始化环境
    env = LightingEnv(
        min_lights=args.min_lights,
        max_lights=args.max_lights,
        simple_layout=args.simple_layout,
        max_n_peaks=args.max_n_peaks,
        max_hue_similarity=args.max_hue_similarity
    )
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 创建模型并加载权重
    policy = SeqLight(
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers
    ).to(device)
    policy.load_state_dict(torch.load(args.model_path, map_location=device))
    print(f"Model loaded from {args.model_path}")

    # 评估
    evaluate(env, policy, args.num_episodes, device,
             deterministic=args.deterministic,
             plot=args.plot,
             t=args.t,
             mode=args.mode,
             scale_values=args.scale_values,
             scale_range=(args.scale_min, args.scale_max),
             scale_criterion=args.scale_criterion,
             extra_metrics=args.extra_metrics)


if __name__ == "__main__":
    main()