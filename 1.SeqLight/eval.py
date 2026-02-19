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
    """
    可视化：
    1. 每个灯光单独的光照图（带 hue 着色）
    2. 所有灯光混合后的亮度图
    3. （可选）混合后的代表色相图
    """
    N = len(hues)
    if N == 0:
        print("No lights provided.")
        return

    # 统一 value 到 [0,1]
    values_norm = np.asarray(values, dtype=float)
    if values_norm.max() > 1.5:
        values_norm /= 255.03
    values_norm = np.clip(values_norm, 0, 1)

    # 计算混合结果（复用原有函数）
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

    # 准备画布
    n_plots = N + 1
    if show_mixed_hue:
        n_plots += 1

    nrows = (n_plots + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=figsize,
        dpi=dpi,
        squeeze=False
    )
    axes = axes.ravel()  # 展平，方便索引

    # 逐个灯画单独光照
    h, w = grid_size
    yy, xx = np.mgrid[0:h, 0:w]
    grid_yx = np.stack([xx, yy], axis=-1).astype(float)

    pos_norm = positions.copy()
    if pos_norm.max() > 2.0:
        pos_norm[:, 0] /= w
        pos_norm[:, 1] /= h

    diag = np.sqrt(w**2 + h**2)
    sigma_pix = sigma * diag

    plot_idx = 0

    for i in range(N):
        # 计算单个灯的权重（和混合函数里一样）
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

        weight = np.clip(weight / (weight.max() + 1e-8), 0, 1)  # 归一到[0,1]

        # 用 HSV 着色：H = hue/360, S=饱和度（可调）, V=weight * value
        hsv = np.zeros((h, w, 3))
        hsv[..., 0] = hues[i] / 360.0
        hsv[..., 1] = 0.85                      # 饱和度，可调低一点更自然
        hsv[..., 2] = weight * values_norm[i]

        rgb = hsv_to_rgb(hsv)

        ax = axes[plot_idx]
        ax.imshow(rgb)
        ax.scatter(pos_norm[i, 0] * w, pos_norm[i, 1] * h,
                   c='white', s=80, edgecolor='black', zorder=10)
        ax.set_title(f"{title_prefix}Light {i+1}\nHue={hues[i]:.0f}°  Val={values_norm[i]:.2f}")
        ax.axis('off')
        plot_idx += 1

    # 混合后的亮度图（热力图）
    ax = axes[plot_idx]
    im = ax.imshow(value_map_mixed, cmap='hot', vmin=0, vmax=value_map_mixed.max())
    ax.scatter(positions[:, 0] * w, positions[:, 1] * h,
               c='white', s=80, edgecolor='black', zorder=10)
    ax.set_title(f"{title_prefix}Mixed Brightness")
    ax.axis('off')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Intensity')
    plot_idx += 1

    # （可选）混合后的代表色相图
    if show_mixed_hue:
        ax = axes[plot_idx]
        # hue_map_mixed 是 0~360，需要转成 HSV 图像
        hsv_mixed = np.zeros((h, w, 3))
        hsv_mixed[..., 0] = hue_map_mixed / 360.0
        # hsv_mixed[..., 1] = 0.9
        hsv_mixed[..., 1] = 1.0
        hsv_mixed[..., 2] = value_map_mixed
        # hsv_mixed[..., 2] = np.clip(value_map_mixed / (value_map_mixed.max() + 1e-8), 0, 1)

        rgb_mixed = hsv_to_rgb(hsv_mixed)
        ax.imshow(rgb_mixed)
        ax.scatter(positions[:, 0] * w, positions[:, 1] * h,
                   c='white', s=80, edgecolor='black', zorder=10)
        ax.set_title(f"{title_prefix}Mixed Approximate Color")
        ax.axis('off')
        plot_idx += 1

    # 隐藏多余的 subplot
    for i in range(plot_idx, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


def visualize_target_as_color_block(target_hue, target_value, figsize=(3,3)):
    """
    将目标分布（直方图）简化为一个色块显示。
    使用峰值色调和峰值亮度合成纯色。
    """
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


# ------------------------------ 新增：搜索最优缩放因子 ------------------------------
def find_best_scale(final_hues, final_values, target_hue, target_value, env,
                    scale_min=0.5, scale_max=2.0, num_steps=100, criterion='l1'):
    """
    在 [scale_min, scale_max] 范围内线性搜索最优缩放因子。
    criterion: 'l1' 最小化 value L1 距离，'mean' 最小化均值差异。
    返回 (best_scale, best_value)
    """
    best_scale = 1.0
    best_value = float('inf')
    scales = np.linspace(scale_min, scale_max, num_steps)

    # 预先计算目标均值（如果 criterion 为 mean）
    if criterion == 'mean':
        bin_centers = np.arange(100) * 0.01 + 0.005  # 每个 bin 的中心值
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


def evaluate(env, policy, num_episodes, device, deterministic=True, t=1.0,
             plot=False, mode=None,
             scale_values=False, scale_range=(0.5, 2.0), scale_criterion='l1'):
    """
    在环境中运行策略，评估目标分布与最终分布的差异。
    如果 scale_values 为 True，则对每个 episode 的最终灯光 values 进行全局缩放，
    寻找最优缩放因子使 value 差异最小（根据 scale_criterion），并输出缩放后的指标。
    绘图时始终使用缩放后的结果（如果启用缩放）。
    """
    policy.eval()
    hue_dists_raw = []
    value_dists_raw = []
    hue_dists_scaled = [] if scale_values else None
    value_dists_scaled = [] if scale_values else None
    best_scales = [] if scale_values else None

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

        # 获取原始最终分布和灯光参数
        final_hue_raw = info['hue_hist_after']
        final_value_raw = info['value_hist_after']
        target_hue = info['target_hue_hist']
        target_value = info['target_value_hist']

        hue_dist_raw = circular_l1_distance(final_hue_raw, target_hue)
        value_dist_raw = linear_l1_distance(final_value_raw, target_value)

        hue_dists_raw.append(hue_dist_raw)
        value_dists_raw.append(value_dist_raw)

        msg = f"Episode {ep+1}: Hue L1 = {hue_dist_raw:.4f}, Value L1 = {value_dist_raw:.4f}"

        # 缩放处理
        if scale_values:
            # 搜索最优缩放因子（基于原始值）
            best_scale, _ = find_best_scale(
                env.hues.copy(), env.values.copy(),
                target_hue, target_value, env,
                scale_min=scale_range[0], scale_max=scale_range[1],
                criterion=scale_criterion
            )
            best_scales.append(best_scale)

            # 应用缩放并重新计算分布
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
            final_hue_scaled = scaled_result['hue_histogram']  # 理论上与 final_hue_raw 相同
            final_value_scaled = scaled_result['value_histogram']

            hue_dist_scaled = circular_l1_distance(final_hue_scaled, target_hue)
            value_dist_scaled = linear_l1_distance(final_value_scaled, target_value)

            hue_dists_scaled.append(hue_dist_scaled)
            value_dists_scaled.append(value_dist_scaled)

            msg += f", Best scale = {best_scale:.3f}, Scaled Value L1 = {value_dist_scaled:.4f}"

            # 后续绘图使用缩放后的数据和分布
            plot_hue = final_hue_scaled
            plot_value = final_value_scaled
            plot_values = scaled_values
        else:
            # 未缩放，使用原始数据
            plot_hue = final_hue_raw
            plot_value = final_value_raw
            plot_values = env.values

        print(msg)

        if plot:
            # 绘制直方图对比（使用缩放后的值或原始值）
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            axes[0].bar(np.arange(360), target_hue, width=1, alpha=0.7, label='Target')
            axes[0].bar(np.arange(360), plot_hue, width=1, alpha=0.7, label='Final')
            title_hue = f'Hue Distribution (L1={hue_dist_raw:.4f})'
            if scale_values:
                title_hue += f' [Scaled L1={hue_dist_scaled:.4f}]'
            axes[0].set_title(title_hue)
            axes[0].set_xlabel('Hue bin')
            axes[0].set_ylabel('Probability')
            axes[0].legend()

            axes[1].bar(np.arange(100), target_value, width=1, alpha=0.7, label='Target')
            axes[1].bar(np.arange(100), plot_value, width=1, alpha=0.7, label='Final')
            title_val = f'Value Distribution (L1={value_dist_raw:.4f})'
            if scale_values:
                title_val += f' [Scaled L1={value_dist_scaled:.4f}]'
            axes[1].set_title(title_val)
            axes[1].set_xlabel('Value bin')
            axes[1].set_ylabel('Probability')
            axes[1].legend()
            plt.tight_layout()
            plt.show()

            # 显示目标色块
            visualize_target_as_color_block(target_hue, target_value)

            # 显示 ground truth 灯光效果（如果存在）
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

            # 显示策略生成的灯光效果（使用缩放后的 values 或原始 values）
            visualize_individual_and_mixed_lights(
                positions=env.positions_raw,
                hues=env.hues,
                values=plot_values,  # 关键：使用缩放后的值
                grid_size=(env.grid_h, env.grid_w),
                decay_model=env.decay_model,
                sigma=env.sigma,
                ncols=3,
                show_mixed_hue=True,
                title_prefix=f"Episode {ep+1} Policy{' (Scaled)' if scale_values else ''} - "
            )

    # 统计输出
    avg_hue_raw = np.mean(hue_dists_raw)
    avg_value_raw = np.mean(value_dists_raw)
    print(f"\nAverage over {num_episodes} episodes: Hue L1 = {avg_hue_raw:.4f}, Value L1 = {avg_value_raw:.4f}")
    if scale_values:
        avg_hue_scaled = np.mean(hue_dists_scaled)
        avg_value_scaled = np.mean(value_dists_scaled)
        avg_scale = np.mean(best_scales)
        print(f"After scaling: Average Hue L1 = {avg_hue_scaled:.4f}, Average Value L1 = {avg_value_scaled:.4f}, Average scale = {avg_scale:.3f}")
    return hue_dists_raw, value_dists_raw

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
    parser.add_argument('--model_path', type=str, default="./ppo_latest.pth", help='Path to trained model weights')

    # 评估参数
    parser.add_argument('--num_episodes', type=int, default=5, help='Number of episodes to evaluate')
    parser.add_argument('--deterministic', action='store_true', default=True, help='Use deterministic actions (mean)')
    parser.add_argument('--t', type=float, default=1, help='Temperature')
    parser.add_argument('--mode', type=int, default=None)
    parser.add_argument('--plot', action='store_true', default=True, help='Plot distribution comparisons for each episode')
    parser.add_argument('--seed', type=int, default=1, help='Random seed for reproducibility')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='Disable CUDA')

    # 值缩放选项
    parser.add_argument('--scale_values', action='store_true', default=True, help='Enable global scaling of final values')
    parser.add_argument('--scale_min', type=float, default=0.01, help='Minimum scaling factor to search')
    parser.add_argument('--scale_max', type=float, default=10.0, help='Maximum scaling factor to search')
    parser.add_argument('--scale_criterion', type=str, default='mean', choices=['l1', 'mean'],
                        help='Criterion for optimal scaling: l1 (minimize L1 distance) or mean (match mean)')

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
             scale_criterion=args.scale_criterion)


if __name__ == "__main__":
    main()