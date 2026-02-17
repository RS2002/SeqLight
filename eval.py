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

# ------------------------------ 灯光可视化函数（从您提供的代码复制） ------------------------------
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
        hsv_mixed[..., 1] = 0.9
        hsv_mixed[..., 2] = np.clip(value_map_mixed / (value_map_mixed.max() + 1e-8), 0, 1)

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

# ------------------------------ 目标分布可视化辅助函数 ------------------------------
def visualize_target_as_color_block(target_hue, target_value, figsize=(3,3)):
    """
    将目标分布（直方图）简化为一个色块显示。
    使用峰值色调和峰值亮度合成纯色。
    """
    peak_hue_bin = np.argmax(target_hue)
    peak_hue = peak_hue_bin  # bin index 对应色调度数（0~359）
    peak_value_bin = np.argmax(target_value)
    # value 直方图的 bin 中心：0~1 分成100份，每个bin宽度0.01，中心为 (bin+0.5)/100
    peak_value = (peak_value_bin + 0.5) / 100.0

    # HSV 转 RGB
    hsv = np.array([[[peak_hue/360.0, 1.0, peak_value]]], dtype=np.float32)
    rgb = hsv_to_rgb(hsv)[0,0]

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow([[rgb]], extent=[0,1,0,1])
    ax.set_title(f"Target Distribution\nPeak Hue={peak_hue}°, Peak Value={peak_value:.2f}")
    ax.axis('off')
    plt.show()

# ------------------------------ 评估主函数 ------------------------------
def evaluate(env, policy, num_episodes, device, deterministic=True, plot=False):
    """
    在环境中运行策略，评估目标分布与最终分布的差异。
    返回每个episode的hue距离、value距离列表。
    """
    policy.eval()
    hue_dists = []
    value_dists = []

    for ep in range(num_episodes):
        # 重置环境（随机目标分布）
        state = env.reset(mode=1)
        done = False
        while not done:
            # 构建批处理字典（增加batch维度）
            state_tensor = {}
            for k, v in state.items():
                if isinstance(v, np.ndarray):
                    state_tensor[k] = torch.from_numpy(v).unsqueeze(0).to(device)
                else:
                    state_tensor[k] = torch.tensor([v]).to(device)

            with torch.no_grad():
                _, hue_action, val_action, _, _ = policy(state_tensor, deterministic=deterministic)
                action = np.array([hue_action.cpu().numpy()[0], val_action.cpu().numpy()[0]])

            next_state, _, done, info = env.step(action)
            state = next_state

        # episode结束，info中包含了最后的分布信息
        final_hue = info['hue_hist_after']
        final_value = info['value_hist_after']
        target_hue = info['target_hue_hist']
        target_value = info['target_value_hist']

        hue_dist = circular_l1_distance(final_hue, target_hue)
        value_dist = linear_l1_distance(final_value, target_value)

        hue_dists.append(hue_dist)
        value_dists.append(value_dist)

        print(f"Episode {ep+1}: Hue L1 = {hue_dist:.4f}, Value L1 = {value_dist:.4f}")

        if plot:
            # 绘制直方图对比
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            axes[0].bar(np.arange(360), target_hue, width=1, alpha=0.7, label='Target')
            axes[0].bar(np.arange(360), final_hue, width=1, alpha=0.7, label='Final')
            axes[0].set_title(f'Hue Distribution (L1={hue_dist:.4f})')
            axes[0].set_xlabel('Hue bin')
            axes[0].set_ylabel('Probability')
            axes[0].legend()

            axes[1].bar(np.arange(100), target_value, width=1, alpha=0.7, label='Target')
            axes[1].bar(np.arange(100), final_value, width=1, alpha=0.7, label='Final')
            axes[1].set_title(f'Value Distribution (L1={value_dist:.4f})')
            axes[1].set_xlabel('Value bin')
            axes[1].set_ylabel('Probability')
            axes[1].legend()

            plt.tight_layout()
            plt.show()

            # 显示目标色块
            visualize_target_as_color_block(target_hue, target_value)

            # 显示策略生成的灯光效果
            # 从环境中获取最终灯光参数
            final_positions = env.positions_raw
            final_hues = env.hues
            final_values = env.values
            visualize_individual_and_mixed_lights(
                positions=final_positions,
                hues=final_hues,
                values=final_values,
                grid_size=(env.grid_h, env.grid_w),
                decay_model=env.decay_model,
                sigma=env.sigma,
                ncols=3,
                show_mixed_hue=True,
                title_prefix=f"Episode {ep+1} - "
            )

    avg_hue = np.mean(hue_dists)
    avg_value = np.mean(value_dists)
    print(f"\nAverage over {num_episodes} episodes: Hue L1 = {avg_hue:.4f}, Value L1 = {avg_value:.4f}")
    return hue_dists, value_dists

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
    parser.add_argument('--model_path', type=str, default="./airl_latest.pth", help='Path to trained model weights')

    # 评估参数
    parser.add_argument('--num_episodes', type=int, default=3, help='Number of episodes to evaluate')
    parser.add_argument('--deterministic', action='store_true', default=True, help='Use deterministic actions (mean)')
    parser.add_argument('--plot', action='store_true', default=True, help='Plot distribution comparisons for each episode')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='Disable CUDA')

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
    evaluate(env, policy, args.num_episodes, device, deterministic=args.deterministic, plot=args.plot)

if __name__ == "__main__":
    main()