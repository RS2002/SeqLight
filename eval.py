import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.colors import hsv_to_rgb

# 从你的训练文件导入
from train import LightingEnv, SeqLight, compute_mixed_lighting

torch.serialization.add_safe_globals([argparse.Namespace])


# ====================== 你提供的专业可视化函数 ======================
def visualize_individual_and_mixed_lights(
        positions: np.ndarray,
        hues: np.ndarray,  # 0~360
        values: np.ndarray,  # 0~1
        grid_size=(120, 160),
        decay_model="gaussian",
        sigma=0.20,
        ncols=4,
        show_mixed_hue=True,
        figsize=(16, 10),
        title_prefix=""
):
    """
    真实灯光可视化（单个灯 + 混合亮度 + 混合色相）
    """
    N = len(hues)
    if N == 0:
        print("No lights provided.")
        return

    values_norm = np.clip(np.asarray(values, dtype=float), 0, 1)

    # 计算混合结果
    mixed = compute_mixed_lighting(
        positions=positions,
        hues=hues,
        values=values,
        grid_size=grid_size,
        decay_model=decay_model,
        sigma=sigma
    )

    value_map_mixed = mixed['value_map']
    hue_map_mixed = mixed['mixed_hue_map']

    n_plots = N + 1 + (1 if show_mixed_hue else 0)
    nrows = (n_plots + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=120, squeeze=False)
    axes = axes.ravel()

    h, w = grid_size
    yy, xx = np.mgrid[0:h, 0:w]
    grid_yx = np.stack([xx, yy], axis=-1).astype(float)

    pos_norm = positions.copy()
    if pos_norm.max() > 1.5:
        pos_norm[:, 0] /= w
        pos_norm[:, 1] /= h

    diag = np.sqrt(w ** 2 + h ** 2)
    sigma_pix = sigma * diag

    plot_idx = 0

    for i in range(N):
        dy = grid_yx[..., 1] - pos_norm[i, 1] * h
        dx = grid_yx[..., 0] - pos_norm[i, 0] * w
        dist2 = dx * dx + dy * dy

        if decay_model == "gaussian":
            weight = np.exp(-dist2 / (2 * sigma_pix ** 2))
        elif decay_model == "inverse_square":
            dist = np.sqrt(dist2 + 1e-8)
            weight = 1.0 / (dist ** 2)
        else:
            weight = np.ones((h, w))

        weight = np.clip(weight / (weight.max() + 1e-8), 0, 1)

        hsv = np.zeros((h, w, 3))
        hsv[..., 0] = hues[i] / 360.0
        hsv[..., 1] = 0.88
        hsv[..., 2] = weight * values_norm[i]

        rgb = hsv_to_rgb(hsv)

        ax = axes[plot_idx]
        ax.imshow(rgb)
        ax.scatter(pos_norm[i, 0] * w, pos_norm[i, 1] * h,
                   c='white', s=100, edgecolor='black', linewidth=1.5, zorder=10)
        ax.set_title(f"Light {i + 1}\nHue={hues[i]:.0f}°  Val={values_norm[i]:.2f}")
        ax.axis('off')
        plot_idx += 1

    # 混合亮度图
    ax = axes[plot_idx]
    im = ax.imshow(value_map_mixed, cmap='hot', vmin=0, vmax=value_map_mixed.max())
    ax.scatter(pos_norm[:, 0] * w, pos_norm[:, 1] * h,
               c='white', s=100, edgecolor='black', linewidth=1.5, zorder=10)
    ax.set_title("Mixed Brightness")
    ax.axis('off')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Intensity')
    plot_idx += 1

    # 混合色相图
    if show_mixed_hue:
        ax = axes[plot_idx]
        hsv_mixed = np.zeros((h, w, 3))
        hsv_mixed[..., 0] = hue_map_mixed / 360.0
        hsv_mixed[..., 1] = 0.92
        hsv_mixed[..., 2] = np.clip(value_map_mixed / (value_map_mixed.max() + 1e-8), 0, 1)
        rgb_mixed = hsv_to_rgb(hsv_mixed)
        ax.imshow(rgb_mixed)
        ax.scatter(pos_norm[:, 0] * w, pos_norm[:, 1] * h,
                   c='white', s=100, edgecolor='black', linewidth=1.5, zorder=10)
        ax.set_title("Mixed Color")
        ax.axis('off')
        plot_idx += 1

    for i in range(plot_idx, len(axes)):
        axes[i].axis('off')

    fig.suptitle(f"{title_prefix}  (N={N})", fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


# ====================== 从直方图真实采样灯光配置 ======================
def sample_lights_from_hist(target_hue_hist, target_value_hist, N, positions, seed=42):
    """从目标直方图中真实采样灯光配置，使 GT 视觉效果更接近真实分布"""
    np.random.seed(seed)

    # Hue 采样（CDF 方式，保证分布一致）
    hue_cdf = np.cumsum(target_hue_hist)
    hue_cdf /= hue_cdf[-1]
    u = np.random.rand(N)
    hues = np.searchsorted(hue_cdf, u).astype(float)
    hues += np.random.normal(0, 4, N)  # 轻微扩散，更自然
    hues = np.clip(hues % 360, 0, 359)

    # Value 采样
    value_cdf = np.cumsum(target_value_hist)
    value_cdf /= value_cdf[-1]
    u = np.random.rand(N)
    value_bins = np.linspace(0, 1, 100)
    values = value_bins[np.searchsorted(value_cdf, u)]
    values += np.random.normal(0, 0.04, N)
    values = np.clip(values, 0.05, 1.0)

    # # 位置：轻微扰动的网格 + 随机，避免完全随机太乱
    # grid = np.linspace(0.15, 0.85, int(np.ceil(np.sqrt(N))))
    # xx, yy = np.meshgrid(grid, grid)
    # positions = np.column_stack([xx.ravel()[:N], yy.ravel()[:N]])
    # positions += np.random.normal(0, 0.08, positions.shape)
    # positions = np.clip(positions, 0.05, 0.95)

    return positions, hues, values


# ====================== 原有测试函数（基本不变） ======================
def parse_test_args():
    parser = argparse.ArgumentParser(description='Test Lighting Decomposition Model')
    parser.add_argument('--model_path', type=str, default='./models/model.pth')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    # 环境参数（与训练完全一致）
    parser.add_argument('--grid_size', type=int, nargs=2, default=[120, 160])
    parser.add_argument('--decay_model', type=str, default='gaussian')
    parser.add_argument('--sigma', type=float, default=0.18)
    parser.add_argument('--value_power', type=float, default=1.0)
    parser.add_argument('--eps', type=float, default=1e-8)
    parser.add_argument('--min_lights', type=int, default=8)
    parser.add_argument('--max_lights', type=int, default=8)
    parser.add_argument('--target_gen_modes', type=int, nargs='+', default=[1])
    parser.add_argument('--simple_layout', action='store_true', default=True)

    # 模型参数
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=3)

    # 其他训练参数（保持完整性）
    parser.add_argument('--total_timesteps', type=int, default=200000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--buffer_capacity', type=int, default=100000)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--lr_actor', type=float, default=3e-4)
    parser.add_argument('--lr_critic', type=float, default=3e-4)
    parser.add_argument('--policy_delay', type=int, default=2)
    parser.add_argument('--exploration_noise', type=float, default=0.1)
    parser.add_argument('--target_noise', type=float, default=0.1)
    parser.add_argument('--noise_clip', type=float, default=0.1)
    parser.add_argument('--updates_per_step', type=int, default=1)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)

    parser.add_argument('--log_file', type=str, default='training_log.txt')
    parser.add_argument('--log_file_eval', type=str, default='eval_log.txt')
    parser.add_argument('--save_dir', type=str, default='./models')
    parser.add_argument('--model_name', type=str, default='model.pth')
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--no_cuda', action='store_true')

    # 奖励参数
    parser.add_argument('--step_dist_coef', type=float, default=2.0, help='每步距离负奖励系数')
    parser.add_argument('--hue_improve_coef', type=float, default=5.0, help='hue改善奖励系数')
    parser.add_argument('--value_improve_coef', type=float, default=5.0, help='value改善奖励系数')
    parser.add_argument('--terminal_hue_coef', type=float, default=10.0, help='终端hue得分系数')
    parser.add_argument('--terminal_value_coef', type=float, default=10.0, help='终端value得分系数')
    parser.add_argument('--success_hue_thresh', type=float, default=0.15, help='成功hue距离阈值')
    parser.add_argument('--success_value_thresh', type=float, default=0.15, help='成功value距离阈值')
    parser.add_argument('--success_bonus', type=float, default=20.0, help='成功额外奖励')

    return parser.parse_args()


def load_model_and_env(args):
    device = torch.device(args.device)
    print(f"Using device: {device}")

    env = LightingEnv(args)
    env.seed(args.seed)

    model = SeqLight(
        d_model=64, nhead=4, num_layers=3
    ).to(device)

    if not Path(args.model_path).exists():
        raise FileNotFoundError(f"Model not found: {args.model_path}")

    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint.get('policy_state_dict', checkpoint))
    model.eval()
    print(f"Loaded model from: {args.model_path}")

    return model, env, device


def run_one_episode(model, env, device, N=8):
    state = env.reset(N=N)
    done = False

    while not done:
        with torch.no_grad():
            state_tensor = {
                k: torch.from_numpy(v).unsqueeze(0).to(device).float()
                if isinstance(v, np.ndarray) else torch.tensor([v]).to(device)
                for k, v in state.items()
            }
            action = model(state_tensor).cpu().numpy().flatten()
            action = np.clip(action, 0.0, 1.0)

        next_state, _, done, info = env.step(action)
        state = next_state

    final_hue_hist, final_value_hist = env._compute_current_mixed()

    return {
        'target_hue': env.target_hue_hist,
        'target_value': env.target_value_hist,
        'pred_hue': final_hue_hist,
        'pred_value': final_value_hist,
        'final_hue_err': info['hue_err'],
        'final_value_err': info['value_err'],
        'mode': env.target_gen_mode,
        'N': env.N
    }


def plot_compare(gt_hue, gt_value, pred_hue, pred_value, title_info=""):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes[0, 0].plot(gt_hue, color='C0');
    axes[0, 0].set_title("Ground Truth Hue");
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 1].plot(pred_hue, color='C1');
    axes[0, 1].set_title("Predicted Hue");
    axes[0, 1].grid(True, alpha=0.3)
    axes[1, 0].plot(gt_value, color='C0');
    axes[1, 0].set_title("Ground Truth Value");
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 1].plot(pred_value, color='C1');
    axes[1, 1].set_title("Predicted Value");
    axes[1, 1].grid(True, alpha=0.3)
    fig.suptitle(title_info, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


# ====================== 主程序 ======================
def main():
    args = parse_test_args()
    model, env, device = load_model_and_env(args)

    for i in range(3):  # 跑 3 次不同随机目标
        # print(f"\n{'=' * 60}\n=== Test Episode {i + 1} (N=10) ===\n{'=' * 60}")

        result = run_one_episode(model, env, device, N=8)

        title = (f"N={result['N']} | mode={result['mode']} | "
                 f"HueErr={result['final_hue_err']:.4f} | ValueErr={result['final_value_err']:.4f}")

        plot_compare(
            result['target_hue'], result['target_value'],
            result['pred_hue'], result['pred_value'],
            title_info=title
        )

        # ==================== 更真实的 Ground Truth 可视化 ====================
        final_mask = env.values != 0

        pos_gt, hues_gt, vals_gt = sample_lights_from_hist(
            result['target_hue'], result['target_value'], result['N'], env.positions_raw[final_mask], seed=100 + i
        )
        visualize_individual_and_mixed_lights(
            positions=pos_gt, hues=hues_gt, values=vals_gt,
            grid_size=args.grid_size, decay_model=args.decay_model, sigma=args.sigma,
            ncols=4, show_mixed_hue=True, figsize=(17, 11),
            title_prefix="Ground Truth"
        )

        # ==================== Agent 生成结果可视化 ====================
        visualize_individual_and_mixed_lights(
            positions=env.positions_raw[final_mask],
            hues=env.hues[final_mask],
            values=env.values[final_mask],
            grid_size=args.grid_size, decay_model=args.decay_model, sigma=args.sigma,
            ncols=4, show_mixed_hue=True, figsize=(17, 11),
            title_prefix="Generation Result"
        )


if __name__ == "__main__":
    main()