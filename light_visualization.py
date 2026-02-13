import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from light_mix import compute_mixed_lighting

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
    dpi=100
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
        values_norm /= 255.0
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
    # 单个灯图 + 混合亮度图 + （可选）混合色相图
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
        ax.set_title(f"Light {i+1}\nHue={hues[i]:.0f}°  Val={values_norm[i]:.2f}")
        ax.axis('off')
        plot_idx += 1

    # 混合后的亮度图（热力图）
    ax = axes[plot_idx]
    im = ax.imshow(value_map_mixed, cmap='hot', vmin=0, vmax=value_map_mixed.max())
    ax.scatter(positions[:, 0] * w, positions[:, 1] * h,
               c='white', s=80, edgecolor='black', zorder=10)
    ax.set_title("Mixed Brightness")
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
        ax.set_title("Mixed Approximate Color")
        ax.axis('off')
        plot_idx += 1

    # 隐藏多余的 subplot
    for i in range(plot_idx, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


# ------------------ 使用示例 ------------------
if __name__ == '__main__':
    positions = np.array([
        [0.3, 0.2],   # 左前
        [0.7, 0.2],   # 右前
        [0.5, 0.8],   # 后方顶光
        [0.1, 0.6],   # 左侧边光
    ])

    hues   = np.array([ 30, 210,   0, 280])   # 暖白、冷白、红、紫
    values = np.array([0.90, 0.65, 0.40, 0.55])

    visualize_individual_and_mixed_lights(
        positions=positions,
        hues=hues,
        values=values,
        grid_size=(120, 160),
        decay_model="gaussian",
        sigma=0.20,
        ncols=3,
        show_mixed_hue=True,
        figsize=(14, 9)
    )