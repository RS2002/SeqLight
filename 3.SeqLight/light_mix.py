import numpy as np
import matplotlib.pyplot as plt

def compute_mixed_lighting(
    positions: np.ndarray,       # (N, 2) normalized [0,1]
    hues: np.ndarray,            # (N,) degrees 0~360
    values: np.ndarray,          # (N,) 0~1
    grid_size=(120, 160),
    decay_model="gaussian",
    sigma=0.18,                  # for gaussian, relative to diag
    alpha=1.0,                   # for inverse_square, weight = 1 / (dist^alpha + eps)
    value_power=1.0,
    eps=1e-6,
    max_value_clip=True,
    clip_factor=2.5,             # for soft clip: 1 - exp(-v * clip_factor)
) -> dict:
    """
    计算多点光源叠加后的 HV 分布（改进版）
    """
    N = len(hues)
    if N == 0:
        return {
            'hue_histogram': np.zeros(360, dtype=np.float32),
            'value_histogram': np.zeros(100, dtype=np.float32),
            'value_map': np.zeros(grid_size, dtype=np.float32),
            'mean_value': 0.0,
            'max_value': 0.0,
            'mixed_hue_map': np.zeros(grid_size, dtype=np.float32),
            'peak_hue': 0.0
        }

    # 预处理
    positions = np.asarray(positions, dtype=np.float32)
    hues = np.asarray(hues, dtype=np.float32) % 360
    values = np.clip(np.asarray(values, dtype=np.float32), 0.0, None)
    values = values ** value_power

    # 网格
    h, w = grid_size
    yy, xx = np.mgrid[0:h, 0:w]
    grid_yx = np.stack([xx, yy], axis=-1).astype(np.float32)  # (h,w,2)

    # 向量化距离计算
    pos_scaled = positions * np.array([w, h])  # (N,2)
    pos_scaled = pos_scaled[:, None, None, :]  # (N,1,1,2)
    grid_exp = grid_yx[None, ...]              # (1,h,w,2)
    diff = grid_exp - pos_scaled               # (N,h,w,2)
    dist2 = np.sum(diff**2, axis=-1)           # (N,h,w)

    # 距离权重
    if decay_model == "gaussian":
        diag = np.sqrt(w**2 + h**2)
        sigma_pix = sigma * diag
        weight_map = np.exp(-dist2 / (2 * sigma_pix**2))
    elif decay_model == "inverse_square":
        weight_map = 1.0 / (dist2 + eps) ** (alpha / 2.0)
        weight_map /= weight_map.sum(axis=(1,2), keepdims=True) + eps  # 可选norm per light
    elif decay_model == "none":
        weight_map = np.full((N, h, w), 1.0 / N, dtype=np.float32)
    else:
        raise ValueError(f"Unknown decay_model: {decay_model}")

    # 亮度贡献
    contrib = weight_map * values[:, None, None]  # (N, h, w)
    value_per_pixel = contrib.sum(axis=0)         # (h, w)

    if max_value_clip:
        value_per_pixel = 1.0 - np.exp(-value_per_pixel * clip_factor)

    # Hue 混合
    weight_for_hue = contrib / (value_per_pixel[None, ...] + eps)  # (N,h,w)
    hue_rad = np.deg2rad(hues)[:, None, None]
    avg_sin = (weight_for_hue * np.sin(hue_rad)).sum(axis=0)
    avg_cos = (weight_for_hue * np.cos(hue_rad)).sum(axis=0)
    mixed_hue = np.rad2deg(np.arctan2(avg_sin, avg_cos)) % 360
    mixed_hue = np.where(value_per_pixel < 1e-5, 0, mixed_hue)

    # 直方图（clip value for hist to [0,1]）
    hist_value = np.clip(value_per_pixel, 0, 1)
    hue_hist_raw, _ = np.histogram(mixed_hue.ravel(), bins=360, range=(0, 360))
    hue_hist = hue_hist_raw.astype(np.float32)
    hue_hist /= hue_hist.sum() + eps

    value_hist_raw, _ = np.histogram(hist_value.ravel(), bins=100, range=(0.0, 1.0))
    value_hist = value_hist_raw.astype(np.float32)
    value_hist /= value_hist.sum() + eps

    # 统计
    peak_bin = np.argmax(hue_hist)
    peak_hue = float(peak_bin)

    return {
        'hue_histogram': hue_hist,
        'value_histogram': value_hist,
        'value_map': value_per_pixel,
        'mean_value': float(value_per_pixel.mean()),
        'max_value': float(value_per_pixel.max()),
        'mixed_hue_map': mixed_hue,
        'peak_hue': peak_hue
    }

# ============================= 测试代码 =============================
if __name__ == '__main__':
    positions = np.array([
        [0.3, 0.2], [0.7, 0.2], [0.5, 0.8], [0.1, 0.6],
    ])

    hues = np.array([30, 210, 50, 280])
    values = np.array([0.9, 0.65, 0.1, 0.55])

    result = compute_mixed_lighting(
        positions, hues, values,
        grid_size=(120, 160),
        decay_model="gaussian",
        sigma=0.20
    )

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))

    axs[0].imshow(result['value_map'], cmap='hot', vmin=0, vmax=1)
    axs[0].set_title("Brightness map")
    axs[0].scatter(positions[:, 0] * 160, positions[:, 1] * 120, c='white', edgecolor='black')

    axs[1].bar(np.arange(360), result['hue_histogram'], width=1)
    axs[1].set_title(f"Hue distribution\n(peak hue = {result['peak_hue']:.0f}°)")
    axs[1].set_xlim(0, 360)

    # Value 占比直方图
    bin_centers = np.linspace(0, 1, 101)[:-1] + 0.5/100
    axs[2].bar(bin_centers, result['value_histogram'], width=1/100, color='purple')
    axs[2].set_title("Value distribution (proportion)")
    axs[2].set_xlim(0, 1)
    axs[2].set_ylim(0, result['value_histogram'].max() * 1.1)

    plt.tight_layout()
    plt.show()

    print(f"平均亮度: {result['mean_value']:.3f}")
    print(f"最大亮度: {result['max_value']:.3f}")
    print(f"hue 峰值位置: {result['peak_hue']:.0f}°")
    print(f"value_histogram sum: {result['value_histogram'].sum():.6f}")   # 应接近 1.0
    print(f"hue_histogram sum: {result['hue_histogram'].sum():.6f}")       # 应接近 1.0