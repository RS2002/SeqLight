import numpy as np
import matplotlib.pyplot as plt


def compute_mixed_lighting(
    positions: np.ndarray,       # (N, 2)  [x, y] 或 [列, 行]
    hues: np.ndarray,            # (N,)     0~1
    values: np.ndarray,          # (N,)     建议 0~1 范围，内部会归一化
    grid_size=(120, 160),        # (高度, 宽度) 建议比例接近舞台
    decay_model="gaussian",
    sigma=0.18,                  # 相对画面对角线的比例
    value_power=1.0,             # value 的非线性指数（可选 gamma 校正）
    eps=1e-8
) -> dict:
    """
    计算多灯叠加后的整体 hue 分布 与 value 空间分布
    """
    N = len(hues)
    if N == 0:
        return {
            'hue_histogram': np.zeros(360),
            'value_map': np.zeros(grid_size),
            'mean_value': 0.0,
            'max_value': 0.0
        }

    # 统一 value & hue 到 [0,1]
    values = np.asarray(values, dtype=float)
    if values.max() > 1.5:           # 粗暴判断是否 0~255
        values = values / 255.0
    values = np.clip(values, 0, None)
    values **= value_power           # 可选 gamma 校正

    if hues.max() > 1.5:           # 粗暴判断是否 0~359
        hues = hues / 360.0
    hues = np.clip(hues, 0, None)
    hues *= 360.0

    # 建立网格坐标 (像素中心)
    h, w = grid_size
    yy, xx = np.mgrid[0:h, 0:w]
    grid_yx = np.stack([xx, yy], axis=-1).astype(float)   # (h,w,2)

    # 灯光位置归一化到 [0,1]×[0,1] 区间更稳定
    pos_norm = positions.copy()
    if pos_norm.max() > 2.0:   # 粗判是否已经是归一化坐标
        pos_norm[:, 0] /= w
        pos_norm[:, 1] /= h

    # 计算每个灯对每个网格的权重（贡献比例）
    weight_map = np.zeros((N, h, w), dtype=np.float32)

    if decay_model == "none":
        weight_map[:] = 1.0 / N

    elif decay_model == "gaussian":
        # 每个灯贡献一个高斯
        diag = np.sqrt(w**2 + h**2)          # 对角线长度 ≈ 最大距离
        sigma_pix = sigma * diag

        for i in range(N):
            dy = grid_yx[..., 1] - pos_norm[i, 1] * h
            dx = grid_yx[..., 0] - pos_norm[i, 0] * w
            dist2 = dx*dx + dy*dy
            weight_map[i] = np.exp(-dist2 / (2 * sigma_pix**2))

    elif decay_model == "inverse_square":
        for i in range(N):
            dy = grid_yx[..., 1] - pos_norm[i, 1] * h
            dx = grid_yx[..., 0] - pos_norm[i, 0] * w
            dist = np.sqrt(dx*dx + dy*dy + eps)
            weight_map[i] = 1.0 / (dist ** 2)

    else:
        raise ValueError(f"unknown decay_model: {decay_model}")

    # 归一化权重（每像素独立 softmax）
    weight_sum = weight_map.sum(axis=0) + eps
    weight_map /= weight_sum[None, ...]

    # ------------------ hue 加权混合 ------------------
    # 方法1：每个像素取加权平均色相（最常用，但跨 0/360 会断裂）
    hue_rad = np.deg2rad(hues)[:, None, None]   # (N,1,1)
    hue_x = np.sin(hue_rad)                     # (N,1,1)
    hue_y = np.cos(hue_rad)

    avg_sin = (weight_map * hue_x).sum(axis=0)  # (h,w)
    avg_cos = (weight_map * hue_y).sum(axis=0)
    mixed_hue_rad = np.arctan2(avg_sin, avg_cos)
    mixed_hue = np.rad2deg(mixed_hue_rad) % 360

    # 收集所有像素的 hue，用于直方图
    hue_flat = mixed_hue.ravel()
    hue_hist, _ = np.histogram(hue_flat, bins=360, range=(0, 360), density=True)

    # ------------------ value 叠加 ------------------
    # 每个像素的亮度 = Σ (weight_i * value_i)
    value_per_pixel = (weight_map * values[:, None, None]).sum(axis=0)  # (h,w)

    # 统计
    mean_v = float(value_per_pixel.mean())
    max_v = float(value_per_pixel.max())

    return {
        'hue_histogram': hue_hist,             # shape (360,), 归一化密度
        'value_map': value_per_pixel,          # shape (h, w)
        'mean_value': mean_v,
        'max_value': max_v,
        'mixed_hue_map': mixed_hue             # 可选：每个像素的代表色相 (h,w)
    }

if __name__ == '__main__':
    # test
    # 假设舞台是 160×120 的归一化坐标系
    positions = np.array([
        [0.3, 0.2],  # 左前
        [0.7, 0.2],  # 右前
        [0.5, 0.8],  # 后方顶光
        [0.1, 0.6],  # 左侧边光
    ])

    hues = np.array([30, 210, 0, 280])  # 暖白、冷白、红、紫
    values = np.array([0.9, 0.65, 0.4, 0.55])

    result = compute_mixed_lighting(
        positions, hues, values,
        grid_size=(120, 160),
        decay_model="gaussian",
        sigma=0.20
    )

    # 可视化
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    axs[0].imshow(result['value_map'], cmap='hot', vmin=0, vmax=1)
    axs[0].set_title("Brightness map")
    axs[0].scatter(positions[:, 0] * 160, positions[:, 1] * 120, c='white', edgecolor='black')

    axs[1].bar(np.arange(360), result['hue_histogram'], width=1)
    axs[1].set_title("Global hue distribution")
    axs[1].set_xlim(0, 360)
    axs[1].set_xlabel("Hue (degree)")

    plt.tight_layout()
    plt.show()

    print(f"平均亮度: {result['mean_value']:.3f}")
    print(f"最大亮度: {result['max_value']:.3f}")