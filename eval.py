import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import argparse
import os
import random

from models import SeqLight
from light_mix import compute_mixed_lighting

# ===================== 参数解析 =====================
parser = argparse.ArgumentParser(description='Evaluate SeqLight model with visualization (mimic training env)')
parser.add_argument('--model_path', type=str, default='./models/best_model.pth')
parser.add_argument('--grid_size', type=int, nargs=2, default=[120, 160])
parser.add_argument('--decay_model', type=str, default='gaussian', choices=['gaussian', 'inverse_square'])
parser.add_argument('--sigma', type=float, default=0.18)
parser.add_argument('--min_lights', type=int, default=5)
parser.add_argument('--max_lights', type=int, default=15)
parser.add_argument('--d_model', type=int, default=64)
parser.add_argument('--nhead', type=int, default=4)
parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--seed', type=int, default=123)
args = parser.parse_args()

# np.random.seed(args.seed)
# random.seed(args.seed)
# torch.manual_seed(args.seed)
# torch.cuda.manual_seed_all(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ===================== 加载模型 =====================
def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    torch.serialization.add_safe_globals([argparse.Namespace])

    checkpoint = torch.load(model_path, map_location=device, weights_only=True)

    model = SeqLight(
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
    ).to(device)
    model.load_state_dict(checkpoint['policy_state_dict'])
    model.eval()
    print(f"Loaded model from {model_path} (episode {checkpoint.get('episode', 'unknown')})")
    return model


# ===================== 模拟环境状态生成（完全复制训练中的 reset/step 逻辑） =====================
class TestEnv:
    def __init__(self, args):
        self.args = args
        self.max_lights = args.max_lights
        self.grid_size = args.grid_size
        self.decay_model = args.decay_model
        self.sigma = args.sigma

    def reset(self):
        self.N = np.random.randint(args.min_lights, args.max_lights + 1)
        self.positions_raw = np.random.uniform(0, 1, size=(self.N, 2)).astype(np.float32)
        self.positions_padded = np.zeros((self.max_lights, 2), dtype=np.float32)
        self.positions_padded[:self.N] = self.positions_raw
        self.all_mask = np.zeros(self.max_lights, dtype=bool)
        self.all_mask[:self.N] = True
        self.hues = np.zeros(self.N, dtype=np.float32)
        self.values = np.zeros(self.N, dtype=np.float32)
        self.light_indices = np.random.permutation(self.N)
        self.current_idx = 0

        # 随机目标（模仿训练）
        mode = random.choice([0, 1])
        if mode == 0:
            num_src = np.random.randint(1, 6)
            src_pos = np.random.uniform(0, 1, (num_src, 2))
            src_hues = np.random.uniform(0, 1.0, num_src)
            src_vals = np.random.uniform(0, 1.0, num_src)
            result = compute_mixed_lighting(
                src_pos, src_hues, src_vals,
                grid_size=self.grid_size, decay_model=self.decay_model, sigma=self.sigma
            )
        else:
            n_peaks = np.random.randint(1, 6)
            hue_hist = np.ones(360) * 0.001
            for _ in range(n_peaks):
                peak_pos = np.random.randint(0, 360)
                peak_weight = np.random.uniform(0.5, 1.0)
                for offset in range(-10, 11):
                    idx = (peak_pos + offset) % 360
                    hue_hist[idx] += peak_weight * np.exp(-0.5 * (offset / 3.0) ** 2)
            hue_hist /= hue_hist.sum()
            mean_value = np.random.uniform(0.0, 1.0)
            max_value = min(1.0, mean_value * np.random.uniform(1.2, 1.5))
            result = {'hue_histogram': hue_hist, 'mean_value': mean_value, 'max_value': max_value}

        self.target_hue_hist = result['hue_histogram'].astype(np.float32)
        self.target_mean_value = result['mean_value']
        self.target_max_value = result['max_value']

        self.history_positions = np.zeros((self.max_lights, 2), dtype=np.float32)
        self.history_actions = np.zeros((self.max_lights, 2), dtype=np.float32)
        self.history_mixed_hue = np.zeros((self.max_lights, 360), dtype=np.float32)
        self.history_mixed_value = np.zeros((self.max_lights, 1), dtype=np.float32)

        current_hue_hist, current_mean_val, _ = self._compute_current_mixed()
        state = self._build_state(
            current_position=self.positions_raw[self.light_indices[0]],
            current_hue_hist=current_hue_hist,
            current_mean_val=current_mean_val
        )
        return state, self.positions_raw, self.N, self.target_hue_hist, self.target_mean_value

    def _compute_current_mixed(self):
        mask = self.values > 0
        if not np.any(mask):
            return np.ones(360) / 360.0, 0.0, 0.0
        result = compute_mixed_lighting(
            positions=self.positions_raw[mask],
            hues=self.hues[mask],
            values=self.values[mask],
            grid_size=self.grid_size,
            decay_model=self.decay_model,
            sigma=self.sigma
        )
        return result['hue_histogram'], result['mean_value'], result['max_value']

    def _build_state(self, current_position, current_hue_hist, current_mean_val):
        state = {
            'target_hue': self.target_hue_hist,
            'target_value': np.array([self.target_mean_value], dtype=np.float32),
            'all_positions': self.positions_padded,
            'all_mask': self.all_mask,
            'history_positions': self.history_positions,
            'history_actions': self.history_actions,
            'history_mixed_hue': self.history_mixed_hue,
            'history_mixed_value': self.history_mixed_value,
            'current_position': current_position.reshape(1, 2).astype(np.float32),
            'current_mixed_hue': current_hue_hist.reshape(1, 360).astype(np.float32),
            'current_mixed_value': np.array([current_mean_val], dtype=np.float32).reshape(1, 1),
            't': self.current_idx + 1
        }
        return state

    def step(self, action, positions_raw, light_indices):
        hue = np.clip(action[0], 0.0, 1.0)
        value = np.clip(action[1], 0.0, 1.0)
        light_id = light_indices[self.current_idx]

        hue_before, mean_before, _ = self._compute_current_mixed()
        self.hues[light_id] = hue
        self.values[light_id] = value
        hue_after, mean_after, _ = self._compute_current_mixed()

        self.history_positions[self.current_idx] = positions_raw[light_id]
        self.history_actions[self.current_idx] = [hue, value]
        self.history_mixed_hue[self.current_idx] = hue_before
        self.history_mixed_value[self.current_idx] = [mean_before]

        self.current_idx += 1
        done = self.current_idx >= self.N

        if not done:
            next_position = positions_raw[light_indices[self.current_idx]]
            next_state = self._build_state(
                current_position=next_position,
                current_hue_hist=hue_after,
                current_mean_val=mean_after
            )
        else:
            next_state = None

        return next_state, done, mean_after


# ===================== 模型推理 =====================
def run_inference(model, env):
    state, positions_raw, N, gt_hist, gt_mean = env.reset()
    light_indices = env.light_indices  # 保存顺序

    hues_pred = np.zeros(N, dtype=np.float32)
    values_pred = np.zeros(N, dtype=np.float32)

    step = 0
    done = False
    while not done:
        state_tensor = {
            k: torch.from_numpy(v).unsqueeze(0).to(device) if isinstance(v, np.ndarray) else v
            for k, v in state.items()
        }

        with torch.no_grad():
            action_raw = model(state_tensor)
            action = action_raw.squeeze(0).cpu().numpy()

        next_state, done, mean_after = env.step(action, positions_raw, light_indices)

        hue, value = action
        light_id = light_indices[step]
        hues_pred[light_id] = hue
        values_pred[light_id] = value

        state = next_state
        step += 1

    return hues_pred * 360, values_pred, gt_hist, gt_mean

def visualize_comparison(positions, hues_pred, values_pred, gt_hist, gt_mean):
    pred_result = compute_mixed_lighting(
        positions, hues_pred, values_pred,
        grid_size=args.grid_size, decay_model=args.decay_model, sigma=args.sigma
    )

    # 计算 GT 主色相（加权平均 hue）
    hues = np.arange(360)
    main_gt_hue = np.sum(hues * gt_hist) % 360

    # 创建 GT 单一色块（全图一个颜色）
    h, w = args.grid_size
    hsv_gt = np.full((h, w, 3), 0.0)  # 全图初始化
    hsv_gt[..., 0] = main_gt_hue / 360.0
    hsv_gt[..., 1] = 0.85           # 饱和度稍高，看得清楚
    hsv_gt[..., 2] = np.clip(gt_mean * 1.5, 0.1, 1.0)  # 亮度稍放大，避免太暗
    rgb_gt = hsv_to_rgb(hsv_gt)

    fig = plt.figure(figsize=(18, 12))

    # 1. GT Hue Histogram
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.bar(np.arange(360), gt_hist, width=1, color='skyblue')
    ax1.set_title(f"Ground Truth Hue Histogram\n(mean={gt_mean:.3f})")
    ax1.set_xlim(0, 360)
    ax1.set_xlabel("Hue (degree)")

    # 2. Pred Hue Histogram
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.bar(np.arange(360), pred_result['hue_histogram'], width=1, color='lightcoral')
    ax2.set_title(f"Predicted Hue Histogram")
    ax2.set_xlim(0, 360)
    ax2.set_xlabel("Hue (degree)")

    # 3. GT Approx Color Block（右上角：单一色块）
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.imshow(rgb_gt)
    ax3.set_title(f"Ground Truth Approx. Color Block")
    ax3.axis('off')

    # 4. Pred Approx Color Map
    ax5 = fig.add_subplot(2, 2, 4)
    hsv_pred = np.zeros((*args.grid_size, 3))
    hsv_pred[..., 0] = pred_result['mixed_hue_map'] / 360.0
    hsv_pred[..., 1] = 0.9
    hsv_pred[..., 2] = np.clip(pred_result['value_map'] / (pred_result['value_map'].max() + 1e-8), 0, 1)
    rgb_pred = hsv_to_rgb(hsv_pred)
    ax5.imshow(rgb_pred)
    ax5.scatter(positions[:,0]*args.grid_size[1], positions[:,1]*args.grid_size[0], c='white', s=50, edgecolor='black')
    ax5.set_title("Predicted Approx. Color Map")
    ax5.axis('off')


    plt.suptitle("Ground Truth vs Model Prediction Comparison", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

# ===================== 主程序 =====================
if __name__ == "__main__":
    model = load_model(args.model_path)
    env = TestEnv(args)

    print("Running model inference (greedy, no noise)...")
    hues_pred, values_pred, gt_hist, gt_mean = run_inference(model, env)

    # 随机生成 gt 灯光用于可视化对比
    N = len(hues_pred)
    positions = env.positions_raw  # 从 env 拿真实的测试位置

    visualize_comparison(positions, hues_pred, values_pred, gt_hist, gt_mean)