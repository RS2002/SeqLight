import numpy as np
import random
import torch
from light_mix import compute_mixed_lighting

class LightingEnv:
    def __init__(
        self,
        grid_size=(120, 160),
        decay_model='gaussian',
        sigma=0.18,
        value_power=1.0,
        eps=1e-8,
        min_lights=8,
        max_lights=8,
        target_gen_modes=[1],          # 仅支持模式1（随机稀疏峰）
        simple_layout=True,
        max_n_peaks=3,                  # 新增：随机稀疏峰的最大峰值数
        max_hue_similarity=3,            # 新增：专家式目标的最大色调相似度
    ):
        self.grid_h, self.grid_w = grid_size
        self.decay_model = decay_model
        self.sigma = sigma
        self.value_power = value_power
        self.eps = eps
        self.min_lights = min_lights
        self.max_lights = max_lights
        self.target_gen_modes = target_gen_modes
        self.simple_layout = simple_layout
        self.max_n_peaks = max_n_peaks
        self.max_hue_similarity = max_hue_similarity

        # 内部状态（将在 reset 中初始化）
        self.N = None
        self.positions_raw = None
        self.positions_padded = None
        self.all_mask = None
        self.hues = None
        self.values = None
        self.light_indices = None
        self.current_idx = 0
        self.target_hue_hist = None
        self.target_value_hist = None

        # 固定大小历史缓冲区（填充0）
        self.history_positions = None
        self.history_actions = None
        self.history_mixed_hue = None
        self.history_mixed_value = None

    def seed(self, seed=None):
        if seed is None:
            return
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

    def _gen_target_random_sparse(self, n_peaks):
        """
        生成稀疏高斯峰目标分布，峰值数由 n_peaks 指定。
        """
        hue_hist = np.ones(360) * 0.001
        for _ in range(n_peaks):
            peak_pos = np.random.randint(0, 360)
            peak_weight = np.random.uniform(0.5, 1.0)
            for offset in range(-10, 11):
                idx = (peak_pos + offset) % 360
                hue_hist[idx] += peak_weight * np.exp(-0.5 * (offset / 3.0) ** 2)
        hue_hist /= hue_hist.sum()

        value_hist = np.ones(100) * 0.001
        peak_bin = np.random.randint(20, 100 - 20)
        peak_weight = np.random.uniform(0.5, 1.0)
        for offset in range(-30, 31):
            idx = (peak_bin + offset) % 100
            value_hist[idx] += peak_weight * np.exp(-0.5 * (offset / 15.0) ** 2)
        value_hist /= value_hist.sum()

        return hue_hist, value_hist

    def _gen_target_expert(self, N, hue_similarity):
        """
        生成专家式目标分布：随机生成 N 个灯光的最终参数（受 hue_similarity 控制），
        计算其混合分布，返回两个直方图（不修改环境状态）。
        """
        # 生成最终灯光参数（内部调用 _generate_final_hues_values）
        final_hues, final_values = self._generate_final_hues_values(N, hue_similarity)

        # 生成灯光位置（与 reset 中 simple_layout 逻辑一致）
        if self.simple_layout:
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
            positions = np.array([
                [0.5 + 0.5 * np.cos(angle), 0.5 + 0.5 * np.sin(angle)]
                for angle in angles
            ]).astype(np.float32)
        else:
            positions = np.random.uniform(0, 1, size=(N, 2)).astype(np.float32)

        # 计算混合分布
        result = compute_mixed_lighting(
            positions=positions,
            hues=final_hues,
            values=final_values,
            grid_size=(self.grid_h, self.grid_w),
            decay_model=self.decay_model,
            sigma=self.sigma,
            value_power=self.value_power,
            eps=self.eps
        )
        return result['hue_histogram'], result['value_histogram']

    def _generate_target_distribution(self,mode=None):
        """
        随机选择目标生成方式：
        - 方式0：随机稀疏峰（峰值数从 1 ~ max_n_peaks 随机）
        - 方式1：专家式（色调相似度从 1 ~ max_hue_similarity 随机）
        """
        if mode is None:
            mode = np.random.choice([0, 1])  # 0: 稀疏峰, 1: 专家式
        if mode == 0:
            n_peaks = np.random.randint(1, self.max_n_peaks + 1)
            return self._gen_target_random_sparse(n_peaks)
        else:
            hue_similarity = np.random.randint(1, self.max_hue_similarity + 1)
            # 注意：这里需要使用 self.N，因为 reset 中已设置灯光数量
            return self._gen_target_expert(self.N, hue_similarity)

    def _compute_current_mixed(self):
        """计算当前已设置灯光的混合分布"""
        mask = (self.values != 0)
        if not np.any(mask):
            # 全黑情况：均匀 hue 直方图，value 集中在0
            hue_hist = np.ones(360) / 360
            value_hist = np.zeros(100)
            value_hist[0] = 1.0
            return hue_hist, value_hist
        result = compute_mixed_lighting(
            positions=self.positions_raw[mask],
            hues=self.hues[mask],
            values=self.values[mask],
            grid_size=(self.grid_h, self.grid_w),
            decay_model=self.decay_model,
            sigma=self.sigma,
            value_power=self.value_power,
            eps=self.eps
        )
        return result['hue_histogram'], result['value_histogram']

    def _build_state(self, current_position, current_hue_hist, current_value_hist):
        """构建状态字典（与模型输入格式一致）"""
        state = {
            'target_hue': self.target_hue_hist.astype(np.float32),
            'target_value': self.target_value_hist.astype(np.float32),
            'all_positions': self.positions_padded,
            'all_mask': self.all_mask,
            'history_positions': self.history_positions,
            'history_actions': self.history_actions,
            'history_mixed_hue': self.history_mixed_hue,
            'history_mixed_value': self.history_mixed_value,
            'current_position': current_position.reshape(1, 2).astype(np.float32),
            'current_mixed_hue': current_hue_hist.reshape(1, 360).astype(np.float32),
            'current_mixed_value': current_value_hist.reshape(1, 100).astype(np.float32),
            't': self.current_idx + 1
        }
        return state

    def reset(self, N=None, mode=None):
        """重置环境（随机生成目标分布），返回初始状态"""
        if N is None:
            self.N = np.random.randint(self.min_lights, self.max_lights + 1)
        else:
            self.N = N

        # 生成灯光位置
        if self.simple_layout:
            angles = np.linspace(0, 2*np.pi, self.N, endpoint=False)
            self.positions_raw = np.array([
                [0.5 + 0.5 * np.cos(angle), 0.5 + 0.5 * np.sin(angle)]
                for angle in angles
            ]).astype(np.float32)
            self.light_indices = np.arange(self.N)   # 顺序按索引
        else:
            self.positions_raw = np.random.uniform(0, 1, size=(self.N, 2)).astype(np.float32)
            self.light_indices = np.random.permutation(self.N)

        # 填充到固定大小
        self.positions_padded = np.zeros((self.max_lights, 2), dtype=np.float32)
        self.positions_padded[:self.N] = self.positions_raw
        self.all_mask = np.zeros(self.max_lights, dtype=bool)
        self.all_mask[:self.N] = True

        # 初始化灯光参数（全0）
        self.hues = np.zeros(self.N, dtype=np.float32)
        self.values = np.zeros(self.N, dtype=np.float32)
        self.current_idx = 0

        # 生成目标分布（随机选择方式）
        self.target_hue_hist, self.target_value_hist = self._generate_target_distribution(mode)

        # 历史缓冲区（用0填充）
        self.history_positions = np.zeros((self.max_lights, 2), dtype=np.float32)
        self.history_actions = np.zeros((self.max_lights, 2), dtype=np.float32)
        self.history_mixed_hue = np.zeros((self.max_lights, 360), dtype=np.float32)
        self.history_mixed_value = np.zeros((self.max_lights, 100), dtype=np.float32)

        # 初始混合分布（全黑）
        current_hue_hist, current_value_hist = self._compute_current_mixed()
        state = self._build_state(
            current_position=self.positions_raw[self.light_indices[0]],
            current_hue_hist=current_hue_hist,
            current_value_hist=current_value_hist
        )
        return state

    def step(self, action):
        """
        执行一步动作，更新环境。
        参数 action: [hue_norm, value_norm]，范围 [0,1]
        返回 (next_state, reward, done, info)，其中 reward 固定为 0.0（外部提供）
        info 包含当前分布和目标分布，用于外部奖励计算。
        """
        hue_norm = np.clip(action[0], 0.0, 1.0)
        value_norm = np.clip(action[1], 0.0, 1.0)
        hue = hue_norm * 360.0
        value = value_norm

        light_id = self.light_indices[self.current_idx]

        # 记录动作前的分布（用于历史）
        hue_before, value_hist_before = self._compute_current_mixed()

        # 设置灯光参数
        self.hues[light_id] = hue
        self.values[light_id] = value

        # 计算动作后的分布
        hue_after, value_hist_after = self._compute_current_mixed()

        # 更新历史缓冲区（使用动作前的分布作为该步的历史观测）
        self.history_positions[self.current_idx] = self.positions_raw[light_id]
        self.history_actions[self.current_idx] = [hue_norm, value_norm]
        self.history_mixed_hue[self.current_idx] = hue_before
        self.history_mixed_value[self.current_idx] = value_hist_before

        self.current_idx += 1
        done = (self.current_idx >= self.N)

        info = {
            'hue_hist_before': hue_before,
            'hue_hist_after': hue_after,
            'target_hue_hist': self.target_hue_hist,
            'value_hist_before': value_hist_before,
            'value_hist_after': value_hist_after,
            'target_value_hist': self.target_value_hist,
            'light_id': light_id,
        }

        if not done:
            next_position = self.positions_raw[self.light_indices[self.current_idx]]
            next_state = self._build_state(
                current_position=next_position,
                current_hue_hist=hue_after,
                current_value_hist=value_hist_after
            )
        else:
            next_state = None

        return next_state, 0.0, done, info

    def _copy_state(self, state):
        """深拷贝状态字典，避免后续修改影响已存储的轨迹"""
        new_state = {}
        for k, v in state.items():
            if isinstance(v, np.ndarray):
                new_state[k] = v.copy()
            else:
                new_state[k] = v  # 标量（如 t）直接复制
        return new_state

    # ==================== 新增辅助方法：生成带相似度控制的最终灯光参数 ====================
    def _generate_final_hues_values(self, N, hue_similarity=None, hue_noise_scale=10.0):
        """
        根据相似度控制生成最终的 hue 和 value。
        参数:
            N: 灯光数量
            hue_similarity: 控制色调聚类的整数 k，表示大致有 k 种颜色；若为 None 则完全随机。
            hue_noise_scale: 组内噪声的标准差（度），默认10度。
        返回:
            hues: (N,) float32, 范围 [0,360)
            values: (N,) float32, 范围 [0,1] 完全随机
        """
        values = np.random.uniform(0, 1, size=N).astype(np.float32)

        if hue_similarity is None or hue_similarity <= 0:
            # 完全随机
            hues = np.random.uniform(0, 360, size=N).astype(np.float32)
            return hues, values

        k = int(hue_similarity)
        # 随机选择 k 个中心色调
        centers = np.random.uniform(0, 360, size=k)
        # 为每个灯光随机分配一个中心（可以均匀分配，这里用随机分配）
        assignments = np.random.randint(0, k, size=N)
        # 在每个中心附近添加高斯噪声，并取模360以保证环形
        noises = np.random.normal(0, hue_noise_scale, size=N)
        hues = centers[assignments] + noises
        hues = hues % 360  # 保持范围
        return hues.astype(np.float32), values

    # ==================== 新增重置方法：基于最终参数设定目标分布 ====================
    def reset_expert(self, N=None, hue_similarity=None, final_hues=None, final_values=None,
                     positions=None, light_indices=None):
        """
        以“专家模式”重置环境：目标分布由一组最终灯光参数确定（若无则随机生成，支持相似度控制）。
        返回初始状态（所有灯光尚未设置，hue=0, value=0）。
        参数:
            N: 灯光数量（若未提供 final_hues/final_values 则必须指定）
            hue_similarity: 控制最终色调聚类的整数 k，仅在随机生成时有效
            final_hues: 可选，预定义的最终 hues (N,) 范围 [0,360)
            final_values: 可选，预定义的最终 values (N,) 范围 [0,1]
            positions: 可选，灯光位置 (N,2)；若未提供则按 simple_layout 生成
            light_indices: 可选，设置灯光的顺序索引 (N,)；若未提供则默认按位置顺序
        """
        if final_hues is None or final_values is None:
            # 需要随机生成最终参数
            if N is None:
                N = np.random.randint(self.min_lights, self.max_lights + 1)
            final_hues, final_values = self._generate_final_hues_values(N, hue_similarity)
        else:
            N = len(final_hues)

        # 生成位置
        if positions is None:
            if self.simple_layout:
                angles = np.linspace(0, 2*np.pi, N, endpoint=False)
                positions = np.array([
                    [0.5 + 0.5 * np.cos(angle), 0.5 + 0.5 * np.sin(angle)]
                    for angle in angles
                ]).astype(np.float32)
            else:
                positions = np.random.uniform(0, 1, size=(N, 2)).astype(np.float32)

        # 设置灯光顺序
        if light_indices is None:
            if self.simple_layout:
                light_indices = np.arange(N)  # 保持位置顺序
            else:
                light_indices = np.random.permutation(N)

        # 计算目标分布（由最终参数混合而成）
        final_result = compute_mixed_lighting(
            positions=positions,
            hues=final_hues,
            values=final_values,
            grid_size=(self.grid_h, self.grid_w),
            decay_model=self.decay_model,
            sigma=self.sigma,
            value_power=self.value_power,
            eps=self.eps
        )
        target_hue = final_result['hue_histogram']
        target_value = final_result['value_histogram']

        # 设置环境内部状态
        self.N = N
        self.positions_raw = positions
        self.positions_padded = np.zeros((self.max_lights, 2), dtype=np.float32)
        self.positions_padded[:N] = positions
        self.all_mask = np.zeros(self.max_lights, dtype=bool)
        self.all_mask[:N] = True
        self.hues = np.zeros(N, dtype=np.float32)      # 初始全未设置
        self.values = np.zeros(N, dtype=np.float32)
        self.light_indices = light_indices
        self.current_idx = 0
        self.target_hue_hist = target_hue
        self.target_value_hist = target_value

        # 清空历史缓冲区
        self.history_positions = np.zeros((self.max_lights, 2), dtype=np.float32)
        self.history_actions = np.zeros((self.max_lights, 2), dtype=np.float32)
        self.history_mixed_hue = np.zeros((self.max_lights, 360), dtype=np.float32)
        self.history_mixed_value = np.zeros((self.max_lights, 100), dtype=np.float32)

        # 初始混合分布（全黑）
        current_hue, current_value = self._compute_current_mixed()
        state = self._build_state(
            current_position=positions[light_indices[0]],
            current_hue_hist=current_hue,
            current_value_hist=current_value
        )
        return state

    # ==================== 修改专家轨迹生成，支持相似度控制 ====================
    def generate_expert_trajectory(self, N=None, hue_similarity=None):
        """
        生成一条专家轨迹，每一步包含状态、动作、以及动作后的混合分布。
        返回:
            trajectory: list of (state, action, next_hue_hist, next_value_hist)
        """
        if N is None:
            N = np.random.randint(self.min_lights, self.max_lights + 1)

        # 生成最终灯光参数（支持相似度控制）
        final_hues, final_values = self._generate_final_hues_values(N, hue_similarity)

        # 生成灯光位置
        if self.simple_layout:
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
            positions = np.array([
                [0.5 + 0.5 * np.cos(angle), 0.5 + 0.5 * np.sin(angle)]
                for angle in angles
            ]).astype(np.float32)
            light_indices = np.arange(N)  # 顺序按角度递增
        else:
            positions = np.random.uniform(0, 1, size=(N, 2)).astype(np.float32)
            light_indices = np.random.permutation(N)  # 随机顺序

        # 计算目标分布（实际上并不用于环境内部，但可用于检验）
        final_result = compute_mixed_lighting(
            positions=positions,
            hues=final_hues,
            values=final_values,
            grid_size=(self.grid_h, self.grid_w),
            decay_model=self.decay_model,
            sigma=self.sigma,
            value_power=self.value_power,
            eps=self.eps
        )
        target_hue = final_result['hue_histogram']
        target_value = final_result['value_histogram']

        # 手动初始化环境状态
        self.N = N
        self.positions_raw = positions
        self.positions_padded = np.zeros((self.max_lights, 2), dtype=np.float32)
        self.positions_padded[:N] = positions
        self.all_mask = np.zeros(self.max_lights, dtype=bool)
        self.all_mask[:N] = True
        self.hues = np.zeros(N, dtype=np.float32)
        self.values = np.zeros(N, dtype=np.float32)
        self.light_indices = light_indices
        self.current_idx = 0
        self.target_hue_hist = target_hue
        self.target_value_hist = target_value

        self.history_positions = np.zeros((self.max_lights, 2), dtype=np.float32)
        self.history_actions = np.zeros((self.max_lights, 2), dtype=np.float32)
        self.history_mixed_hue = np.zeros((self.max_lights, 360), dtype=np.float32)
        self.history_mixed_value = np.zeros((self.max_lights, 100), dtype=np.float32)

        # 逐步设置灯光，记录轨迹
        trajectory = []
        for i in range(N):
            light_idx = light_indices[i]
            hue = final_hues[light_idx]
            value = final_values[light_idx]
            action = np.array([hue / 360.0, value], dtype=np.float32)

            # 动作前的分布
            current_hue, current_value = self._compute_current_mixed()
            current_position = positions[light_idx]

            state = self._build_state(
                current_position=current_position,
                current_hue_hist=current_hue,
                current_value_hist=current_value
            )

            # 执行动作
            self.hues[light_idx] = hue
            self.values[light_idx] = value

            # 动作后的分布
            next_hue, next_value = self._compute_current_mixed()

            # 存储 (state, action, next_hue, next_value) —— 全部深拷贝
            trajectory.append((
                self._copy_state(state),
                action.copy(),
                next_hue.copy(),
                next_value.copy()
            ))

            # 更新历史缓冲区
            self.history_positions[i] = positions[light_idx]
            self.history_actions[i] = action
            self.history_mixed_hue[i] = current_hue
            self.history_mixed_value[i] = current_value

            self.current_idx = i + 1

        return trajectory


if __name__ == '__main__':
    # 测试新功能
    env = LightingEnv(simple_layout=True, max_n_peaks=4, max_hue_similarity=3)

    # 测试 reset（现在可能随机生成两种目标）
    print("=== reset (random mode) ===")
    state = env.reset(N=8)
    print("t =", state['t'])
    print("target_hue histogram sum =", state['target_hue'].sum())
    print("target_value histogram sum =", state['target_value'].sum())

    # 测试 reset_expert（随机生成，相似度控制 k=3）
    print("\n=== reset_expert with hue_similarity=3 ===")
    state = env.reset_expert(N=8, hue_similarity=3)
    print("t =", state['t'])
    print("target_hue histogram sum =", state['target_hue'].sum())
    print("target_value histogram sum =", state['target_value'].sum())

    # 测试生成专家轨迹，相似度控制 k=2
    traj = env.generate_expert_trajectory(N=6, hue_similarity=2)
    for step, (s, a, next_hue, next_value) in enumerate(traj):
        print(f"Step {step}: action {a}, t={s['t']}, next_hue sum={next_hue.sum():.2f}")
    print("轨迹长度:", len(traj))