import numpy as np
import random
import torch
from light_mix import compute_mixed_lighting
from scipy.stats import beta as beta_dist  # 用于偏向采样

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
        target_gen_modes=[1],
        simple_layout=True,
        max_n_peaks=3,
        max_hue_similarity=3,
        value_range=(0, 1),
        default_value_bias=5.0,  # 新增：默认值采样偏向性，1.0=均匀，>1 偏向低值
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
        self.value_range = value_range
        self.default_value_bias = default_value_bias

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

        self.ground_truth_hues = None
        self.ground_truth_values = None

    def seed(self, seed=None):
        if seed is None:
            return
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

    def get_state(self):
        """返回当前状态字典，用于 GRPO 轨迹收集"""
        current_position = self.positions_raw[self.light_indices[self.current_idx]]
        current_hue, current_value = self._compute_current_mixed()
        return self._build_state(current_position, current_hue, current_value)

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
        peak_bin = np.random.randint(0, 100)
        peak_weight = np.random.uniform(0.5, 1.0)
        for offset in range(-30, 31):
            idx = peak_bin + offset
            if 0 <= idx < 100:  # 只保留有效范围内的索引
                value_hist[idx] += peak_weight * np.exp(-0.5 * (offset / 15.0) ** 2)
        value_hist /= value_hist.sum()

        return hue_hist, value_hist

    def _gen_target_expert(self, N, hue_similarity, value_range=None, value_bias=None):
        """
        生成专家式目标分布：随机生成 N 个灯光的最终参数（受 hue_similarity 控制），
        计算其混合分布，返回两个直方图（不修改环境状态）。
        value_range: 控制最终灯光 values 的采样范围，若为 None 则使用 self.value_range。
        value_bias: 控制 values 的采样偏向性，若为 None 则使用 self.default_value_bias。
        """
        final_hues, final_values = self._generate_final_hues_values(
            N, hue_similarity, value_range=value_range, value_bias=value_bias
        )

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

    def _generate_target_distribution(self, mode=None, n=None, value_range=None, value_bias=None):
        """
        随机选择目标生成方式：
        - 方式0：随机稀疏峰（峰值数从 1 ~ max_n_peaks 随机）
        - 方式1：专家式（色调相似度从 1 ~ max_hue_similarity 随机）
        value_range: 仅在专家式生成时使用。
        value_bias: 仅在专家式生成时使用。
        """
        if mode is None:
            mode = np.random.choice([0, 1])
        if mode == 0:
            if n is not None:
                n_peaks = n
            else:
                n_peaks = np.random.randint(1, self.max_n_peaks + 1)
            return self._gen_target_random_sparse(n_peaks)
        else:
            if n is not None:
                hue_similarity = n
            else:
                hue_similarity = np.random.randint(1, self.max_hue_similarity + 1)
            return self._gen_target_expert(self.N, hue_similarity, value_range=value_range, value_bias=value_bias)

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

    def _resample_histogram(self, hist, target_bins):
        """
        将直方图重新采样到目标 bin 数，并归一化。
        hist: 原始直方图数组，长度为任意。
        target_bins: 目标 bin 数量。
        返回: 归一化后的 target_bins 长度数组。
        """
        src_len = len(hist)
        if src_len == target_bins:
            # 如果长度已匹配，仅归一化返回
            return (hist.astype(np.float32) / (hist.sum() + 1e-8)).astype(np.float32)

        # 原始 bin 中心位置（假设 bin 均匀分布在 [0,1] 区间）
        src_centers = np.linspace(0, 1, src_len, endpoint=False) + 0.5 / src_len
        # 目标 bin 中心位置
        tgt_centers = np.linspace(0, 1, target_bins, endpoint=False) + 0.5 / target_bins

        # 线性插值
        new_hist = np.interp(tgt_centers, src_centers, hist.astype(np.float64))
        new_hist = np.maximum(new_hist, 0)  # 去除可能的负值
        new_hist /= new_hist.sum() + 1e-8
        return new_hist.astype(np.float32)

    def reset(self, N=None, mode=None, n=None, value_range=None, value_bias=None,
              goal_hue=None, goal_value=None):
        """重置环境，返回初始状态。
        参数:
            goal_hue: 可选，目标色调分布，可以是任意长度的数组，会自动插值到 360 bin。
            goal_value: 可选，目标亮度分布，可以是任意长度的数组，会自动插值到 100 bin。
            如果提供了 goal_hue 和 goal_value，则直接使用它们作为目标分布，忽略 mode 和 n。
        """
        # 确定灯光数量
        if N is None:
            self.N = np.random.randint(self.min_lights, self.max_lights + 1)
        else:
            self.N = N

        # 生成灯光位置（保持不变）
        if self.simple_layout:
            angles = np.linspace(0, 2 * np.pi, self.N, endpoint=False)
            self.positions_raw = np.array([
                [0.5 + 0.5 * np.cos(angle), 0.5 + 0.5 * np.sin(angle)]
                for angle in angles
            ]).astype(np.float32)
            self.light_indices = np.arange(self.N)
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

        # 设置目标分布
        if goal_hue is not None and goal_value is not None:
            # 使用给定的目标分布，必要时进行插值
            self.target_hue_hist = self._resample_histogram(goal_hue, 360)
            self.target_value_hist = self._resample_histogram(goal_value, 100)
            self.ground_truth_hues = None
            self.ground_truth_values = None
        else:
            # 原有随机生成逻辑
            if mode is None:
                mode = np.random.choice([0, 1])
            if mode == 0:  # 随机稀疏峰
                if n is not None:
                    n_peaks = n
                else:
                    n_peaks = np.random.randint(1, self.max_n_peaks + 1)
                self.target_hue_hist, self.target_value_hist = self._gen_target_random_sparse(n_peaks)
                self.ground_truth_hues = None
                self.ground_truth_values = None
            else:  # 专家模式
                if n is not None:
                    hue_similarity = n
                else:
                    hue_similarity = np.random.randint(1, self.max_hue_similarity + 1)
                self.ground_truth_hues, self.ground_truth_values = self._generate_final_hues_values(
                    self.N, hue_similarity, value_range=value_range, value_bias=value_bias
                )
                final_result = compute_mixed_lighting(
                    positions=self.positions_raw,
                    hues=self.ground_truth_hues,
                    values=self.ground_truth_values,
                    grid_size=(self.grid_h, self.grid_w),
                    decay_model=self.decay_model,
                    sigma=self.sigma,
                    value_power=self.value_power,
                    eps=self.eps
                )
                self.target_hue_hist = final_result['hue_histogram']
                self.target_value_hist = final_result['value_histogram']

        # 历史缓冲区（保持不变）
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

    # ==================== 辅助方法：生成带相似度控制的最终灯光参数 ====================
    def _generate_final_hues_values(self, N, hue_similarity=None, hue_noise_scale=10.0,
                                     value_range=None, value_bias=None):
        """
        根据相似度控制生成最终的 hue 和 value。
        参数:
            N: 灯光数量
            hue_similarity: 控制色调聚类的整数 k，表示大致有 k 种颜色；若为 None 则完全随机。
            hue_noise_scale: 组内噪声的标准差（度），默认10度。
            value_range: (low, high) 元组，指定 value 的采样范围，若为 None 则使用 self.value_range。
            value_bias: 控制 values 的偏向性，若为 None 则使用 self.default_value_bias。
                        value_bias=1.0 时均匀采样；>1.0 时采样偏向低值（使用 Beta 分布）；
                        <1.0 时偏向高值（但一般不用）。
        返回:
            hues: (N,) float32, 范围 [0,360)
            values: (N,) float32, 范围 [low, high]，分布由 value_bias 控制
        """
        if value_range is None:
            value_range = self.value_range
        low, high = value_range

        # 处理 value_bias
        if value_bias is None:
            # value_bias = self.default_value_bias
            # value_bias = 1.0
            # value_bias = random.uniform(0.2, 5.0)
            value_bias = random.uniform(1.0, self.default_value_bias)

        if value_bias == 1.0:
            # 均匀采样
            values = np.random.uniform(low, high, size=N).astype(np.float32)
        else:
            # 使用 Beta 分布，参数 a=1, b=value_bias，概率密度偏向 0
            # 生成值在 [0,1] 的 Beta 样本，然后线性映射到 [low, high]
            beta_samples = beta_dist.rvs(1, value_bias, size=N)
            values = low + (high - low) * beta_samples
            values = values.astype(np.float32)

        if hue_similarity is None or hue_similarity <= 0:
            hues = np.random.uniform(0, 360, size=N).astype(np.float32)
            return hues, values

        k = int(hue_similarity)
        centers = np.random.uniform(0, 360, size=k)
        assignments = np.random.randint(0, k, size=N)
        noises = np.random.normal(0, hue_noise_scale, size=N)
        hues = centers[assignments] + noises
        hues = hues % 360
        return hues.astype(np.float32), values

    # ==================== 重置方法：基于最终参数设定目标分布 ====================
    def reset_expert(self, N=None, hue_similarity=None, final_hues=None, final_values=None,
                     positions=None, light_indices=None, value_range=None, value_bias=None):
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
            value_range: 若随机生成最终参数，用于控制 values 的采样范围
            value_bias: 若随机生成最终参数，用于控制 values 的偏向性
        """
        if final_hues is None or final_values is None:
            if N is None:
                N = np.random.randint(self.min_lights, self.max_lights + 1)
            final_hues, final_values = self._generate_final_hues_values(
                N, hue_similarity, value_range=value_range, value_bias=value_bias
            )
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
                light_indices = np.arange(N)
            else:
                light_indices = np.random.permutation(N)

        # 计算目标分布
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
        self.hues = np.zeros(N, dtype=np.float32)
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

    # ==================== 专家轨迹生成，支持相似度控制和亮度偏向 ====================
    def generate_expert_trajectory(self, N=None, hue_similarity=None, value_range=None, value_bias=None):
        """
        生成一条专家轨迹，每一步包含状态、动作、以及动作后的混合分布。
        返回:
            trajectory: list of (state, action, next_hue_hist, next_value_hist)
        """
        if N is None:
            N = np.random.randint(self.min_lights, self.max_lights + 1)

        # 生成最终灯光参数（支持相似度控制和亮度偏向）
        final_hues, final_values = self._generate_final_hues_values(
            N, hue_similarity, value_range=value_range, value_bias=value_bias
        )

        # 生成灯光位置
        if self.simple_layout:
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
            positions = np.array([
                [0.5 + 0.5 * np.cos(angle), 0.5 + 0.5 * np.sin(angle)]
                for angle in angles
            ]).astype(np.float32)
            light_indices = np.arange(N)
        else:
            positions = np.random.uniform(0, 1, size=(N, 2)).astype(np.float32)
            light_indices = np.random.permutation(N)

        # 计算目标分布（仅用于参考，实际不用于环境内部）
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

            next_hue, next_value = self._compute_current_mixed()

            trajectory.append((
                self._copy_state(state),
                action.copy(),
                next_hue.copy(),
                next_value.copy()
            ))

            self.history_positions[i] = positions[light_idx]
            self.history_actions[i] = action
            self.history_mixed_hue[i] = current_hue
            self.history_mixed_value[i] = current_value

            self.current_idx = i + 1

        return trajectory


if __name__ == '__main__':
    # 假设已有目标分布数组
    target_hue = np.random.rand(150)
    target_hue /= target_hue.sum()
    target_value = np.random.rand(80)
    target_value /= target_value.sum()

    env = LightingEnv()
    state = env.reset(goal_hue=target_hue, goal_value=target_value)