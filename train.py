import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.nn import functional as F
from collections import deque, namedtuple
import random
import os
from datetime import datetime
from light_mix import compute_mixed_lighting      # 假设该模块存在
from models import SeqLight                        # 假设该模块存在
from itertools import groupby

# ====================== 距离度量函数 ======================
def circular_l1_distance(p, q, bins=360):
    """
    计算两个环形离散分布的平均L1距离（通过循环移位取最小）。
    """
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
    """
    计算两个线性离散分布的平均L1距离。
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p /= p.sum() + 1e-8
    q /= q.sum() + 1e-8
    return np.sum(np.abs(p - q)) / len(p)

# ====================== 环境类（增强版，支持专家模式） ======================
class LightingEnv:
    def __init__(self, args):
        self.args = args
        self.grid_h, self.grid_w = args.grid_size
        self.decay_model = args.decay_model
        self.sigma = args.sigma
        self.value_power = args.value_power
        self.eps = args.eps
        self.min_lights = args.min_lights
        self.max_lights = args.max_lights
        self.target_gen_modes = args.target_gen_modes
        self.simple_layout = args.simple_layout

        # 内部状态
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
        self.target_gen_mode = None

        # 历史记录（固定大小）
        self.history_positions = None
        self.history_actions = None
        self.history_mixed_hue = None
        self.history_mixed_value = None

        # 专家数据（用于生成专家轨迹）
        self.expert_hues = None
        self.expert_values = None

    def seed(self, seed=None):
        if seed is None:
            return
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

    def _gen_target_random_sparse(self):
        # 生成稀疏高斯峰目标分布（与原代码相同）
        n_peaks = np.random.randint(1, 5)
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

    def _generate_target_distribution(self):
        return self._gen_target_random_sparse()

    def _compute_current_mixed(self):
        """根据当前非零灯光计算混合分布"""
        mask = (self.values != 0)
        if not np.any(mask):
            # 全黑情况
            hue_hist = np.ones(360)
            hue_hist /= 360
            value_hist = np.zeros(100)
            value_hist[0] = 1
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

    def reset(self, N=None):
        """普通模式：随机目标分布，灯光初始为零"""
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
            self.light_indices = np.arange(self.N)
        else:
            self.positions_raw = np.random.uniform(0, 1, size=(self.N, 2)).astype(np.float32)
            self.light_indices = np.random.permutation(self.N)

        self.positions_padded = np.zeros((self.max_lights, 2), dtype=np.float32)
        self.positions_padded[:self.N] = self.positions_raw
        self.all_mask = np.zeros(self.max_lights, dtype=bool)
        self.all_mask[:self.N] = True

        self.hues = np.zeros(self.N, dtype=np.float32)
        self.values = np.zeros(self.N, dtype=np.float32)
        self.current_idx = 0

        # 生成随机目标分布
        self.target_hue_hist, self.target_value_hist = self._generate_target_distribution()

        # 清空历史
        self.history_positions = np.zeros((self.max_lights, 2), dtype=np.float32)
        self.history_actions = np.zeros((self.max_lights, 2), dtype=np.float32)
        self.history_mixed_hue = np.zeros((self.max_lights, 360), dtype=np.float32)
        self.history_mixed_value = np.zeros((self.max_lights, 100), dtype=np.float32)

        # 计算初始混合分布（全黑）
        current_hue_hist, current_value_hist = self._compute_current_mixed()
        state = self._build_state(
            current_position=self.positions_raw[self.light_indices[0]],
            current_hue_hist=current_hue_hist,
            current_value_hist=current_value_hist
        )
        return state

    def reset_with_expert(self, N=None):
        """
        专家模式：随机生成光源参数，以其混合分布作为目标分布，并重置所有灯光为零。
        返回初始状态，后续可通过 get_expert_action 获取当前步的专家动作。
        """
        if N is None:
            self.N = np.random.randint(self.min_lights, self.max_lights + 1)
        else:
            self.N = N

        # 生成灯光位置（与 reset 相同）
        if self.simple_layout:
            angles = np.linspace(0, 2*np.pi, self.N, endpoint=False)
            self.positions_raw = np.array([
                [0.5 + 0.5 * np.cos(angle), 0.5 + 0.5 * np.sin(angle)]
                for angle in angles
            ]).astype(np.float32)
            self.light_indices = np.arange(self.N)
        else:
            self.positions_raw = np.random.uniform(0, 1, size=(self.N, 2)).astype(np.float32)
            self.light_indices = np.random.permutation(self.N)

        self.positions_padded = np.zeros((self.max_lights, 2), dtype=np.float32)
        self.positions_padded[:self.N] = self.positions_raw
        self.all_mask = np.zeros(self.max_lights, dtype=bool)
        self.all_mask[:self.N] = True

        # 随机生成专家光源参数
        self.expert_hues = np.random.uniform(0, 360, size=self.N).astype(np.float32)
        self.expert_values = np.random.uniform(0, 1, size=self.N).astype(np.float32)

        # 临时设置灯光值，计算目标分布
        self.hues = self.expert_hues.copy()
        self.values = self.expert_values.copy()
        self.target_hue_hist, self.target_value_hist = self._compute_current_mixed()

        # 重置灯光为零，准备逐步设置
        self.hues = np.zeros(self.N, dtype=np.float32)
        self.values = np.zeros(self.N, dtype=np.float32)
        self.current_idx = 0

        # 清空历史
        self.history_positions = np.zeros((self.max_lights, 2), dtype=np.float32)
        self.history_actions = np.zeros((self.max_lights, 2), dtype=np.float32)
        self.history_mixed_hue = np.zeros((self.max_lights, 360), dtype=np.float32)
        self.history_mixed_value = np.zeros((self.max_lights, 100), dtype=np.float32)

        # 构建初始状态（全黑）
        current_hue_hist, current_value_hist = self._compute_current_mixed()
        state = self._build_state(
            current_position=self.positions_raw[self.light_indices[0]],
            current_hue_hist=current_hue_hist,
            current_value_hist=current_value_hist
        )
        return state

    def get_expert_action(self):
        """返回当前步的专家动作（归一化 hue, value）"""
        light_id = self.light_indices[self.current_idx]
        hue_norm = self.expert_hues[light_id] / 360.0
        value_norm = self.expert_values[light_id]
        return np.array([hue_norm, value_norm])

    def _build_state(self, current_position, current_hue_hist, current_value_hist):
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

    def _compute_reward(self, step_info):
        """
        增强版奖励函数：
        - 每步即时距离负奖励
        - 改善奖励
        - 终端正得分 + 成功奖励
        """
        hue_dist_before = circular_l1_distance(step_info['hue_hist_before'], step_info['target_hue_hist'], bins=360)
        hue_dist_after = circular_l1_distance(step_info['hue_hist_after'], step_info['target_hue_hist'], bins=360)
        value_dist_before = linear_l1_distance(step_info['value_hist_before'], step_info['target_value_hist'])
        value_dist_after = linear_l1_distance(step_info['value_hist_after'], step_info['target_value_hist'])

        reward = 0.0

        # 1. 即时距离惩罚
        step_dist_penalty = self.args.step_dist_coef * (hue_dist_after + value_dist_after)
        reward -= step_dist_penalty

        # 2. 改善奖励
        hue_improve = hue_dist_before - hue_dist_after
        value_improve = value_dist_before - value_dist_after
        reward += self.args.hue_improve_coef * hue_improve
        reward += self.args.value_improve_coef * value_improve

        # 3. 终端奖励
        if step_info['is_terminal']:
            hue_score = max(0.0, 1.0 - hue_dist_after / 0.2)
            value_score = max(0.0, 1.0 - value_dist_after / 0.2)
            terminal_reward = self.args.terminal_hue_coef * hue_score + self.args.terminal_value_coef * value_score
            reward += terminal_reward

            if (hue_dist_after < self.args.success_hue_thresh and
                    value_dist_after < self.args.success_value_thresh):
                reward += self.args.success_bonus

        return reward

    def step(self, action):
        hue_norm = np.clip(action[0], 0.0, 1.0)
        value_norm = np.clip(action[1], 0.0, 1.0)
        hue = hue_norm * 360.0
        value = value_norm

        light_id = self.light_indices[self.current_idx]

        hue_before, value_hist_before = self._compute_current_mixed()

        self.hues[light_id] = hue
        self.values[light_id] = value

        hue_after, value_hist_after = self._compute_current_mixed()

        step_info = {
            'hue_hist_before': hue_before,
            'hue_hist_after': hue_after,
            'target_hue_hist': self.target_hue_hist,
            'value_hist_before': value_hist_before,
            'value_hist_after': value_hist_after,
            'target_value_hist': self.target_value_hist,
            'is_terminal': (self.current_idx == self.N - 1),
        }

        reward = self._compute_reward(step_info)

        # 更新历史
        self.history_positions[self.current_idx] = self.positions_raw[light_id]
        self.history_actions[self.current_idx] = [hue_norm, value_norm]
        self.history_mixed_hue[self.current_idx] = hue_before
        self.history_mixed_value[self.current_idx] = value_hist_before

        self.current_idx += 1
        done = (self.current_idx >= self.N)
        info = {
            'hue_err': np.abs(hue_after - self.target_hue_hist).sum() / 2.0,
            'value_err': np.abs(value_hist_after - self.target_value_hist).sum() / 2.0,
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

        return next_state, reward, done, info


# ====================== 经验回放缓冲区 ======================
Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return self._collate(batch)

    def _collate(self, batch):
        states = [t.state for t in batch]
        actions = np.stack([t.action for t in batch])
        rewards = np.stack([t.reward for t in batch])
        dones = np.stack([t.done for t in batch])
        next_states = []
        for t in batch:
            if t.next_state is None:
                zero_state = {k: np.zeros_like(v) for k, v in t.state.items()}
                zero_state['t'] = 0
                next_states.append(zero_state)
            else:
                next_states.append(t.next_state)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


# ====================== 批处理转换函数 ======================
def batch_to_device(batch_dicts, device):
    if not batch_dicts:
        return {}
    keys = batch_dicts[0].keys()
    batched = {}
    for k in keys:
        if k == 't':
            batched[k] = torch.tensor([d[k] for d in batch_dicts], device=device)
        else:
            arrs = [d[k] for d in batch_dicts]
            stacked = np.stack(arrs, axis=0).astype(np.float32)
            batched[k] = torch.from_numpy(stacked).to(device)
    return batched


# ====================== TD3 训练主程序 ======================
def train(args):
    eval_interval = 50
    num_eval_episodes = 5
    best_avg_reward = -np.inf

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")
    env = LightingEnv(args)
    if args.seed >= 0:
        env.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    policy = SeqLight(
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
    ).to(device)
    target = SeqLight(
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
    ).to(device)
    target.load_state_dict(policy.state_dict())
    target.eval()

    all_params = dict(policy.named_parameters())
    actor_param_names = [name for name in all_params.keys() if 'critic_head' not in name]
    critic_param_names = [name for name in all_params.keys() if 'actor_head' not in name]
    actor_params = [all_params[name] for name in actor_param_names]
    critic_params = [all_params[name] for name in critic_param_names]

    actor_optim = optim.Adam(actor_params, lr=args.lr_actor)
    critic_optim = optim.Adam(critic_params, lr=args.lr_critic)

    replay_buffer = ReplayBuffer(args.buffer_capacity)

    # ========== 生成专家样本 ==========
    if args.use_expert:
        print(f"Generating {args.num_expert_episodes} expert episodes...")
        for ep in range(args.num_expert_episodes):
            state = env.reset_with_expert()
            done = False
            while not done:
                expert_action = env.get_expert_action()
                next_state, reward, done, info = env.step(expert_action)
                replay_buffer.push(state, expert_action, reward, next_state, done)
                state = next_state
        print(f"Expert buffer size: {len(replay_buffer)}")

    # 重置环境为普通模式，开始训练
    state = env.reset()

    log_path = args.log_file
    log_file_eval = args.log_file_eval
    with open(log_path, 'w') as f:
        f.write(f"TD3 Training Log - Started at {datetime.now()}\n")
        f.write(f"Args: {args}\n\n")
        f.write("Episode\tTotalReward\tAvgReward\tAvgQ1\tAvgQ2\tFinalHueErr\tFinalValueErr\n")

    os.makedirs(args.save_dir, exist_ok=True)

    total_steps = 0
    episode_rewards = []
    episode_len = env.N
    episode_reward = 0.0
    episode_q1_last = 0.0
    episode_q2_last = 0.0
    actor_update_counter = 0

    while total_steps < args.total_timesteps:
        # ----- 选择动作 -----
        with torch.no_grad():
            state_tensor = {
                k: torch.from_numpy(v).unsqueeze(0).to(device) if isinstance(v, np.ndarray) else v
                for k, v in state.items()
            }
            action = policy(state_tensor)
            if torch.isnan(action).any():
                action = torch.zeros_like(action)
            action = action.cpu().numpy().flatten()

            noise_std = max(args.exploration_noise * (1 - total_steps / args.total_timesteps), 0.01)
            noise = np.random.normal(0, noise_std, size=action.shape)
            hue_noisy = (action[0] + noise[0]) % 1.0
            value_noisy = action[1] + noise[1]
            value_noisy = np.clip(value_noisy, 0.0, 1.0)
            action = np.array([hue_noisy, value_noisy])

        next_state, reward, done, info = env.step(action)
        replay_buffer.push(state, action, reward, next_state, done)

        state = next_state if not done else env.reset()
        episode_reward += reward
        total_steps += 1

        # ----- 训练更新 -----
        if len(replay_buffer) > args.batch_size:
            for _ in range(args.updates_per_step):
                states, actions, rewards, next_states, dones = replay_buffer.sample(args.batch_size)
                batch_zip = list(zip(states, actions, rewards, next_states, dones))
                done_indices = [i for i, d in enumerate(dones) if d]
                not_done_indices = [i for i, d in enumerate(dones) if not d]

                sub_groups_states = []  # 用于 Actor 更新
                total_critic_loss = 0.0

                # 处理非终止样本
                if not_done_indices:
                    batch_not_done = [batch_zip[i] for i in not_done_indices]
                    batch_not_done_sorted = sorted(batch_not_done, key=lambda x: x[0]['t'])
                    for t_val, group in groupby(batch_not_done_sorted, key=lambda x: x[0]['t']):
                        sub_group = list(group)
                        sub_states, sub_actions, sub_rewards, sub_next_states, _ = zip(*sub_group)

                        batch_state = batch_to_device(list(sub_states), device)
                        batch_next_state = batch_to_device(list(sub_next_states), device)
                        batch_actions = torch.from_numpy(np.stack(sub_actions)).float().to(device)
                        batch_rewards = torch.from_numpy(np.stack(sub_rewards)).float().to(device).unsqueeze(1)

                        sub_groups_states.append((t_val, batch_state))

                        with torch.no_grad():
                            target_actions = target(batch_next_state)
                            if torch.isnan(target_actions).any():
                                target_actions = torch.zeros_like(target_actions)
                            noise = torch.randn_like(batch_actions) * args.target_noise
                            noise = torch.clamp(noise, -args.noise_clip, args.noise_clip)
                            hue_target = (target_actions[..., 0] + noise[..., 0]) % 1.0
                            value_target = target_actions[..., 1] + noise[..., 1]
                            value_target = torch.clamp(value_target, 0.0, 1.0)
                            target_actions_noisy = torch.stack([hue_target, value_target], dim=-1)

                            target_q1, target_q2 = target(batch_next_state, target_actions_noisy)
                            target_q = torch.min(target_q1, target_q2)
                            target_q = batch_rewards + args.gamma * target_q

                        current_q1, current_q2 = policy(batch_state, batch_actions)
                        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
                        total_critic_loss += critic_loss

                        with torch.no_grad():
                            episode_q1_last = current_q1.mean().item()
                            episode_q2_last = current_q2.mean().item()

                # 处理终止样本
                if done_indices:
                    done_samples = []
                    for i in done_indices:
                        state_i, action_i, reward_i, _, _ = batch_zip[i]
                        done_samples.append((state_i, action_i, reward_i, state_i['t']))
                    done_samples_sorted = sorted(done_samples, key=lambda x: x[3])
                    for t_val, group in groupby(done_samples_sorted, key=lambda x: x[3]):
                        sub_group = list(group)
                        sub_states = [item[0] for item in sub_group]
                        sub_actions = np.stack([item[1] for item in sub_group])
                        sub_rewards = np.stack([item[2] for item in sub_group])

                        batch_state = batch_to_device(sub_states, device)
                        batch_actions = torch.from_numpy(sub_actions).float().to(device)
                        batch_rewards = torch.from_numpy(sub_rewards).float().to(device).unsqueeze(1)

                        sub_groups_states.append((t_val, batch_state))

                        current_q1, current_q2 = policy(batch_state, batch_actions)
                        target_q = batch_rewards
                        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
                        total_critic_loss += critic_loss

                # 更新 Critic
                if total_critic_loss != 0.0:
                    critic_optim.zero_grad()
                    total_critic_loss.backward()
                    torch.nn.utils.clip_grad_norm_(critic_params, args.max_grad_norm)
                    critic_optim.step()

                # 策略更新（延迟）
                actor_update_counter += 1
                if actor_update_counter % args.policy_delay == 0 and sub_groups_states:
                    total_actor_loss = 0.0
                    for _, batch_state in sub_groups_states:
                        actions_pred = policy(batch_state)
                        if torch.isnan(actions_pred).any():
                            continue
                        q1, _ = policy(batch_state, actions_pred)
                        actor_loss = -q1.mean()
                        total_actor_loss += actor_loss

                    if total_actor_loss != 0.0:
                        actor_optim.zero_grad()
                        total_actor_loss.backward()
                        torch.nn.utils.clip_grad_norm_(actor_params, args.max_grad_norm)
                        actor_optim.step()

                        for target_param, param in zip(target.parameters(), policy.parameters()):
                            target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

        # ----- 日志与保存 -----
        if done:
            episode_rewards.append(episode_reward)
            with open(args.log_file, 'a') as f:
                f.write(f"{len(episode_rewards)}\t{episode_reward:.4f}\t"
                        f"{episode_reward / episode_len:.4f}\t"
                        f"{episode_q1_last:.4f}\t{episode_q2_last:.4f}\t"
                        f"{info['hue_err']:.4f}\t{info['value_err']:.4f}\n")

            torch.save({
                'policy_state_dict': policy.state_dict(),
                'target_state_dict': target.state_dict(),
                'actor_optim_state_dict': actor_optim.state_dict(),
                'critic_optim_state_dict': critic_optim.state_dict(),
                'args': args,
                'episode': len(episode_rewards),
                'total_steps': total_steps
            }, os.path.join(args.save_dir, args.model_name))

            episode_reward = 0.0
            episode_len = env.N

            if len(episode_rewards) % args.print_freq == 0:
                avg_reward = np.mean(episode_rewards[-min(args.print_freq, len(episode_rewards)):])
                print(f"Episode {len(episode_rewards)} | Steps {total_steps} | "
                      f"AvgReward {avg_reward:.2f} | FinalHueErr {info['hue_err']:.4f} | "
                      f"FinalValueErr {info['value_err']:.2f}")

            if len(episode_rewards) % eval_interval == 0:
                print("\n=== Evaluation ===")
                eval_rewards = []
                eval_hue_errs = []
                eval_value_errs = []

                for _ in range(num_eval_episodes):
                    state_eval = env.reset(N=8)
                    ep_reward = 0.0
                    done = False
                    while not done:
                        with torch.no_grad():
                            state_tensor = {k: torch.from_numpy(v).unsqueeze(0).to(device).float()
                                            if isinstance(v, np.ndarray) else v
                                            for k, v in state_eval.items()}
                            action = policy(state_tensor).cpu().numpy().flatten()
                            next_state, reward, done, info_eval = env.step(action)
                            ep_reward += reward
                            state_eval = next_state
                    eval_rewards.append(ep_reward)
                    eval_hue_errs.append(info_eval['hue_err'])
                    eval_value_errs.append(info_eval['value_err'])

                avg_eval_reward = np.mean(eval_rewards)
                print(f"Eval Avg Reward: {avg_eval_reward:.2f} | "
                      f"Avg Final Hue Err: {np.mean(eval_hue_errs):.4f} | "
                      f"Avg Final Value Err: {np.mean(eval_value_errs):.4f}")

                with open(log_file_eval, 'a') as f:
                    f.write(f"Episode {len(episode_rewards)}\tSteps {total_steps}\t"
                            f"Eval Avg Reward: {avg_eval_reward:.2f}\t"
                            f"Avg Final Hue Err: {np.mean(eval_hue_errs):.4f}\t"
                            f"Avg Final Value Err: {np.mean(eval_value_errs):.4f}\n")

                if avg_eval_reward > best_avg_reward:
                    best_avg_reward = avg_eval_reward
                    torch.save({
                        'policy_state_dict': policy.state_dict(),
                        'target_state_dict': target.state_dict(),
                        'actor_optim_state_dict': actor_optim.state_dict(),
                        'critic_optim_state_dict': critic_optim.state_dict(),
                        'args': args,
                        'episode': len(episode_rewards),
                        'total_steps': total_steps
                    }, os.path.join(args.save_dir, "best_model.pth"))
                    print("New best model saved!")

            state = env.reset()

    print("Training finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TD3 for Lighting Control')
    # 环境参数
    parser.add_argument('--grid_size', type=int, nargs=2, default=[120, 160])
    parser.add_argument('--decay_model', type=str, default='gaussian', choices=['gaussian', 'inverse_square'])
    parser.add_argument('--sigma', type=float, default=0.18)
    parser.add_argument('--value_power', type=float, default=1.0)
    parser.add_argument('--eps', type=float, default=1e-8)
    parser.add_argument('--min_lights', type=int, default=8)
    parser.add_argument('--max_lights', type=int, default=8)
    parser.add_argument('--target_gen_modes', type=int, nargs='+', default=[1])
    parser.add_argument('--simple_layout', type=bool, default=True, help='使用简单布局：灯光均匀分布在圆上，顺序顺时针')

    # 专家样本参数
    parser.add_argument('--use_expert', type=bool, default=True, help='是否使用专家样本')
    parser.add_argument('--num_expert_episodes', type=int, default=10, help='专家轨迹数量')

    # 模型参数
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=3)

    # 训练超参数
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

    # 日志与保存
    parser.add_argument('--log_file', type=str, default='training_log.txt')
    parser.add_argument('--log_file_eval', type=str, default='eval_log.txt')
    parser.add_argument('--save_dir', type=str, default='./models')
    parser.add_argument('--model_name', type=str, default='model.pth')
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no_cuda', action='store_true')

    # 奖励函数参数
    parser.add_argument('--step_dist_coef', type=float, default=2.0, help='每步距离负奖励系数')
    parser.add_argument('--hue_improve_coef', type=float, default=5.0, help='hue改善奖励系数')
    parser.add_argument('--value_improve_coef', type=float, default=5.0, help='value改善奖励系数')
    parser.add_argument('--terminal_hue_coef', type=float, default=10.0, help='终端hue得分系数')
    parser.add_argument('--terminal_value_coef', type=float, default=10.0, help='终端value得分系数')
    parser.add_argument('--success_hue_thresh', type=float, default=0.15, help='成功hue距离阈值')
    parser.add_argument('--success_value_thresh', type=float, default=0.15, help='成功value距离阈值')
    parser.add_argument('--success_bonus', type=float, default=20.0, help='成功额外奖励')

    args = parser.parse_args()
    train(args)