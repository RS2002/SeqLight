import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.nn import functional as F
from collections import deque, namedtuple
import random
import os
from datetime import datetime
from light_mix import compute_mixed_lighting
from models import SeqLight
from itertools import groupby

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
        self.target_mean_value = None
        self.target_max_value = None
        self.target_gen_mode = None
        # 固定大小历史（padding with 0）
        self.history_positions = None
        self.history_actions = None
        self.history_mixed_hue = None
        self.history_mixed_value = None
    def seed(self, seed=None):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
    # ---------- 目标分布生成方式0：虚拟光源混合 ----------
    def _gen_target_mixed_lights(self):
        num_src = np.random.randint(1, 6)
        src_pos = np.random.uniform(0, 1, size=(num_src, 2))
        src_hues = np.random.uniform(0, 1.0, size=num_src)
        src_vals = np.random.uniform(0, 1.0, size=num_src)
        result = compute_mixed_lighting(
            positions=src_pos,
            hues=src_hues,
            values=src_vals,
            grid_size=(self.grid_h, self.grid_w),
            decay_model=self.decay_model,
            sigma=self.sigma,
            value_power=self.value_power,
            eps=self.eps
        )
        return result['hue_histogram'], result['mean_value'], result['max_value']
    # ---------- 目标分布生成方式1：随机稀疏分布 ----------
    def _gen_target_random_sparse(self):
        n_peaks = np.random.randint(1, 6)
        hue_hist = np.ones(360) * 0.001
        for _ in range(n_peaks):
            peak_pos = np.random.randint(0, 360)
            peak_weight = np.random.uniform(0.5, 1.0)
            for offset in range(-10, 11):
                idx = (peak_pos + offset) % 360
                hue_hist[idx] += peak_weight * np.exp(-0.5 * (offset / 3.0) ** 2)
        hue_hist /= hue_hist.sum()
        # 随机亮度目标
        mean_value = np.random.uniform(0.0, 1.0)
        # 对于稀疏分布，假设最大亮度 ≈ 平均亮度 * (1.2~1.5) 并 clip 至 ≤1.0
        max_value = min(1.0, mean_value * np.random.uniform(1.2, 1.5))
        return hue_hist, mean_value, max_value
    def _generate_target_distribution(self):
        self.target_gen_mode = random.choice(self.target_gen_modes)
        if self.target_gen_mode == 0:
            return self._gen_target_mixed_lights()
        elif self.target_gen_mode == 1:
            return self._gen_target_random_sparse()
        else:
            raise ValueError(f"Unknown target generation mode: {self.target_gen_mode}")
    def _compute_current_mixed(self):
        """返回 (hue_hist, mean_value, max_value)"""
        mask = self.values > 0
        if not np.any(mask):
            hue_hist = np.ones(360) / 360.0
            mean_val = 0.0
            max_val = 0.0
            return hue_hist, mean_val, max_val
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
        return result['hue_histogram'], result['mean_value'], result['max_value']
    def reset(self):
        self.N = np.random.randint(self.min_lights, self.max_lights + 1)
        self.positions_raw = np.random.uniform(0, 1, size=(self.N, 2)).astype(np.float32)
        self.positions_padded = np.zeros((self.max_lights, 2), dtype=np.float32)
        self.positions_padded[:self.N] = self.positions_raw
        self.all_mask = np.zeros(self.max_lights, dtype=bool)
        self.all_mask[:self.N] = True
        self.hues = np.zeros(self.N, dtype=np.float32)
        self.values = np.zeros(self.N, dtype=np.float32)
        self.light_indices = np.random.permutation(self.N)
        self.current_idx = 0
        # 生成目标分布（包含 mean 和 max）
        (self.target_hue_hist,
         self.target_mean_value,
         self.target_max_value) = self._generate_target_distribution()
        # 初始历史 padded with 0
        self.history_positions = np.zeros((self.max_lights, 2), dtype=np.float32)
        self.history_actions = np.zeros((self.max_lights, 2), dtype=np.float32)
        self.history_mixed_hue = np.zeros((self.max_lights, 360), dtype=np.float32)
        self.history_mixed_value = np.zeros((self.max_lights, 1), dtype=np.float32)
        # 初始状态（无灯亮）
        current_hue_hist, current_mean_val, current_max_val = self._compute_current_mixed()
        state = self._build_state(
            current_position=self.positions_raw[self.light_indices[0]],
            current_hue_hist=current_hue_hist,
            current_mean_val=current_mean_val
        )
        return state
    def _build_state(self, current_position, current_hue_hist, current_mean_val):
        """状态字典（与之前完全一致）"""
        state = {
            'target_hue': self.target_hue_hist.astype(np.float32),
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
    # ---------- 新版奖励函数（改善奖励 + 终端惩罚/奖励）----------
    def _compute_reward(self, step_info):
        """
        奖励规则：
          - 非终端步：
              * 色相分布距离（L1/2）减小 → 正奖励，增大 → 负奖励
              * 亮度绝对误差减小 → 正奖励，增大 → 负奖励
          - 终端步：
              * 直接惩罚最终色相距离和亮度误差（负奖励）
              * 若同时低于阈值，给予额外成功奖励
        """
        # 色相分布距离（L1/2，范围 [0,1]）
        hue_dist_before = np.abs(step_info['hue_hist_before'] - step_info['target_hue_hist']).sum() / 2.0
        hue_dist_after = np.abs(step_info['hue_hist_after'] - step_info['target_hue_hist']).sum() / 2.0
        # 亮度绝对误差
        value_err_before = abs(step_info['mean_value_before'] - step_info['target_mean_value'])
        value_err_after = abs(step_info['mean_value_after'] - step_info['target_mean_value'])
        reward = 0.0
        hue_improve = hue_dist_before - hue_dist_after
        reward += self.args.hue_improve_coef * hue_improve
        value_improve = value_err_before - value_err_after
        reward += self.args.value_improve_coef * value_improve
        if step_info['is_terminal']:
            # 终端：惩罚最终误差
            reward -= self.args.terminal_hue_coef * hue_dist_after
            reward -= self.args.terminal_value_coef * value_err_after
            # 成功奖励（同时满足阈值）
            if (hue_dist_after < self.args.success_hue_thresh and
                    value_err_after < self.args.success_value_thresh):
                reward += self.args.success_bonus
        return reward
    def step(self, action):
        # print(action)
        """
        执行动作，返回 (next_state, reward, done, info)
        action: np.ndarray shape (2,), 取值范围 [0,1]
        """
        # 安全处理动作
        hue = np.clip(action[0] , 0.0, 1.0)
        value = np.clip(action[1], 0.0, 1.0)
        light_id = self.light_indices[self.current_idx]
        # ----- 记录动作前的状态 -----
        hue_before, mean_before, max_before = self._compute_current_mixed()
        # ----- 更新灯光参数 -----
        self.hues[light_id] = hue
        self.values[light_id] = value
        # ----- 记录动作后的状态 -----
        hue_after, mean_after, max_after = self._compute_current_mixed()
        # ----- 构造 step_info 用于奖励计算 -----
        step_info = {
            'hue_hist_before': hue_before,
            'hue_hist_after': hue_after,
            'target_hue_hist': self.target_hue_hist,
            'mean_value_before': mean_before,
            'mean_value_after': mean_after,
            'max_value_before': max_before,
            'max_value_after': max_after,
            'target_mean_value': self.target_mean_value,
            'target_max_value': self.target_max_value,
            'is_terminal': (self.current_idx == self.N - 1) # 当前步是最后一盏灯？
        }
        reward = self._compute_reward(step_info)
        # 更新历史（fixed index）
        self.history_positions[self.current_idx] = self.positions_raw[light_id]
        self.history_actions[self.current_idx] = [hue, value]
        self.history_mixed_hue[self.current_idx] = hue_before
        self.history_mixed_value[self.current_idx] = [mean_before]
        # 推进步数
        self.current_idx += 1
        done = (self.current_idx >= self.N)
        info = {
            'hue_err': np.abs(hue_after - self.target_hue_hist).sum() / 2.0, # 仍保留用于日志
            'value_err': (mean_after - self.target_mean_value) ** 2,
            'mean_value': mean_after,
            'max_value': max_after,
            'light_id': light_id,
            'target_gen_mode': self.target_gen_mode
        }
        if not done:
            next_position = self.positions_raw[self.light_indices[self.current_idx]]
            next_state = self._build_state(
                current_position=next_position,
                current_hue_hist=hue_after,
                current_mean_val=mean_after
            )
        else:
            next_state = None
        return next_state, reward, done, info
# ------------------------------ 经验回放缓冲区 ------------------------------
Transition = namedtuple('Transition',
                        ['state', 'action', 'reward', 'next_state', 'done'])
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
            if t.next_state is None: # TODO: note it's not valid manner (but we process it in training process)
                zero_state = {k: np.zeros_like(v) for k, v in t.state.items()}
                zero_state['t'] = t.state['t']  # 保持 t
                next_states.append(zero_state)
            else:
                next_states.append(t.next_state)
        return states, actions, rewards, next_states, dones
    def __len__(self):
        return len(self.buffer)
# ------------------------------ 批处理转换函数------------------------
def batch_to_device(batch_dicts, device):
    if not batch_dicts:
        return {}
    keys = batch_dicts[0].keys()
    batched = {}
    for k in keys:
        if k == 't':
            batched[k] = batch_dicts[0][k]  # same t in subbatch
        else:
            arrs = [d[k] for d in batch_dicts]
            stacked = np.stack(arrs, axis=0).astype(np.float32)
            batched[k] = torch.from_numpy(stacked).to(device)
    return batched
# ------------------------------ TD3 训练主程序----------------
def train(args):
    eval_interval = 50  # 每 50 个 episode 评估一次
    num_eval_episodes = 5  # 每次评估跑 5 局无噪声
    best_avg_reward = -np.inf

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")
    env = LightingEnv(args)
    if args.seed >= 0:
        env.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
    # ---------- 网络初始化（两个 SeqLight 实例）----------
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
    # ---------- 优化器参数分组 ----------
    all_params = dict(policy.named_parameters())
    actor_param_names = [name for name in all_params.keys() if 'critic_head' not in name]
    critic_param_names = [name for name in all_params.keys() if 'actor_head' not in name]
    actor_params = [all_params[name] for name in actor_param_names]
    critic_params = [all_params[name] for name in critic_param_names]
    actor_optim = optim.Adam(actor_params, lr=args.lr_actor)
    critic_optim = optim.Adam(critic_params, lr=args.lr_critic)
    replay_buffer = ReplayBuffer(args.buffer_capacity)
    # 日志
    log_path = args.log_file
    log_file_eval = args.log_file_eval
    with open(log_path, 'w') as f:
        f.write(f"TD3 Training Log - Started at {datetime.now()}\n")
        f.write(f"Args: {args}\n\n")
        f.write("Episode\tTotalReward\tAvgReward\tAvgQ1\tAvgQ2\tFinalHueErr\tFinalValueErr\n")
    os.makedirs(args.save_dir, exist_ok=True)
    model_save_path = os.path.join(args.save_dir, args.model_name)
    total_steps = 0
    episode_rewards = []
    state = env.reset()
    episode_len = env.N
    episode_reward = 0.0
    episode_q1_last = 0.0
    episode_q2_last = 0.0
    while total_steps < args.total_timesteps:
        # ---------- 选择动作 ----------
        with torch.no_grad():
            state_tensor = {
                k: torch.from_numpy(v).unsqueeze(0).to(device) if isinstance(v, np.ndarray) else v
                for k, v in state.items()
            }
            action = policy(state_tensor)
            if torch.isnan(action).any():
                action = torch.zeros_like(action)
            action = action.cpu().numpy().flatten()
            noise = np.random.normal(0, args.exploration_noise, size=action.shape)
            action = np.clip(action + noise, 0.0, 1.0)
        next_state, reward, done, info = env.step(action)
        replay_buffer.push(state, action, reward, next_state, done)
        state = next_state if not done else env.reset()
        episode_reward += reward
        total_steps += 1
        # ---------- 训练 ----------
        if len(replay_buffer) > args.batch_size:
            for _ in range(args.updates_per_step):
                states, actions, rewards, next_states, dones = replay_buffer.sample(args.batch_size)
                # group by t to handle variable history length
                batch_zip = list(zip(states, actions, rewards, next_states, dones))
                batch_zip_sorted = sorted(batch_zip, key=lambda x: x[0]['t'])
                actor_loss_total = 0.0
                critic_loss_total = 0.0
                batch_size_actual = 0
                for t, group in groupby(batch_zip_sorted, key=lambda x: x[0]['t']):
                    sub_group = list(group)
                    sub_size = len(sub_group)
                    sub_states, sub_actions, sub_rewards, sub_next_states, sub_dones = zip(*sub_group)
                    batch_state = batch_to_device(list(sub_states), device)
                    batch_next_state = batch_to_device(list(sub_next_states), device)
                    batch_actions = torch.from_numpy(np.stack(sub_actions)).float().to(device)
                    batch_rewards = torch.from_numpy(np.stack(sub_rewards)).float().to(device).unsqueeze(1)
                    batch_dones = torch.from_numpy(np.stack(sub_dones)).float().to(device).unsqueeze(1)
                    # ----- 更新 Critic -----
                    with torch.no_grad():
                        noise = torch.randn_like(batch_actions) * args.target_noise
                        noise = torch.clamp(noise, -args.noise_clip, args.noise_clip)
                        target_actions = target(batch_next_state)
                        if torch.isnan(target_actions).any():
                            target_actions = torch.zeros_like(target_actions)
                        target_actions = torch.clamp(target_actions + noise, 0.0, 1.0)
                        target_q1, target_q2 = target(batch_next_state, target_actions)
                        target_q = torch.min(target_q1, target_q2)
                        target_q = torch.where(
                            batch_dones.bool(),
                            torch.zeros_like(target_q),  # done=1 时未来价值=0
                            target_q
                        )
                        target_q = batch_rewards + args.gamma * (1 - batch_dones) * target_q
                        # print(target_q, batch_dones)
                    current_q1, current_q2 = policy(batch_state, batch_actions)
                    critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
                    if torch.isnan(critic_loss):
                        print("Invalid Critic Loss")
                        continue
                    critic_loss_total += critic_loss.item() * sub_size
                    critic_optim.zero_grad()
                    critic_loss.backward()
                    torch.nn.utils.clip_grad_norm_(critic_params, args.max_grad_norm)
                    critic_optim.step()
                    # 记录最后一次更新的 Q 值（用于日志）
                    with torch.no_grad():
                        episode_q1_last = current_q1.mean().item()
                        episode_q2_last = current_q2.mean().item()
                    # ----- 延迟更新 Actor -----
                    if total_steps % args.policy_delay == 0:
                        actions_pred = policy(batch_state)
                        if torch.isnan(actions_pred).any():
                            print("Invalid Actor Prediction")
                            continue
                        q1 = policy(batch_state, actions_pred)[0]
                        actor_loss = -q1.mean()
                        if torch.isnan(actor_loss):
                            print("Invalid Actor Loss")
                            continue
                        actor_loss_total += actor_loss.item() * sub_size
                        actor_optim.zero_grad()
                        actor_loss.backward()
                        torch.nn.utils.clip_grad_norm_(actor_params, args.max_grad_norm)
                        actor_optim.step()
                        # 软更新
                        for target_param, param in zip(target.parameters(), policy.parameters()):
                            target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                    batch_size_actual += sub_size
                # 平均 loss (if needed, but since update per sub, ok)
        # ---------- Episode 结束 ----------
        if done:
            episode_rewards.append(episode_reward)
            episode_final_hue_err = info['hue_err']
            episode_final_value_err = info['value_err']
            with open(log_path, 'a') as f:
                f.write(f"{len(episode_rewards)}\t{episode_reward:.4f}\t"
                        f"{episode_reward / episode_len:.4f}\t"
                        f"{episode_q1_last:.4f}\t{episode_q2_last:.4f}\t"
                        f"{episode_final_hue_err:.4f}\t{episode_final_value_err:.4f}\n")
            torch.save({
                'policy_state_dict': policy.state_dict(),
                'target_state_dict': target.state_dict(),
                'actor_optim_state_dict': actor_optim.state_dict(),
                'critic_optim_state_dict': critic_optim.state_dict(),
                'args': args,
                'episode': len(episode_rewards),
                'total_steps': total_steps
            }, model_save_path)
            # 重置
            episode_reward = 0.0
            episode_len = env.N
            if len(episode_rewards) % args.print_freq == 0:
                avg_reward = np.mean(episode_rewards[-min(args.print_freq, len(episode_rewards)):])
                print(f"Episode {len(episode_rewards)} | Steps {total_steps} | "
                      f"AvgReward {avg_reward:.2f} | FinalHueErr {episode_final_hue_err:.4f} | "
                      f"AvgQ1 {episode_q1_last:.2f}")

            # evaluation
            if len(episode_rewards) % eval_interval == 0 and len(episode_rewards) > 0:
                print("\n=== Evaluation ===")
                eval_rewards = []
                eval_hue_errs = []
                eval_value_errs = []

                for _ in range(num_eval_episodes):
                    state = env.reset()
                    ep_reward = 0.0
                    done = False
                    while not done:
                        with torch.no_grad():
                            state_tensor = {k: torch.from_numpy(v).unsqueeze(0).to(device).float()
                            if isinstance(v, np.ndarray) else v
                                            for k, v in state.items()}
                            action = policy(state_tensor).cpu().numpy().flatten()  # 无噪声！
                            next_state, reward, done, info = env.step(action)
                            ep_reward += reward
                            state = next_state
                    eval_rewards.append(ep_reward)
                    eval_hue_errs.append(info['hue_err'])
                    eval_value_errs.append(info['value_err'])

                avg_eval_reward = np.mean(eval_rewards)
                print(f"Eval Avg Reward: {avg_eval_reward:.2f} | "
                      f"Avg Final Hue Err: {np.mean(eval_hue_errs):.4f} | "
                      f"Avg Final Value Err: {np.mean(eval_value_errs):.4f}")

                with open(log_file_eval, 'a') as f:
                    f.write(f"Episode {len(episode_rewards)}\t"
                      f"Steps {total_steps}\t"      
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
# ------------------------------ 参数解析 ------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TD3 for Lighting Control - Wasserstein Reward')
    # ---------- 环境参数 ----------
    parser.add_argument('--grid_size', type=int, nargs=2, default=[120, 160],
                        help='Grid size (height, width)')
    parser.add_argument('--decay_model', type=str, default='gaussian',
                        choices=['gaussian', 'inverse_square'],
                        help='Light decay model')
    parser.add_argument('--sigma', type=float, default=0.18,
                        help='Sigma for Gaussian decay')
    parser.add_argument('--value_power', type=float, default=1.0,
                        help='Gamma correction for value')
    parser.add_argument('--eps', type=float, default=1e-8,
                        help='Numerical stability epsilon')
    parser.add_argument('--min_lights', type=int, default=5,
                        help='Minimum number of lights per episode')
    parser.add_argument('--max_lights', type=int, default=15,
                        help='Maximum number of lights per episode (padding size)')
    parser.add_argument('--target_gen_modes', type=int, nargs='+', default=[0, 1],
                        help='Available target generation modes (0: mixed lights, 1: random sparse)')
    # ---------- 模型参数 ----------
    parser.add_argument('--d_model', type=int, default=64,
                        help='Transformer embedding dimension')
    parser.add_argument('--nhead', type=int, default=4,
                        help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='Number of transformer layers')
    # ---------- 训练超参数 ----------
    parser.add_argument('--total_timesteps', type=int, default=100000,
                        help='Total environment steps')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--buffer_capacity', type=int, default=100000,
                        help='Replay buffer size')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor')
    parser.add_argument('--tau', type=float, default=0.005,
                        help='Soft update coefficient')
    parser.add_argument('--lr_actor', type=float, default=3e-4,
                        help='Actor learning rate')
    parser.add_argument('--lr_critic', type=float, default=3e-4,
                        help='Critic learning rate')
    parser.add_argument('--policy_delay', type=int, default=2,
                        help='Delay steps for policy update')
    parser.add_argument('--exploration_noise', type=float, default=0.2,
                        help='Exploration noise std')
    parser.add_argument('--target_noise', type=float, default=0.2,
                        help='Target policy smoothing noise std')
    parser.add_argument('--noise_clip', type=float, default=0.5,
                        help='Noise clip limit')
    parser.add_argument('--updates_per_step', type=int, default=1,
                        help='Number of updates per environment step')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Gradient clipping norm')
    # ---------- 日志与保存 ----------
    parser.add_argument('--log_file', type=str, default='training_log.txt',
                        help='Log file path')
    parser.add_argument('--log_file_eval', type=str, default='eval_log.txt',
                        help='Log file path (evaluation)')
    parser.add_argument('--save_dir', type=str, default='./models',
                        help='Directory to save model')
    parser.add_argument('--model_name', type=str, default='model.pth',
                        help='Model filename (overwritten)')
    parser.add_argument('--print_freq', type=int, default=10,
                        help='Print progress every N episodes')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed, -1 for no seed')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Disable CUDA')
    # ---------- 奖励函数参数（全新）----------
    parser.add_argument('--hue_improve_coef', type=float, default=4.0,
                        help='Coefficient for hue improvement reward (positive = reward for reduction)')
    parser.add_argument('--value_improve_coef', type=float, default=2.0,
                        help='Coefficient for value improvement reward')
    parser.add_argument('--terminal_hue_coef', type=float, default=5.0,
                        help='Coefficient for terminal hue distance penalty')
    parser.add_argument('--terminal_value_coef', type=float, default=3.0,
                        help='Coefficient for terminal value error penalty')
    parser.add_argument('--success_hue_thresh', type=float, default=0.08,
                        help='Hue error threshold for success bonus')
    parser.add_argument('--success_value_thresh', type=float, default=0.02,
                        help='Value error threshold for success bonus')
    parser.add_argument('--success_bonus', type=float, default=10.0,
                        help='Extra reward when terminal errors below thresholds')
    args = parser.parse_args()
    train(args)