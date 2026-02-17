import os
import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from datetime import datetime
from collections import namedtuple
from env import LightingEnv
from models import SeqLight
from light_mix import compute_mixed_lighting

# ------------------------------ 数据结构 ------------------------------
Transition = namedtuple('Transition',
    ['state', 'action', 'reward', 'next_state', 'done',
     'log_prob', 'value', 'next_hue', 'next_value'])

class RolloutBuffer:
    def __init__(self):
        self.buffer = []

    def push(self, **kwargs):
        self.buffer.append(Transition(**kwargs))

    def clear(self):
        self.buffer = []

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]

    def __len__(self):
        return len(self.buffer)

    def get_all(self):
        return self.buffer

# ------------------------------ 动态专家数据集（从 AIRL 复制） ------------------------------
class DynamicExpertDataset:
    def __init__(self, env, num_trajectories_per_iter, min_lights, max_lights,
                 hue_similarity_range=(1, 3)):
        self.env = env
        self.num_trajectories_per_iter = num_trajectories_per_iter
        self.min_lights = min_lights
        self.max_lights = max_lights
        self.hue_similarity_range = hue_similarity_range
        self.buffer = []

    def generate(self):
        self.buffer = []
        for _ in range(self.num_trajectories_per_iter):
            N = np.random.randint(self.min_lights, self.max_lights + 1)
            hue_sim = np.random.randint(self.hue_similarity_range[0],
                                        self.hue_similarity_range[1] + 1)
            traj = self.env.generate_expert_trajectory(N=N, hue_similarity=hue_sim)
            for step, (state, action, next_hue, next_value) in enumerate(traj):
                self.buffer.append((state, action, next_hue, next_value))
        return self.buffer

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        return collate_states_with_next(batch)

# ------------------------------ 批处理函数 ------------------------------
def collate_states_with_next(batch):
    """合并 (state, action, next_hue, next_value) 列表"""
    states = [item[0] for item in batch]
    actions = np.stack([item[1] for item in batch])
    next_hues = np.stack([item[2] if item[2] is not None else np.zeros(360) for item in batch])
    next_values = np.stack([item[3] if item[3] is not None else np.zeros(100) for item in batch])

    batched = {}
    keys = states[0].keys()
    for k in keys:
        if k == 't':
            batched[k] = torch.tensor([s[k] for s in states], dtype=torch.long)
        elif isinstance(states[0][k], np.ndarray):
            arr = np.stack([s[k] for s in states])
            if k == 'all_mask':
                batched[k] = torch.from_numpy(arr).bool()
            else:
                batched[k] = torch.from_numpy(arr).float()
        else:
            batched[k] = torch.tensor([s[k] for s in states])
    return batched, torch.from_numpy(actions).float(), \
           torch.from_numpy(next_hues).float(), torch.from_numpy(next_values).float()

def collate_policy_trajectory(batch):
    """合并 Transition 列表用于 PPO 更新"""
    states = [t.state for t in batch]
    actions = np.stack([t.action for t in batch])
    rewards = np.stack([t.reward for t in batch])
    dones = np.stack([t.done for t in batch])
    old_log_probs = np.stack([t.log_prob for t in batch])
    old_values = np.stack([t.value for t in batch])
    next_hues = np.stack([t.next_hue for t in batch])
    next_values = np.stack([t.next_value for t in batch])

    next_states = []
    for t in batch:
        if t.next_state is None:
            zero_state = {}
            for k, v in t.state.items():
                if isinstance(v, np.ndarray):
                    zero_state[k] = np.zeros_like(v)
                else:
                    zero_state[k] = 0
            zero_state['t'] = 0
            next_states.append(zero_state)
        else:
            next_states.append(t.next_state)

    batched_states = {}
    keys = states[0].keys()
    for k in keys:
        if k == 't':
            batched_states[k] = torch.tensor([s[k] for s in states], dtype=torch.long)
        elif isinstance(states[0][k], np.ndarray):
            arr = np.stack([s[k] for s in states])
            if k == 'all_mask':
                batched_states[k] = torch.from_numpy(arr).bool()
            else:
                batched_states[k] = torch.from_numpy(arr).float()
        else:
            batched_states[k] = torch.tensor([s[k] for s in states])

    batched_next_states = {}
    for k in keys:
        if k == 't':
            batched_next_states[k] = torch.tensor([s[k] for s in next_states], dtype=torch.long)
        elif isinstance(next_states[0][k], np.ndarray):
            arr = np.stack([s[k] for s in next_states])
            if k == 'all_mask':
                batched_next_states[k] = torch.from_numpy(arr).bool()
            else:
                batched_next_states[k] = torch.from_numpy(arr).float()
        else:
            batched_next_states[k] = torch.tensor([s[k] for s in next_states])

    return (batched_states,
            torch.from_numpy(actions).float(),
            torch.from_numpy(rewards).float().unsqueeze(1),
            torch.from_numpy(dones).float().unsqueeze(1),
            torch.from_numpy(old_log_probs).float().unsqueeze(1),
            torch.from_numpy(old_values).float().unsqueeze(1),
            batched_next_states,
            torch.from_numpy(next_hues).float(),
            torch.from_numpy(next_values).float())

# ------------------------------ 轨迹收集 ------------------------------
def collect_trajectories(env, policy, num_steps, device, deterministic=False):
    """收集 num_steps 步轨迹，同时使用固定的判别器计算奖励"""
    buffer = RolloutBuffer()
    state = env.reset()
    steps = 0

    while steps < num_steps:
        state_tensor = {}
        for k, v in state.items():
            if isinstance(v, np.ndarray):
                state_tensor[k] = torch.from_numpy(v).unsqueeze(0).to(device)
            else:
                state_tensor[k] = torch.tensor([v]).to(device)

        with torch.no_grad():
            value, hue_action, val_action, log_prob, _ = policy(state_tensor, deterministic=deterministic)
            action = np.array([hue_action.cpu().numpy()[0], val_action.cpu().numpy()[0]])

        next_state, _, done, info = env.step(action)  # env 返回的 reward 为 0

        # 使用固定的判别器计算奖励
        action_tensor = torch.from_numpy(action).unsqueeze(0).float().to(device)
        with torch.no_grad():
            reward_logit, _, _ = policy.discriminate(state_tensor, action_tensor)
            reward = reward_logit.item()  # logits 直接作为奖励

        buffer.push(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            log_prob=log_prob.item(),
            value=value.item(),
            next_hue=info['hue_hist_after'],
            next_value=info['value_hist_after']
        )

        state = next_state if not done else env.reset()
        steps += 1

    return buffer

# ------------------------------ GAE 优势计算 ------------------------------
def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    advantages = []
    gae = 0
    values = values.squeeze(1).cpu().numpy()
    rewards = rewards.squeeze(1).cpu().numpy()
    dones = dones.squeeze(1).cpu().numpy()

    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0.0
        else:
            next_value = values[t+1] * (1 - dones[t+1])
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)
    advantages = np.array(advantages).reshape(-1, 1)
    returns = advantages + values.reshape(-1, 1)
    return torch.from_numpy(advantages).float(), torch.from_numpy(returns).float()

# ------------------------------ PPO 更新（支持 BC loss） ------------------------------
def ppo_update(policy, optimizer, batch, clip_epsilon, value_coef, entropy_coef,
               max_grad_norm, device, bc_loss=None, bc_coef=0.0):
    (states, actions, rewards, dones, old_log_probs, old_values,
     next_states, next_hues, next_values) = batch

    for k in states:
        if isinstance(states[k], torch.Tensor):
            states[k] = states[k].to(device)
    actions = actions.to(device)
    rewards = rewards.to(device)
    dones = dones.to(device)
    old_log_probs = old_log_probs.to(device)
    old_values = old_values.to(device)

    advantages, returns = compute_gae(rewards, old_values, dones, gamma=0.99, lam=0.95)
    advantages = advantages.to(device)
    returns = returns.to(device)

    values_pred, _, _, log_prob, entropy = policy(states, action=actions.squeeze(1))
    values_pred = values_pred.unsqueeze(1)
    log_prob = log_prob.unsqueeze(1)

    ratio = torch.exp(log_prob - old_log_probs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    value_loss = F.mse_loss(values_pred, returns)

    entropy_loss = -entropy.mean()

    loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss
    if bc_loss is not None:
        loss = loss + bc_coef * bc_loss

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
    optimizer.step()

    return loss.item()

# ------------------------------ 评估函数（返回距离和累积奖励） ------------------------------
def evaluate_policy(env, policy, device, num_episodes=5):
    policy.eval()
    hue_dists = []
    value_dists = []
    episode_rewards = []
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            state_tensor = {}
            for k, v in state.items():
                if isinstance(v, np.ndarray):
                    state_tensor[k] = torch.from_numpy(v).unsqueeze(0).to(device)
                else:
                    state_tensor[k] = torch.tensor([v]).to(device)
            with torch.no_grad():
                _, hue_act, val_act, _, _ = policy(state_tensor, deterministic=True)
                action = np.array([hue_act.cpu().numpy()[0], val_act.cpu().numpy()[0]])
            next_state, _, done, info = env.step(action)
            # 计算这一步的奖励（使用判别器）
            action_tensor = torch.from_numpy(action).unsqueeze(0).float().to(device)
            with torch.no_grad():
                reward_logit, _, _ = policy.discriminate(state_tensor, action_tensor)
                step_reward = reward_logit.item()
            ep_reward += step_reward
            state = next_state
        final_hue = info['hue_hist_after']
        final_value = info['value_hist_after']
        target_hue = info['target_hue_hist']
        target_value = info['target_value_hist']
        hue_dists.append(circular_l1_distance(final_hue, target_hue))
        value_dists.append(linear_l1_distance(final_value, target_value))
        episode_rewards.append(ep_reward)
    policy.train()
    return np.mean(hue_dists), np.mean(value_dists), np.mean(episode_rewards)

# ------------------------------ 距离度量 ------------------------------
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

# ------------------------------ 主训练函数 ------------------------------
def finetune_ppo(args):
    log_file = args.log_file
    with open(log_file, 'w') as f:
        f.write(f"PPO Finetuning Log - Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Arguments: {args}\n")
        f.write("Iter\tPolicy Loss\tAvg Reward\tAvg Eval Reward\tAvg Hue L1\tAvg Value L1\n")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")

    # 环境（更复杂配置）
    env = LightingEnv(
        min_lights=args.min_lights,
        max_lights=args.max_lights,
        simple_layout=args.simple_layout,
        max_n_peaks=args.max_n_peaks,
        max_hue_similarity=args.max_hue_similarity
    )
    if args.seed >= 0:
        env.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    # 加载预训练模型
    policy = SeqLight(
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers
    ).to(device)
    if args.pretrained_path:
        policy.load_state_dict(torch.load(args.pretrained_path, map_location=device))
        print(f"Loaded pretrained model from {args.pretrained_path}")

    # 冻结除 actor_head 和 critic_head 以外的所有参数
    for name, param in policy.named_parameters():
        if 'actor_head' not in name and 'critic_head' not in name:
            param.requires_grad = False
    trainable_params = [p for p in policy.parameters() if p.requires_grad]
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params)}")

    optimizer = optim.Adam(trainable_params, lr=args.lr)

    # 如果需要 BC loss，创建专家数据集
    if args.use_bc_loss:
        expert_dataset = DynamicExpertDataset(
            env,
            num_trajectories_per_iter=args.expert_trajs_per_iter,
            min_lights=args.min_lights,
            max_lights=args.max_lights,
            hue_similarity_range=(1, args.max_hue_similarity)
        )
    else:
        expert_dataset = None

    best_score = float('inf')

    for iteration in range(args.iterations):
        print(f"Iteration {iteration+1}/{args.iterations}")

        # 如果需要 BC loss，生成新的专家数据
        if args.use_bc_loss:
            expert_dataset.generate()

        # 收集轨迹
        print("  Collecting trajectories...")
        rollout_buffer = collect_trajectories(env, policy, args.policy_steps_per_iter, device)

        # PPO 更新多次
        policy_loss_total = 0.0
        transitions = rollout_buffer.get_all()
        for _ in range(args.policy_updates):
            indices = np.random.choice(len(transitions), args.batch_size, replace=False)
            batch = [transitions[i] for i in indices]
            batched = collate_policy_trajectory(batch)

            # 如果需要 BC loss，从专家数据集中采样并计算 BC loss
            if args.use_bc_loss:
                bc_batch = expert_dataset.sample(args.batch_size)
                bc_states, bc_actions, _, _ = bc_batch
                for k in bc_states:
                    if isinstance(bc_states[k], torch.Tensor):
                        bc_states[k] = bc_states[k].to(device)
                bc_actions = bc_actions.to(device)
                _, _, _, bc_log_prob, _ = policy(bc_states, action=bc_actions)
                bc_loss_val = -bc_log_prob.mean()
            else:
                bc_loss_val = None

            loss = ppo_update(policy, optimizer, batched,
                              args.clip_epsilon, args.value_coef, args.entropy_coef,
                              args.max_grad_norm, device,
                              bc_loss=bc_loss_val, bc_coef=args.bc_coef if args.use_bc_loss else 0.0)
            policy_loss_total += loss

        avg_policy_loss = policy_loss_total / args.policy_updates

        # 评估
        avg_hue_l1, avg_value_l1, avg_eval_reward = evaluate_policy(env, policy, device, num_episodes=5)

        # 平均奖励（来自收集的轨迹）
        avg_reward = np.mean([t.reward for t in transitions])
        print(f"  Policy Loss: {avg_policy_loss:.4f}, Avg Reward: {avg_reward:.4f}, "
              f"Eval Reward: {avg_eval_reward:.4f}, Hue L1: {avg_hue_l1:.4f}, Value L1: {avg_value_l1:.4f}")

        with open(log_file, 'a') as f:
            f.write(f"{iteration+1}\t{avg_policy_loss:.6f}\t{avg_reward:.6f}\t"
                    f"{avg_eval_reward:.6f}\t{avg_hue_l1:.6f}\t{avg_value_l1:.6f}\n")

        # 保存最新模型
        torch.save(policy.state_dict(), args.latest_save_path)
        print(f"  Latest model saved to {args.latest_save_path}")

        current_score = avg_hue_l1 + avg_value_l1
        if current_score < best_score:
            best_score = current_score
            torch.save(policy.state_dict(), args.best_save_path)
            print(f"  New best model saved to {args.best_save_path}")

    print("PPO finetuning finished.")
    with open(log_file, 'a') as f:
        f.write(f"Training finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# ------------------------------ 主程序入口 ------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO Finetuning with Fixed Reward")
    # 环境参数（更复杂场景）
    parser.add_argument('--min_lights', type=int, default=8, help='Minimum number of lights')
    parser.add_argument('--max_lights', type=int, default=8, help='Maximum number of lights')
    parser.add_argument('--simple_layout', action='store_true', default=False, help='Use simple circular layout')
    parser.add_argument('--max_n_peaks', type=int, default=3)
    parser.add_argument('--max_hue_similarity', type=int, default=3)

    # 模型参数
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--pretrained_path', type=str, default='airl_latest.pth', help='Path to trained AIRL model')

    # 训练参数
    parser.add_argument('--iterations', type=int, default=200)
    parser.add_argument('--policy_steps_per_iter', type=int, default=1000, help='Steps collected per iteration')
    parser.add_argument('--policy_updates', type=int, default=10, help='Number of PPO updates per iteration')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate for actor/critic')
    parser.add_argument('--clip_epsilon', type=float, default=0.2)
    parser.add_argument('--value_coef', type=float, default=0.5)
    parser.add_argument('--entropy_coef', type=float, default=0.01)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no_cuda', action='store_true')

    # BC loss 选项
    parser.add_argument('--use_bc_loss', action='store_true', help='Whether to add BC loss during policy update')
    parser.add_argument('--bc_coef', type=float, default=0.1, help='Coefficient for BC loss')
    parser.add_argument('--expert_trajs_per_iter', type=int, default=20, help='Expert trajectories per iteration for BC (if used)')

    # 保存与日志
    parser.add_argument('--best_save_path', type=str, default='ppo_best.pth')
    parser.add_argument('--latest_save_path', type=str, default='ppo_latest.pth')
    parser.add_argument('--log_file', type=str, default='ppo_log.txt')

    args = parser.parse_args()
    finetune_ppo(args)