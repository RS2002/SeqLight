import os
import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
from datetime import datetime
from collections import namedtuple, deque
import copy
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

# HER样本数据结构：包含状态、动作以及该动作后的实际分布（与目标无关）
class HERSample:
    def __init__(self, state, action, next_hue, next_value):
        self.state = state
        self.action = action
        self.next_hue = next_hue
        self.next_value = next_value

def collate_her_samples(batch):
    """合并HER样本列表，返回 (states, actions, next_hues, next_values) 的批处理字典"""
    states = [item.state for item in batch]
    actions = np.stack([item.action for item in batch])
    next_hues = np.stack([item.next_hue for item in batch])
    next_values = np.stack([item.next_value for item in batch])
    # 构造状态字典
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

# ------------------------------ 动态专家数据集 ------------------------------
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
    """合并 (state, action, next_hue, next_value) 列表，用于专家或策略数据"""
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
    buffer = RolloutBuffer()
    state = env.reset(mode=1)
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

        next_state, _, done, info = env.step(action)

        buffer.push(
            state=state,
            action=action,
            reward=0.0,
            next_state=next_state,
            done=done,
            log_prob=log_prob.item(),
            value=value.item(),
            next_hue=info['hue_hist_after'],
            next_value=info['value_hist_after']
        )

        state = next_state if not done else env.reset(mode=1)
        steps += 1

    return buffer

# ------------------------------ 从缓冲区提取完整轨迹 ------------------------------
def extract_trajectories(buffer):
    """将缓冲区中的transition按done分割成完整轨迹列表，每条轨迹是一个Transition列表"""
    transitions = buffer.get_all()
    trajectories = []
    current_traj = []
    for t in transitions:
        current_traj.append(t)
        if t.done:
            trajectories.append(current_traj)
            current_traj = []
    if current_traj:
        # 如果最后一条轨迹未完成，丢弃（因为无法获取最终分布）
        pass
    return trajectories

# ------------------------------ 生成HER样本 ------------------------------
def generate_her_samples(trajectories):
    """
    从完整轨迹列表生成HER样本。
    对每条轨迹，用最终分布的next_hue/next_value替换每个transition的状态目标，生成 (new_state, action, next_hue, next_value) 样本。
    其中 next_hue/next_value 是原始轨迹中该动作后的实际分布（保持不变）。
    返回 HERSample 列表。
    """
    her_samples = []
    for traj in trajectories:
        # 最终分布
        final_hue = traj[-1].next_hue
        final_value = traj[-1].next_value
        for t in traj:
            # 深拷贝状态
            new_state = copy.deepcopy(t.state)
            new_state['target_hue'] = final_hue
            new_state['target_value'] = final_value
            her_samples.append(HERSample(new_state, t.action, t.next_hue, t.next_value))
    return her_samples

# ------------------------------ 判别器奖励计算 ------------------------------
def compute_discriminator_rewards(buffer, policy, device, batch_size=64):
    transitions = buffer.get_all()
    new_buffer = RolloutBuffer()

    for i in range(0, len(transitions), batch_size):
        batch = transitions[i:i+batch_size]
        states = [t.state for t in batch]
        actions = np.stack([t.action for t in batch])
        batched_state, batched_actions, _, _ = collate_states_with_next(
            [(states[j], actions[j], None, None) for j in range(len(batch))]
        )
        for k in batched_state:
            if isinstance(batched_state[k], torch.Tensor):
                batched_state[k] = batched_state[k].to(device)
        batched_actions = batched_actions.to(device)

        with torch.no_grad():
            logits, _, _ = policy.discriminate(batched_state, batched_actions)
            rewards = logits.cpu().numpy().flatten()

        for j, t in enumerate(batch):
            new_buffer.push(
                state=t.state,
                action=t.action,
                reward=rewards[j],
                next_state=t.next_state,
                done=t.done,
                log_prob=t.log_prob,
                value=t.value,
                next_hue=t.next_hue,
                next_value=t.next_value
            )

    return new_buffer

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

# ------------------------------ 评估函数 ------------------------------
def evaluate_policy(env, policy, device, num_episodes=5):
    policy.eval()
    hue_dists = []
    value_dists = []
    for _ in range(num_episodes):
        state = env.reset(mode=1,n=3)
        done = False
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
            state = next_state
        final_hue = info['hue_hist_after']
        final_value = info['value_hist_after']
        target_hue = info['target_hue_hist']
        target_value = info['target_value_hist']
        hue_dists.append(circular_l1_distance(final_hue, target_hue))
        value_dists.append(linear_l1_distance(final_value, target_value))
    policy.train()
    return np.mean(hue_dists), np.mean(value_dists)

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
def train_airl(args):
    log_file = args.log_file
    with open(log_file, 'w') as f:
        f.write(f"AIRL Training Log - Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Arguments: {args}\n")
        f.write("Iter\tDisc Loss\tPolicy Loss\tAvg Reward\tAvg Hue L1\tAvg Value L1\n")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")

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

    policy = SeqLight(
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers
    ).to(device)
    if args.pretrained_path:
        policy.load_state_dict(torch.load(args.pretrained_path, map_location=device))
        print(f"Loaded pretrained model from {args.pretrained_path}")
    policy.train()

    optimizer = optim.Adam(policy.parameters(), lr=args.lr)

    # 学习率调度器
    if args.lr_scheduler == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    elif args.lr_scheduler == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.iterations, eta_min=args.lr_min)
    else:
        scheduler = None

    expert_dataset = DynamicExpertDataset(
        env,
        num_trajectories_per_iter=args.expert_trajs_per_iter,
        min_lights=args.min_lights,
        max_lights=args.max_lights,
        hue_similarity_range=(1, args.max_hue_similarity)
    )

    best_score = float('inf')

    for iteration in range(args.iterations):
        print(f"Iteration {iteration+1}/{args.iterations}")

        # 1. 生成专家数据
        expert_dataset.generate()

        # 2. 收集策略轨迹
        print("  Collecting policy trajectories...")
        rollout_buffer = collect_trajectories(env, policy, args.policy_steps_per_iter, device)

        # 3. 如果启用HER，从策略轨迹中提取完整轨迹并生成HER样本
        her_samples = []
        if args.use_her:
            trajectories = extract_trajectories(rollout_buffer)
            her_samples = generate_her_samples(trajectories)
            print(f"    Generated {len(her_samples)} HER samples")

        # 4. 计算判别器奖励（基于原始目标，用于PPO）
        rollout_buffer = compute_discriminator_rewards(rollout_buffer, policy, device, batch_size=args.batch_size)

        # 5. 训练判别器
        disc_loss_total = 0.0
        for _ in range(args.disc_updates):
            # 从真实专家数据中采样
            expert_batch = expert_dataset.sample(args.batch_size)

            # 从策略轨迹中采样（作为负样本）
            policy_transitions = rollout_buffer.sample(args.batch_size)
            policy_batch = collate_states_with_next(
                [(t.state, t.action, t.next_hue, t.next_value) for t in policy_transitions]
            )

            # 如果启用HER并用于判别器，则从HER样本中采样作为正样本的一部分
            # HER样本也有真实的下一步分布，可以直接与专家合并
            if args.use_her and args.her_in_expert and len(her_samples) > 0:
                her_batch_size = int(args.batch_size * args.her_ratio)
                if her_batch_size > 0:
                    # 从HER样本中采样
                    her_indices = np.random.choice(len(her_samples), her_batch_size, replace=False)
                    her_batch = [her_samples[i] for i in her_indices]
                    her_states, her_actions, her_next_hue, her_next_value = collate_her_samples(her_batch)

                    # 从真实专家中采样剩余数量
                    expert_batch_real = expert_dataset.sample(args.batch_size - her_batch_size)

                    # 合并HER和真实专家
                    batch_states_e = {}
                    for k in expert_batch_real[0].keys():
                        if k == 't':
                            batch_states_e[k] = torch.cat([expert_batch_real[0][k], her_states[k]], dim=0)
                        elif isinstance(expert_batch_real[0][k], torch.Tensor):
                            batch_states_e[k] = torch.cat([expert_batch_real[0][k], her_states[k]], dim=0)
                    batch_actions_e = torch.cat([expert_batch_real[1], her_actions], dim=0)
                    batch_next_hue_e = torch.cat([expert_batch_real[2], her_next_hue], dim=0)
                    batch_next_value_e = torch.cat([expert_batch_real[3], her_next_value], dim=0)
                else:
                    # her_batch_size为0，退化为纯专家
                    batch_states_e, batch_actions_e, batch_next_hue_e, batch_next_value_e = expert_batch
            else:
                # 不使用HER，直接使用专家
                batch_states_e, batch_actions_e, batch_next_hue_e, batch_next_value_e = expert_batch

            # 将专家数据（可能包含HER）移到设备
            for k in batch_states_e:
                if isinstance(batch_states_e[k], torch.Tensor):
                    batch_states_e[k] = batch_states_e[k].to(device)
            batch_actions_e = batch_actions_e.to(device)
            batch_next_hue_e = batch_next_hue_e.to(device)
            batch_next_value_e = batch_next_value_e.to(device)

            # 策略数据
            batch_states_p, batch_actions_p, batch_next_hue_p, batch_next_value_p = policy_batch
            for k in batch_states_p:
                if isinstance(batch_states_p[k], torch.Tensor):
                    batch_states_p[k] = batch_states_p[k].to(device)
            batch_actions_p = batch_actions_p.to(device)
            batch_next_hue_p = batch_next_hue_p.to(device)
            batch_next_value_p = batch_next_value_p.to(device)

            logits_e, pred_hue_e, pred_value_e = policy.discriminate(batch_states_e, batch_actions_e)
            logits_p, pred_hue_p, pred_value_p = policy.discriminate(batch_states_p, batch_actions_p)

            labels_e = torch.ones_like(logits_e)
            labels_p = torch.zeros_like(logits_p)
            loss_bce = F.binary_cross_entropy_with_logits(
                torch.cat([logits_e, logits_p]),
                torch.cat([labels_e, labels_p])
            )

            # AUX loss：所有样本（包括HER）都有真实的下一步分布，统一计算
            loss_aux_e = -(batch_next_hue_e * torch.log(pred_hue_e + 1e-8)).sum(dim=-1).mean() \
                         - (batch_next_value_e * torch.log(pred_value_e + 1e-8)).sum(dim=-1).mean()
            loss_aux_p = -(batch_next_hue_p * torch.log(pred_hue_p + 1e-8)).sum(dim=-1).mean() \
                         - (batch_next_value_p * torch.log(pred_value_p + 1e-8)).sum(dim=-1).mean()
            loss_aux = (loss_aux_e + loss_aux_p) / 2

            disc_loss = loss_bce + args.aux_weight * loss_aux

            optimizer.zero_grad()
            disc_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
            optimizer.step()

            disc_loss_total += disc_loss.item()

        avg_disc_loss = disc_loss_total / args.disc_updates

        # 6. 训练策略（PPO），可选择加入 BC loss 和 HER BC loss
        policy_loss_total = 0.0
        transitions = rollout_buffer.get_all()
        for _ in range(args.policy_updates):
            # 从策略轨迹中采样一个 batch
            indices = np.random.choice(len(transitions), args.batch_size, replace=False)
            batch = [transitions[i] for i in indices]
            batched = collate_policy_trajectory(batch)

            # 计算BC loss：可能来自真实专家和/或HER
            bc_loss_val = None
            if args.use_bc_loss:
                # 从真实专家采样
                bc_expert_batch = expert_dataset.sample(args.batch_size)
                bc_states_e, bc_actions_e, _, _ = bc_expert_batch
                for k in bc_states_e:
                    if isinstance(bc_states_e[k], torch.Tensor):
                        bc_states_e[k] = bc_states_e[k].to(device)
                bc_actions_e = bc_actions_e.to(device)
                _, _, _, bc_log_prob_e, _ = policy(bc_states_e, action=bc_actions_e)
                bc_loss_e = -bc_log_prob_e.mean()

                # 如果启用HER BC，则同时从HER采样并计算BC loss
                if args.use_her and args.her_in_bc and len(her_samples) > 0:
                    her_indices = np.random.choice(len(her_samples), args.batch_size, replace=False)
                    her_batch = [her_samples[i] for i in her_indices]
                    her_states, her_actions, _, _ = collate_her_samples(her_batch)  # 只需要状态和动作
                    for k in her_states:
                        if isinstance(her_states[k], torch.Tensor):
                            her_states[k] = her_states[k].to(device)
                    her_actions = her_actions.to(device)
                    _, _, _, bc_log_prob_h, _ = policy(her_states, action=her_actions)
                    bc_loss_h = -bc_log_prob_h.mean()
                    # 合并两个BC loss，可以加权平均
                    bc_loss_val = (bc_loss_e + bc_loss_h) / 2
                else:
                    bc_loss_val = bc_loss_e

            loss = ppo_update(policy, optimizer, batched,
                              args.clip_epsilon, args.value_coef, args.entropy_coef,
                              args.max_grad_norm, device,
                              bc_loss=bc_loss_val, bc_coef=args.bc_coef if args.use_bc_loss else 0.0)
            policy_loss_total += loss

        avg_policy_loss = policy_loss_total / args.policy_updates

        # 7. 评估
        avg_hue_l1, avg_value_l1 = evaluate_policy(env, policy, device, num_episodes=5)

        avg_reward = np.mean([t.reward for t in transitions])
        print(f"  Disc Loss: {avg_disc_loss:.4f}, Policy Loss: {avg_policy_loss:.4f}, "
              f"Avg Reward: {avg_reward:.4f}, Hue L1: {avg_hue_l1:.4f}, Value L1: {avg_value_l1:.4f}")

        with open(log_file, 'a') as f:
            f.write(f"{iteration+1}\t{avg_disc_loss:.6f}\t{avg_policy_loss:.6f}\t"
                    f"{avg_reward:.6f}\t{avg_hue_l1:.6f}\t{avg_value_l1:.6f}\n")

        torch.save(policy.state_dict(), args.latest_save_path)
        print(f"  Latest model saved to {args.latest_save_path}")

        current_score = avg_hue_l1 + avg_value_l1
        if current_score < best_score:
            best_score = current_score
            torch.save(policy.state_dict(), args.best_save_path)
            print(f"  New best model saved to {args.best_save_path}")

        # 学习率调度器步进
        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            print(f"  Current LR: {current_lr:.6f}")

    print("AIRL training finished.")
    with open(log_file, 'a') as f:
        f.write(f"Training finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# ------------------------------ 主程序入口 ------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AIRL + PPO for Lighting Control with HER")
    # 环境参数
    parser.add_argument('--min_lights', type=int, default=8)
    parser.add_argument('--max_lights', type=int, default=8)
    parser.add_argument('--simple_layout', action='store_true')
    parser.add_argument('--max_n_peaks', type=int, default=3)
    parser.add_argument('--max_hue_similarity', type=int, default=3)

    # 模型参数
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--pretrained_path', type=str, default='bc_best.pth')

    # 训练参数
    parser.add_argument('--iterations', type=int, default=200)
    parser.add_argument('--expert_trajs_per_iter', type=int, default=50)
    parser.add_argument('--policy_steps_per_iter', type=int, default=1000)
    parser.add_argument('--disc_updates', type=int, default=5)
    parser.add_argument('--policy_updates', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--aux_weight', type=float, default=0.1)
    parser.add_argument('--clip_epsilon', type=float, default=0.2)
    parser.add_argument('--value_coef', type=float, default=0.5)
    parser.add_argument('--entropy_coef', type=float, default=0.01)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no_cuda', action='store_true')

    # BC loss 选项
    parser.add_argument('--use_bc_loss', action='store_true', help='Whether to add BC loss during policy update')
    parser.add_argument('--bc_coef', type=float, default=0.1, help='Coefficient for BC loss')

    # HER 选项
    parser.add_argument('--use_her', action='store_true', help='Enable Hindsight Experience Replay')
    parser.add_argument('--her_in_expert', action='store_true', help='Use HER samples as expert data for discriminator')
    parser.add_argument('--her_in_bc', action='store_true', help='Use HER samples for BC loss')
    parser.add_argument('--her_ratio', type=float, default=0.2, help='Proportion of HER samples in expert batch (if used)')

    # 学习率调度器
    parser.add_argument('--lr_scheduler', type=str, default='step', choices=['none', 'step', 'cosine'],
                        help='Learning rate scheduler type')
    parser.add_argument('--lr_step_size', type=int, default=50, help='StepLR step size')
    parser.add_argument('--lr_gamma', type=float, default=0.5, help='StepLR gamma')
    parser.add_argument('--lr_min', type=float, default=1e-5, help='Minimum LR for cosine annealing')

    # 保存与日志
    parser.add_argument('--best_save_path', type=str, default='airl_best.pth')
    parser.add_argument('--latest_save_path', type=str, default='airl_latest.pth')
    parser.add_argument('--log_file', type=str, default='airl_log.txt')

    args = parser.parse_args()
    train_airl(args)