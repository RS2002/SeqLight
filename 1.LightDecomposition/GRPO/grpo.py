import os
import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
from datetime import datetime
from collections import namedtuple
import copy
from env import LightingEnv
from models import SeqLight
from light_mix import compute_mixed_lighting

# ------------------------------ 数据结构 ------------------------------
Transition = namedtuple('Transition',
    ['state', 'action', 'reward', 'next_state', 'done',
     'log_prob', 'next_hue', 'next_value'])  # 移除 value 字段

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

# HER样本数据结构（保持不变）
class HERSample:
    def __init__(self, state, action):
        self.state = state
        self.action = action

def collate_her_samples(batch):
    """合并HER样本列表，返回 (states, actions) 的批处理字典"""
    states = [item.state for item in batch]
    actions = np.stack([item.action for item in batch])
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
    return batched, torch.from_numpy(actions).float()

# ------------------------------ 动态专家数据集（保持不变） ------------------------------
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
    """合并 Transition 列表用于 GRPO 更新（已移除 value）"""
    states = [t.state for t in batch]
    actions = np.stack([t.action for t in batch])
    rewards = np.stack([t.reward for t in batch])
    dones = np.stack([t.done for t in batch])
    old_log_probs = np.stack([t.log_prob for t in batch])
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
            batched_next_states,
            torch.from_numpy(next_hues).float(),
            torch.from_numpy(next_values).float())

# ------------------------------ 分组轨迹收集（用于 GRPO） ------------------------------
def collect_grouped_trajectories(env, policy, state_seeds, group_size, device, deterministic=False):
    """
    对每个种子，采样 group_size 条完整轨迹。
    每次重置前设置环境种子，确保相同的初始状态。
    需要环境实现 get_state() 方法。
    """
    all_trajectories = []
    for seed in state_seeds:
        for _ in range(group_size):
            env.seed(seed)            # 设置种子，保证初始状态一致
            env.reset(mode=1)          # 重置环境（假设专家模式）
            states = []
            actions = []
            log_probs = []
            rewards = []
            next_hues = []
            next_values = []
            done = False
            while not done:
                state = env.get_state()   # 获取当前状态字典（需要环境实现）
                state_tensor = {}
                for k, v in state.items():
                    if isinstance(v, np.ndarray):
                        state_tensor[k] = torch.from_numpy(v).unsqueeze(0).to(device)
                    else:
                        state_tensor[k] = torch.tensor([v]).to(device)

                with torch.no_grad():
                    _, hue_action, val_action, log_prob, _ = policy(state_tensor, deterministic=deterministic)
                    action = np.array([hue_action.cpu().numpy()[0], val_action.cpu().numpy()[0]])

                next_state, _, done, info = env.step(action)

                # 使用固定的判别器计算奖励
                action_tensor = torch.from_numpy(action).unsqueeze(0).float().to(device)
                with torch.no_grad():
                    reward_logit, _, _ = policy.discriminate(state_tensor, action_tensor)
                    step_reward = reward_logit.item()

                states.append(state)
                actions.append(action)
                log_probs.append(log_prob.item())
                rewards.append(step_reward)
                next_hues.append(info['hue_hist_after'])
                next_values.append(info['value_hist_after'])

                state = next_state   # 更新（实际环境已更新）

            all_trajectories.append({
                'states': states,
                'actions': actions,
                'log_probs': log_probs,
                'rewards': rewards,
                'next_hues': next_hues,
                'next_values': next_values
            })
    return all_trajectories

# ------------------------------ 从缓冲区提取完整轨迹（用于HER） ------------------------------
def extract_trajectories_from_transitions(transitions):
    """将 Transition 列表按 done 分割成完整轨迹列表"""
    trajectories = []
    current_traj = []
    for t in transitions:
        current_traj.append(t)
        if t.done:
            trajectories.append(current_traj)
            current_traj = []
    if current_traj:
        pass
    return trajectories

def generate_her_samples(trajectories):
    """
    从完整轨迹列表生成HER样本。
    对每条轨迹，用最终分布的next_hue/next_value替换每个transition的状态目标，生成 (new_state, action) 样本。
    返回 HERSample 列表。
    """
    her_samples = []
    for traj in trajectories:
        final_hue = traj[-1].next_hue
        final_value = traj[-1].next_value
        for t in traj:
            new_state = copy.deepcopy(t.state)
            new_state['target_hue'] = final_hue
            new_state['target_value'] = final_value
            her_samples.append(HERSample(new_state, t.action))
    return her_samples

# ------------------------------ GRPO 更新 ------------------------------
def grpo_update(policy, optimizer, trajectories, group_size, clip_epsilon, entropy_coef,
                max_grad_norm, device, bc_loss=None, bc_coef=0.0):
    """
    使用分组轨迹进行 GRPO 更新。
    trajectories: 列表，每个元素是一条轨迹的字典（包含 states, actions, log_probs, rewards 等）。
    group_size: 每组轨迹数（同一初始状态的轨迹数）。
    """
    all_states = []
    all_actions = []
    all_old_log_probs = []
    step_group_ids = []          # 每个 step 所属的组 id
    step_traj_ids = []            # 每个 step 所属的轨迹 id
    traj_returns = []              # 每条轨迹的总回报
    traj_group_ids = []            # 每条轨迹所属的组 id (轨迹级别)

    num_groups = len(trajectories) // group_size
    for group_idx in range(num_groups):
        group_trajs = trajectories[group_idx * group_size : (group_idx + 1) * group_size]
        for traj_idx, traj in enumerate(group_trajs):
            traj_return = np.sum(traj['rewards'])
            traj_returns.append(traj_return)
            traj_group_ids.append(group_idx)   # 轨迹级别的组标签
            for step in range(len(traj['states'])):
                all_states.append(traj['states'][step])
                all_actions.append(traj['actions'][step])
                all_old_log_probs.append(traj['log_probs'][step])
                step_group_ids.append(group_idx)
                step_traj_ids.append(len(traj_returns) - 1)  # 当前轨迹的索引

    # 合并状态字典
    batched_states = {}
    keys = all_states[0].keys()
    for k in keys:
        if k == 't':
            batched_states[k] = torch.tensor([s[k] for s in all_states], dtype=torch.long)
        elif isinstance(all_states[0][k], np.ndarray):
            arr = np.stack([s[k] for s in all_states])
            if k == 'all_mask':
                batched_states[k] = torch.from_numpy(arr).bool()
            else:
                batched_states[k] = torch.from_numpy(arr).float()
        else:
            batched_states[k] = torch.tensor([s[k] for s in all_states])

    batched_actions = torch.from_numpy(np.stack(all_actions)).float()
    batched_old_log_probs = torch.from_numpy(np.stack(all_old_log_probs)).float().unsqueeze(1)

    # 转换为 tensor
    traj_returns = torch.tensor(traj_returns, dtype=torch.float)
    traj_group_ids = torch.tensor(traj_group_ids, dtype=torch.long)
    step_traj_ids = torch.tensor(step_traj_ids, dtype=torch.long)

    # 计算每个组的轨迹回报均值和标准差
    unique_groups = traj_group_ids.unique()
    group_mean = {}
    group_std = {}
    for g in unique_groups:
        mask = (traj_group_ids == g)
        group_ret = traj_returns[mask]
        group_mean[g.item()] = group_ret.mean()
        group_std[g.item()] = group_ret.std() + 1e-8

    # 构建每个轨迹的优势
    traj_advantages = torch.zeros_like(traj_returns)
    for i, g in enumerate(traj_group_ids):
        traj_advantages[i] = (traj_returns[i] - group_mean[g.item()]) / group_std[g.item()]

    # 将轨迹优势广播到每个 step
    step_advantages = traj_advantages[step_traj_ids].unsqueeze(1).to(device)

    # 将数据移到设备
    for k in batched_states:
        if isinstance(batched_states[k], torch.Tensor):
            batched_states[k] = batched_states[k].to(device)
    batched_actions = batched_actions.to(device)
    batched_old_log_probs = batched_old_log_probs.to(device)

    # 当前策略前向
    values_pred, _, _, log_prob, entropy = policy(batched_states, action=batched_actions.squeeze(1))
    log_prob = log_prob.unsqueeze(1)

    # PPO clip 目标
    ratio = torch.exp(log_prob - batched_old_log_probs)
    surr1 = ratio * step_advantages
    surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * step_advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    # 熵奖励
    entropy_loss = -entropy.mean()

    loss = policy_loss + entropy_coef * entropy_loss
    if bc_loss is not None:
        loss = loss + bc_coef * bc_loss

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
    optimizer.step()

    return loss.item()

# ------------------------------ 评估函数（保持不变） ------------------------------
def evaluate_policy(env, policy, device, num_episodes=5):
    policy.eval()
    hue_dists = []
    value_dists = []
    episode_rewards = []
    for _ in range(num_episodes):
        state = env.reset(mode=1, n=3)
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

# ------------------------------ 距离度量（保持不变） ------------------------------
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

# ------------------------------ 主训练函数（GRPO 微调） ------------------------------
def finetune_grpo(args):
    log_file = args.log_file
    with open(log_file, 'w') as f:
        f.write(f"GRPO Finetuning Log - Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
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

    # 冻结除 actor_head 和 critic_head 以外的所有参数（critic_head 虽不再使用，但保留冻结）
    for name, param in policy.named_parameters():
        if 'actor_head' not in name and 'critic_head' not in name:
            param.requires_grad = False
    trainable_params = [p for p in policy.parameters() if p.requires_grad]
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params)}")

    optimizer = optim.Adam(trainable_params, lr=args.lr)

    # 学习率调度器
    if args.lr_scheduler == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    elif args.lr_scheduler == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.iterations, eta_min=args.lr_min)
    else:
        scheduler = None

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

        # 生成一组随机种子，用于重现初始状态
        state_seeds = [np.random.randint(0, 2**31) for _ in range(args.num_prompts)]

        # 收集分组轨迹
        print("  Collecting grouped trajectories...")
        all_trajectories = collect_grouped_trajectories(
            env, policy, state_seeds, args.group_size, device, deterministic=False
        )

        # 如果需要 BC loss，生成新的专家数据
        if args.use_bc_loss:
            expert_dataset.generate()

        # 从轨迹中提取所有 transition，用于后续可能的 HER 生成（但 HER 需要完整轨迹）
        # 我们将所有轨迹转换为 Transition 列表，便于 HER 生成
        transitions = []
        for traj in all_trajectories:
            for step in range(len(traj['states'])):
                trans = Transition(
                    state=traj['states'][step],
                    action=traj['actions'][step],
                    reward=traj['rewards'][step],
                    next_state=None,
                    done=(step == len(traj['states'])-1),
                    log_prob=traj['log_probs'][step],
                    next_hue=traj['next_hues'][step],
                    next_value=traj['next_values'][step]
                )
                transitions.append(trans)

        # 如果启用HER，生成HER样本（用于 BC loss）
        her_samples = []
        if args.use_her and args.her_in_bc:
            trajectories_list = extract_trajectories_from_transitions(transitions)
            her_samples = generate_her_samples(trajectories_list)
            print(f"    Generated {len(her_samples)} HER samples")

        # GRPO 更新多次（可以多次利用同一批数据进行更新）
        policy_loss_total = 0.0
        for _ in range(args.policy_updates):
            # 计算 BC loss（如果需要）
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

                # 如果启用HER，从HER样本采样并计算BC loss
                if args.use_her and args.her_in_bc and len(her_samples) > 0:
                    her_batch_size = int(args.batch_size * args.her_ratio)
                    if her_batch_size > 0:
                        her_indices = np.random.choice(len(her_samples), her_batch_size, replace=False)
                        her_batch = [her_samples[i] for i in her_indices]
                        her_states, her_actions = collate_her_samples(her_batch)
                        for k in her_states:
                            if isinstance(her_states[k], torch.Tensor):
                                her_states[k] = her_states[k].to(device)
                        her_actions = her_actions.to(device)
                        _, _, _, bc_log_prob_h, _ = policy(her_states, action=her_actions)
                        bc_loss_h = -bc_log_prob_h.mean()
                        bc_loss_val = (bc_loss_e + bc_loss_h) / 2
                    else:
                        bc_loss_val = bc_loss_e
                else:
                    bc_loss_val = bc_loss_e

            # 执行 GRPO 更新
            loss = grpo_update(
                policy, optimizer, all_trajectories, args.group_size,
                args.clip_epsilon, args.entropy_coef, args.max_grad_norm, device,
                bc_loss=bc_loss_val, bc_coef=args.bc_coef if args.use_bc_loss else 0.0
            )
            policy_loss_total += loss

        avg_policy_loss = policy_loss_total / args.policy_updates

        # 评估
        avg_hue_l1, avg_value_l1, avg_eval_reward = evaluate_policy(env, policy, device, num_episodes=5)

        # 平均奖励（来自收集的轨迹）
        all_rewards = [r for traj in all_trajectories for r in traj['rewards']]
        avg_reward = np.mean(all_rewards)

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

        # 学习率调度器步进
        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            print(f"  Current LR: {current_lr:.6f}")

    print("GRPO finetuning finished.")
    with open(log_file, 'a') as f:
        f.write(f"Training finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# ------------------------------ 主程序入口 ------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GRPO Finetuning with Fixed Reward and HER")
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
    parser.add_argument('--pretrained_path', type=str, default='airl_grpo_latest.pth', help='Path to trained AIRL model')

    # 训练参数
    parser.add_argument('--iterations', type=int, default=500)
    parser.add_argument('--num_prompts', type=int, default=32, help='Number of initial states per iteration')
    parser.add_argument('--group_size', type=int, default=8, help='Number of trajectories per prompt for GRPO')
    parser.add_argument('--policy_updates', type=int, default=10, help='Number of GRPO updates per iteration')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for actor')
    parser.add_argument('--clip_epsilon', type=float, default=0.2)
    parser.add_argument('--entropy_coef', type=float, default=0.01)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no_cuda', action='store_true')

    # BC loss 选项
    parser.add_argument('--use_bc_loss', action='store_true', help='Whether to add BC loss during policy update')
    parser.add_argument('--bc_coef', type=float, default=0.1, help='Coefficient for BC loss')
    parser.add_argument('--expert_trajs_per_iter', type=int, default=20, help='Expert trajectories per iteration for BC (if used)')

    # HER 选项
    parser.add_argument('--use_her', action='store_true', help='Enable Hindsight Experience Replay')
    parser.add_argument('--her_in_bc', action='store_true', help='Use HER samples for BC loss')
    parser.add_argument('--her_ratio', type=float, default=0.2, help='Proportion of HER samples in BC batch (if used)')

    # 学习率调度器
    parser.add_argument('--lr_scheduler', type=str, default='step', choices=['none', 'step', 'cosine'],
                        help='Learning rate scheduler type')
    parser.add_argument('--lr_step_size', type=int, default=50, help='StepLR step size')
    parser.add_argument('--lr_gamma', type=float, default=0.5, help='StepLR gamma')
    parser.add_argument('--lr_min', type=float, default=5e-6, help='Minimum LR for cosine annealing')

    # 保存与日志
    parser.add_argument('--best_save_path', type=str, default='grpo_best.pth')
    parser.add_argument('--latest_save_path', type=str, default='grpo_latest.pth')
    parser.add_argument('--log_file', type=str, default='grpo_log.txt')

    args = parser.parse_args()
    finetune_grpo(args)