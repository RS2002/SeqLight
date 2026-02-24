import os
import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from env import LightingEnv
from models import SeqLight
from light_mix import compute_mixed_lighting


# ------------------------------ 动态数据集生成器 ------------------------------
class DynamicExpertDataset:
    """
    每个 epoch 重新生成一批轨迹，并提供 DataLoader
    """
    def __init__(self, env, num_trajectories_per_epoch, min_lights, max_lights,
                 hue_similarity_range=(1, 3)):
        self.env = env
        self.num_trajectories_per_epoch = num_trajectories_per_epoch
        self.min_lights = min_lights
        self.max_lights = max_lights
        self.hue_similarity_range = hue_similarity_range

    def generate_epoch_data(self):
        """生成当前 epoch 的数据，返回按 t 分组的样本列表"""
        samples_by_t = {t: [] for t in range(1, self.max_lights + 1)}
        for _ in range(self.num_trajectories_per_epoch):
            N = np.random.randint(self.min_lights, self.max_lights + 1)
            hue_sim = np.random.randint(self.hue_similarity_range[0],
                                        self.hue_similarity_range[1] + 1)
            # 生成轨迹，每个元素为 (state, action, next_hue, next_value)
            traj = self.env.generate_expert_trajectory(N=N, hue_similarity=hue_sim)
            for step, (state, action, next_hue, next_value) in enumerate(traj):
                t = step + 1
                samples_by_t[t].append((state, action, next_hue, next_value))
        return samples_by_t


def collate_states_with_next(batch):
    """
    合并一批 (state, action, next_hue, next_value) 为批处理字典。
    """
    states = [item[0] for item in batch]
    actions = np.stack([item[1] for item in batch])
    next_hues = np.stack([item[2] for item in batch])
    next_values = np.stack([item[3] for item in batch])

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
    return batched, torch.from_numpy(actions).float(), torch.from_numpy(next_hues).float(), torch.from_numpy(
        next_values).float()


# ------------------------------ 训练函数 ------------------------------
def train_bc(args):
    # 设置日志文件
    log_file = args.log_file
    with open(log_file, 'w') as f:
        f.write(f"BC Training Log - Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Arguments: {args}\n")
        f.write("Epoch\tBC Loss\tAux Loss\tTotal Loss\tBest So Far\tLR\n")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")
    with open(log_file, 'a') as f:
        f.write(f"Device: {device}\n")

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

    dataset = DynamicExpertDataset(
        env,
        num_trajectories_per_epoch=args.trajectories_per_epoch,
        min_lights=args.min_lights,
        max_lights=args.max_lights,
        hue_similarity_range=(1, args.max_hue_similarity)
    )

    model = SeqLight(
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers
    ).to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 学习率调度器
    if args.lr_scheduler == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    elif args.lr_scheduler == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr_min)
    else:
        scheduler = None

    best_loss = float('inf')
    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}: generating new expert data...")
        samples_by_t = dataset.generate_epoch_data()

        epoch_loss_bc = 0.0
        epoch_loss_aux = 0.0
        num_batches = 0
        skipped_batches = 0

        for t in range(1, args.max_lights + 1):
            samples = samples_by_t[t]
            if len(samples) == 0:
                continue

            # 手动构造 mini-batch
            indices = np.random.permutation(len(samples))
            for start in range(0, len(samples), args.batch_size):
                batch_indices = indices[start:start + args.batch_size]
                batch = [samples[i] for i in batch_indices]
                batched_state, batch_actions, batch_next_hue, batch_next_value = collate_states_with_next(batch)

                # 移到设备
                for k in batched_state:
                    if isinstance(batched_state[k], torch.Tensor):
                        batched_state[k] = batched_state[k].to(device)
                batch_actions = batch_actions.to(device)
                batch_next_hue = batch_next_hue.to(device)
                batch_next_value = batch_next_value.to(device)

                # 检查输入是否包含 NaN
                if torch.isnan(batch_actions).any():
                    print("Warning: batch_actions contains NaN, skipping this batch")
                    skipped_batches += 1
                    continue
                nan_in_state = False
                for k, v in batched_state.items():
                    if isinstance(v, torch.Tensor) and torch.isnan(v).any():
                        print(f"Warning: {k} contains NaN, skipping this batch")
                        nan_in_state = True
                        break
                if nan_in_state:
                    skipped_batches += 1
                    continue

                try:
                    # 前向：获得动作的对数概率和判别器输出
                    _, _, _, log_prob, _ = model(batched_state, action=batch_actions)
                    _, pred_hue, pred_value = model.discriminate(batched_state, batch_actions)

                    # BC 损失：负对数似然
                    loss_bc = -log_prob.mean()

                    # 辅助预测损失：交叉熵
                    loss_aux_hue = -(batch_next_hue * torch.log(pred_hue + 1e-8)).sum(dim=-1).mean()
                    loss_aux_value = -(batch_next_value * torch.log(pred_value + 1e-8)).sum(dim=-1).mean()
                    loss_aux = loss_aux_hue + loss_aux_value

                    # 总损失
                    loss = loss_bc + args.aux_weight * loss_aux

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()

                    epoch_loss_bc += loss_bc.item()
                    epoch_loss_aux += loss_aux.item()
                    num_batches += 1

                except ValueError as e:
                    print(f"ValueError in forward pass: {e}")
                    print("Skipping this batch due to numerical error.")
                    # 打印调试信息
                    print("batch_actions stats:", batch_actions.min().item(), batch_actions.max().item(), batch_actions.mean().item())
                    for k, v in batched_state.items():
                        if isinstance(v, torch.Tensor):
                            print(f"{k}: min={v.min().item():.3f}, max={v.max().item():.3f}, mean={v.mean().item():.3f}, any_nan={torch.isnan(v).any()}")
                    skipped_batches += 1
                    continue

        if num_batches == 0:
            print(f"Warning: No valid batches in epoch {epoch+1}, all skipped.")
            continue

        avg_loss_bc = epoch_loss_bc / num_batches
        avg_loss_aux = epoch_loss_aux / num_batches
        avg_total = avg_loss_bc + args.aux_weight * avg_loss_aux

        best_so_far = "Yes" if avg_total < best_loss else "No"

        # 获取当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        print(f"  BC Loss: {avg_loss_bc:.4f} | Aux Loss: {avg_loss_aux:.4f} | Total: {avg_total:.4f} | LR: {current_lr:.6f} | Skipped: {skipped_batches}")

        # 写入日志
        with open(log_file, 'a') as f:
            f.write(f"{epoch + 1}\t{avg_loss_bc:.6f}\t{avg_loss_aux:.6f}\t{avg_total:.6f}\t{best_so_far}\t{current_lr:.6f}\n")

        # 保存最新模型（覆盖）
        torch.save(model.state_dict(), args.latest_save_path)
        print(f"  Latest model saved to {args.latest_save_path}")

        if avg_total < best_loss:
            best_loss = avg_total
            torch.save(model.state_dict(), args.best_save_path)
            print(f"  New best model saved to {args.best_save_path}")

        # 学习率调度器步进
        if scheduler is not None:
            scheduler.step()

    print("BC training finished.")
    with open(log_file, 'a') as f:
        f.write(f"Training finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


# ------------------------------ 主程序 ------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced BC with auxiliary prediction and scheduler")
    # 环境参数
    parser.add_argument('--min_lights', type=int, default=8)
    parser.add_argument('--max_lights', type=int, default=8)
    parser.add_argument('--simple_layout', action='store_true')
    parser.add_argument('--max_n_peaks', type=int, default=3)
    parser.add_argument('--max_hue_similarity', type=int, default=3)

    # 数据生成
    parser.add_argument('--trajectories_per_epoch', type=int, default=256,
                        help='Number of expert trajectories generated each epoch')

    # 模型参数
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=3)

    # 训练超参数
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--aux_weight', type=float, default=0.1,
                        help='Weight for auxiliary prediction loss')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no_cuda', action='store_true')

    # 学习率调度器
    parser.add_argument('--lr_scheduler', type=str, default='step', choices=['none', 'step', 'cosine'],
                        help='Learning rate scheduler type')
    parser.add_argument('--lr_step_size', type=int, default=50, help='StepLR step size')
    parser.add_argument('--lr_gamma', type=float, default=0.5, help='StepLR gamma')
    parser.add_argument('--lr_min', type=float, default=1e-5, help='Minimum LR for cosine annealing')

    # 保存路径
    parser.add_argument('--best_save_path', type=str, default='bc_best.pth')
    parser.add_argument('--latest_save_path', type=str, default='bc_latest.pth')
    # 日志文件
    parser.add_argument('--log_file', type=str, default='bc_log.txt', help='Path to save training log')

    args = parser.parse_args()
    train_bc(args)