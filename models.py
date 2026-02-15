import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, layer_sizes = [64,64,64,1], arl = False, dropout = 0.0, bias = True):
        super().__init__()
        self.arl = arl
        if self.arl:
            self.attention = nn.Sequential(
                nn.Linear(layer_sizes[0],layer_sizes[0]),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(layer_sizes[0],layer_sizes[0])
            )

        self.layer_sizes = layer_sizes
        if len(layer_sizes) < 2:
            raise ValueError()
        self.layers = nn.ModuleList()
        self.act = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.dropout = nn.Dropout(dropout)
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1], bias = bias))

    def forward(self, x):
        if self.arl:
            x = x * self.attention(x)
        for layer in self.layers[:-1]:
            x = self.dropout(self.act(layer(x)))
        x = self.layers[-1](x)
        return x


class GlobalPositionEncoder(nn.Module):
    """
    Set Transformer 风格的全局位置汇总
    改进：将灯光数量编码拼接到每个位置的嵌入中，使数量信息参与自注意力计算
    """
    def __init__(self, pos_enc, d_model=64, nhead=4):
        super().__init__()
        self.pos_enc = pos_enc
        self.d_model = d_model

        # 数量编码器：标量数量 (0~max_lights) → d_model 维向量
        self.num_encoder = MLP([1,d_model,d_model])

        # 拼接后维度为 2*d_model，投影回 d_model
        self.input_proj = nn.Linear(d_model * 2, d_model)

        # Transformer Encoder（与原版一致）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.0,
            activation='gelu',
            norm_first=True,
            batch_first=True
        )
        self.transformer_enc = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # 池化：平均池化得到全局向量
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, positions, mask=None):
        """
        positions: (B, N, 2)  — 所有灯光的位置（已填充至 N = max_lights）
        mask:      (B, N)     — bool, True=有效灯光
        """
        B, N, _ = positions.shape
        device = positions.device

        # 1. 位置编码 (B, N, d_model)
        x = self.pos_enc(positions)

        # 2. 计算有效灯光数量 (B, 1)
        if mask is not None:
            num_valid = mask.sum(dim=1, keepdim=True).float()
        else:
            num_valid = torch.full((B, 1), N, dtype=torch.float, device=device)

        # 3. 数量编码并扩展到每个位置 (B, N, d_model)
        num_feat = self.num_encoder(num_valid)                 # (B, d_model)
        num_feat = num_feat.unsqueeze(1).expand(-1, N, -1)    # (B, N, d_model)

        # 4. 拼接位置编码与数量编码 (B, N, 2*d_model)
        x = torch.cat([x, num_feat], dim=-1)

        # 5. 投影回 d_model (B, N, d_model)
        x = self.input_proj(x)

        # 6. Transformer 编码（支持 mask）
        if mask is not None:
            mask = mask.bool()
            key_padding_mask = ~mask
            x = self.transformer_enc(x, src_key_padding_mask=key_padding_mask)
        else:
            x = self.transformer_enc(x)

        # 7. 全局平均池化 (B, d_model)
        x = x.transpose(1, 2)          # (B, d_model, N)
        x = self.pool(x).squeeze(-1)   # (B, d_model)

        return x


class SeqLight(nn.Module):
    def __init__(self, d_model=64, nhead=4, num_layers=3):
        super().__init__()
        self.d_model = d_model

        self.pos_encoder = MLP([2,d_model,d_model])
        self.action_encoder = MLP([2,d_model,d_model])
        self.hue_encoder = MLP([360,d_model*2,d_model])
        self.value_encoder = MLP([100,d_model,d_model])
        self.global_pos_encoder = GlobalPositionEncoder(self.pos_encoder, d_model, nhead)

        self.dummy_act = nn.Parameter(torch.randn(d_model))

        # 新增：将四个 embedding concat 后投影回 d_model
        self.token_proj = nn.Sequential(
            nn.Linear(d_model * 4, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model)
        )

        # Learnable positional embedding (支持最长32个token)
        self.pos_embedding = nn.Parameter(torch.randn(1, 32, d_model))

        # Transformer (Decoder-only)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.0,
            activation='gelu',
            norm_first=True,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Actor head
        self.actor_head =  MLP([d_model,d_model,2]) # hue_raw, value_raw

        # Twin Critic heads
        self.critic_head1 =  MLP([d_model*2,d_model,1])
        self.critic_head2 = MLP([d_model*2,d_model,1])

    def _build_sequence(self, batch_dict):
        """
        适配 fixed-size padded 历史版本
        历史是 (B, max_lights, ...) 但只使用前 (t-1) 步
        """
        B = batch_dict['target_hue'].shape[0]
        t = batch_dict['t']
        # 将张量 t 转换为标量（批内所有样本 t 相同）
        if isinstance(t, torch.Tensor):
            t = t[0].item()
        # device = next(self.parameters()).device
        # max_hist = batch_dict['history_positions'].shape[1]  # max_lights

        # ----- 1. 目标 token（固定1个）-----
        global_pos = self.global_pos_encoder(
            batch_dict['all_positions'],
            batch_dict.get('all_mask', None)
        )  # (B, d_model)
        target_hue_emb = self.hue_encoder(batch_dict['target_hue'])  # (B, d_model)
        target_value_emb = self.value_encoder(batch_dict['target_value'])  # (B, d_model)
        dummy_act_emb = self.dummy_act.unsqueeze(0).expand(B, -1)  # (B, d_model)
        target_concat = torch.cat([global_pos, dummy_act_emb, target_hue_emb, target_value_emb], dim=-1)
        target_token = self.token_proj(target_concat).unsqueeze(1)  # (B, 1, d_model)

        # ----- 2. 当前 token（固定1个）-----
        cur_pos_emb = self.pos_encoder(batch_dict['current_position'].squeeze(1))  # (B, d_model)
        cur_hue_emb = self.hue_encoder(batch_dict['current_mixed_hue'].squeeze(1))
        cur_val_emb = self.value_encoder(batch_dict['current_mixed_value'].squeeze(1))
        cur_concat = torch.cat([cur_pos_emb, dummy_act_emb, cur_hue_emb, cur_val_emb], dim=-1)
        current_token = self.token_proj(cur_concat).unsqueeze(1)  # (B, 1, d_model)

        # ----- 3. 历史 token（取前 t-1 步，有效长度） -----
        if t > 1:
            hist_len = t - 1
            # 截取前 hist_len 步（因为是按决策顺序填充的）
            hist_pos = batch_dict['history_positions'][:, :hist_len, :]  # (B, hist_len, 2)
            hist_act = batch_dict['history_actions'][:, :hist_len, :]  # (B, hist_len, 2)
            hist_hue = batch_dict['history_mixed_hue'][:, :hist_len, :]  # (B, hist_len, 360)
            hist_val = batch_dict['history_mixed_value'][:, :hist_len, :]  # (B, hist_len, 100)

            # 展平后批量编码
            B_flat = B * hist_len
            hist_pos_flat = hist_pos.reshape(B_flat, 2)
            hist_act_flat = hist_act.reshape(B_flat, 2)
            hist_hue_flat = hist_hue.reshape(B_flat, 360)
            hist_val_flat = hist_val.reshape(B_flat, 100)

            pos_emb_flat = self.pos_encoder(hist_pos_flat)  # (B*hist_len, d_model)
            act_emb_flat = self.action_encoder(hist_act_flat)  # (B*hist_len, d_model)
            hue_emb_flat = self.hue_encoder(hist_hue_flat)
            val_emb_flat = self.value_encoder(hist_val_flat)

            # 恢复形状
            pos_emb = pos_emb_flat.view(B, hist_len, self.d_model)
            act_emb = act_emb_flat.view(B, hist_len, self.d_model)
            hue_emb = hue_emb_flat.view(B, hist_len, self.d_model)
            val_emb = val_emb_flat.view(B, hist_len, self.d_model)

            hist_concat = torch.cat([pos_emb, act_emb, hue_emb, val_emb], dim=-1)  # (B, hist_len, d_model*4)
            hist_tokens = self.token_proj(hist_concat)  # (B, hist_len, d_model)

            # 完整序列：目标 + 历史 + 当前
            seq = torch.cat([target_token, hist_tokens, current_token], dim=1)
        else:
            # 无历史，仅目标 + 当前
            seq = torch.cat([target_token, current_token], dim=1)  # (B, 2, d_model)

        return seq

    def forward(self, batch_dict, action=None):
        seq = self._build_sequence(batch_dict)
        seq_len = seq.shape[1]

        # positional embedding
        pos_emb = self.pos_embedding[:, :seq_len, :]
        seq = seq + pos_emb

        # causal mask
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float('-inf'), device=seq.device),
            diagonal=1
        )

        hidden = self.transformer(seq, mask=causal_mask)
        last_hidden = hidden[:, -1]

        if action is None:
            # Actor
            raw = self.actor_head(last_hidden)
            return torch.sigmoid(raw)
        else:
            # Critic
            act_emb = self.action_encoder(action)
            combined = torch.cat([last_hidden, act_emb], dim=1)
            q1 = self.critic_head1(combined)
            q2 = self.critic_head2(combined)
            return q1, q2


   # ============================= 新增：并行轨迹推理 =============================
    def forward_trajectory(self, batch_dict):
        """
        直接并行处理多条完整轨迹，输出每个时间步的动作及对应的双Q值。

        参数 batch_dict 必须包含:
            target_hue:      (B, 360)
            target_value:    (B, 100)
            all_positions:   (B, N, 2)
            all_mask:        (B, N)          # bool，有效位置掩码
            step_positions:  (B, T, 2)       # 每个时间步被调节的灯光位置
            step_mixed_hue:  (B, T, 360)     # 每个时间步混合后的色相直方图
            step_mixed_value:(B, T, 100)       # 每个时间步混合后的平均亮度

        返回:
            dict {
                'actions':      (B, T, 2)    归一化到 [-1,1] 的动作（可直接输入环境）
                'actions_raw': (B, T, 2)    未归一化的 actor 原始输出
                'q1':          (B, T)       第一个 critic 的 Q 值
                'q2':          (B, T)       第二个 critic 的 Q 值
            }
        """
        # ----- 1. 构建目标 token -----
        B = batch_dict['target_hue'].shape[0]
        device = next(self.parameters()).device

        # 全局位置编码（所有灯的固定布局）
        global_pos = self.global_pos_encoder(
            batch_dict['all_positions'],
            batch_dict.get('all_mask', None)
        )  # (B, d_model)

        target_hue_emb = self.hue_encoder(batch_dict['target_hue'])   # (B, d_model)
        target_value_emb = self.value_encoder(batch_dict['target_value']) # (B, d_model)
        dummy_act_emb = self.dummy_act.unsqueeze(0).expand(B, -1)  # (B, d_model)
        target_concat = torch.cat([global_pos, dummy_act_emb, target_hue_emb, target_value_emb], dim=-1)  # (B, d_model*4)
        target_token = self.token_proj(target_concat).unsqueeze(1)   # (B, 1, d_model)

        # ----- 2. 构建每个时间步的 token -----
        T = batch_dict['step_positions'].size(1)

        # 将 (B, T, ...) 合并为 (B*T, ...) 一次性编码
        pos_flat = batch_dict['step_positions'].view(B * T, 2)               # (B*T, 2)
        hue_flat = batch_dict['step_mixed_hue'].view(B * T, 360)            # (B*T, 360)
        val_flat = batch_dict['step_mixed_value'].view(B * T, 100)            # (B*T, 100)

        pos_emb = self.pos_encoder(pos_flat)            # (B*T, d_model)
        hue_emb = self.hue_encoder(hue_flat)            # (B*T, d_model)
        val_emb = self.value_encoder(val_flat)          # (B*T, d_model)
        act_emb_flat = self.dummy_act.unsqueeze(0).expand(B * T, -1)  # (B*T, d_model)

        step_concat = torch.cat([pos_emb, act_emb_flat, hue_emb, val_emb], dim=-1)   # (B*T, d_model*4)
        step_token = self.token_proj(step_concat)                     # (B*T, d_model)
        step_tokens = step_token.view(B, T, self.d_model)             # (B, T, d_model)

        # ----- 3. 组合完整序列 -----
        seq = torch.cat([target_token, step_tokens], dim=1)  # (B, 1+T, d_model)

        # ----- 4. 添加位置编码与因果掩码 -----
        seq_len = seq.size(1)
        pos_emb = self.pos_embedding[:, :seq_len, :]        # (1, seq_len, d_model)
        seq = seq + pos_emb

        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float('-inf'), device=device),
            diagonal=1
        )
        hidden = self.transformer(seq, mask=causal_mask)   # (B, seq_len, d_model)

        # 取出每个时间步的隐状态（跳过目标token）
        step_hidden = hidden[:, 1:, :]  # (B, T, d_model)

        # ----- 5. 预测动作及 Q 值 -----
        # 原始 actor 输出（未归一化）
        actions_raw = self.actor_head(step_hidden)        # (B, T, 2)
        # 将动作归一化到 (0, 1)（与环境动作空间一致）
        actions = torch.sigmoid(actions_raw)                 # (B, T, 2)

        # 编码归一化后的动作，用于 critic
        act_flat = actions.view(B * T, 2)                # (B*T, 2)
        act_emb = self.action_encoder(act_flat)          # (B*T, d_model)
        act_emb = act_emb.view(B, T, self.d_model)       # (B, T, d_model)

        # 拼接隐状态与动作编码
        combined = torch.cat([step_hidden, act_emb], dim=-1)  # (B, T, d_model*2)
        combined_flat = combined.view(B * T, self.d_model * 2)

        # 双 Q 值
        q1_flat = self.critic_head1(combined_flat)       # (B*T, 1)
        q2_flat = self.critic_head2(combined_flat)       # (B*T, 1)

        q1 = q1_flat.view(B, T)                          # (B, T)
        q2 = q2_flat.view(B, T)                          # (B, T)

        return {
            'actions': actions,          # 归一化动作，可直接用于环境
            'actions_raw': actions_raw,  # 原始 actor logits
            'q1': q1,
            'q2': q2
        }





# ============================= 测试代码 =============================
if __name__ == "__main__":
    torch.manual_seed(42)

    # 1. 创建模型，并设置为训练模式（确保梯度计算）
    model = SeqLight(d_model=64, nhead=4, num_layers=3).cuda()
    model.train()   # 显式启用 dropout / 梯度计算
    print("[Info] Model created and set to train mode.")

    # 参数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")

    B = 4
    t = 3  # 当前是第 3 步

    batch_dict = {
        'target_hue': torch.randn(B, 360).cuda(),
        'target_value': torch.randn(B, 100).cuda(),
        'all_positions': torch.rand(B, 12, 2).cuda(),
        'all_mask': torch.ones(B, 12, dtype=torch.bool).cuda(),
        'history_positions': torch.rand(B, t - 1, 2).cuda(),
        'history_actions': (torch.rand(B, t - 1, 2) * 2 - 1).cuda(),
        'history_mixed_hue': torch.randn(B, t - 1, 360).cuda(),
        'history_mixed_value': torch.randn(B, t - 1, 100).cuda(),
        'current_position': torch.rand(B, 1, 2).cuda(),
        'current_mixed_hue': torch.randn(B, 1, 360).cuda(),
        'current_mixed_value': torch.randn(B, 1, 100).cuda(),
        't': t,
    }

    # 测试 Actor
    action = model(batch_dict)
    print("Actor output shape:", action.shape)

    # 测试 Critic
    fake_action = torch.rand(B, 2).cuda() * 2 - 1
    q1, q2 = model(batch_dict, action=fake_action)
    print(q1)
    print("Critic Q1 shape:", q1.shape)
    print("Critic Q2 shape:", q2.shape)


    # 2. 构造一个 batch 的完整轨迹数据
    B = 4          # batch size
    T = 5          # 轨迹长度
    N = 12         # 舞台上的灯光总数（固定）

    batch_dict = {
        'target_hue': torch.randn(B, 360).cuda(),
        'target_value': torch.randn(B, 100).cuda(),
        'all_positions': torch.rand(B, N, 2).cuda(),
        'all_mask': torch.ones(B, N, dtype=torch.bool).cuda(),
        'step_positions': torch.rand(B, T, 2).cuda(),
        'step_mixed_hue': torch.randn(B, T, 360).cuda(),
        'step_mixed_value': torch.randn(B, T, 100).cuda(),
    }

    # ------------------- 前向测试（不计算梯度，仅验证形状） -------------------
    with torch.no_grad():
        out_no_grad = model.forward_trajectory(batch_dict)
        print("\n[Forward without grad]")
        print(f"actions shape:      {out_no_grad['actions'].shape}")
        print(f"actions_raw shape:  {out_no_grad['actions_raw'].shape}")
        print(f"q1 shape:           {out_no_grad['q1'].shape}")
        print(f"q2 shape:           {out_no_grad['q2'].shape}")
        print(f"actions range:      [{out_no_grad['actions'].min():.3f}, {out_no_grad['actions'].max():.3f}]")

    # ------------------- 反向传播测试（必须启用梯度） -------------------
    out = model.forward_trajectory(batch_dict)   # 此时所有输出张量均 requires_grad=True
    loss = (out['q1'].mean() + out['q2'].mean()) * 0.1 + out['actions'].pow(2).mean()
    loss.backward()
    print("\n[Backward]")
    print(f"Loss value: {loss.item():.4f}")
    # 检查第一个参数的梯度是否存在
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"  {name:30s} grad norm: {param.grad.norm().item():.4e}")
            break

    # ------------------- 与原始 forward 的一致性验证 -------------------
    print("\n[Consistency Check with original forward]")
    B1 = 2
    batch_single = {
        'target_hue': torch.randn(B1, 360).cuda(),
        'target_value': torch.randn(B1, 100).cuda(),
        'all_positions': torch.rand(B1, N, 2).cuda(),
        'all_mask': torch.ones(B1, N, dtype=torch.bool).cuda(),
        'step_positions': torch.rand(B1, 1, 2).cuda(),
        'step_mixed_hue': torch.randn(B1, 1, 360).cuda(),
        'step_mixed_value': torch.randn(B1, 1, 100).cuda(),
    }
    # 轨迹方式（不跟踪梯度以便比较）
    with torch.no_grad():
        out_traj = model.forward_trajectory(batch_single)
        action_traj = out_traj['actions']
        q1_traj = out_traj['q1']

        # 原始单步方式
        batch_original = {
            'target_hue': batch_single['target_hue'],
            'target_value': batch_single['target_value'],
            'all_positions': batch_single['all_positions'],
            'all_mask': batch_single['all_mask'],
            'history_positions': torch.empty(B1, 0, 2).cuda(),
            'history_actions': torch.empty(B1, 0, 2).cuda(),
            'history_mixed_hue': torch.empty(B1, 0, 360).cuda(),
            'history_mixed_value': torch.empty(B1, 0, 100).cuda(),
            'current_position': batch_single['step_positions'],
            'current_mixed_hue': batch_single['step_mixed_hue'],
            'current_mixed_value': batch_single['step_mixed_value'],
            't': 1,
        }
        action_original = model(batch_original)
        q1_original, _ = model(batch_original, action=action_original)

    print(f"Action difference:  {(action_traj.squeeze(1) - action_original).abs().max().item():.6f}")
    print(f"Q1 difference:      {(q1_traj.squeeze(1) - q1_original.squeeze(1)).abs().max().item():.6f}")
    print("[Consistency] Should be near zero (difference < 1e-5).")