from transformers import BartModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from peft import get_peft_model, LoraConfig
import math
from torch.distributions import VonMises, Beta


class MLP(nn.Module):
    def __init__(self, layer_sizes=[64, 64, 64, 1], arl=False, dropout=0.0):
        super().__init__()
        self.arl = arl
        if arl:
            self.attention = nn.Sequential(
                nn.Linear(layer_sizes[0], layer_sizes[0]),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(layer_sizes[0], layer_sizes[0])
            )

        self.layer_sizes = layer_sizes
        if len(layer_sizes) < 2:
            raise ValueError()
        self.layers = nn.ModuleList()
        self.act = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.dropout = nn.Dropout(dropout)
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))

    def forward(self, x):
        if self.arl:
            x = x * self.attention(x)
        for layer in self.layers[:-1]:
            x = self.dropout(self.act(layer(x)))
        x = self.layers[-1](x)
        return x



class BART(nn.Module):
    def __init__(self, bartconfig, class_num=100):
        super().__init__()
        d_model = bartconfig.d_model
        self.decoder_emb = nn.Embedding(class_num, d_model)
        self.bart = BartModel(bartconfig)

    def forward(self, x_encoder, x_decoder, attn_mask_encoder=None, attn_mask_decoder=None):
        emb_encoder = x_encoder
        emb_decoder = self.decoder_emb(x_decoder)
        y = self.bart(inputs_embeds=emb_encoder, decoder_inputs_embeds=emb_decoder,
                      attention_mask=attn_mask_encoder, decoder_attention_mask=attn_mask_decoder,
                      output_hidden_states=False)
        y = y.last_hidden_state
        return y

    def encode(self, x_encoder, attn_mask_encoder=None):
        emb_encoder = x_encoder
        y = self.bart.encoder(inputs_embeds=emb_encoder, attention_mask=attn_mask_encoder, output_hidden_states=False)
        y = y.last_hidden_state
        return y


class ML_BART(nn.Module):
    def __init__(self, bartconfig, class_num=[180, 100], pretrain=False, music_dim=512):
        super().__init__()
        d_model = bartconfig.d_model
        # self.decoder_emb = nn.ModuleList([
        #     MLP(int(class_num[0]), d_model // 4),
        #     MLP(int(class_num[1]), d_model // 4)
        # ])
        self.decoder_emb2 = nn.ModuleList([
            MLP([int(class_num[0]), d_model // 4]),
            MLP([int(class_num[1]), d_model // 4])
        ])
        self.decoder = MLP([music_dim, d_model // 2])

        self.sos = nn.Parameter(torch.randn([1,1,d_model]),requires_grad=True)

        self.bart = BartModel(bartconfig)
        self.pretrain = pretrain

        self.encoder = MLP([music_dim, d_model])
        self.lora_config = LoraConfig(
            r=4,
            lora_alpha=16,
            lora_dropout=0.1
        )

    def forward(self, x_encoder, x_decoder, attn_mask_encoder=None, attn_mask_decoder=None):
        # emb_encoder = x_encoder
        emb_encoder = self.encoder(x_encoder)

        if self.pretrain:
            # emb_decoder = x_decoder
            emb_decoder = self.encoder(x_decoder)
        else:
            emb_decoder = torch.concatenate(
                [self.decoder_emb2[0](x_decoder[0]), self.decoder_emb2[1](x_decoder[1]), self.decoder(x_encoder)],
                dim=-1)
            sos = self.sos.repeat(x_encoder.shape[0],1,1)
            emb_decoder = torch.concat([sos,emb_decoder[:,1:,:]],dim=1)

        y = self.bart(inputs_embeds=emb_encoder, decoder_inputs_embeds=emb_decoder,
                      attention_mask=attn_mask_encoder, decoder_attention_mask=attn_mask_decoder,
                      output_hidden_states=False)
        y = y.last_hidden_state
        return y

    def encode(self, x_encoder, attn_mask_encoder=None):
        # emb_encoder = x_encoder
        emb_encoder = self.encoder(x_encoder)

        y = self.bart.encoder(inputs_embeds=emb_encoder, attention_mask=attn_mask_encoder, output_hidden_states=False)
        y = y.last_hidden_state
        return y

    def reset_decoder(self):
        for name, param in self.bart.decoder.named_parameters():
            if param.dim() >= 2:
                init.xavier_uniform_(param)
            elif param.dim() == 1:
                init.zeros_(param)


class ML_Classifier(nn.Module):
    def __init__(self, hidden_dim=512, class_num=[360, 100]):
        super().__init__()
        self.classifier = nn.ModuleList([
            MLP([hidden_dim, hidden_dim, class_num[0]]),
            MLP([hidden_dim, hidden_dim, class_num[1]])
        ])
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        h = self.classifier[0](x)
        v = self.classifier[1](x)
        return self.softmax(h), self.softmax(v)


class SelfAttention(nn.Module):
    def __init__(self, input_dim, da, r):
        super().__init__()
        self.ws1 = nn.Linear(input_dim, da, bias=False)
        self.ws2 = nn.Linear(da, r, bias=False)

    def forward(self, h):
        attn_mat = F.softmax(self.ws2(torch.tanh(self.ws1(h))), dim=1)
        attn_mat = attn_mat.permute(0, 2, 1)
        return attn_mat


class Sequence_Classifier(nn.Module):
    def __init__(self, class_num=1, hs=512, da=512, r=8):
        super().__init__()
        self.attention = SelfAttention(hs, da, r)
        self.classifier = MLP([hs * r, (hs * r + class_num) // 2, class_num])

    def forward(self, x):
        attn_mat = self.attention(x)
        m = torch.bmm(attn_mat, x)
        flatten = m.view(m.size()[0], -1)
        res = self.classifier(flatten)
        return res


class Token_Predictor(nn.Module):
    def __init__(self, hidden_dim=512, class_num=1):
        super().__init__()
        self.classifier = MLP([hidden_dim, (hidden_dim + class_num) // 2, class_num])

    def forward(self, x):
        x = self.classifier(x)
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
        self.actor_head =  MLP([d_model,d_model,4]) # hue, value

        # Critic head
        self.critic_head =  MLP([d_model,d_model,1])

        # Reward head
        self.reward_head =  MLP([d_model*2,d_model,1])

        # Mixture head
        self.hue_head = MLP([d_model*2, d_model, 360])
        self.value_head = MLP([d_model*2, d_model, 100])
        self.softmax = nn.Softmax(dim=-1)

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

    def forward(self, batch_dict, action=None, deterministic=False, t_h=1.0, t_v=1.0, output_dist=True):
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

        v_value = self.critic_head(last_hidden).squeeze(-1)
        raw_action = self.actor_head(last_hidden)
        # 分割四个原始输出
        mu_hue_raw, kappa_hue_raw, alpha_raw, beta_raw = torch.split(raw_action, 1, dim=-1)
        # 应用激活函数确保参数有效
        mu_hue = torch.tanh(mu_hue_raw) * math.pi  # 映射到 [-π, π]
        kappa_hue = F.softplus(kappa_hue_raw) + 1e-6  # 确保 >0
        alpha_val = F.softplus(alpha_raw) + 1e-6  # 确保 >0
        beta_val = F.softplus(beta_raw) + 1e-6  # 确保 >0

        # print(f"Alpha: {alpha_val.mean().item():.4f}, Beta: {beta_val.mean().item():.4f}")

        hue_dist = VonMises(mu_hue.squeeze(-1), kappa_hue.squeeze(-1) / t_h)
        if t_v == 1:
            val_dist = Beta(alpha_val.squeeze(-1), beta_val.squeeze(-1))
        else:
            eps = 1e-6
            alpha_temp = (alpha_val - 1) / t_v + 1
            beta_temp = (beta_val - 1) / t_v + 1
            alpha_temp = torch.clamp(alpha_temp, min=eps)
            beta_temp = torch.clamp(beta_temp, min=eps)
            val_dist = Beta(alpha_temp.squeeze(-1), beta_temp.squeeze(-1))

        if output_dist:
            return hue_dist,val_dist


        if action is None:
            if deterministic:
                # 确定性：取均值（注意hue的均值可能在圆上，直接使用）
                hue_action = hue_dist.mean
                val_action = val_dist.mean
            else:
                hue_action = hue_dist.sample()   # 采样
                val_action = val_dist.sample()   # 采样
        else:
            hue_action = action[...,0] * (2 * math.pi) - math.pi
            val_action = action[...,1]

        # 计算对数概率（分别计算后求和，假设独立）
        log_prob_hue = hue_dist.log_prob(hue_action)
        log_prob_val = val_dist.log_prob(val_action)
        log_prob = log_prob_hue + log_prob_val

        # 手动计算von Mises分布的熵（PyTorch内置未实现）
        kappa = hue_dist.concentration
        kappa = torch.clamp(kappa, max=100)
        i0 = torch.special.i0(kappa)
        i1 = torch.special.i1(kappa)
        ratio = i1 / i0
        entropy_hue = torch.log(2 * math.pi * i0) - kappa * ratio
        # Beta分布的熵直接可用
        entropy_val = val_dist.entropy()
        entropy = entropy_hue + entropy_val

        return v_value, ((hue_action + math.pi) % (2 * math.pi)) / (2 * math.pi), val_action, log_prob, entropy

    def discriminate(self, batch_dict, action):

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

        act_emb = self.action_encoder(action)
        combined = torch.cat([last_hidden, act_emb], dim=1)

        reward = self.reward_head(combined).squeeze(-1)
        hue_dist = self.softmax(self.hue_head(combined))
        value_dist = self.softmax(self.value_head(combined))

        return reward, hue_dist, value_dist