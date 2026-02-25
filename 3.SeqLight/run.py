import os
import pickle
from model import ML_BART, ML_Classifier, SeqLight
from transformers import BartConfig
import argparse
import tqdm
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
from dataset import load_data
from peft import get_peft_model, LoraConfig
from env import LightingEnv
from light_mix import compute_mixed_lighting

pad = -1000


def get_args():
    parser = argparse.ArgumentParser(description='')

    parser.add_argument("--music_dim", type=int, default=128)
    parser.add_argument("--light_dim", type=int, nargs='+', default=[180, 100])
    parser.add_argument('--gap', type=int, default=0)

    parser.add_argument("--t", type=float, nargs='+', default=[0.1, 1.0])
    parser.add_argument("--p", type=float, nargs='+', default=[0.5, 0.9])
    parser.add_argument("--h_range", type=int, default=50)
    parser.add_argument("--v_range", type=int, default=30)

    parser.add_argument('--layers', type=int, default=8)
    parser.add_argument('--max_len', type=int, default=1024)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--hs', type=int, default=1024)
    parser.add_argument('--ffn_dims', type=int, default=2048)

    parser.add_argument("--cpu", action="store_true", default=False)
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=[1])
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--converge_epoch', type=int, default=30)

    parser.add_argument('--data_path', type=str, default="./data")
    parser.add_argument('--train_prop', type=float, default=0.9)

    parser.add_argument('--bart_path', type=str,
                        default="./model/bart_finetune.pth")
    parser.add_argument('--head_path', type=str,
                        default="./model/head_finetune.pth")

    parser.add_argument("--shuffle", action="store_true", default=False)
    parser.add_argument('--random_seed', type=int, default=42)

    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--rl_model_path', type=str,
                        default="./model/grpo_latest.pth")
    parser.add_argument('--num_lights', type=int, default=8)

    args = parser.parse_args()
    return args


def nucleus(probs, p):
    probs /= (sum(probs) + 1e-5)
    sorted_probs = np.sort(probs)[::-1]
    sorted_index = np.argsort(probs)[::-1]
    cusum_sorted_probs = np.cumsum(sorted_probs)
    p = min(p, 1.0)
    after_threshold = cusum_sorted_probs >= p
    if sum(after_threshold) > 0:
        last_index = np.where(after_threshold)[0][0] + 1
        candi_index = sorted_index[:last_index]
    else:
        candi_index = sorted_index[0:1]
    candi_probs = [probs[i] for i in candi_index]
    candi_probs /= sum(candi_probs)
    word = np.random.choice(candi_index, size=1, p=candi_probs)[0]
    return word

def find_best_scale(final_hues, final_values, target_hue, target_value, env,
                    scale_min=0.5, scale_max=2.0, num_steps=100, criterion='mean'):
    best_scale = 1.0
    best_value = float('inf')
    scales = np.linspace(scale_min, scale_max, num_steps)

    if criterion == 'mean':
        bin_centers = np.arange(100) * 0.01 + 0.005
        target_mean = np.sum(target_value * bin_centers)

    for s in scales:
        scaled_values = np.clip(final_values * s, 0, 1)
        result = compute_mixed_lighting(
            positions=env.positions_raw,
            hues=final_hues,
            values=scaled_values,
            grid_size=(env.grid_h, env.grid_w),
            decay_model=env.decay_model,
            sigma=env.sigma,
            value_power=env.value_power,
            eps=env.eps
        )
        if criterion == 'mean':
            scaled_mean = np.sum(result['value_histogram'] * bin_centers)
            dist = abs(scaled_mean - target_mean)
        else:
            raise ValueError(f"Unknown criterion: {criterion}")

        if dist < best_value:
            best_value = dist
            best_scale = s

    return best_scale, best_value


def _normalize_distribution(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.clip(x, 0.0, None)
    s = np.sum(x, axis=-1, keepdims=True)
    return np.divide(x, s + eps, out=np.zeros_like(x, dtype=np.float32), where=s > 0)


def _downsample_distribution_to_bins(x: np.ndarray, out_bins: int) -> np.ndarray:
    in_bins = x.shape[0]
    in_edges = np.linspace(0, 1, in_bins + 1)
    out_edges = np.linspace(0, 1, out_bins + 1)

    cdf = np.concatenate(([0.0], np.cumsum(x, dtype=np.float32)))
    cdf_rs = np.interp(out_edges, in_edges, cdf).astype(np.float32)
    y = np.diff(cdf_rs)
    return _normalize_distribution(y)


def iteration(data_loader, device, bart, model, env, policy, light_num, t, p, h_range=None, v_range=None):
    output_light_hue = []
    output_light_value = []

    output_hue = []
    output_value = []

    # pbar = tqdm.tqdm(data_loader, disable=False)
    for music, _, f_name in data_loader:
        music = music.float().to(device)
        light = [torch.zeros([music.shape[0], music.shape[1], 180]).to(device),
                 torch.zeros([music.shape[0], music.shape[1], 100]).to(device)]

        non_pad = (music != pad).to(device)
        attn_mask = non_pad[..., 0].float()
        attn_mask_light = torch.zeros_like(attn_mask)
        attn_mask_light[:, 1:] = attn_mask[:, :-1]
        attn_mask_light[:, 0] = attn_mask[:, 0]

        batch_size, seq_len, _ = music.shape
        result = [torch.zeros([batch_size, seq_len, 180]).to(device),
                  torch.zeros([batch_size, seq_len, 100]).to(device)]
        result_light = [torch.zeros([batch_size, seq_len, light_num]), torch.zeros([batch_size, seq_len, light_num])]
        last_state = [[None for _ in range(light_num)] for _ in range(batch_size)]

        # for i in range(seq_len):
        for i in tqdm.tqdm(range(seq_len)):
            h, v = model(bart(music, light, attn_mask, attn_mask_light))

            for w in range(batch_size):
                h_temp, v_temp = h[w, i], v[w, i]
                state = env.reset(goal_hue=h_temp.detach().cpu().numpy(), goal_value=v_temp.detach().cpu().numpy())
                done = False
                j = -1
                while not done:
                    j += 1
                    state_tensor = {}
                    for k, v in state.items():
                        if isinstance(v, np.ndarray):
                            state_tensor[k] = torch.from_numpy(v).unsqueeze(0).to(device)
                        else:
                            state_tensor[k] = torch.tensor([v]).to(device)
                    hue_dist, val_dist = policy(state_tensor, t_h=t[0], t_v=t[1], output_dist=True)

                    hue = torch.linspace(0, 2 * torch.pi, 180).to(device)
                    val = torch.linspace(0.0001, 0.9999, 100).to(device)
                    h_temp = hue_dist.log_prob(hue.unsqueeze(0)).squeeze(0)
                    h_temp = torch.exp(h_temp)
                    v_temp = val_dist.log_prob(val.unsqueeze(0)).squeeze(0)
                    v_temp = torch.exp(v_temp)

                    if h_range is not None and last_state[w][j] is not None:
                        h_last = last_state[w][j][0]
                        h_left = h_last - h_range
                        h_right = h_last + h_range
                        if h_left >= 0 and h_right <= 179:
                            h_temp[:h_left] = 1e-8
                            h_temp[h_right:] = 1e-8
                        elif h_left < 0 and h_right <= 179:
                            h_left = 180 + h_left
                            if h_left < h_right:
                                h_temp[h_left:h_right] = 1e-8
                        elif h_left >= 0 and h_right > 179:
                            h_right = h_right - 179
                            if h_right < h_left:
                                h_temp[h_right:h_left] = 1e-8

                    if v_range is not None and last_state[w][j] is not None:
                        v_last = last_state[w][j][1]
                        v_left = max(0, v_last - v_range)
                        v_right = min(100, v_last + v_range)
                        v_temp[:v_left] = 1e-8
                        v_temp[v_right:] = 1e-8

                    h_temp = h_temp / (torch.sum(h_temp) + 1e-8)
                    v_temp = v_temp / (torch.sum(v_temp) + 1e-8)
                    # h_idx = torch.multinomial(h_temp, num_samples=1).item()
                    # v_idx = torch.multinomial(v_temp, num_samples=1).item()
                    h_idx = nucleus(h_temp.detach().cpu().numpy(), p[0])
                    v_idx = nucleus(v_temp.detach().cpu().numpy(), p[1])
                    last_state[w][j] = [h_idx, v_idx]

                    hue_action = h_idx / 180
                    val_action = v_idx / 100
                    action = np.array([hue_action, val_action])
                    next_state, _, done, info = env.step(action)
                    state = next_state

                target_hue = info['target_hue_hist']
                target_value = info['target_value_hist']

                best_scale, _ = find_best_scale(
                    env.hues.copy(), env.values.copy(),
                    target_hue, target_value, env,
                    scale_min=0.1, scale_max=10.0, num_steps=100,
                    criterion='mean'
                )

                scaled_values = np.clip(env.values.copy() * best_scale, 0, 1)
                scaled_result = compute_mixed_lighting(
                    positions=env.positions_raw,
                    hues=env.hues,
                    values=scaled_values,
                    grid_size=(env.grid_h, env.grid_w),
                    decay_model=env.decay_model,
                    sigma=env.sigma,
                    value_power=env.value_power,
                    eps=env.eps
                )
                final_hue_scaled = scaled_result['hue_histogram']
                final_value_scaled = scaled_result['value_histogram']
                final_hue_scaled = _downsample_distribution_to_bins(final_hue_scaled, 180)

                h_temp, v_temp = torch.from_numpy(final_hue_scaled).float().to(device), torch.from_numpy(
                    final_value_scaled).float().to(device)

                if attn_mask[w, i] == 1:
                    result[0][w, i], result[1][w, i] = h_temp, v_temp
                    if i != seq_len - 1:
                        light[0][w, i + 1], light[1][w, i + 1] = h_temp, v_temp

                light_hues = env.hues.copy()
                light_values = env.values.copy()
                result_light[0][w, i] = torch.tensor(light_hues)
                result_light[1][w, i] = torch.tensor(light_values)


        output_hue.append(result[0].cpu().detach())
        output_value.append(result[1].cpu().detach())
        output_light_hue.append(result_light[0].detach())
        output_light_value.append(result_light[1].detach())


    return torch.cat(output_hue, dim=0), torch.cat(output_value, dim=0), torch.cat(output_light_hue, dim=0), torch.cat(
        output_light_value, dim=0)


def main():
    args = get_args()
    cuda_devices = args.cuda_devices
    if not args.cpu and cuda_devices is not None and len(cuda_devices) >= 1:
        device_name = "cuda:" + str(cuda_devices[0])
    else:
        device_name = "cpu"
    device = torch.device(device_name)

    # folder_path = os.path.dirname(args.bart_path)
    folder_path = "./"

    bartconfig = BartConfig(max_position_embeddings=args.max_len,
                            d_model=args.hs,
                            encoder_layers=args.layers,
                            encoder_ffn_dim=args.ffn_dims,
                            encoder_attention_heads=args.heads,
                            decoder_layers=args.layers,
                            decoder_ffn_dim=args.ffn_dims,
                            decoder_attention_heads=args.heads
                            )

    bart = ML_BART(bartconfig, class_num=args.light_dim).to(device)
    model = ML_Classifier(hidden_dim=args.hs, class_num=args.light_dim).to(device)

    bart.bart = get_peft_model(bart.bart, bart.lora_config)

    if len(cuda_devices) > 1 and not args.cpu:
        bart = nn.DataParallel(bart, device_ids=cuda_devices)
        model = nn.DataParallel(model, device_ids=cuda_devices)

    bart.load_state_dict(torch.load(args.bart_path, map_location=device), strict=False)
    model.load_state_dict(torch.load(args.head_path, map_location=device), strict=False)

    policy = SeqLight(
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers
    ).to(device)
    policy.load_state_dict(torch.load(args.rl_model_path, map_location=device), strict=False)

    torch.set_grad_enabled(False)
    bart.eval()
    model.eval()
    policy.eval()

    env = LightingEnv(
        min_lights=args.num_lights,
        max_lights=args.num_lights,
        simple_layout=True
    )

    _, test_data = load_data(args.data_path, args.train_prop, args.max_len, args.gap, args.shuffle, args.random_seed,
                             fix_start=0)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=5)

    hue_gt = []
    value_gt = []
    f_names = []
    for i in range(len(test_data)):
        music, hv, f_name = test_data[i]
        hue, value = hv
        hue_gt.append(hue)
        value_gt.append(value)
        f_names.append(f_name)
    hue_gt = np.stack(hue_gt, axis=0)
    value_gt = np.stack(value_gt, axis=0)
    print(hue_gt.shape, value_gt.shape)

    output_hue, output_value, light_hue, light_value = iteration(test_loader, device, bart, model, env, policy,
                                                                 args.num_lights, args.t, args.p, args.h_range, args.v_range)

    output_hue, output_value, light_hue, light_value = output_hue.numpy(), output_value.numpy(), light_hue.numpy(), light_value.numpy()
    print(output_hue.shape, output_value.shape, light_hue.shape, light_value.shape)
    res = {
        'hue_gt': hue_gt,
        'value_gt': value_gt,
        'output_hue': output_hue,
        'output_value': output_value,
        'light_hue': light_hue,
        'light_value': light_value,
        'f_names': f_names
    }
    info = f'h_range={args.h_range}, v_range={args.v_range}, t={args.t[0]}-{args.t[1]}, p={args.p[0]}-{args.p[1]}'
    with open(os.path.join(folder_path, f'light_pred_{info}.pkl'), 'wb') as f:
        pickle.dump(res, f)


if __name__ == '__main__':
    main()
