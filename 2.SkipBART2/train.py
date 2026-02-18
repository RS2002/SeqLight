import datetime
import os
from model import ML_BART, ML_Classifier
from transformers import BartConfig
import argparse
import tqdm
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
from dataset import load_data
import random
from torch.optim import AdamW
from peft import get_peft_model, LoraConfig
import torch.nn.functional as F

pad = -1000


def get_args():
    parser = argparse.ArgumentParser(description='')

    parser.add_argument("--music_dim", type=int, default=512)
    parser.add_argument("--light_dim", type=int, nargs='+', default=[360, 100])
    parser.add_argument('--gap', type=int, default=0)

    parser.add_argument('--layers', type=int, default=8)
    parser.add_argument('--max_len', type=int, default=1024)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--hs', type=int, default=1024)
    parser.add_argument('--ffn_dims', type=int, default=2048)

    parser.add_argument("--cpu", action="store_true", default=False)
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=[0])
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--converge_epoch', type=int, default=30)
    parser.add_argument('--min_epoch', type=int, default=50)

    parser.add_argument('--data_path', type=str, default="./discard/test/data")
    parser.add_argument('--train_prop', type=float, default=0.9)

    parser.add_argument('--model_path', type=str, default=None)

    parser.add_argument("--shuffle", action="store_true", default=False)
    parser.add_argument('--random_seed', type=int, default=42)

    parser.add_argument("--reset_decoder", action="store_true", default=False)

    args = parser.parse_args()
    return args


def iteration(data_loader, device, bart, model, optim, train=True, weight=[1.0, 1.0]):
    if train:
        torch.set_grad_enabled(True)
        bart.train()
        model.train()
    else:
        torch.set_grad_enabled(False)
        bart.eval()
        model.eval()

    h_loss_list = []
    v_loss_list = []
    loss_list = []

    pbar = tqdm.tqdm(data_loader, disable=False)
    for music, light, f_name in pbar:
        music = music.float().to(device)
        light = light.numpy()

        # 1. Tokenize Light
        light[light[..., 0, :] < 0, :, 0] = 0
        light[light[..., 1, :] < 0, :, 1] = 0

        light = torch.from_numpy(light)
        light = torch.round(light)
        light = light.long().to(device)

        light_input = torch.zeros_like(light)
        light_input[:, 1:, :] = light[:, :-1, :]
        light_input[:, 0, :] = light[:, 0, :]

        # 2. Process Music Emb
        non_pad = (music != pad).to(device)
        batch_size, seq_len, input_dim = music.shape
        rand_word = torch.randn((batch_size, seq_len, input_dim)).to(device)
        avg = torch.sum(music * non_pad, dim=1, keepdim=True) / (torch.sum(non_pad, dim=1, keepdim=True) + 1e-8)
        std = torch.sqrt(torch.sum(((music - avg) ** 2) * non_pad, dim=1, keepdim=True) / (
                torch.sum(non_pad, dim=1, keepdim=True) + 1e-8))
        rand_word = (rand_word + avg) * std
        # rand_word = torch.clip(rand_word,0,1)
        music[~non_pad.bool()] = rand_word[~non_pad.bool()]
        attn_mask = non_pad[..., 0].float()
        attn_mask_light = torch.zeros_like(attn_mask)
        attn_mask_light[:, 1:] = attn_mask[:, :-1]
        attn_mask_light[:, 0] = attn_mask[:, 0]

        # 3. train
        h_hat, v_hat = model(bart(music, light_input, attn_mask, attn_mask_light))
        h_hat, v_hat = h_hat.reshape(batch_size * seq_len, -1), v_hat.reshape(batch_size * seq_len, -1)
        h, v = light[..., 0].reshape(batch_size * seq_len, -1), light[..., 1].reshape(batch_size * seq_len, -1)
        attn_mask = attn_mask.reshape(batch_size * seq_len)
        loss_h = torch.sum(F.kl_div(torch.log(h_hat + 1e-8), h, reduction='batchmean') * attn_mask) / torch.sum(attn_mask)
        loss_v = torch.sum(F.kl_div(torch.log(v_hat + 1e-8), v, reduction='batchmean') * attn_mask) / torch.sum(attn_mask)
        loss = loss_h * weight[0] + loss_v * weight[1]
        loss_list.append((loss_h + loss_v).item())
        h_loss_list.append(loss_h.item())
        v_loss_list.append(loss_v.item())

        # 4. calculate loss
        if train:
            optim.zero_grad()
            loss.backward()
            optim.step()

    return np.mean(h_loss_list), np.mean(v_loss_list), np.mean(loss_list)


def main():
    args = get_args()
    cuda_devices = args.cuda_devices
    if not args.cpu and cuda_devices is not None and len(cuda_devices) >= 1:
        device_name = "cuda:" + str(cuda_devices[0])
    else:
        device_name = "cpu"
    device = torch.device(device_name)

    date_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    date_str += '_FINETUNE'
    os.makedirs("results/{}".format(date_str), exist_ok=True)

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

    if args.model_path is not None:
        bart.load_state_dict(torch.load(args.model_path),strict=False)
        print("Load Model from ", args.model_path)
        if args.reset_decoder:
            bart.reset_decoder()
    else:
        print("No Pre-train Model")

    params = set(bart.parameters()) | set(model.parameters())
    total_params = sum(p.numel() for p in params if p.requires_grad)
    print('total parameters:', total_params)
    optim = AdamW(params, lr=args.lr)

    loss_best = 1e8
    loss_epoch = 0
    j = 0

    train_data, test_data = load_data(args.data_path, args.train_prop, args.max_len, args.gap, args.shuffle,
                                      args.random_seed)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=5)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True, num_workers=5)

    weight = [1.0, 1.0]
    while True:
        j += 1
        loss_h, loss_v, loss = iteration(train_loader, device, bart, model, optim, train=True, weight=weight)
        log = "Epoch {} | Training Loss_H {:06f} , Loss_V {:06f} , Loss {:} | ".format(j, loss_h, loss_v, loss)
        print(log)
        with open("results/{}/log.txt".format(date_str), 'a') as file:
            file.write(log)
        loss_h, loss_v, loss = iteration(test_loader, device, bart, model, optim, train=False, weight=weight)
        log = "Test Loss_H {:06f} , Loss_V {:06f} , Loss {:} ".format(loss_h, loss_v, loss)
        print(log)
        with open("results/{}/log.txt".format(date_str), 'a') as file:
            file.write(log + "\n")

        # weight = [(loss_h + 1e-8) / (loss + 1e-8), (loss_v + 1e-8) / (loss + 1e-8)]

        if j > args.min_epoch:
            if loss <= loss_best:
                torch.save(bart.state_dict(), "results/{}/bart_finetune.pth".format(date_str))
                torch.save(model.state_dict(), "results/{}/head_finetune.pth".format(date_str))

            if loss <= loss_best:
                loss_best = loss
                loss_epoch = 0
            else:
                loss_epoch += 1

            if loss_epoch > args.converge_epoch:
                break
        print("Loss Epoch {:}".format(loss_epoch))


if __name__ == '__main__':
    import time
    start = time.time()
    main()
    end = time.time()
    print("Time:", time.strftime("%H:%M:%S", time.gmtime(end - start)))