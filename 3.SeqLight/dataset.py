import torch
from torch.utils.data import Dataset
import os
import pickle
import numpy as np
import random

pad = -1000


def _normalize_distribution(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.clip(x, 0.0, None)
    s = np.sum(x, axis=-1, keepdims=True)
    return np.divide(x, s + eps, out=np.zeros_like(x, dtype=np.float32), where=s > 0)


def _downsample_distribution_to_bins(x: np.ndarray, out_bins: int) -> np.ndarray:
    t, in_bins = x.shape
    x = _normalize_distribution(x)
    in_edges = np.linspace(0, 1, in_bins + 1)
    out_edges = np.linspace(0, 1, out_bins + 1)

    y = np.zeros((t, out_bins), dtype=np.float32)
    for i in range(t):
        cdf = np.concatenate(([0.0], np.cumsum(x[i], dtype=np.float32)))
        cdf_rs = np.interp(out_edges, in_edges, cdf).astype(np.float32)
        y[i] = np.diff(cdf_rs)
    return _normalize_distribution(y)


class ML_Dataset(Dataset):
    def __init__(self, file_path, max_len=600, gap=0, fix_start=None):
        self.file_path = file_path
        self.max_len = int(max_len)
        self.gap = int(gap)
        self.sample_len = self.max_len * (gap + 1)
        self.fix_start = fix_start

    def __len__(self):
        return len(self.file_path)

    def __getitem__(self, index):
        with open(self.file_path[index], 'rb') as file:
            data = pickle.load(file)
        music = data['music']
        music_len = music.shape[0]

        # use average value as light
        sample = data['sample']
        light = data['light']

        h = [light[int(i)][60][0] for i in sample]
        v = [light[int(i)][0][2] for i in sample]
        h = np.array(h)
        v = np.array(v)

        # set the hue=0 as the average of the hue=1 and hue=179
        h[:, 0] = (h[:, 1] + h[:, 179]) // 2
        # h = np.argmax(h, axis=1)

        v[:, 0] = 0
        # values = np.arange(v.shape[1])
        # weighted_sums = np.dot(v, values)
        # counts = np.sum(v, axis=1)
        # v = np.divide(weighted_sums, counts, out=np.zeros_like(weighted_sums, dtype=float), where=counts != 0)

        h = _normalize_distribution(h)
        v = _downsample_distribution_to_bins(v, out_bins=100)

        if music_len > self.sample_len:
            if self.fix_start is not None:
                start_index = self.fix_start
            else:
                start_index = np.random.randint(0, music_len - self.sample_len + 1)
            music = music[start_index:start_index + self.sample_len, :]
            h = h[start_index:start_index + self.sample_len]
            v = v[start_index:start_index + self.sample_len]
        elif music_len < self.sample_len:
            music = np.concatenate([music, np.zeros([self.sample_len - music_len, music.shape[1]]) + pad], axis=0)
            h = np.concatenate([h, np.zeros([self.sample_len - music_len,h.shape[1]]) + pad], axis=0)
            v = np.concatenate([v, np.zeros([self.sample_len - music_len,v.shape[1]]) + pad], axis=0)

        music = music[::(self.gap + 1)]
        h = h[::(self.gap + 1)]
        v = v[::(self.gap + 1)]
        # hv = np.stack([h, v], axis=1)

        f_name = self.file_path[index].split('/')[-1]
        return music, [h, v], f_name

class Pretrain_Dataset(Dataset):
    def __init__(self, file_path, max_len=600, gap=0):
        self.file_path = file_path
        self.max_len = int(max_len)
        self.gap = int(gap)
        self.sample_len = self.max_len * (gap + 1)
        self.num = len(self.file_path)

    def __len__(self):
        return self.num

    def __getitem__(self, index):
        with open(self.file_path[index], 'rb') as file:
            data = pickle.load(file)
        music_ori = data['music']
        music_len = music_ori.shape[0]

        if music_len >= self.sample_len:
            start_index = np.random.randint(0, music_len - self.sample_len + 1)
            music = music_ori[start_index:start_index + self.sample_len, :]
        else:
            music = np.concatenate([music_ori, np.zeros([self.sample_len - music_len, music_ori.shape[1]]) + pad],
                                   axis=0)
        music = music[::(self.gap + 1)]

        if music_len >= self.sample_len:
            start_index = np.random.randint(0, music_len - self.sample_len + 1)
            pos = music_ori[start_index:start_index + self.sample_len, :]
        else:
            pos = np.concatenate([music_ori, np.zeros([self.sample_len - music_len, music_ori.shape[1]]) + pad], axis=0)
        pos = pos[::(self.gap + 1)]

        neg_index = random.randint(0, self.num - 1)
        while neg_index == index:
            neg_index = random.randint(0, self.num - 1)
        with open(self.file_path[neg_index], 'rb') as file:
            data = pickle.load(file)
        music_ori = data['music']
        music_len = music_ori.shape[0]
        if music_len >= self.sample_len:
            start_index = np.random.randint(0, music_len - self.sample_len + 1)
            neg = music_ori[start_index:start_index + self.sample_len, :]
        else:
            neg = np.concatenate([music_ori, np.zeros([self.sample_len - music_len, music_ori.shape[1]]) + pad], axis=0)
        neg = neg[::(self.gap + 1)]

        return music, pos, neg


def load_data(root_path, train_prop=0.9, max_len=600, gap=0, shuffle=False, random_seed=42, fix_start=None):
    file_path = []
    for dirpath, _, filenames in os.walk(root_path):
        for file in filenames:
            if file[-4:] == ".pkl":
                file_path.append(os.path.join(dirpath, file))
    if train_prop == 1.0:
        return ML_Dataset(file_path, max_len, gap, fix_start)
    else:
        train_num = round(len(file_path) * train_prop)
        if shuffle:
            np.random.seed(random_seed)
            np.random.shuffle(file_path)
        return ML_Dataset(file_path[:train_num], max_len, gap, fix_start), ML_Dataset(file_path[train_num:], max_len,
                                                                                      gap, fix_start)


def load_pretrain(root_path, train_prop=0.9, max_len=600, gap=0):
    file_path = []
    for dirpath, _, filenames in os.walk(root_path):
        for file in filenames:
            if file[-4:] == ".pkl":
                file_path.append(os.path.join(dirpath, file))
    if train_prop == 1.0:
        return Pretrain_Dataset(file_path, max_len, gap)
    else:
        train_num = round(len(file_path) * train_prop)
        np.random.shuffle(file_path)
        return Pretrain_Dataset(file_path[:train_num], max_len, gap), Pretrain_Dataset(file_path[train_num:], max_len,
                                                                                       gap)

if __name__ == "__main__":
    dataset = load_data("/mnt/disk/dian/m2l_data/output", train_prop=1.0, max_len=600, gap=0, shuffle=False, random_seed=42, fix_start=0)
    for music, (h, v), f_name in dataset:
        print(music.shape, h.shape, v.shape, f_name)
        try:
            import matplotlib.pyplot as plt

            # Plot the first frame distributions for quick sanity check.
            n_f = 100
            h0 = h[n_f]
            v0 = v[n_f]
            fig, axes = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True)
            axes[0].plot(np.arange(h0.shape[-1]), h0, linewidth=1.0)
            axes[0].set_title("H distribution (frame 0)")
            axes[0].set_xlabel("bin")
            axes[0].set_ylabel("prob")
            axes[0].set_ylim(0, max(1e-6, float(np.max(h0)) * 1.1))

            axes[1].plot(np.arange(v0.shape[-1]), v0, linewidth=1.0)
            axes[1].set_title("V distribution (frame 0, downsampled to 100)")
            axes[1].set_xlabel("bin")
            axes[1].set_ylabel("prob")
            axes[1].set_ylim(0, max(1e-6, float(np.max(v0)) * 1.1))

            out_path = f"cache/first_frame_dist_{os.path.splitext(f_name)[0]}.png"
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            print("Saved first-frame distribution plot to:", out_path)
        except Exception as e:
            print("Skip plotting (matplotlib missing or runtime error):", type(e).__name__, str(e))
        break

