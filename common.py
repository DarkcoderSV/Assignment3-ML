"""Shared data, split, transforms, models and training loops for all questions."""
import json
import os
import re
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]


def load_config():
    with open(os.path.join(ROOT, "configs", "config.json")) as f:
        cfg = json.load(f)
    digits = re.sub(r"\D", "", cfg["roll_number"])
    assert digits, "roll_number must contain at least one digit"
    cfg["seed"] = int(digits)
    return cfg


def device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed):
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)


def path(*parts):
    p = os.path.join(ROOT, *parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def save_json(obj, *parts):
    with open(path(*parts), "w") as f:
        json.dump(obj, f, indent=1)


def load_json(*parts):
    with open(os.path.join(ROOT, *parts)) as f:
        return json.load(f)


# --------------------------------------------------------------------------- data

def _raw_cifar():
    import torchvision
    tr = torchvision.datasets.CIFAR10(os.path.join(ROOT, "data"), train=True, download=True)
    te = torchvision.datasets.CIFAR10(os.path.join(ROOT, "data"), train=False, download=True)
    return (tr.data, np.array(tr.targets)), (te.data, np.array(te.targets))


def get_split(cfg):
    """Stratified 45k/5k split of the official training set, saved to splits/."""
    fn = path("splits", f"split_seed{cfg['seed']}.npz")
    if os.path.exists(fn):
        s = np.load(fn)
        return s["train_idx"], s["val_idx"]
    from sklearn.model_selection import train_test_split
    (_, ytr), _ = _raw_cifar()
    idx = np.arange(len(ytr))
    train_idx, val_idx = train_test_split(idx, test_size=cfg["val_size"], stratify=ytr,
                                          random_state=cfg["seed"] % (2 ** 32))
    train_idx, val_idx = np.sort(train_idx), np.sort(val_idx)
    np.savez(fn, train_idx=train_idx, val_idx=val_idx)
    return train_idx, val_idx


def load_data(cfg, dev=None):
    """Returns dict split -> (x float [N,3,32,32] in [0,1], y long) on `dev`."""
    dev = dev or device()
    (xtr, ytr), (xte, yte) = _raw_cifar()
    tri, vai = get_split(cfg)

    def to_t(x, y):
        x = torch.from_numpy(x).permute(0, 3, 1, 2).float().div(255.0)
        return x.to(dev), torch.from_numpy(y).long().to(dev)

    return {"train": to_t(xtr[tri], ytr[tri]), "val": to_t(xtr[vai], ytr[vai]),
            "test": to_t(xte, yte)}


# --------------------------------------------------------------------------- transforms
# Each transform is a fixed deterministic function applied identically to train/val/test.

def make_transform(name, seed):
    g = np.random.RandomState((seed + 7919) % (2 ** 32))
    if name == "clean":
        return lambda x: x
    if name == "pixel_perm":
        # Permute the 1024 spatial positions; the RGB triple of each pixel stays together,
        # so the multiset of pixel values of every image is unchanged.
        perm = torch.from_numpy(g.permutation(32 * 32))

        def f(x):
            n = x.shape[0]
            return x.reshape(n, 3, 1024)[:, :, perm.to(x.device)].reshape(n, 3, 32, 32)
        return f
    if name == "block_perm":
        # Permute the 16 non-overlapping 8x8 blocks; local structure inside blocks is kept.
        perm = g.permutation(16)
        while (perm == np.arange(16)).any():  # a derangement, so every block moves
            perm = g.permutation(16)

        def f(x):
            n = x.shape[0]
            b = x.reshape(n, 3, 4, 8, 4, 8).permute(0, 1, 2, 4, 3, 5).reshape(n, 3, 16, 8, 8)
            b = b[:, :, torch.from_numpy(perm).to(x.device)]
            return b.reshape(n, 3, 4, 4, 8, 8).permute(0, 1, 2, 4, 3, 5).reshape(n, 3, 32, 32)
        return f
    if name == "gray":
        w = torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)

        def f(x):
            return (x * w.to(x.device)).sum(1, keepdim=True).expand(-1, 3, -1, -1).contiguous()
        return f
    raise ValueError(name)


def apply_transform(data, tf, bs=5000):
    out = {}
    for k, (x, y) in data.items():
        out[k] = (torch.cat([tf(x[i:i + bs]) for i in range(0, len(x), bs)]), y)
    return out


def augment(x):
    """Random crop (pad 4) + horizontal flip, on device. Used only where stated."""
    n = x.shape[0]
    xp = F.pad(x, (4, 4, 4, 4))
    dy = torch.randint(0, 9, (n,), device=x.device)
    dx = torch.randint(0, 9, (n,), device=x.device)
    ar = torch.arange(32, device=x.device)
    rows = (dy[:, None] + ar)[:, None, :, None]
    cols = (dx[:, None] + ar)[:, None, None, :]
    out = xp[torch.arange(n, device=x.device)[:, None, None, None],
             torch.arange(3, device=x.device)[None, :, None, None], rows, cols]
    flip = torch.rand(n, device=x.device) < 0.5
    out[flip] = out[flip].flip(3)
    return out


def add_noise(x, sigma, gen=None):
    noise = torch.randn(x.shape, generator=gen, device="cpu").to(x.device) if gen is not None \
        else torch.randn_like(x)
    return (x + sigma * noise).clamp(0, 1)


# --------------------------------------------------------------------------- models

class Normalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32).view(1, 3, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std


def channel_stats(x):
    return x.mean((0, 2, 3)).cpu().tolist(), (x.std((0, 2, 3)) + 1e-6).cpu().tolist()


class MLP(nn.Module):
    def __init__(self, hidden, mean=(0, 0, 0), std=(1, 1, 1)):
        super().__init__()
        self.norm = Normalize(mean, std)
        layers, d = [], 3072
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(inplace=True)]
            d = h
        self.features = nn.Sequential(*layers)
        self.head = nn.Linear(d, 10)

    def forward(self, x):
        return self.head(self.features(self.norm(x).flatten(1)))


class CNN(nn.Module):
    """3 stages of (conv3x3-BN-ReLU)x2 + maxpool, global average pool, linear head."""

    def __init__(self, widths=(64, 160, 320), mean=(0, 0, 0), std=(1, 1, 1)):
        super().__init__()
        self.norm = Normalize(mean, std)
        layers, c = [], 3
        for w in widths:
            for _ in range(2):
                layers += [nn.Conv2d(c, w, 3, padding=1, bias=False), nn.BatchNorm2d(w),
                           nn.ReLU(inplace=True)]
                c = w
            layers.append(nn.MaxPool2d(2))
        self.features = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Linear(c, 10)

    def forward(self, x):
        return self.head(self.features(self.norm(x)))


class ConvAE(nn.Module):
    """Conv encoder 32->16->8->4, linear bottleneck of size d, mirrored decoder.
    Only the two bottleneck linear layers depend on d."""

    def __init__(self, d):
        super().__init__()
        self.d = d

        def down(i, o):
            return [nn.Conv2d(i, o, 4, 2, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True)]

        def up(i, o):
            return [nn.ConvTranspose2d(i, o, 4, 2, 1, bias=False), nn.BatchNorm2d(o),
                    nn.ReLU(inplace=True)]
        self.enc_conv = nn.Sequential(*down(3, 64), *down(64, 128), *down(128, 256), nn.Flatten())
        self.enc_fc = nn.Linear(256 * 16, d)
        self.dec_fc = nn.Sequential(nn.Linear(d, 256 * 16), nn.ReLU(inplace=True))
        self.dec_conv = nn.Sequential(*up(256, 128), *up(128, 64),
                                      nn.ConvTranspose2d(64, 3, 4, 2, 1), nn.Sigmoid())

    def encode(self, x):
        return self.enc_fc(self.enc_conv(x))

    def decode(self, z):
        return self.dec_conv(self.dec_fc(z).view(-1, 256, 4, 4))

    def forward(self, x):
        return self.decode(self.encode(x))


def n_params(m):
    return sum(p.numel() for p in m.parameters())


# --------------------------------------------------------------------------- training

@torch.no_grad()
def predict(model, x, bs=1000):
    model.eval()
    return torch.cat([model(x[i:i + bs]) for i in range(0, len(x), bs)])


@torch.no_grad()
def accuracy(model, x, y):
    return (predict(model, x).argmax(1) == y).float().mean().item()


def train_classifier(model, data, lr, epochs, wd, bs=256, aug=False, seed=0,
                     train_labels=None, schedule="cosine", track=None, log=None):
    """Train with AdamW. Keeps the state with best validation accuracy (early-stopping
    checkpoint). `train_labels` overrides training labels (label-noise experiments).
    `track` maps name -> (x, y) evaluated every epoch (e.g. noisy-label subsets)."""
    dev = next(model.parameters()).device
    xtr, ytr = data["train"]
    if train_labels is not None:
        ytr = train_labels
    xva, yva = data["val"]
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    steps = epochs * ((len(xtr) + bs - 1) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.15) \
        if schedule == "cosine" else None
    g = torch.Generator(device="cpu").manual_seed(seed)
    hist = {k: [] for k in ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "time"]}
    for k in (track or {}):
        hist[k] = []
    best, best_state, t0 = -1, None, time.time()
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(xtr), generator=g).to(dev)
        tl, tc, n = 0.0, 0, 0
        for i in range(0, len(xtr), bs):
            idx = perm[i:i + bs]
            xb, yb = xtr[idx], ytr[idx]
            if aug:
                xb = augment(xb)
            out = model(xb)
            loss = F.cross_entropy(out, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if sched:
                sched.step()
            tl += loss.item() * len(idx)
            tc += (out.argmax(1) == yb).sum().item()
            n += len(idx)
        vo = predict(model, xva)
        va = (vo.argmax(1) == yva).float().mean().item()
        hist["epoch"].append(ep)
        hist["train_loss"].append(tl / n)
        hist["train_acc"].append(tc / n)
        hist["val_loss"].append(F.cross_entropy(vo, yva).item())
        hist["val_acc"].append(va)
        hist["time"].append(time.time() - t0)
        for k, (xs, ys) in (track or {}).items():
            hist[k].append(accuracy(model, xs, ys))
        if va > best:
            best, best_ep = va, ep
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if log:
            extra = " ".join(f"{k}={hist[k][-1]:.3f}" for k in (track or {}))
            print(f"[{log}] ep {ep:3d} loss {tl / n:.3f} train {tc / n:.3f} val {va:.3f} "
                  f"{extra} ({time.time() - t0:.0f}s)", flush=True)
    hist["best_val_acc"], hist["best_epoch"] = best, best_ep
    hist["final_state"] = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return hist


def strip(hist):
    return {k: v for k, v in hist.items() if k != "final_state"}
