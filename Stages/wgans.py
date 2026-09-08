"""
1D WGAN-GP for GPVS fault-detection augmentation.

Adapted from eriklindernoren's wgan_gp.py to the GPVS domain:
  * MLP generator + MLP critic for 13 UNORDERED tabular features (no convolutions).
  * Wasserstein loss with gradient penalty: the critic has NO sigmoid and NO
    BatchNorm; the 5:1 critic:generator update ratio and the GP term give usable
    gradients even when real and fake are well separated (the failure mode that
    killed the vanilla DCGAN at the sparse end).
  * Per-class, unconditional — same shape as dcgans.py, so this module exposes the
    SAME public API (train_class_gans / augment / generate) and is a drop-in in the
    GAN registry.
  * Diagnostics: instead of d_acc, the key fidelity signal is the Wasserstein
    estimate w_dist = E[C(real)] - E[C(fake)] (large -> poor generator, shrinking
    toward 0 -> improving).
"""

import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Generator1D(nn.Module):
    def __init__(self, latent_dim=64, n_features=13, hidden=128):
        super().__init__()
        def block(i, o, norm=True):
            layers = [nn.Linear(i, o)]
            if norm:
                layers.append(nn.BatchNorm1d(o, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        self.model = nn.Sequential(
            *block(latent_dim, hidden, norm=False),
            *block(hidden, hidden * 2),
            *block(hidden * 2, hidden * 2),
            nn.Linear(hidden * 2, n_features),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.model(z)


class Critic1D(nn.Module):
    def __init__(self, n_features=13, hidden=128):
        super().__init__()
        # No BatchNorm (breaks GP) and no Sigmoid (Wasserstein critic outputs a score)
        self.model = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.model(x)


def _gradient_penalty(critic, real, fake, device):
    alpha = torch.rand(real.size(0), 1, device=device)
    inter = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d_inter = critic(inter)
    grads = torch.autograd.grad(
        outputs=d_inter, inputs=inter,
        grad_outputs=torch.ones_like(d_inter),
        create_graph=True, retain_graph=True)[0]
    return ((grads.norm(2, dim=1) - 1) ** 2).mean()


def train_wgan(X_raw, seed, device, n_epochs=300, batch_size=64, latent_dim=64,
               lr=2e-4, n_critic=5, lambda_gp=10.0, b1=0.5, b2=0.9, verbose=False):
    set_seed(seed)
    N, n_features = X_raw.shape
    if N < 2:
        raise ValueError(f"Need >=2 real samples, got {N}.")

    scaler = MinMaxScaler(feature_range=(-1, 1)).fit(X_raw)
    Xs = scaler.transform(X_raw).astype("float32")

    bs = min(batch_size, N)
    drop_last = (N % bs == 1)
    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(torch.from_numpy(Xs)),
                        batch_size=bs, shuffle=True, drop_last=drop_last, generator=g)

    G = Generator1D(latent_dim, n_features).to(device)
    C = Critic1D(n_features).to(device)
    opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(b1, b2))
    opt_C = torch.optim.Adam(C.parameters(), lr=lr, betas=(b1, b2))

    history = {"c_loss": [], "g_loss": [], "w_dist": []}

    for epoch in range(n_epochs):
        ep_c = ep_g = ep_w = 0.0
        n_batches = 0
        for (real,) in loader:
            real = real.to(device)
            n = real.size(0)

            # --- critic: n_critic updates on this batch ---
            for _ in range(n_critic):
                opt_C.zero_grad()
                z = torch.randn(n, latent_dim, device=device)
                fake = G(z).detach()
                d_real = C(real).mean()
                d_fake = C(fake).mean()
                gp = _gradient_penalty(C, real, fake, device)
                c_loss = d_fake - d_real + lambda_gp * gp
                c_loss.backward()
                opt_C.step()

            # --- generator: one update ---
            opt_G.zero_grad()
            z = torch.randn(n, latent_dim, device=device)
            g_loss = -C(G(z)).mean()
            g_loss.backward()
            opt_G.step()

            ep_c += c_loss.item()
            ep_g += g_loss.item()
            ep_w += (d_real - d_fake).item()        # Wasserstein estimate
            n_batches += 1

        history["c_loss"].append(ep_c / n_batches)
        history["g_loss"].append(ep_g / n_batches)
        history["w_dist"].append(ep_w / n_batches)

        if verbose and (epoch == 0 or (epoch + 1) % 50 == 0):
            print(f"    [ep {epoch+1:3d}/{n_epochs}] "
                  f"C_loss {history['c_loss'][-1]:.3f} "
                  f"G_loss {history['g_loss'][-1]:.3f} "
                  f"W_dist {history['w_dist'][-1]:.3f}")

    return G, scaler, history


def generate(G, scaler, n_samples, seed, device, latent_dim=64):
    if n_samples <= 0:
        return np.empty((0, scaler.n_features_in_), dtype="float32")
    G.eval()
    g = torch.Generator().manual_seed(seed + 10_000)
    z = torch.randn(n_samples, latent_dim, generator=g).to(device)
    with torch.no_grad():
        xs = G(z).cpu().numpy()
    return scaler.inverse_transform(xs).astype("float32")


# ---- public API mirrors dcgans.py so the registry treats them identically ----
def train_class_gans(train_array, seed, device, n_classes=8,
                     feat_slice=(1, 14), label_col=-1, **gan_kw):
    feats = train_array[:, feat_slice[0]:feat_slice[1]].astype("float32")
    labels = train_array[:, label_col].astype(int)
    gans, histories = {}, {}
    for c in range(n_classes):
        Xc = feats[labels == c]
        if len(Xc) < 2:
            continue
        G, scaler, hist = train_wgan(Xc, seed=seed, device=device, **gan_kw)
        gans[c] = (G, scaler, len(Xc))
        histories[c] = hist
    return gans, histories


def augment(train_array, gans, ratio, seed, device,
            feat_slice=(1, 14), label_col=-1, latent_dim=64):
    if ratio <= 0 or not gans:
        return train_array
    width = train_array.shape[1]
    blocks = [train_array]
    for c, (G, scaler, n_real) in gans.items():
        n_gen = int(round(ratio * n_real))
        gen = generate(G, scaler, n_gen, seed=seed, device=device, latent_dim=latent_dim)
        if len(gen) == 0:
            continue
        block = np.zeros((len(gen), width), dtype=train_array.dtype)
        block[:, feat_slice[0]:feat_slice[1]] = gen
        block[:, label_col] = c
        blocks.append(block)
    return np.vstack(blocks)