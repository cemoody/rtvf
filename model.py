import torch
import numpy as np

from torch import nn
from torch_trainer.trainer import Trainer
from torch_trainer.callbacks import rms_callback
from torch.optim import Adam
from torch.autograd import Variable
import torch.nn.functional as F


# V shape is [n_frames, nx, nx]
# V is deconvolutional from latent
# V = B + F
# s.t. B is low-rank and:
# F is fused-sparse
# V = u * v, where
# L = (I_i -
# Soft image:
# F = B + (c_i0 * v0.v0T + c_i1 * v1.v1T + ...)
# B [nx, nx, 3] <real>
# c = [n_frames, k]
# v = [nx, k]
# Hard image
# H (n_frames, nx, nx, 3) <real>
# No regularization on H
# Mask
# M (n_frames, nx, nx) <binary>
# TV Regularize the binary mask
# Also L1 regularize all of M down to zero
# Total image:
# I = F * (1 - M) + H * M


def total_variation(T):
    diff0 = T[1:, :, :] - T[:-1, :, :]
    diff1 = T[:, 1:, :] - T[:, :-1, :]
    diff2 = T[:, :, 1:] - T[:, :, :-1]
    d = (torch.abs(diff0).sum() +
         torch.abs(diff1).sum() +
         torch.abs(diff2).sum())
    return d


def sample_gumbel(size):
    noise = torch.rand(size)
    eps = 1e-20
    noise.add_(eps).log_().neg_()
    noise.add_(eps).log_().neg_()
    return Variable(noise)


def gumbel_softmax_sample(input, dim=0, temperature=.1, noise=None):
    if noise is None:
        noise = sample_gumbel(input.size())
    x = (input + noise) / temperature
    x = F.softmax(x, dim=dim)
    return x.view_as(input)


class RTVF(nn.Module):
    def __init__(self, n_frames, n_pixels, n_channels, k=1, l1=1e-4,
                 lmbda=1e-2, ltv=1e-3):
        super().__init__()
        self.B = nn.Parameter(torch.randn(n_pixels, n_pixels, 3))
        self.V = nn.Parameter(torch.randn(k, n_pixels, n_pixels, n_channels))
        self.C = nn.Parameter(torch.randn(n_frames, k))
        self.Mmu = nn.Parameter(torch.randn(n_frames, n_pixels, n_pixels) - 2.)
        self.Mlv = nn.Parameter(torch.randn(n_frames, n_pixels, n_pixels))
        self.H = nn.Parameter(torch.randn(n_frames, n_pixels, n_pixels,
                                          n_channels))
        self.lmbda = lmbda
        self.n_frames = n_frames
        self.ltv = ltv
        self.l1 = l1

    def soft_image(self, index=None, bg_only=False):
        if index is None:
            index = Variable(torch.arange(0, self.n_frames).long())
        if bg_only:
            return self.B[None, ...]
        # self.V is (nx, k)
        # V is (k, nx, nx)
        # V = self.V.t()[:, None, :] * self.V.t()[:, :, None]
        # cV = self.C[index] @ self.V
        cV = self.C[index][..., None, None] * self.V
        # V is (1, nx, nx, k)
        # V = torch.transpose(V, 0, 2)[None, ...]
        # c is (bs, 1, 1, k)
        # c = self.C[index][:, None, None, :]
        # cV is (bs, nx, nx, k)
        # cV = c * V
        return self.B[None, ...] + cV

    def hard_image(self, index=None):
        if index is None:
            index = Variable(torch.arange(0, self.n_frames).long())
        return self.H[index]

    def mask_multiply(self, index, image, sign=1.0):
        mu = torch.sigmoid(self.Mmu[index][..., None] * sign)
        lv = self.Mlv[index][..., None]
        noise = Variable(torch.rand(lv.size()))
        mi = mu * image
        vi = torch.exp(lv) * image * image
        sample = mi + vi * noise
        return sample

    def forward(self, index, img):
        S = self.soft_image(index)
        H = self.hard_image(index)
        SM = self.mask_multiply(index, S, sign=-1.0)
        HM = self.mask_multiply(index, H)
        return torch.sigmoid(SM + HM)

    def loss(self, prediction, index, img):
        # log likelihood of observed image
        llh = (img - prediction).norm(2).sum()
        # L2 Regularize V, c
        la = self.V.norm(2) + self.C.norm(2)
        # L1 Regualrize H
        lb = self.H.norm(1) * self.l1
        # TV regularize M
        tv = total_variation(self.Mmu) * self.ltv
        # l1 regularize M
        l1 = torch.abs(torch.sigmoid(self.Mmu)).sum() * self.l1
        # import pdb; pdb.set_trace()
        return llh + (self.lmbda * (la + lb + tv + l1) /
                      self.n_frames * len(index))


def fit(X, n_epochs=100):
    n_frames, n_pixels, _, n_channels = X.shape
    index = np.arange(n_frames)
    model = RTVF(n_frames, n_pixels, n_channels)
    optim = Adam(model.parameters(), lr=1e-1)
    callbacks = {'rms': rms_callback}
    t = Trainer(model, optim, batchsize=32, callbacks=callbacks,
                seed=42, print_every=1)
    for epoch in range(n_epochs):
        t.fit(index, X)
        background = model.soft_image(bg_only=True)
        foreground = model.hard_image()
        mask = torch.sigmoid(model.Mmu)
        mask = mask.data.numpy()
        bg = background.data.numpy()
        fg = foreground.data.numpy()
        np.savez("checkpoint", bg=bg, fg=fg, mask=mask)
        if epoch > 4:
            optim.lr = 1e-4
            t.batchsize = 128
    return bg, fg, mask
