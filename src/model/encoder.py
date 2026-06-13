"""
Frozen VecSet encoder based on the KLAutoEncoder from 3DShape2VecSet.

All classes are self-contained — no imports from the 3DShape2VecSet directory.
The KLAutoEncoder architecture (including decoder-side parameters) is reproduced
exactly so that pretrained checkpoints can be loaded with strict=True.

Reference: "3D-Shape2VecSet" — https://github.com/1zb/3DShape2VecSet
"""

from dataclasses import dataclass
from functools import wraps

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch_cluster import fps


ENCODER_ARCH = dict(
    depth=24,
    dim=512,
    queries_dim=512,
    output_dim=1,
    num_inputs=2048,
    num_latents=512,
    heads=8,
    dim_head=64,
)


@dataclass
class EncoderCfg:
    latent_dim: int = 64
    weights_path: str = ""


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _exists(val):
    return val is not None


def _default(val, d):
    return val if _exists(val) else d


def _cache_fn(f):
    cache = None
    @wraps(f)
    def cached_fn(*args, _cache=True, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache
    return cached_fn


# ---------------------------------------------------------------------------
# DropPath (inline — avoids timm dependency)
# ---------------------------------------------------------------------------

class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        noise = torch.rand(shape, dtype=x.dtype, device=x.device)
        noise.floor_(noise + keep_prob)          # Bernoulli mask
        return x / keep_prob * noise


# ---------------------------------------------------------------------------
# Building blocks (exact structural match with 3DShape2VecSet/models_ae.py)
# ---------------------------------------------------------------------------

class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim=None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if _exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)
        if _exists(self.norm_context):
            context = kwargs['context']
            kwargs.update(context=self.norm_context(context))
        return self.fn(x, **kwargs)


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, drop_path_rate=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim),
        )
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

    def forward(self, x):
        return self.drop_path(self.net(x))


class Attention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, drop_path_rate=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = _default(context_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

    def forward(self, x, context=None, mask=None):
        h = self.heads
        q = self.to_q(x)
        context = _default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale

        if _exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg)

        attn = sim.softmax(dim=-1)
        out = torch.einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.drop_path(self.to_out(out))


class PointEmbed(nn.Module):
    def __init__(self, hidden_dim=48, dim=128):
        super().__init__()
        assert hidden_dim % 6 == 0
        self.embedding_dim = hidden_dim
        e = torch.pow(2, torch.arange(hidden_dim // 6)).float() * np.pi
        e = torch.stack([
            torch.cat([e, torch.zeros(hidden_dim // 6), torch.zeros(hidden_dim // 6)]),
            torch.cat([torch.zeros(hidden_dim // 6), e, torch.zeros(hidden_dim // 6)]),
            torch.cat([torch.zeros(hidden_dim // 6), torch.zeros(hidden_dim // 6), e]),
        ])
        self.register_buffer('basis', e)   # (3, hidden_dim//2)
        self.mlp = nn.Linear(hidden_dim + 3, dim)

    @staticmethod
    def embed(input, basis):
        proj = torch.einsum('bnd,de->bne', input, basis)
        return torch.cat([proj.sin(), proj.cos()], dim=2)

    def forward(self, input):
        # input: (B, N, 3)
        embed = self.mlp(torch.cat([self.embed(input, self.basis), input], dim=2))
        return embed


class DiagonalGaussianDistribution:
    def __init__(self, mean, logvar, deterministic=False):
        self.mean = mean
        self.logvar = torch.clamp(logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if deterministic:
            self.var = self.std = torch.zeros_like(self.mean)

    def sample(self):
        return self.mean + self.std * torch.randn_like(self.mean)

    def kl(self, other=None):
        if self.deterministic:
            return torch.tensor([0.0])
        if other is None:
            return 0.5 * torch.mean(
                self.mean.pow(2) + self.var - 1.0 - self.logvar, dim=[1, 2]
            )
        return 0.5 * torch.mean(
            (self.mean - other.mean).pow(2) / other.var
            + self.var / other.var - 1.0 - self.logvar + other.logvar,
            dim=[1, 2, 3],
        )

    def mode(self):
        return self.mean


# ---------------------------------------------------------------------------
# KLAutoEncoder — exact structural copy of 3DShape2VecSet/models_ae.py
# (parameter names must match the pretrained checkpoint)
# ---------------------------------------------------------------------------

class KLAutoEncoder(nn.Module):
    def __init__(
        self,
        *,
        depth=24,
        dim=512,
        queries_dim=512,
        output_dim=1,
        num_inputs=2048,
        num_latents=512,
        latent_dim=64,
        heads=8,
        dim_head=64,
        weight_tie_layers=False,
        decoder_ff=False,
    ):
        super().__init__()
        self.depth = depth
        self.num_inputs = num_inputs
        self.num_latents = num_latents

        self.cross_attend_blocks = nn.ModuleList([
            PreNorm(dim, Attention(dim, dim, heads=1, dim_head=dim), context_dim=dim),
            PreNorm(dim, FeedForward(dim)),
        ])

        self.point_embed = PointEmbed(dim=dim)

        get_latent_attn = lambda: PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, drop_path_rate=0.1))
        get_latent_ff   = lambda: PreNorm(dim, FeedForward(dim, drop_path_rate=0.1))
        get_latent_attn, get_latent_ff = map(_cache_fn, (get_latent_attn, get_latent_ff))

        cache_args = {'_cache': weight_tie_layers}
        self.layers = nn.ModuleList([
            nn.ModuleList([get_latent_attn(**cache_args), get_latent_ff(**cache_args)])
            for _ in range(depth)
        ])

        self.decoder_cross_attn = PreNorm(
            queries_dim,
            Attention(queries_dim, dim, heads=1, dim_head=dim),
            context_dim=dim,
        )
        self.decoder_ff = PreNorm(queries_dim, FeedForward(queries_dim)) if decoder_ff else None
        self.to_outputs = nn.Linear(queries_dim, output_dim) if _exists(output_dim) else nn.Identity()

        self.proj = nn.Linear(latent_dim, dim)
        self.mean_fc   = nn.Linear(dim, latent_dim)
        self.logvar_fc = nn.Linear(dim, latent_dim)

    def encode(self, pc: torch.Tensor):
        """
        Args:
            pc: (B, num_inputs, 3) xyz point cloud
        Returns:
            kl: KL divergence (scalar per batch element)
            z:  (B, num_latents, latent_dim) sampled latent
        """
        B, N, D = pc.shape
        assert N == self.num_inputs

        # FPS: select num_latents anchor points
        flat = pc.reshape(B * N, D)
        batch = torch.arange(B, device=pc.device).repeat_interleave(N)
        ratio = self.num_latents / self.num_inputs
        idx = fps(flat, batch, ratio=ratio)
        sampled = flat[idx].reshape(B, -1, 3)

        # Embed
        sampled_emb = self.point_embed(sampled)   # (B, num_latents, dim)
        pc_emb      = self.point_embed(pc)         # (B, N, dim)

        # Cross-attention: anchors ← full cloud
        cross_attn, cross_ff = self.cross_attend_blocks
        x = cross_attn(sampled_emb, context=pc_emb, mask=None) + sampled_emb
        x = cross_ff(x) + x

        # KL bottleneck
        mean   = self.mean_fc(x)
        logvar = self.logvar_fc(x)
        posterior = DiagonalGaussianDistribution(mean, logvar)
        z  = posterior.sample()
        kl = posterior.kl()
        return kl, z

    def decode(self, x: torch.Tensor, queries: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        for self_attn, self_ff in self.layers:
            x = self_attn(x) + x
            x = self_ff(x) + x
        queries_emb = self.point_embed(queries)
        latents = self.decoder_cross_attn(queries_emb, context=x)
        if _exists(self.decoder_ff):
            latents = latents + self.decoder_ff(latents)
        return self.to_outputs(latents)

    def forward(self, pc, queries):
        kl, x = self.encode(pc)
        o = self.decode(x, queries).squeeze(-1)
        return {'logits': o, 'kl': kl}

