from typing import Tuple
import torch
from einops import rearrange, repeat



def rotate_half(x, interleaved=False):
    if not interleaved:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    else:
        x1, x2 = x[..., ::2], x[..., 1::2]
        return rearrange(
            torch.stack((-x2, x1), dim=-1), "... d two -> ... (d two)", two=2
        )


def apply_rotary_emb_torch(x, cos, sin, interleaved=False, _inplace=False):
    """
    x: (batch_size, seqlen, nheads, headdim)
    cos, sin: (seqlen, rotary_dim / 2)
    """
    ro_dim = cos.shape[-1] * 2
    assert ro_dim <= x.shape[-1]
    seqlen = x.size(1)
    cos = cos[:seqlen]
    sin = sin[:seqlen]
    cos = repeat(cos, "s d -> s 1 (2 d)")
    sin = repeat(sin, "s d -> s 1 (2 d)")
    return torch.cat(
        [
            x[..., :ro_dim] * cos + rotate_half(x[..., :ro_dim], interleaved) * sin,
            x[..., ro_dim:],
        ],
        dim=-1,
    )


class RotaryEmbedding(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        base=10000.0,
        interleaved=False,
        scale_base=None,
        scaling_factor=1.0,
        pos_idx_in_fp32=True,
        device=None,
    ):
        super().__init__()
        self.dim = dim
        self.base = float(base)
        self.pos_idx_in_fp32 = pos_idx_in_fp32
        self.interleaved = interleaved
        self.scale_base = scale_base
        self.scaling_factor = scaling_factor
        self.device = device

        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
        self._cos_k_cached = None
        self._sin_k_cached = None
        self.reset_parameters()

    def reset_parameters(self):
        inv_freq = self._compute_inv_freq(self.device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        arange = torch.arange(0, self.dim, 2, device=self.device, dtype=torch.float32)
        scale = (
            (arange + 0.4 * self.dim) / (1.4 * self.dim)
            if self.scale_base is not None
            else None
        )
        self.register_buffer("scale", scale)

    def _compute_inv_freq(self, device=None):
        return 1 / (
            self.base
            ** (
                torch.arange(0, self.dim, 2, device=device, dtype=torch.float32)
                / self.dim
            )
        )

    def _update_cos_sin_cache(self, positions, device=None, dtype=None):
        """
        positions: 1D tensor of shape (seqlen,) giving absolute positions
        """
        seqlen = positions.shape[0]
        if self.pos_idx_in_fp32:
            t = positions.to(dtype=torch.float32, device=device) / self.scaling_factor
            if self.inv_freq.dtype != torch.float32:
                inv_freq = self.inv_freq.to(torch.float32)
            else:
                inv_freq = self.inv_freq
        else:
            t = positions.to(dtype=self.inv_freq.dtype, device=device) / self.scaling_factor
            inv_freq = self.inv_freq

        freqs = torch.outer(t, inv_freq)  # (seqlen, dim/2)

        if self.scale is None:
            self._cos_cached = torch.cos(freqs).to(dtype)
            self._sin_cached = torch.sin(freqs).to(dtype)
        else:
            power = (positions - seqlen // 2) / self.scale_base
            scale = self.scale.to(device=power.device) ** power.unsqueeze(-1)
            self._cos_cached = (torch.cos(freqs) * scale).to(dtype)
            self._sin_cached = (torch.sin(freqs) * scale).to(dtype)
            self._cos_k_cached = (torch.cos(freqs) / scale).to(dtype)
            self._sin_k_cached = (torch.sin(freqs) / scale).to(dtype)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        q: (batch, seqlen, nheads, headdim)
        k: (batch, seqlen, nheads, headdim)
        positions: optional LongTensor of shape (seqlen,)
                   If None, defaults to [0,1,2,...]
        """
        seqlen = q.shape[1]
        if positions is None:
            positions = torch.arange(seqlen, device=q.device, dtype=torch.long)

        self._update_cos_sin_cache(positions, device=q.device, dtype=q.dtype)
        assert self._cos_cached is not None
        assert self._sin_cached is not None

        return (
            apply_rotary_emb_torch(
                q,
                self._cos_cached,
                self._sin_cached,
                self.interleaved,
                True,
            ),
            apply_rotary_emb_torch(
                k,
                self._cos_cached,
                self._sin_cached,
                self.interleaved,
                True,
            ),
        )
