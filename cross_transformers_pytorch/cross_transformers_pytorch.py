import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange

class CrossTransformer(nn.Module):
    def __init__(
        self,
        dim = 512,
        dim_key = 128,
        dim_value = 128
    ):
        super().__init__()
        self.to_qk = nn.Conv2d(dim, dim_key, 1, bias = False)
        self.to_v = nn.Conv2d(dim, dim_value, 1, bias = False)

    def forward(self, model, img_query, img_supports):
        b, *_ = img_supports.shape

        query_repr = model(img_query)
        *_, h, w = query_repr.shape

        img_supports = rearrange(img_supports, 'b n c h w -> (b n) c h w', b = b)
        supports_repr = model(img_supports)

        query_q, query_v = self.to_qk(query_repr), self.to_v(query_repr)

        supports_k, supports_v = self.to_qk(supports_repr), self.to_v(supports_repr)
        supports_k, supports_v = map(lambda t: rearrange(t, '(b n) c h w -> b n c h w', b = b), (supports_k, supports_v))

        sim = einsum('b c h w, b n c i j -> b h w n i j', query_q, supports_k)
        sim = rearrange(sim, 'b h w n i j -> b h w (n i j)')

        attn = sim.softmax(dim = -1)
        attn = rearrange(attn, 'b h w (n i j) -> b h w n i j', i = h, j = w)

        out = einsum('b h w n i j, b n c i j -> b c h w', attn, supports_v)

        euclidian_dist = ((query_v - out) ** 2).sum() / (h * w)
        return euclidian_dist
