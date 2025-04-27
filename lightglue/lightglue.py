import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

try:
    # from flash_attn.module.mha import FlashCrossAttention
    from flash_attn.modules.mha import FlashCrossAttention
except ModuleNotFoundError:
    FlashCrossAttention = None

if FlashCrossAttention or hasattr(F, "scaled_dot_product_attention"):
    FLASH_AVAILABLE = True
else:
    FLASH_AVAILABLE = False

torch.backends.cudnn.deterministic = True


@torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
def normalize_keypoints(
    kpts: torch.Tensor, size: Optional[torch.Tensor] = None
) -> torch.Tensor:
    if size is None:
        size = 1 + kpts.max(-2).values - kpts.min(-2).values
    elif not isinstance(size, torch.Tensor):
        size = torch.tensor(size, device=kpts.device, dtype=kpts.dtype)
    size = size.to(kpts)
    shift = size / 2
    scale = size.max(-1).values / 2
    kpts = (kpts - shift[..., None, :]) / scale[..., None, None]
    return kpts


def pad_to_length(x: torch.Tensor, length: int) -> Tuple[torch.Tensor]:
    if length <= x.shape[-2]:
        return x, torch.ones_like(x[..., :1], dtype=torch.bool)
    pad = torch.ones(
        *x.shape[:-2], length - x.shape[-2], x.shape[-1], device=x.device, dtype=x.dtype
    )
    y = torch.cat([x, pad], dim=-2)
    mask = torch.zeros(*y.shape[:-1], 1, dtype=torch.bool, device=x.device)
    mask[..., : x.shape[-2], :] = True
    return y, mask


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x = x.unflatten(-1, (-1, 2))
    x1, x2 = x.unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)


def apply_cached_rotary_emb(freqs: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return (t * freqs[0]) + (rotate_half(t) * freqs[1])


class LearnableFourierPositionalEncoding(nn.Module):
    def __init__(self, M: int, dim: int, F_dim: Optional[int] = None, gamma: float = 1.0):
        super().__init__()
        F_dim = F_dim if F_dim is not None else dim
        self.gamma = gamma
        self.F_dim = F_dim

        self.Wr = nn.Linear(M, F_dim // 2, bias=False)
        nn.init.normal_(self.Wr.weight.data, mean=0.0, std=self.gamma ** -2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.Wr(x)  # (N, D/2)

        cosines = torch.cos(projected)
        sines = torch.sin(projected)
        emb = torch.stack([cosines, sines], 0).unsqueeze(-3)
        return emb.repeat_interleave(2, dim=-1)



class TokenConfidence(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.token = nn.Sequential(nn.Linear(dim, 1), nn.Sigmoid())

    def forward(self, desc0: torch.Tensor, desc1: torch.Tensor):
        """get confidence tokens"""
        return (
            self.token(desc0.detach()).squeeze(-1),
            self.token(desc1.detach()).squeeze(-1),
        )


class Attention(nn.Module):
    def __init__(self, allow_flash: bool) -> None:
        super().__init__()
        if allow_flash and not FLASH_AVAILABLE:
            warnings.warn(
                "FlashAttention is not available. For optimal speed, "
                "consider installing torch >= 2.0 or flash-attn.",
                stacklevel=2,
            )
        self.enable_flash = allow_flash and FLASH_AVAILABLE
        self.has_sdp = hasattr(F, "scaled_dot_product_attention")
        if allow_flash and FlashCrossAttention:
            self.flash_ = FlashCrossAttention()
        if self.has_sdp:
            torch.backends.cuda.enable_flash_sdp(allow_flash)

    def forward(self, q, k, v, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        
        v = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return v if mask is None else v.nan_to_num()


class SelfBlock(nn.Module):
    def __init__(
        self, embed_dim: int, num_heads: int, flash: bool = False, bias: bool = True
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert self.embed_dim % num_heads == 0
        self.head_dim = self.embed_dim // num_heads
        self.Wqkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.inner_attn = Attention(flash)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.ffn = nn.Sequential(
            nn.Linear(2 * embed_dim, 2 * embed_dim),
            nn.LayerNorm(2 * embed_dim, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2 * embed_dim, embed_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        encoding: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        qkv = self.Wqkv(x)
        qkv = qkv.unflatten(-1, (self.num_heads, -1, 3)).transpose(1, 2)
        q, k, v = qkv[..., 0], qkv[..., 1], qkv[..., 2]
        q = apply_cached_rotary_emb(encoding, q)
        k = apply_cached_rotary_emb(encoding, k)
        context = self.inner_attn(q, k, v, mask=mask)
        message = self.out_proj(context.transpose(1, 2).flatten(start_dim=-2))
        return x + self.ffn(torch.cat([x, message], -1))


class CrossBlock(nn.Module):
    def __init__(
        self, embed_dim: int, num_heads: int, flash: bool = False, bias: bool = True
    ) -> None:
        super().__init__()
        self.heads = num_heads
        dim_head = embed_dim // num_heads
        self.scale = dim_head**-0.5
        inner_dim = dim_head * num_heads
        self.to_qk = nn.Linear(embed_dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(embed_dim, inner_dim, bias=bias)
        self.to_out = nn.Linear(inner_dim, embed_dim, bias=bias)
        self.ffn = nn.Sequential(
            nn.Linear(2 * embed_dim, 2 * embed_dim),
            nn.LayerNorm(2 * embed_dim, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2 * embed_dim, embed_dim),
        )
        if flash and FLASH_AVAILABLE:
            self.flash = Attention(True)
        else:
            self.flash = None

    def unflatten(self, x: torch.Tensor) -> torch.Tensor:
        return x.unflatten(-1, (self.heads, -1)).transpose(1, 2)
    
    def flatten(self, x: torch.Tensor) -> torch.Tensor:
        return x.transpose(1, 2).flatten(start_dim=-2)

    def forward(
        self, x0: torch.Tensor, x1: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> List[torch.Tensor]:
        qk0 = self.unflatten(self.to_qk(x0))
        qk1 = self.unflatten(self.to_qk(x1))
        v0 = self.unflatten(self.to_v(x0))
        v1 = self.unflatten(self.to_v(x1))

        if self.flash is not None and qk0.device.type == "cuda":
            m0 = self.flash(qk0, qk1, v1, mask)
            m1 = self.flash(
                qk1, qk0, v0, mask.transpose(-1, -2) if mask is not None else None
            )
        else:
            qk0, qk1 = qk0 * self.scale**0.5, qk1 * self.scale**0.5
            sim = torch.einsum("bhid, bhjd -> bhij", qk0, qk1)
            if mask is not None:
                sim = sim.masked_fill(~mask, -float("inf"))
            attn01 = F.softmax(sim, dim=-1)
            attn10 = F.softmax(sim.transpose(-2, -1).contiguous(), dim=-1)
            m0 = torch.einsum("bhij, bhjd -> bhid", attn01, v1)
            m1 = torch.einsum("bhji, bhjd -> bhid", attn10.transpose(-2, -1), v0)
            if mask is not None:
                m0, m1 = m0.nan_to_num(), m1.nan_to_num()
        m0 = self.to_out(self.flatten(m0))
        m1 = self.to_out(self.flatten(m1))
        x0 = x0 + self.ffn(torch.cat([x0, m0], -1))
        x1 = x1 + self.ffn(torch.cat([x1, m1], -1))
        return x0, x1



class TransformerLayer(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.self_attn = SelfBlock(*args, **kwargs)
        self.cross_attn = CrossBlock(*args, **kwargs)

    def forward(
        self,
        desc0,
        desc1,
        encoding0,
        encoding1,
        mask0: Optional[torch.Tensor] = None,
        mask1: Optional[torch.Tensor] = None,
    ):
        if mask0 is not None and mask1 is not None:
            return self.masked_forward(desc0, desc1, encoding0, encoding1, mask0, mask1)
        else:
            desc0 = self.self_attn(desc0, encoding0)
            desc1 = self.self_attn(desc1, encoding1)
            return self.cross_attn(desc0, desc1)

    # This part is compiled and allows padding inputs
    def masked_forward(self, desc0, desc1, encoding0, encoding1, mask0, mask1):
        mask = mask0 & mask1.transpose(-1, -2)
        mask0 = mask0 & mask0.transpose(-1, -2)
        mask1 = mask1 & mask1.transpose(-1, -2)
        desc0 = self.self_attn(desc0, encoding0, mask0)
        desc1 = self.self_attn(desc1, encoding1, mask1)
        return self.cross_attn(desc0, desc1, mask)


def sigmoid_log_double_softmax(
    sim: torch.Tensor, z0: torch.Tensor, z1: torch.Tensor
) -> torch.Tensor:
    """create the log assignment matrix from logits and similarity"""
    b, m, n = sim.shape
    certainties = F.logsigmoid(z0) + F.logsigmoid(z1).transpose(1, 2)
    scores0 = F.log_softmax(sim, 2)
    scores1 = F.log_softmax(sim.transpose(-1, -2).contiguous(), 2).transpose(-1, -2)
    scores = sim.new_full((b, m + 1, n + 1), 0)
    scores[:, :m, :n] = scores0 + scores1 + certainties
    scores[:, :-1, -1] = F.logsigmoid(-z0.squeeze(-1))
    scores[:, -1, :-1] = F.logsigmoid(-z1.squeeze(-1))
    return scores


class MatchAssignment(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.matchability = nn.Linear(dim, 1, bias=True)
        self.final_proj = nn.Linear(dim, dim, bias=True)

    def forward(self, desc0: torch.Tensor, desc1: torch.Tensor):
        """build assignment matrix from descriptors"""
        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)
        _, _, d = mdesc0.shape
        mdesc0, mdesc1 = mdesc0 / d**0.25, mdesc1 / d**0.25
        sim = torch.einsum("bmd,bnd->bmn", mdesc0, mdesc1)
        z0 = self.matchability(desc0)
        z1 = self.matchability(desc1)
        scores = sigmoid_log_double_softmax(sim, z0, z1)
        return scores, sim

    def get_matchability(self, desc: torch.Tensor):
        return torch.sigmoid(self.matchability(desc)).squeeze(-1)


def filter_matches(scores: torch.Tensor, th: float):
    """obtain matches from a log assignment matrix [Bx M+1 x N+1]"""
    max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
    m0, m1 = max0.indices, max1.indices
    indices0 = torch.arange(m0.shape[1], device=m0.device)[None]
    indices1 = torch.arange(m1.shape[1], device=m1.device)[None]
    mutual0 = indices0 == m1.gather(1, m0)
    mutual1 = indices1 == m0.gather(1, m1)
    max0_exp = max0.values.exp()
    zero = torch.zeros_like(max0_exp)
    mscores0 = torch.where(mutual0, max0_exp, zero)
    mscores1 = torch.where(mutual1, mscores0.gather(1, m1), zero)
    valid0 = mutual0 & (mscores0 > th)
    valid1 = mutual1 & valid0.gather(1, m1)
    m0 = torch.where(valid0, m0, -1)
    m1 = torch.where(valid1, m1, -1)
    return m0, m1, mscores0, mscores1


class LightGlue(nn.Module):
    version = "v0.1_arxiv"
    url = "https://github.com/cvg/LightGlue/releases/download/{}/{}_lightglue.pth"

    def __init__(self, features="superpoint", **conf) -> None:
        super().__init__()
        # self.conf = conf = {**self.default_conf, **conf}
        self.flash_available = FLASH_AVAILABLE
        self.pruning_keypoint_thresholds = {
            "cpu": -1,
            "mps": -1,
            "cuda": 1024,
            "flash": 1536,
        }

        self.input_proj = nn.Identity()

        head_dim = 256 // 4
        self.posenc = LearnableFourierPositionalEncoding(2, head_dim, head_dim)

        h, n, d = 4, 9, 256

        self.transformers = nn.ModuleList(
            [TransformerLayer(d, h, True) for _ in range(n)]
        )

        self.log_assignment = nn.ModuleList([MatchAssignment(d) for _ in range(n)])
        self.token_confidence = nn.ModuleList(
            [TokenConfidence(d) for _ in range(n - 1)]
        )
        self.register_buffer(
            "confidence_thresholds",
            torch.Tensor(
                [self.confidence_threshold(i) for i in range(9)]
            ),
        )

        state_dict = None
        if features is not None:
            fname = "superpoint_lightglue_v0-1_arxiv.pth"
            state_dict = torch.hub.load_state_dict_from_url(
                self.url.format(self.version, features), file_name=fname
            )
            self.load_state_dict(state_dict, strict=False)

        if state_dict:
            # rename old state dict entries
            for i in range(9):
                pattern = f"self_attn.{i}", f"transformers.{i}.self_attn"
                state_dict = {k.replace(*pattern): v for k, v in state_dict.items()}
                pattern = f"cross_attn.{i}", f"transformers.{i}.cross_attn"
                state_dict = {k.replace(*pattern): v for k, v in state_dict.items()}
            self.load_state_dict(state_dict, strict=False)

        # static lengths LightGlue is compiled for (only used with torch.compile)
        self.static_lengths = None

    @torch.jit.unused
    def transformer_loop(
                self,
                desc0, desc1, encoding0, encoding1,
                mask0, mask1, m:int, n:int,
                pruning_th:int, do_early_stop:bool,
                ind0, ind1, prune0, prune1):

        token0, token1 = None, None
        i = 0
        # for i, transformer in enumerate(self.transformers):
        num_layers = len(self.transformers)
        for i in range(num_layers):
            transformer = self.transformers[i]
            if desc0.shape[1] == 0 or desc1.shape[1] == 0:
                return i, desc0, desc1, token0, token1, i, ind0, ind1, prune0, prune1

            desc0, desc1 = transformer(desc0, desc1, encoding0, encoding1, mask0=mask0, mask1=mask1)

            if i != 8:
                if do_early_stop:
                    token0, token1 = self.token_confidence[i](desc0, desc1)
                    if self.check_if_stop(token0[..., :m], token1[..., :n], i, m + n):
                        return i, desc0, desc1, token0, token1, i, ind0, ind1, prune0, prune1

                if desc0.shape[-2] > pruning_th:
                    scores0 = self.log_assignment[i].get_matchability(desc0)
                    prunemask0 = self.get_pruning_mask(token0, scores0, i)
                    keep0 = torch.where(prunemask0)[1]
                    ind0 = ind0.index_select(1, keep0)
                    desc0 = desc0.index_select(1, keep0)
                    encoding0 = encoding0.index_select(-2, keep0)
                    prune0[:, ind0] += 1

                if desc1.shape[-2] > pruning_th:
                    scores1 = self.log_assignment[i].get_matchability(desc1)
                    prunemask1 = self.get_pruning_mask(token1, scores1, i)
                    keep1 = torch.where(prunemask1)[1]
                    ind1 = ind1.index_select(1, keep1)
                    desc1 = desc1.index_select(1, keep1)
                    encoding1 = encoding1.index_select(-2, keep1)
                    prune1[:, ind1] += 1

        return i, desc0, desc1, token0, token1, len(self.transformers) - 1, ind0, ind1, prune0, prune1


    def forward(self, kpts0, desc0, kpts1, desc1):
        with torch.autocast(enabled=False, device_type="cuda"):
            b, m, _ = kpts0.shape
            b, n, _ = kpts1.shape
            device = kpts0.device

            mask0 = torch.ones((b, m), dtype=torch.bool, device=device)
            mask1 = torch.ones((b, n), dtype=torch.bool, device=device)
            c = max(m, n)

            desc0 = self.input_proj(desc0)
            desc1 = self.input_proj(desc1)
            # cache positional embeddings
            encoding0 = self.posenc(kpts0)
            encoding1 = self.posenc(kpts1)

            # GNN + final_proj + assignment
            do_early_stop = True
            pruning_th = 1536 # self.pruning_min_kpts(device)

            # We store the index of the layer at which pruning is detected.
            ind0 = torch.arange(0, m, device=device)[None]
            ind1 = torch.arange(0, n, device=device)[None]
            prune0 = torch.ones_like(ind0)
            prune1 = torch.ones_like(ind1)

            # result = self.transformer_loop(desc0, desc1, encoding0, encoding1, mask0, mask1, m, n, pruning_th, do_early_stop, ind0, ind1, prune0, prune1)
            """
            BEGIN transform_loop
            """
            token0 = torch.zeros(1, 512)
            token1 = torch.zeros(1, 512)
            i = 0
            last_i = 0
            
            # transformer = self.transformers[0]
            if desc0.shape[1] == 0 or desc1.shape[1] == 0:
                i = 0
                desc0 = desc0
                desc1 = desc1
                token0 = token0
                token1 = token1
                last_i = 0
                ind0 = ind0
                ind1 = ind1
                prune0 = prune0
                prune1 = prune1
            else:
                desc0, desc1 = self.transformers[0](desc0, desc1, encoding0, encoding1, mask0=mask0, mask1=mask1)
                token0, token1 = self.token_confidence[0](desc0, desc1)
                if self.check_if_stop(token0[..., :m], token1[..., :n], i, m + n):
                    i = 0
                    desc0 = desc0
                    desc1 = desc1
                    token0 = token0
                    token1 = token1
                    last_i = 0
                    ind0 = ind0
                    ind1 = ind1
                    prune0 = prune0
                    prune1 = prune1
                else:
                    if desc0.shape[-2] > pruning_th:
                        scores0 = self.log_assignment[0].get_matchability(desc0)
                        prunemask0 = self.get_pruning_mask(token0, scores0, 0)
                        keep0 = torch.where(prunemask0)[1]
                        ind0 = ind0.index_select(1, keep0)
                        desc0 = desc0.index_select(1, keep0)
                        encoding0 = encoding0.index_select(-2, keep0)
                        prune0[:, ind0] += 1
                    if desc1.shape[-2] > pruning_th:
                        scores1 = self.log_assignment[0].get_matchability(desc1)
                        prunemask1 = self.get_pruning_mask(token1, scores1, 0)
                        keep1 = torch.where(prunemask1)[1]
                        ind1 = ind1.index_select(1, keep1)
                        desc1 = desc1.index_select(1, keep1)
                        encoding1 = encoding1.index_select(-2, keep1)
                        prune1[:, ind1] += 1

                    # transformer = self.transformers[1]
                    if desc0.shape[1] == 0 or desc1.shape[1] == 0:
                        i = 1
                        desc0 = desc0
                        desc1 = desc1
                        token0 = token0
                        token1 = token1
                        last_i = 1
                        ind0 = ind0
                        ind1 = ind1
                        prune0 = prune0
                        prune1 = prune1
                    else:
                        desc0, desc1 = self.transformers[1](desc0, desc1, encoding0, encoding1, mask0=mask0, mask1=mask1)
                        token0, token1 = self.token_confidence[1](desc0, desc1)
                        if self.check_if_stop(token0[..., :m], token1[..., :n], i, m + n):
                            i = 1
                            desc0 = desc0
                            desc1 = desc1
                            token0 = token0
                            token1 = token1
                            last_i = 1
                            ind0 = ind0
                            ind1 = ind1
                            prune0 = prune0
                            prune1 = prune1
                        else:
                            if desc0.shape[-2] > pruning_th:
                                scores0 = self.log_assignment[1].get_matchability(desc0)
                                prunemask0 = self.get_pruning_mask(token0, scores0, 1)
                                keep0 = torch.where(prunemask0)[1]
                                ind0 = ind0.index_select(1, keep0)
                                desc0 = desc0.index_select(1, keep0)
                                encoding0 = encoding0.index_select(-2, keep0)
                                prune0[:, ind0] += 1
                            if desc1.shape[-2] > pruning_th:
                                scores1 = self.log_assignment[1].get_matchability(desc1)
                                prunemask1 = self.get_pruning_mask(token1, scores1, 1)
                                keep1 = torch.where(prunemask1)[1]
                                ind1 = ind1.index_select(1, keep1)
                                desc1 = desc1.index_select(1, keep1)
                                encoding1 = encoding1.index_select(-2, keep1)
                                prune1[:, ind1] += 1

                            # transformer = self.transformers[2]
                            if desc0.shape[1] == 0 or desc1.shape[1] == 0:
                                i = 2
                                desc0 = desc0
                                desc1 = desc1
                                token0 = token0
                                token1 = token1
                                last_i = 2
                                ind0 = ind0
                                ind1 = ind1
                                prune0 = prune0
                                prune1 = prune1
                            else:
                                desc0, desc1 = self.transformers[2](desc0, desc1, encoding0, encoding1, mask0=mask0, mask1=mask1)
                                token0, token1 = self.token_confidence[2](desc0, desc1)
                                if self.check_if_stop(token0[..., :m], token1[..., :n], i, m + n):
                                    i = 2
                                    desc0 = desc0
                                    desc1 = desc1
                                    token0 = token0
                                    token1 = token1
                                    last_i = 2
                                    ind0 = ind0
                                    ind1 = ind1
                                    prune0 = prune0
                                    prune1 = prune1
                                else:
                                    if desc0.shape[-2] > pruning_th:
                                        scores0 = self.log_assignment[2].get_matchability(desc0)
                                        prunemask0 = self.get_pruning_mask(token0, scores0, 2)
                                        keep0 = torch.where(prunemask0)[2]
                                        ind0 = ind0.index_select(1, keep0)
                                        desc0 = desc0.index_select(1, keep0)
                                        encoding0 = encoding0.index_select(-2, keep0)
                                        prune0[:, ind0] += 1
                                    if desc1.shape[-2] > pruning_th:
                                        scores1 = self.log_assignment[2].get_matchability(desc1)
                                        prunemask1 = self.get_pruning_mask(token1, scores1, 2)
                                        keep1 = torch.where(prunemask1)[1]
                                        ind1 = ind1.index_select(1, keep1)
                                        desc1 = desc1.index_select(1, keep1)
                                        encoding1 = encoding1.index_select(-2, keep1)
                                        prune1[:, ind1] += 1
                                            
                                    # transformer = self.transformers[3]
                                    if desc0.shape[1] == 0 or desc1.shape[1] == 0:
                                        i = 3
                                        desc0 = desc0
                                        desc1 = desc1
                                        token0 = token0
                                        token1 = token1
                                        last_i = 3
                                        ind0 = ind0
                                        ind1 = ind1
                                        prune0 = prune0
                                        prune1 = prune1
                                    else:
                                        desc0, desc1 = self.transformers[3](desc0, desc1, encoding0, encoding1, mask0=mask0, mask1=mask1)
                                        token0, token1 = self.token_confidence[3](desc0, desc1)
                                        if self.check_if_stop(token0[..., :m], token1[..., :n], i, m + n):
                                            i = 3
                                            desc0 = desc0
                                            desc1 = desc1
                                            token0 = token0
                                            token1 = token1
                                            last_i = 3
                                            ind0 = ind0
                                            ind1 = ind1
                                            prune0 = prune0
                                            prune1 = prune1
                                        else:
                                            if desc0.shape[-2] > pruning_th:
                                                scores0 = self.log_assignment[3].get_matchability(desc0)
                                                prunemask0 = self.get_pruning_mask(token0, scores0, 3)
                                                keep0 = torch.where(prunemask0)[1]
                                                ind0 = ind0.index_select(1, keep0)
                                                desc0 = desc0.index_select(1, keep0)
                                                encoding0 = encoding0.index_select(-2, keep0)
                                                prune0[:, ind0] += 1
                                            if desc1.shape[-2] > pruning_th:
                                                scores1 = self.log_assignment[3].get_matchability(desc1)
                                                prunemask1 = self.get_pruning_mask(token1, scores1, 3)
                                                keep1 = torch.where(prunemask1)[1]
                                                ind1 = ind1.index_select(1, keep1)
                                                desc1 = desc1.index_select(1, keep1)
                                                encoding1 = encoding1.index_select(-2, keep1)
                                                prune1[:, ind1] += 1

                                            # transformer = self.transformers[4]
                                            if desc0.shape[1] == 0 or desc1.shape[1] == 0:
                                                i = 4
                                                desc0 = desc0
                                                desc1 = desc1
                                                token0 = token0
                                                token1 = token1
                                                last_i = 4
                                                ind0 = ind0
                                                ind1 = ind1
                                                prune0 = prune0
                                                prune1 = prune1
                                            else:
                                                desc0, desc1 = self.transformers[4](desc0, desc1, encoding0, encoding1, mask0=mask0, mask1=mask1)
                                                token0, token1 = self.token_confidence[4](desc0, desc1)
                                                if self.check_if_stop(token0[..., :m], token1[..., :n], i, m + n):
                                                    i = 4
                                                    desc0 = desc0
                                                    desc1 = desc1
                                                    token0 = token0
                                                    token1 = token1
                                                    last_i = 4
                                                    ind0 = ind0
                                                    ind1 = ind1
                                                    prune0 = prune0
                                                    prune1 = prune1
                                                else:
                                                    if desc0.shape[-2] > pruning_th:
                                                        scores0 = self.log_assignment[4].get_matchability(desc0)
                                                        prunemask0 = self.get_pruning_mask(token0, scores0, 4)
                                                        keep0 = torch.where(prunemask0)[1]
                                                        ind0 = ind0.index_select(1, keep0)
                                                        desc0 = desc0.index_select(1, keep0)
                                                        encoding0 = encoding0.index_select(-2, keep0)
                                                        prune0[:, ind0] += 1
                                                    if desc1.shape[-2] > pruning_th:
                                                        scores1 = self.log_assignment[4].get_matchability(desc1)
                                                        prunemask1 = self.get_pruning_mask(token1, scores1, 4)
                                                        keep1 = torch.where(prunemask1)[1]
                                                        ind1 = ind1.index_select(1, keep1)
                                                        desc1 = desc1.index_select(1, keep1)
                                                        encoding1 = encoding1.index_select(-2, keep1)
                                                        prune1[:, ind1] += 1
                                                        
                                                    # transformer = self.transformers[5]
                                                    if desc0.shape[1] == 0 or desc1.shape[1] == 0:
                                                        i = 5
                                                        desc0 = desc0
                                                        desc1 = desc1
                                                        token0 = token0
                                                        token1 = token1
                                                        last_i = 5
                                                        ind0 = ind0
                                                        ind1 = ind1
                                                        prune0 = prune0
                                                        prune1 = prune1
                                                    else:
                                                        desc0, desc1 = self.transformers[5](desc0, desc1, encoding0, encoding1, mask0=mask0, mask1=mask1)
                                                        token0, token1 = self.token_confidence[5](desc0, desc1)
                                                        if self.check_if_stop(token0[..., :m], token1[..., :n], i, m + n):
                                                            i = 5
                                                            desc0 = desc0
                                                            desc1 = desc1
                                                            token0 = token0
                                                            token1 = token1
                                                            last_i = 5
                                                            ind0 = ind0
                                                            ind1 = ind1
                                                            prune0 = prune0
                                                            prune1 = prune1
                                                        else:
                                                            if desc0.shape[-2] > pruning_th:
                                                                scores0 = self.log_assignment[5].get_matchability(desc0)
                                                                prunemask0 = self.get_pruning_mask(token0, scores0, 5)
                                                                keep0 = torch.where(prunemask0)[1]
                                                                ind0 = ind0.index_select(1, keep0)
                                                                desc0 = desc0.index_select(1, keep0)
                                                                encoding0 = encoding0.index_select(-2, keep0)
                                                                prune0[:, ind0] += 1
                                                            if desc1.shape[-2] > pruning_th:
                                                                scores1 = self.log_assignment[5].get_matchability(desc1)
                                                                prunemask1 = self.get_pruning_mask(token1, scores1, 5)
                                                                keep1 = torch.where(prunemask1)[1]
                                                                ind1 = ind1.index_select(1, keep1)
                                                                desc1 = desc1.index_select(1, keep1)
                                                                encoding1 = encoding1.index_select(-2, keep1)
                                                                prune1[:, ind1] += 1

                                                            # transformer = self.transformers[6]
                                                            if desc0.shape[1] == 0 or desc1.shape[1] == 0:
                                                                i = 6
                                                                desc0 = desc0
                                                                desc1 = desc1
                                                                token0 = token0
                                                                token1 = token1
                                                                last_i = 6
                                                                ind0 = ind0
                                                                ind1 = ind1
                                                                prune0 = prune0
                                                                prune1 = prune1
                                                            else:
                                                                desc0, desc1 = self.transformers[6](desc0, desc1, encoding0, encoding1, mask0=mask0, mask1=mask1)
                                                                token0, token1 = self.token_confidence[6](desc0, desc1)
                                                                if self.check_if_stop(token0[..., :m], token1[..., :n], i, m + n):
                                                                    i = 6
                                                                    desc0 = desc0
                                                                    desc1 = desc1
                                                                    token0 = token0
                                                                    token1 = token1
                                                                    last_i = 6
                                                                    ind0 = ind0
                                                                    ind1 = ind1
                                                                    prune0 = prune0
                                                                    prune1 = prune1
                                                                else:
                                                                    if desc0.shape[-2] > pruning_th:
                                                                        scores0 = self.log_assignment[6].get_matchability(desc0)
                                                                        prunemask0 = self.get_pruning_mask(token0, scores0, 6)
                                                                        keep0 = torch.where(prunemask0)[1]
                                                                        ind0 = ind0.index_select(1, keep0)
                                                                        desc0 = desc0.index_select(1, keep0)
                                                                        encoding0 = encoding0.index_select(-2, keep0)
                                                                        prune0[:, ind0] += 1
                                                                    if desc1.shape[-2] > pruning_th:
                                                                        scores1 = self.log_assignment[6].get_matchability(desc1)
                                                                        prunemask1 = self.get_pruning_mask(token1, scores1, 6)
                                                                        keep1 = torch.where(prunemask1)[1]
                                                                        ind1 = ind1.index_select(1, keep1)
                                                                        desc1 = desc1.index_select(1, keep1)
                                                                        encoding1 = encoding1.index_select(-2, keep1)
                                                                        prune1[:, ind1] += 1

                                                                    # transformer = self.transformers[7]
                                                                    if desc0.shape[1] == 0 or desc1.shape[1] == 0:
                                                                        i = 7
                                                                        desc0 = desc0
                                                                        desc1 = desc1
                                                                        token0 = token0
                                                                        token1 = token1
                                                                        last_i = 7
                                                                        ind0 = ind0
                                                                        ind1 = ind1
                                                                        prune0 = prune0
                                                                        prune1 = prune1
                                                                    else:
                                                                        desc0, desc1 = self.transformers[7](desc0, desc1, encoding0, encoding1, mask0=mask0, mask1=mask1)
                                                                        token0, token1 = self.token_confidence[7](desc0, desc1)
                                                                        if self.check_if_stop(token0[..., :m], token1[..., :n], i, m + n):
                                                                            i = 7
                                                                            desc0 = desc0
                                                                            desc1 = desc1
                                                                            token0 = token0
                                                                            token1 = token1
                                                                            last_i = 7
                                                                            ind0 = ind0
                                                                            ind1 = ind1
                                                                            prune0 = prune0
                                                                            prune1 = prune1
                                                                        else:
                                                                            if desc0.shape[-2] > pruning_th:
                                                                                scores0 = self.log_assignment[7].get_matchability(desc0)
                                                                                prunemask0 = self.get_pruning_mask(token0, scores0, 7)
                                                                                keep0 = torch.where(prunemask0)[1]
                                                                                ind0 = ind0.index_select(1, keep0)
                                                                                desc0 = desc0.index_select(1, keep0)
                                                                                encoding0 = encoding0.index_select(-2, keep0)
                                                                                prune0[:, ind0] += 1
                                                                            if desc1.shape[-2] > pruning_th:
                                                                                scores1 = self.log_assignment[7].get_matchability(desc1)
                                                                                prunemask1 = self.get_pruning_mask(token1, scores1, 7)
                                                                                keep1 = torch.where(prunemask1)[1]
                                                                                ind1 = ind1.index_select(1, keep1)
                                                                                desc1 = desc1.index_select(1, keep1)
                                                                                encoding1 = encoding1.index_select(-2, keep1)
                                                                                prune1[:, ind1] += 1

                                                                            # transformer = self.transformers[8]
                                                                            if desc0.shape[1] == 0 or desc1.shape[1] == 0:
                                                                                i = 8
                                                                                desc0 = desc0
                                                                                desc1 = desc1
                                                                                token0 = token0
                                                                                token1 = token1
                                                                                last_i = 8
                                                                                ind0 = ind0
                                                                                ind1 = ind1
                                                                                prune0 = prune0
                                                                                prune1 = prune1
                                                                            else:
                                                                                desc0, desc1 = self.transformers[8](desc0, desc1, encoding0, encoding1, mask0=mask0, mask1=mask1)

            """
            END transform_loop
            """
            # i = result[0]
            # desc0 = result[1]
            # desc1 = result[2]
            # token0 = result[3]
            # token1 = result[4]
            # last_i = result[5]
            # ind0 = result[6]
            # ind1 = result[7]
            # prune0 = result[8]
            # prune1 = result[9]

            if desc0.shape[1] == 0 or desc1.shape[1] == 0:  # no keypoints
                m0 = desc0.new_full((b, m), -1, dtype=torch.long)
                m1 = desc1.new_full((b, n), -1, dtype=torch.long)
                mscores0 = desc0.new_zeros((b, m))
                mscores1 = desc1.new_zeros((b, n))
                empty_matches = desc0.new_empty((0, 2), dtype=torch.long)
                empty_scores = desc0.new_empty((0,))
                return empty_matches, empty_scores

            desc0, desc1 = desc0[..., :m, :], desc1[..., :n, :]  # remove padding
            # scores, _ = self.log_assignment[i](desc0, desc1)
            if i == 0:
                scores, _ = self.log_assignment[0](desc0, desc1)
            elif i == 1:
                scores, _ = self.log_assignment[1](desc0, desc1)
            elif i == 2:
                scores, _ = self.log_assignment[2](desc0, desc1)
            elif i == 3:
                scores, _ = self.log_assignment[3](desc0, desc1)
            elif i == 4:
                scores, _ = self.log_assignment[4](desc0, desc1)
            elif i == 5:
                scores, _ = self.log_assignment[5](desc0, desc1)
            elif i == 6:
                scores, _ = self.log_assignment[6](desc0, desc1)
            elif i == 7:
                scores, _ = self.log_assignment[7](desc0, desc1)
            else:
                scores, _ = self.log_assignment[8](desc0, desc1)

            match_filtered = filter_matches(scores, 0.1)
            m0 = match_filtered[0]
            m1 = match_filtered[1]
            mscores0 = match_filtered[2]
            mscores1 = match_filtered[3]
            # matches, mscores = [], []
            # for k in range(b):
            valid = m0[0] > -1
            m_indices_0 = torch.where(valid)[0]
            m_indices_1 = m0[0][valid]
            m_indices_0 = ind0[0, m_indices_0]
            m_indices_1 = ind1[0, m_indices_1]
            matches = torch.stack([m_indices_0, m_indices_1], -1)
            mscores = mscores0[0][valid]

            # # TODO: Remove when hloc switches to the compact format.
            # m0_ = torch.full((b, m), -1, device=m0.device, dtype=m0.dtype)
            # m1_ = torch.full((b, n), -1, device=m1.device, dtype=m1.dtype)
            # m0_[:, ind0] = torch.where(m0 == -1, -1, ind1.gather(1, m0.clamp(min=0)))
            # m1_[:, ind1] = torch.where(m1 == -1, -1, ind0.gather(1, m1.clamp(min=0)))
            # mscores0_ = torch.zeros((b, m), device=mscores0.device)
            # mscores1_ = torch.zeros((b, n), device=mscores1.device)
            # mscores0_[:, ind0] = mscores0
            # mscores1_[:, ind1] = mscores1
            # m0, m1, mscores0, mscores1 = m0_, m1_, mscores0_, mscores1_

            return matches, mscores

    def confidence_threshold(self, layer_index: int) -> float:
        """scaled confidence threshold"""
        threshold = 0.8 + 0.1 * np.exp(-4.0 * layer_index / 9)
        return np.clip(threshold, 0, 1)

    def get_pruning_mask(
        self, confidences: torch.Tensor, scores: torch.Tensor, layer_index: int
    ) -> torch.Tensor:
        """mask points which should be removed"""
        keep = scores > 0.01
        if confidences is not None:  # Low-confidence points are never pruned.
            keep |= confidences <= self.confidence_thresholds[layer_index]
        return keep

    def check_if_stop(
        self,
        confidences0: torch.Tensor,
        confidences1: torch.Tensor,
        layer_index: int,
        num_points: int,
    ) -> torch.Tensor:
        """evaluate stopping condition"""
        confidences = torch.cat([confidences0, confidences1], -1)
        threshold = self.confidence_thresholds[layer_index]
        ratio_confident = 1.0 - (confidences < threshold).float().sum() / num_points
        return ratio_confident > 0.95

    def pruning_min_kpts(self, device: torch.device):
        return 1536
        # if self.flash_available and device.type == "cuda":
        #     return self.pruning_keypoint_thresholds["flash"]
        # else:
        #     return self.pruning_keypoint_thresholds[device.type]
