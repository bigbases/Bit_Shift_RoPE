from typing import Optional

from pathlib import Path
import torch
from torch import nn
from spikingjelly.activation_based import surrogate, neuron, functional

from ..base import NETWORKS
from ...module.positional_encoding import PositionEmbedding
from ...module.spike_encoding import SpikeEncoder
from ...module.spike_attention import Block1D

tau = 2.0  # beta = 1 - 1/tau
backend = "torch"
detach_reset = True


class CyclicShiftRoPE1D(nn.Module):
    """Fixed frequency-decay cyclic shift via a single cached ``gather``.
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        num_groups: int = 4,
        base: float = 64.0,  
        max_len: int | None = None,  
    ):
        super().__init__()
        assert head_dim % num_groups == 0, "head_dim must be divisible by num_groups."
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_groups = num_groups
        self.group_dim = head_dim // num_groups
        self.base = base

    
        if max_len is not None and max_len > 0:
            idx = self._build_index(max_len, device=torch.device("cpu"))
        else:
            idx = torch.empty(0, dtype=torch.long)
        self.register_buffer("_cached_idx", idx, persistent=False)
        self._cached_len = max_len if (max_len is not None and max_len > 0) else -1

    def _build_index(self, N: int, device: torch.device) -> torch.Tensor:
        m = torch.arange(N, device=device, dtype=torch.float32)
        g = torch.arange(self.num_groups, device=device, dtype=torch.float32)

        inv_freq = self.base ** -(g / max(1, self.num_groups - 1))

        float_shifts = torch.einsum('n, g -> gn', m, inv_freq)

        int_shifts = torch.round(float_shifts).long()

        # 차원 확장: [1, 1, G, N] -> [1, H, G, N]
        int_shifts = int_shifts.view(1, 1, self.num_groups, N)
        int_shifts = int_shifts.expand(1, self.num_heads, self.num_groups, N)

        base_idx = torch.arange(self.group_dim, device=device).view(
            1, 1, 1, 1, self.group_dim
        )
        shift_expanded = int_shifts.unsqueeze(-1)
        
        shifted_group_idx = (base_idx - shift_expanded) % self.group_dim

        group_offsets = (
            torch.arange(self.num_groups, device=device) * self.group_dim
        ).view(1, 1, self.num_groups, 1, 1)
        shifted_idx = shifted_group_idx + group_offsets

        final_idx = shifted_idx.permute(0, 1, 3, 2, 4).reshape(
            1, 1, self.num_heads, N, self.head_dim
        )
        return final_idx.contiguous().long()

    def forward(self, x: torch.Tensor):
        # x: [T, B, H, N, D]
        T, B, H, N, D = x.shape

        if self._cached_len == N:
            cached = self._cached_idx
        else:
            cached = self._build_index(N, x.device)

        idx = cached.expand(T, B, H, N, D)
        return torch.gather(x, dim=-1, index=idx)

class ConvEncoder(nn.Module):
    def __init__(self, output_size: int, kernel_size: int = 3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=output_size,
                kernel_size=(1, kernel_size),
                stride=1,
                padding=(0, kernel_size // 2),
            ),
            nn.BatchNorm2d(output_size),
        )
        self.lif = neuron.LIFNode(
            tau=tau,
            step_mode="m",
            detach_reset=detach_reset,
            surrogate_function=surrogate.ATan(),
        )

    def forward(self, inputs: torch.Tensor):
        # inputs: B, L, D
        inputs = inputs.permute(0, 2, 1).unsqueeze(1)  # B, 1, D, L
        enc = self.encoder(inputs)  # B, T, D, L
        enc = enc.permute(1, 0, 2, 3)  # T, B, D, L
        spks = self.lif(enc)  # T, B, D, L
        return spks


@NETWORKS.register_module("Spikformer_BitShift_RoPE_1D")
class Spikformer_BitShift_RoPE_1D(nn.Module):
    _snn_backend = "spikingjelly"

    def __init__(
        self,
        dim: int,
        d_ff: Optional[int] = None,
        num_pe_neuron: int = 10,
        pe_type: str = "none",
        pe_mode: str = "concat",  # "add" or concat
        neuron_pe_scale: float = 1000.0,  # "100" or "1000" or "10000"
        depths: int = 2,
        common_thr: float = 1.0,
        max_length: int = 5000,
        num_steps: int = 4,
        heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: float = 0.125,
        input_size: Optional[int] = None,
        weight_file: Optional[Path] = None,
        num_groups: int = 4,
        base: float = 64.0,
    ):
        super().__init__()
        self.dim = dim
        self.d_ff = d_ff or dim * 4
        self.T = num_steps
        self.depths = depths
        self.pe_type = pe_type
        self.pe_mode = pe_mode
        self.num_pe_neuron = num_pe_neuron

        self.temporal_encoder = SpikeEncoder[self._snn_backend]["conv"](num_steps)
        self.pe = PositionEmbedding(
            pe_type=pe_type,
            pe_mode=pe_mode,
            neuron_pe_scale=neuron_pe_scale,
            input_size=input_size,
            max_len=max_length,
            num_pe_neuron=self.num_pe_neuron,
            dropout=0.1,
            num_steps=num_steps,
        )
        if (self.pe_type == "neuron" and self.pe_mode == "concat") or (
            self.pe_type == "random" and self.pe_mode == "concat"
        ):
            self.encoder = nn.Linear(input_size + num_pe_neuron, dim)
        else:
            self.encoder = nn.Linear(input_size, dim)
        self.init_lif = neuron.LIFNode(
            tau=tau,
            step_mode="m",
            detach_reset=detach_reset,
            surrogate_function=surrogate.ATan(),
            v_threshold=common_thr,
            backend=backend,
        )

        self.num_groups = num_groups
        self.base = base
        self.blocks = nn.ModuleList(
            [
                Block1D(
                    length=max_length,
                    tau=tau,
                    common_thr=common_thr,
                    dim=dim,
                    d_ff=self.d_ff,
                    heads=heads,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    num_groups=num_groups,
                    base=base,
                )
                for _ in range(depths)
            ]
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        functional.reset_net(self)

        x = self.temporal_encoder(x)  # B L C -> T B C L
        x = x.transpose(-2, -1)  # T B L C
        if self.pe_type != "none":
            x = self.pe(x)  # T B L C'
        T, B, L, _ = x.shape

        x = self.encoder(x.flatten(0, 1)).reshape(T, B, L, -1)  # T B L D
        x = self.init_lif(x)

        for blk in self.blocks:
            x = blk(x)  # T B L D
        out = x.mean(0)
        return out, out.mean(dim=1)  # B L D, B D

    @property
    def output_size(self):
        return self.dim

    @property
    def hidden_size(self):
        return self.dim
