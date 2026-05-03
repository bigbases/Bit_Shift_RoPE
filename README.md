# Bit-Shift RoPE for Spiking Transformers

**Multiplication-Free Rotary Positional Embedding for Spiking Neural Networks via Cyclic Bit-Shift**

This repository implements a **bit-shift / cyclic-shift based Rotary Positional Embedding (RoPE)** tailored for **Spiking Neural Networks (SNNs)**. Instead of applying the conventional multiplicative complex rotation used in standard RoPE (which is incompatible with the binary spike domain), we approximate the rotation by an integer (bit-shift) cyclic permutation of feature dimensions. The shift amount is derived from a frequency-decay schedule and realized with a **single cached `torch.gather`**, making it efficient and fully spike-compatible.

## Key Contributions

- **Multiplication-free RoPE for SNNs**
  Replaces the floating-point complex rotation of standard RoPE with **integer cyclic shifts**, preserving the relative-position prior while staying inside the binary spike domain.

- **Frequency-decay grouping**
  The head dimension is split into `num_groups` groups, each receiving an exponentially-decayed shift frequency:
  ```
  inv_freq[g] = base ** -(g / (num_groups - 1)),   g = 0, 1, ..., num_groups-1
  shift[g, n] = round(n * inv_freq[g])
  ```
  Earlier groups encode short-range positions (large shifts), later groups encode long-range positions (small/zero shifts).

- **Single cached `gather`**
  All shifts are precomputed once and stored as an index tensor of shape `[1, 1, H, N, D]`, then applied with a single `torch.gather` along the last dimension — no per-step trigonometric computation, no multiplications.

- **Drop-in replacement** for the attention block of [SeqSNN](https://github.com/microsoft/SeqSNN)'s Spikformer family.

## Repository Structure

```
Bit_Shift_RoPE/
├── README.md                       # This file
└── spikformer_bitshiftRoPE.py      # Main model: Spikformer_BitShift_RoPE_1D + CyclicShiftRoPE1D
```

## Model Overview

```python
@NETWORKS.register_module("Spikformer_BitShift_RoPE_1D")
class Spikformer_BitShift_RoPE_1D(nn.Module):
    ...
```

| Component | Description |
|---|---|
| `CyclicShiftRoPE1D` | Cached integer cyclic-shift positional embedding |
| `Block1D` (from SeqSNN, patched) | Spiking attention block that consumes `CyclicShiftRoPE1D` |
| `Spikformer_BitShift_RoPE_1D` | Full Spikformer encoder using the bit-shift RoPE |

### Important Hyperparameters

| Argument | Default | Description |
|---|---|---|
| `dim` | — | Embedding dimension |
| `d_ff` | `dim * 4` | FFN hidden size |
| `depths` | `2` | Number of stacked attention blocks |
| `heads` | `8` | Number of attention heads |
| `num_steps` | `4` | SNN time steps `T` |
| `num_groups` | `4` | Number of frequency groups for cyclic shift (`head_dim % num_groups == 0` required) |
| `base` | `64.0` | Decay constant for `inv_freq` (larger ⇒ slower decay) |
| `pe_type` | `"none"` | Optional auxiliary positional encoding (`"none" / "neuron" / "random"`) |
| `pe_mode` | `"concat"` | `"add"` or `"concat"` for the auxiliary PE |

## Training with SeqSNN

This model is designed to plug into Microsoft's [SeqSNN](https://github.com/microsoft/SeqSNN) framework.

### 1. Setup SeqSNN Environment

```bash
conda create -n SeqSNN python=3.9
conda activate SeqSNN
git clone https://github.com/microsoft/SeqSNN/
cd SeqSNN
pip install -e .
```

### 2. File Integration Guide

#### 2-1. Copy the model file

Copy the model implementation into SeqSNN's network directory:

```bash
cp /path/to/Bit_Shift_RoPE/spikformer_bitshiftRoPE.py \
   /path/to/SeqSNN/SeqSNN/network/snn/
```

#### 2-2. Patch `spike_attention.py`

`Spikformer_BitShift_RoPE_1D` imports `Block1D` and expects it to accept the
extra arguments `num_groups` and `base`, and to internally use
`CyclicShiftRoPE1D`. You must replace (or patch) the existing
`SeqSNN/SeqSNN/module/spike_attention.py` so that:

1. `CyclicShiftRoPE1D` is defined (the one in `spikformer_bitshiftRoPE.py` of
   this repo is the reference implementation).
2. The 1-D spiking self-attention used inside `Block1D` calls
   `CyclicShiftRoPE1D` on the query/key tensors.
3. `Block1D.__init__` exposes `num_groups: int = 4` and `base: float = 64.0`.

A reference signature looks like:

```python
class Block1D(nn.Module):
    def __init__(
        self,
        length: int,
        tau: float,
        common_thr: float,
        dim: int,
        d_ff: int,
        heads: int,
        qkv_bias: bool = False,
        qk_scale: float = 0.125,
        num_groups: int = 4,        # ← required
        base: float = 64.0,         # ← required
    ):
        ...
        self.attn = SSA1D(
            ...,
            num_groups=num_groups,
            base=base,
        )
```

#### 2-3. Update `network/__init__.py`

Edit `SeqSNN/SeqSNN/network/__init__.py` and add:

```python
from .base import NETWORKS

# ... existing imports ...
from .snn.spikformer_bitshiftRoPE import Spikformer_BitShift_RoPE_1D  # ← Add this line
```

#### 2-4. Resulting layout

```
SeqSNN/
├── SeqSNN/
│   ├── network/
│   │   ├── __init__.py                          # ← import added here
│   │   └── snn/
│   │       └── spikformer_bitshiftRoPE.py       # ← model implementation
│   └── module/
│       └── spike_attention.py                   # ← patched (CyclicShiftRoPE1D + Block1D)
└── exp/
    └── forecast/
        └── spikformer_bitshift_rope/            # ← configs (see below)
            └── spikformer_bitshift_electricity.yml
```

### 3. Configuration File (YAML)

Create a config under `SeqSNN/exp/forecast/spikformer_bitshift_rope/`.

```yaml
_base_:
- ../dataset/electricity.yml
_custom_imports_:
- SeqSNN.network
- SeqSNN.dataset
- SeqSNN.runner

data:
  raw_label: False
  window: 168
  eval_window: 168
  horizon: 24
  normalize: 3

runner:
  type: ts
  task: regression
  optimizer: Adam
  lr: 0.001
  weight_decay: 0.0
  loss_fn: mse
  metrics: [r2, rse]
  observe: loss
  lower_is_better: True
  max_epoches: 1000
  early_stop: 30
  batch_size: 64
  aggregate: True
  gpu_ids: [0]

network:
  type: Spikformer_BitShift_RoPE_1D       # ← Must match the registered name
  dim: 256
  d_ff: 1024
  depths: 2
  num_steps: 4
  heads: 8
  # Bit-shift RoPE specific
  num_groups: 4
  base: 64.0
  # (Optional) auxiliary positional encoding
  pe_type: none
  pe_mode: concat

runtime:
  seed: 41
  output_dir: ./outputs/Spikformer_BitShift_RoPE_1D_electricity_h24
```

> **Important — Network type registration**
> The `network.type` in the YAML must match the name passed to `@NETWORKS.register_module(...)` in the model file:
>
> ```python
> @NETWORKS.register_module("Spikformer_BitShift_RoPE_1D")  # ← must match YAML
> class Spikformer_BitShift_RoPE_1D(nn.Module):
>     ...
> ```

### 4. Running Training

```bash
cd /path/to/SeqSNN

python -m SeqSNN.entry.tsforecast \
    exp/forecast/spikformer_bitshift_rope/spikformer_bitshift_electricity.yml
```

## Hyperparameter Notes

### `num_groups`
Number of frequency groups used to split each head's feature dimension.
Constraint: `head_dim % num_groups == 0`.
- Small `num_groups` (e.g. 2) — coarse positional resolution, larger per-group dim.
- Large `num_groups` (e.g. 16) — fine-grained position bands.

### `base`
Decay constant of the per-group shift frequency:
```
inv_freq[g] = base ** -(g / (num_groups - 1))
shift[g, n] = round(n * inv_freq[g])
```
- Small `base` (e.g. 2, 4) — multiple groups receive non-trivial shifts → emphasizes short-range relations.
- Large `base` (e.g. 10000) — only the first group shifts noticeably → emphasizes long-range relations (closer to standard RoPE schedule).

We recommend starting from `num_groups=4, base=64` and sweeping along
`num_groups ∈ {2, 4, 8, 16}` and `base ∈ {2, 4, 8, 32, 64, 128, 10000}`.

## Troubleshooting

### `ModuleNotFoundError: No module named 'SeqSNN.network.snn.spikformer_bitshiftRoPE'`
- Make sure `spikformer_bitshiftRoPE.py` is copied to `SeqSNN/SeqSNN/network/snn/`.
- Add the import line to `SeqSNN/SeqSNN/network/__init__.py`:
  ```python
  from .snn.spikformer_bitshiftRoPE import Spikformer_BitShift_RoPE_1D
  ```

### `KeyError: 'Spikformer_BitShift_RoPE_1D'`
- The `@NETWORKS.register_module("Spikformer_BitShift_RoPE_1D")` decorator name must exactly match `network.type` in your YAML.
- Confirm the new model file is actually imported in `SeqSNN/SeqSNN/network/__init__.py`.

### `TypeError: __init__() got an unexpected keyword argument 'num_groups'`
- Your `SeqSNN/SeqSNN/module/spike_attention.py` is the unpatched version.
- Apply the patch described in **2-2** so that `Block1D` (and the underlying
  spiking self-attention) accept `num_groups` and `base`.

### `AssertionError: head_dim must be divisible by num_groups.`
- Choose `num_groups` so that `(dim / heads) % num_groups == 0`.
- E.g. `dim=256, heads=8 ⇒ head_dim=32`, valid `num_groups ∈ {1, 2, 4, 8, 16, 32}`.

## Related Work

- Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding*, 2021. [arXiv:2104.09864](https://arxiv.org/abs/2104.09864)
- Lv et al., *Efficient and Effective Time-Series Forecasting with Spiking Neural Networks*, ICML 2024. [arXiv:2402.01533](https://arxiv.org/abs/2402.01533)

## License

This project is built upon Microsoft's [SeqSNN](https://github.com/microsoft/SeqSNN) framework.
Please refer to the SeqSNN repository for license information regarding the
underlying framework.
