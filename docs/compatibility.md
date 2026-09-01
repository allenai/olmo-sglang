# Compatibility

`olmo-sglang` imports SGLang runtime internals, so compatibility is a tested
stack rather than a broad semantic-version promise. Installing this Python
package alone is expected to succeed, but calling `register()` requires a
compatible SGLang installation.

## Integration baseline

The current OLMo-MILES runtime baseline was recorded on 2026-08-29:

| Component | Tested version or revision |
|---|---|
| SGLang | `0.5.19.dev48+g3bbb281` / `3bbb2812e2ca1defdee76f6ec09dbb2456c21c69` |
| flash-linear-attention | `0.5.2` |
| PyTorch | `2.13.0+cu130` |
| Transformers | `5.12.1` |
| Triton | `3.7.1` |
| Python | `3.12` |

This table is a known-good integration point, not a claim that adjacent SGLang
commits are incompatible. Run the validation suite whenever changing the
runtime because model registry, fused-MoE, linear-attention cache, and scheduler
interfaces are not stable extension APIs.

## Checkpoint contract

The Hugging Face configuration must declare
`architectures: ["Olmo3MoeForCausalLM"]` and one `layer_types` entry per model
layer. Supported entries are:

- `full_attention`;
- `sliding_attention`;
- `linear_attention` for OLMo KDA.

KDA configurations must provide the `linear_*` head, width, convolution, norm,
and negative-eigenvalue fields validated in `olmo_sglang.config`. KDA key and
value head counts must currently match, key head dimensions must be at most
256, and both head counts must divide the attention tensor-parallel size.

Routing currently supports softmax gating with L1-normalized expert weights.
The loader accepts HF-layout attention, KDA, dense SwiGLU, routed-expert,
latent-projection, and shared-expert tensors and maps them into SGLang's fused
representations.

## Runtime constraints

- KDA serving is GPU-oriented and requires FLA 0.5.2.
- The package carries a narrow FLA 0.5.2 source compatibility shim for newer
  Triton releases. Replacing or upstreaming that shim remains open work.
- Model and attention TP groups must match for KDA.
- Tensor-parallel head counts must divide evenly by the selected TP size.
- Radix caching for KDA uses SGLang's `extra_buffer` strategy to preserve branch
  states. Production rollout refreshes still require topology-specific testing.
- Speculative verification has a fused tree-capable correctness path, but
  performance depends on a useful trained draft source and has not been
  established as a serving default.

## Dependency policy

`pyproject.toml` deliberately does not resolve SGLang, PyTorch, CUDA, or Triton.
Those packages form one runtime unit and are supplied by the environment. FLA
is also installed explicitly only for KDA checkpoints. This avoids a small
extension package replacing the runtime's validated GPU stack during install.
