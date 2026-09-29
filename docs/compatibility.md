# Compatibility

`olmo-sglang` imports SGLang runtime internals, so compatibility is a tested
stack rather than a broad semantic-version promise. Installing this Python
package alone is expected to succeed, but calling `register()` requires a
compatible SGLang installation.

## Integration baseline

The pinned runtime uses the SGLang source revision recorded in `pyproject.toml`:

| Component | Tested version or revision |
|---|---|
| SGLang | `0.5.19.dev49+g3145136` / `3145136dcd1238754e0ea2b2ffd546532119c71c` |
| flash-linear-attention | `0.5.2` |
| PyTorch | `2.13.0+cu130` |
| Transformers | `5.12.1` |
| Triton | `3.7.1` |
| Python | `3.12` |

The source revision adds a bounded HTTP-readiness wait before optional GC
freezing to the audited `sglang-miles` base
`3bbb2812e2ca1defdee76f6ec09dbb2456c21c69`. It does not change model kernels.

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

Full and sliding attention require `use_head_qk_norm=true`. Shared gains and
`qk_norm_per_head_gains=true` are supported; normalization across the entire Q/K
projection is not. Attention output gates can be absent or `elementwise`, with
projection biases following `attention_bias`. Headwise output gates are not
supported. Sliding attention requires an integer `sliding_window >= 2`.
Unsupported normalization and gate settings are rejected during model setup.

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
  states. Standalone changed-weight publication with radix caching and decode
  graphs passes the [grouped update probe](validation.md#changed-weights-with-radix-caches-and-graphs).
  Admission and policy-version coordination across replicas remain caller-owned.
- Speculative verification has a fused tree-capable correctness path, but
  performance depends on a useful trained draft source and has not been
  established as a serving default.

## Integration boundaries

This package implements model execution and standalone weight updates. The
calling application owns request admission, policy-version tracking, coordinated
updates across replicas, recovery, and trainer topology. Inference TP support
does not qualify a particular distributed training configuration.

Require successful weight updates on every replica and invalidate cached state
before admitting requests for the new policy. Validate cancellation, worker
replacement, realistic request lengths, and concurrency in the deployment that
will use the package.

Exact repeatability across independent kernel tuning, batching, or graph/cache
settings is not guaranteed. Validate the actual checkpoint and runtime when
reference-equivalent probabilities are required; see
[numerical behavior](numerical-findings.md).

## Dependency policy

`pyproject.toml` deliberately does not resolve SGLang, PyTorch, CUDA, or Triton.
Those packages form one runtime unit and are supplied by the environment. FLA
is also installed explicitly only for KDA checkpoints. This avoids a small
extension package replacing the runtime's validated GPU stack during install.
