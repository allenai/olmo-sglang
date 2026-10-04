# Third-party notices

Original `olmo-sglang` code is licensed under Apache 2.0; see [LICENSE](LICENSE).
The distribution also includes adapted MIT-licensed code as described below.
The package license expression `Apache-2.0 AND MIT` records both applicable
licenses; it does not offer a choice of license for the entire project.

## Release scope

This release contains `olmo-sglang` source and the FLA-derived adaptation below,
plus documentation, tests, examples, and license notices. Dependency packages,
compiled third-party libraries, container images, model weights, and datasets
are not bundled. End users install all runtime dependencies separately, including
SGLang, every NVIDIA-maintained or NVIDIA-licensed dependency (`cuda-toolkit`,
`cuda-*`, `nvidia-*`, and `nccl4py`), and all Rust and native components. This
applies regardless of each dependency's declared license.

`easydict` and `pycountry` are also separately installed transitive dependencies;
no code from either package is copied or modified in this repository.

## Flash Linear Attention, via SGLang

`src/olmo_sglang/kda/packed_decode.py` adapts
`fused_recurrent_kda_packed_decode_kernel` from SGLang's
[`fused_recurrent.py`](https://github.com/sgl-project/sglang/blob/3145136dcd1238754e0ea2b2ffd546532119c71c/python/sglang/kernels/ops/attention/fla/fused_recurrent.py).
That source derives from Flash Linear Attention and carries this original notice:

> Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

The local adaptation implements OLMo beta activation semantics and maintains the
recurrent state in place. The upstream attribution is retained in the source.

Flash Linear Attention is MIT licensed. The complete permission and warranty
notice, including the copyright notice from the pinned FLA v0.5.2 release, is
included in [LICENSES/FLA-MIT.txt](LICENSES/FLA-MIT.txt). These notices and license
texts are included in both source and wheel distributions.

`src/olmo_sglang/kda/backend.py` also applies a guarded compatibility shim to
separately installed FLA 0.5.2 under Triton 3.6 or newer. It replaces a constexpr
expression in the installed kernel's in-memory JIT source and wraps its launcher;
it does not bundle the FLA package or write a modified copy to disk.
