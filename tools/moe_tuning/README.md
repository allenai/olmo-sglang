# Tuning the fused-MoE Triton kernel for OLMo expert shapes

`olmo_sglang/moe_configs` ships the tuned tile configurations that SGLang's
`fused_moe` kernel reads per GPU. This directory records how they were made so the
sweep can be rerun after a Triton, kernel, shape or hardware change.

- `olmo3moe_tuner.patch`: applied to `benchmark/kernels/fused_moe_triton/` of the
  pinned SGLang revision. It teaches `get_model_config` the `Olmo3MoeForCausalLM`
  fields (experts operate in `latent_moe_dim`, so that is the GEMM K/N size) and
  lets a plain JSON config be supplied through `OLMO_MOE_TUNE_CONFIG`.
- `search_space.json`: 288 tile configurations (rows 16-128, columns 64-256,
  reduction depth 64/128, 4 or 8 warps, 2-4 stages). The tuner's default space of
  1,920 compiles for several seconds per configuration at the first batch size.
- `run_tune.sh`: Beaker job body. Expects the patched benchmark directory mounted
  at `/tune`; writes `E=...json`, the tuning log and the geometry used to `/output`.
- `run_bench.sh`: times the kernel at each batch size with the default tile and with
  the files mounted at `/moe_configs` (`SGLANG_MOE_CONFIG_DIR`).

Launch with `mason.py` from open-instruct on one GPU of the target type using the
pinned MILES runtime image; Jupiter H100 nodes additionally need the CUDA
compatibility `LD_LIBRARY_PATH` documented in the MILES launch guide. A sweep over
batch sizes 1 to 8192 takes about 17 minutes. Copy the resulting JSON into
`src/olmo_sglang/moe_configs/configs/triton_<version>/` and rerun
`tests/test_moe_configs.py`.
