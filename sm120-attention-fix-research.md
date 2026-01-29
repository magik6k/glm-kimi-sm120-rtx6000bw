# SM120 Attention Kernel Shared Memory Fix - Research Summary

## Problem Statement

RTX PRO 6000 Blackwell (SM120) has only **100KB shared memory** per SM, but Triton attention extend kernels were compiled with configurations requiring **106KB**, causing crashes:

```
triton.runtime.errors.OutOfResources: out of resource: shared memory, 
Required: 106496, Hardware limit: 101376
```

## Root Cause

The `_get_block_sizes_for_extend_attention()` function in `extend_attention.py` checks:
```python
if _is_cuda and CUDA_CAPABILITY[0] >= 9:
    # Hopper architecture (H100, etc.)
    if Lq <= 256:
        BLOCK_M, BLOCK_N = (128, 64)
```

This matches SM120 (12.x >= 9), but uses Hopper-sized blocks that require too much shared memory.

## Hardware Comparison

| Architecture | Compute Capability | Shared Memory per SM |
|-------------|-------------------|---------------------|
| Hopper (H100) | SM90 | 228 KB |
| Blackwell Datacenter (B200) | SM100 | 228 KB |
| Blackwell RTX (RTX 5090, RTX PRO 6000) | SM120 | **100 KB** |
| Ampere (A100) | SM80 | 160 KB |
| Ampere (RTX 3090) | SM86/89 | **100 KB** |

SM120 consumer Blackwell has the same shared memory constraint as sm86/sm89 Ampere, NOT the datacenter Blackwell (SM100).

## The Fix

Add SM120-specific case in `extend_attention.py`:

```python
elif _is_cuda and CUDA_CAPABILITY[0] == 12:
    # SM120 Blackwell RTX (RTX 5090, RTX PRO 6000, etc.)
    # Has only ~100KB shared memory (vs 228KB on SM100 datacenter Blackwell)
    # Use smaller block sizes similar to sm86/sm89 Ampere which also has ~100KB
    if Lq <= 128:
        BLOCK_M, BLOCK_N = (64, 64)
    elif Lq <= 256:
        BLOCK_M, BLOCK_N = (32, 64)
    else:
        BLOCK_M, BLOCK_N = (32, 32)
    num_warps = 4 if Lq <= 64 else 8
```

## Key Insights from Research

### 1. SGLang Attention Backend Architecture

Available backends: `flashinfer`, `triton`, `torch_native`, `flex_attention`, `trtllm_mha`, etc.

The extend attention kernel (`triton_ops/extend_attention.py`) is used for prefill operations when `--attention-backend triton` is selected (or auto-selected).

### 2. PR #16975 (MoE Fix) Pattern

PR #16975 fixed MoE kernels for SM120 using:
- `StridedLayout` instead of TMA block layout
- `is_persistent = False`
- `num_stages = 1`

This reduces shared memory but the fix only applies to MoE kernels, not attention.

### 3. FlashInfer SM120 Support

FlashInfer has partial SM120 support:
- PR #2261 fixed FP8 GEMM padding issues
- Issues #1147, #2166 track incomplete SM120 support
- MLA (Multi-head Latent Attention) not fully supported on SM120

### 4. vLLM SM120 Approach

vLLM uses similar pattern for MXFP4 on SM120:
- Force `is_persistent = False`
- Use `num_stages = 1`
- Select `StridedLayout` for SM120

### 5. Triton Shared Memory Tuning

Shared memory usage is determined by:
- `BLOCK_M`, `BLOCK_N` (tile sizes)
- `num_stages` (pipeline depth)
- Data type size

Formula: `shared_mem ≈ BLOCK_M × BLOCK_K × dtype_bytes × num_stages + BLOCK_K × BLOCK_N × dtype_bytes × num_stages`

Reducing any of these reduces shared memory.

## Recommended Upstream Changes

### Option 1: Minimal Fix (extend_attention.py only)
Add SM120 case to block size selection as shown above.

### Option 2: Comprehensive SM120 Support
1. Patch `extend_attention.py` for attention kernels
2. Merge PR #16975 for MoE kernels  
3. Update FlashInfer to handle SM120 constraints
4. Add SM120 detection utilities

### Files to Modify

| File | Change |
|------|--------|
| `sglang/srt/layers/attention/triton_ops/extend_attention.py` | Add SM120 block sizes |
| `sglang/srt/layers/attention/triton_ops/decode_attention.py` | May need similar fix |
| `sglang/srt/layers/attention/triton_ops/prefill_attention.py` | May need similar fix |

## Test Results

### Before Fix
```
triton.runtime.errors.OutOfResources: out of resource: shared memory, 
Required: 106496, Hardware limit: 101376
```

### After Fix
```
============ Serving Benchmark Result ============
Successful requests:                     64        
Output token throughput (tok/s):         284.68    
Total token throughput (tok/s):          984.04    
==================================================
```

## Alternative Workarounds

1. **Use `--attention-backend torch_native`** - Avoids Triton entirely but slower
2. **Use `--attention-backend trtllm_mha`** - TensorRT-LLM backend, default for Blackwell
3. **Reduce prefill chunk size** - `--chunked_prefill_size 64` to avoid long prefills
4. **Use llama.cpp** - Completely different inference stack, stable on SM120

## References

- SGLang PR #16975: https://github.com/sgl-project/sglang/pull/16975
- FlashInfer SM120 Issues: https://github.com/flashinfer-ai/flashinfer/issues/1147
- vLLM SM120 PR: https://github.com/vllm-project/vllm/pull/31089
- CUDA SM120 Specs: ~100KB shared memory per SM

## Patch File

See `artifacts/patches/sm120-extend-attention.patch`
