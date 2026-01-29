# Kimi K2 on SM120 (RTX PRO 6000 Blackwell) - Research Artifacts

This repository contains research artifacts, patches, and benchmark results from deploying **Kimi K2.5** and **K2-Thinking** on **8x NVIDIA RTX PRO 6000 Blackwell (SM120)** GPUs.

## Key Achievement

Successfully deployed large MoE models on consumer/workstation Blackwell GPUs after developing a fix for the Triton attention kernel shared memory issue:

| Model | Quantization | Peak Throughput |
|-------|--------------|-----------------|
| **Kimi K2-Thinking** | NVFP4 | **5,816 tok/s** @ 2048 concurrency |
| **Kimi K2.5** | INT4 (AWQ) | **985 tok/s** @ 1000 concurrency |

## The Problem

SM120 (consumer Blackwell: RTX 5090, RTX PRO 6000) has only **100KB shared memory** per SM, while SM100 (datacenter Blackwell: B100, B200) has **228KB**. The Triton `extend_attention.py` kernel was compiled with Hopper-sized blocks requiring ~106KB, causing crashes:

```
triton.runtime.errors.OutOfResources: out of resource: shared memory,
Required: 106496, Hardware limit: 101376
```

## The Fix

Add SM120-specific block sizes in SGLang's `extend_attention.py`:

```python
elif _is_cuda and CUDA_CAPABILITY[0] == 12:
    # SM120 Blackwell RTX - use smaller blocks for 100KB shared memory
    if Lq <= 128:
        BLOCK_M, BLOCK_N = (64, 64)
    elif Lq <= 256:
        BLOCK_M, BLOCK_N = (32, 64)
    else:
        BLOCK_M, BLOCK_N = (32, 32)
    num_warps = 4 if Lq <= 64 else 8
```

**PR submitted**: [sgl-project/sglang#TBD](https://github.com/sgl-project/sglang/pulls)

## Repository Contents

```
.
├── README.md                           # This file
├── FINDINGS.md                         # Comprehensive technical findings (32KB)
├── sm120-attention-fix-research.md     # Focused research on the attention fix
├── report-kimi-k2-sm120.html           # Interactive benchmark report with charts
├── report-kimi-k2-sm120.pdf            # PDF version of benchmark report
├── patches/
│   └── sm120-extend-attention.patch    # Unified diff for the fix
├── source/
│   ├── extend_attention_original.py    # Original SGLang file
│   ├── extend_attention_patched.py     # Patched version
│   ├── decode_attention.py             # Related attention kernel
│   ├── mxfp4.py                        # MXFP4 quantization code
│   ├── server_args_sm120.py            # Server configuration
│   └── sm120-detection.py              # SM120 detection utilities
├── configs/
│   ├── gpu-info.txt                    # nvidia-smi output
│   ├── gpu-topology.txt                # GPU interconnect topology
│   ├── nvidia-smi-full.txt             # Full nvidia-smi details
│   ├── cpu-info.txt                    # CPU configuration
│   ├── memory-info.txt                 # System memory info
│   ├── python-packages.txt             # Installed package versions
│   ├── kimi-k25-config.json            # Model configuration
│   └── moe-configs-list.txt            # Available MoE configs
├── moe-configs/
│   ├── E=161,N=192,...json             # Tuned MoE config for RTX 6000
│   ├── rtx6000-blackwell-*.json        # Hardware-specific configs
│   ├── tp4_config.json                 # Tensor parallel 4 config
│   └── tp8_config.json                 # Tensor parallel 8 config
└── logs/
    ├── sglang-k2-thinking-final-bench.log  # K2-Thinking benchmark
    ├── sglang-k25-int4-sm120fix.log        # K2.5 with fix
    ├── sglang-k2-nvfp4-sm120fix.log        # K2 NVFP4 with fix
    └── ...                                  # Other experiment logs
```

## Hardware Configuration

- **GPUs**: 8x NVIDIA RTX PRO 6000 Blackwell Server Edition
- **Architecture**: SM120 (compute capability 12.0)
- **VRAM**: 96GB GDDR6X per GPU (768GB total)
- **Interconnect**: PCIe only (no NVLink)
- **NUMA**: 2 nodes, 4 GPUs each

## Software Versions

- PyTorch 2.9.1
- Triton 3.5.1
- FlashInfer 0.6.1
- SGLang (main branch + SM120 patch)
- Transformers 4.57.1

## Working Launch Command

```bash
python3 -m sglang.launch_server \
    --model-path moonshotai/Kimi-K2-Instruct \
    --quantization nvfp4 \
    --tp 8 \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 30000
```

## Related Issues

- **SGLang #14322**: Kimi K2 crashes on SM120 (our fix addresses this)
- **SGLang PR #16975**: SM120 MXFP4 support (fixes MoE, not attention)
- **vLLM #26211, #31936**: DeepSeek on SM120
- **FlashInfer #1147, #2166**: SM120 support questions

## Key Technical Insight

SM120 consumer Blackwell has the same ~100KB shared memory constraint as sm86/sm89 Ampere GPUs, NOT the 228KB available on SM100 datacenter Blackwell. This means:

1. Kernels tuned for H100/B200 will fail on RTX 5090/RTX PRO 6000
2. Block sizes must be reduced to fit within 100KB
3. The fix pattern applies to all Triton kernels, not just attention

## License

Research artifacts provided as-is for community benefit. The patch is submitted upstream to SGLang under their license terms.

## Author

Research conducted January 2026.
