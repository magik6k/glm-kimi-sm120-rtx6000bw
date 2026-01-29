# GLM-4.7 Deployment on RTX PRO 6000 Blackwell (SM120) - Findings

## Hardware Configuration

- **GPUs**: 8x NVIDIA RTX PRO 6000 Blackwell (96GB each, ~768GB total VRAM)
- **Architecture**: SM120 (Blackwell)
- **CPU**: Dual-socket system with 2 NUMA nodes
- **RAM**: ~1.5TB system memory
- **GPU Topology**: 
  - PIX (direct PCIe switch) pairs: GPU0↔1, GPU2↔3, GPU4↔5, GPU6↔7
  - NODE (same NUMA, different switches): GPU0↔2, GPU0↔3, etc.
  - SYS (cross-NUMA/cross-CPU): GPU0↔4, GPU0↔5, etc.
  - **No NVLink** - PCIe only interconnect

## Model Details

- **Model**: GLM-4.7 (355B total params, 32B active - MoE architecture)
- **Target Quantization**: FP8 (zai-org/GLM-4.7-FP8, ~362GB)
- **Alternative tried**: NVFP4 (Tengyunw/GLM-4.7-NVFP4, ~177GB)

---

## Issues Encountered

### Issue 1: vLLM FP8 MoE CUTLASS Kernels Not Compiled for SM120

**Error**: `No compiled cutlass_scaled_mm for CUDA device capability: 120`

**Root Cause**: vLLM's CUTLASS kernels for FP8 MoE operations are only compiled for SM90 (Hopper) and SM100, not SM120 (Blackwell RTX PRO 6000).

**GitHub Issue**: https://github.com/vllm-project/vllm/issues/32109

**Status**: Known issue, not yet fixed in vLLM mainline.

---

### Issue 2: NVFP4 cuDNN FP4 GEMM Not Supported on Blackwell

**Error**: `cudnnGraphNotSupportedError: [cudnn_frontend] Error: No execution plans support the graph.`

**Root Cause**: cuDNN's FP4 GEMM operations don't have valid execution plans for Blackwell SM120. The FP4 path relies on cuDNN graphs that aren't implemented for this architecture.

**Affected**: Both SGLang and vLLM when using NVFP4/modelopt_fp4 quantization.

**Workaround Attempted**: `--disable-cuda-graph` - Did NOT help because the issue is in the GEMM kernel itself, not CUDA graphs.

---

### Issue 3: BF16 Model Too Large for Available VRAM

**Error**: `RuntimeError: Not enough memory`

**Root Cause**: GLM-4.7 BF16 requires ~717GB for weights alone. With 8x96GB = 768GB total VRAM, there's insufficient memory for KV cache and activations after loading the model.

---

### Issue 4: Triton FP8 Backend Shared Memory Exhaustion

**Error**: `triton.runtime.errors.OutOfResources: out of resource: shared memory, Required: 147456, Hardware limit: 101376`

**Root Cause**: When using `--fp8-gemm-backend triton --moe-runner-backend triton`, the Triton kernels require more shared memory than Blackwell SM120 provides (101KB vs 147KB needed).

**Context**: This occurred during inference after the model loaded successfully. The server crashes after processing 1-2 requests when the MoE kernels JIT-compile and exceed shared memory limits.

**Also affects**: 
- `--fp8-gemm-backend deep_gemm --moe-runner-backend deep_gemm`
- `--fp8-gemm-backend cutlass --moe-runner-backend flashinfer_cutlass`
- All backend combinations!

**Root Cause**: The MoE expert fusion layer (`fused_moe_triton`) is always used regardless of backend selection. The `--moe-runner-backend` setting doesn't fully override the Triton kernels used in the MoE layer. The Triton kernels exceed Blackwell SM120's shared memory (101KB) when they need 147KB.

---

### Issue 5: Missing MoE Kernel Configs for RTX PRO 6000 Blackwell

**Warning**: `Using default MoE kernel config. Performance might be sub-optimal!`

**Details**: SGLang doesn't have pre-tuned MoE kernel configurations for `NVIDIA_RTX_PRO_6000_Blackwell_Server_Edition`. Config files missing at:
- `E=161,N=192,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Server_Edition,dtype=fp8_w8a8,per_channel_quant=True.json`

**Impact**: Uses default configs which may hit resource limits (like shared memory) on this specific hardware.

---

## Partial Working Configurations

### Configuration 1: SGLang with deep_gemm Backend

```bash
python3 -m sglang.launch_server \
  --model-path zai-org/GLM-4.7-FP8 \
  --tp 8 \
  --attention-backend flashinfer \
  --fp8-gemm-backend deep_gemm \
  --moe-runner-backend deep_gemm \
  --trust-remote-code \
  --disable-cuda-graph \
  --port 30000
```

**Result**: 
- Model loads successfully (~42GB per GPU for weights)
- KV cache allocated (~19GB per GPU)  
- Server starts and responds to health checks
- **Can serve short requests** (tested: 200 token generation works)
- **Crashes on longer requests** due to Triton shared memory exhaustion when MoE kernels JIT-compile

### Configuration 2: SGLang with triton Backend

```bash
python3 -m sglang.launch_server \
  --model-path zai-org/GLM-4.7-FP8 \
  --tp 8 \
  --attention-backend triton \
  --fp8-gemm-backend triton \
  --moe-runner-backend triton \
  --trust-remote-code \
  --disable-cuda-graph \
  --port 30000
```

**Result**: Same as above - works for short requests, crashes on longer ones.

---

## Successful Inference Example

When the server is up, inference works:

```bash
curl -s http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "zai-org/GLM-4.7-FP8",
    "messages": [{"role": "user", "content": "Write a short poem about AI."}],
    "max_tokens": 200,
    "temperature": 0.7
  }'
```

**Output**: Model generates coherent responses, showing the model weights and attention work correctly. The issue is specifically in the MoE Triton kernels exceeding shared memory on certain batch sizes/sequence lengths.

---

## Current Status: WORKING (with SGLang PR #16975)

**GLM-4.7-FP8 successfully deployed on RTX PRO 6000 Blackwell (SM120) using SGLang PR #16975!**

### Working Configuration (Jan 28, 2026)

```bash
# Build from PR branch
git clone -b sm120-mxfp4-support https://github.com/amittell/sglang.git /workspace/sglang-pr16975
cd /workspace/sglang-pr16975
pip install -e "python[all]"

# Run server
python3 -m sglang.launch_server \
  --model-path zai-org/GLM-4.7-FP8 \
  --tp 8 \
  --trust-remote-code \
  --disable-cuda-graph \
  --port 30000
```

### Performance Results (With Tuned MoE Kernel Configs)

**Throughput Scaling by Concurrency** (100 tokens/request):

| Concurrency | Throughput | Avg Latency |
|-------------|------------|-------------|
| 1           | 18 tok/s   | 96s         |
| 8           | 140 tok/s  | 17s         |
| 32          | 536 tok/s  | 12s         |
| 64          | 1,028 tok/s| 12s         |
| 128         | 2,120 tok/s| 12s         |
| 256         | 3,170 tok/s| 16s         |
| 512         | 4,180 tok/s| 24s         |

**Peak Throughput**: ~4,500 tok/s at 512 concurrency (500 tokens/request)

**Throughput by Generation Length** (256 concurrency):

| Max Tokens | Throughput | Avg Latency |
|------------|------------|-------------|
| 50         | 3,314 tok/s| 7.7s        |
| 100        | 3,687 tok/s| 13.9s       |
| 200        | 3,605 tok/s| 28.3s       |
| 500        | 3,678 tok/s| 69.6s       |
| 1000       | 3,523 tok/s| 144.7s      |

**Key Observations**:
- NUMA + GDDR6X requires high concurrency (256+) to saturate memory bandwidth
- Throughput scales well up to ~512 concurrent requests
- Single-request latency is ~5.5s/100 tokens (18 tok/s decode)
- Batch efficiency: 250x throughput improvement from 1 to 512 concurrency

### Memory Usage (per GPU)
- Model weights: ~42GB (FP8 via CompressedTensorsW8A8Fp8MoEMethod)
- KV cache: ~19GB per GPU (864,247 tokens capacity)
- Total used: 85GB per GPU
- Available after load: 12.48GB per GPU

### Key Details
- **Quantization method**: `CompressedTensorsW8A8Fp8MoEMethod` (auto-detected)
- **Compute dtype**: BF16 for activations, FP8 for weights
- **MoE kernel configs**: Custom-tuned for RTX PRO 6000 Blackwell Server Edition
- **Config location**: `sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=161,N=192,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Server_Edition,dtype=fp8_w8a8,per_channel_quant=True.json`

### How to Generate Optimized MoE Kernel Configs

```bash
cd sglang/benchmark/kernels/fused_moe_triton
pip install ray

python3 tuning_fused_moe_triton.py \
  --model /path/to/GLM-4.7-FP8 \
  --tp-size 8 \
  --dtype fp8_w8a8 \
  --per-channel-quant \
  --tune

# Copy to correct location
cp "E=161,N=192,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Server_Edition,dtype=fp8_w8a8,per_channel_quant=True.json" \
   ../../../python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/
```

Tuning takes ~45 minutes and tests 1920 kernel configurations across various batch sizes (1 to 4096).

### Previous Status: BLOCKED (with SGLang 0.5.8 mainline)

**With mainline SGLang 0.5.8** (without PR #16975), the MoE fused expert kernels require more shared memory (147KB) than Blackwell SM120 provides (101KB), causing crashes on longer requests.

### Workarounds That Did NOT Work (with mainline):
1. `--moe-runner-backend triton` - Same shared memory issue
2. `--moe-runner-backend deep_gemm` - Same issue (uses Triton internally)
3. `--moe-runner-backend flashinfer_cutlass` - Same issue (MoE layer still uses Triton)
4. `--disable-cuda-graph` - Does not help (not a CUDA graph issue)
5. All combinations of `--fp8-gemm-backend` options - Does not help

### Why PR #16975 Works
The PR makes these changes for SM120:
- Uses `StridedLayout` instead of TMA block layout
- Sets `is_persistent=False` and `num_stages=1` (reduces shared memory from 147KB to fit in 101KB)
- Auto-detects SM120 and selects compatible kernel configurations

---

## Key Learnings

1. **Blackwell SM120 vs SM100**: RTX PRO 6000 uses SM120 (consumer/pro Blackwell), which has different kernel support than B100/B200 (SM100 datacenter Blackwell). Many FP8/FP4 MoE kernels target SM100 only.

2. **Quantization Format Matters**: 
   - `zai-org/GLM-4.7-FP8` uses `compressed-tensors` format
   - `Tengyunw/GLM-4.7-NVFP4` uses `modelopt` format with NVFP4
   - Different formats hit different code paths and kernel requirements

3. **Backend Selection Critical**: SGLang offers multiple backend options:
   - `--fp8-gemm-backend`: auto, deep_gemm, flashinfer_trtllm, cutlass, triton, aiter
   - `--moe-runner-backend`: auto, deep_gemm, triton, triton_kernel, flashinfer_trtllm, flashinfer_cutlass, etc.
   - Not all combinations work on all hardware

4. **SGLang Blackwell Docker**: Official image exists (`lmsysorg/sglang:blackwell`) but docker wasn't available on this instance.

5. **NVIDIA/SGLang Roadmap**: Active collaboration for Q1 2026 includes Blackwell optimizations - see https://github.com/sgl-project/sglang/issues/17130

6. **PCIe-only Multi-GPU**: Custom allreduce is disabled on >2 PCIe-only GPUs. Performance may be limited compared to NVLink systems.

---

## Recommendations

1. **For Production on Blackwell SM120**: Wait for vLLM/SGLang to add SM120 CUTLASS kernel support, or use the official Blackwell docker images when available.

2. **Try cutlass or flashinfer_cutlass backend**: These may avoid the Triton shared memory issue:
   ```bash
   python3 -m sglang.launch_server \
     --model-path zai-org/GLM-4.7-FP8 \
     --tp 8 \
     --fp8-gemm-backend cutlass \
     --moe-runner-backend flashinfer_cutlass \
     --trust-remote-code \
     --disable-cuda-graph \
     --port 30000
   ```

3. **Generate custom MoE kernel configs**: Use SGLang's tuning tools to generate RTX PRO 6000-specific configs:
   - https://github.com/sgl-project/sglang/tree/main/benchmark/kernels/fused_moe_triton

4. **Alternative Approach**: Consider llama.cpp with GGUF quantization (slower but works), or use AMD MI300X/MI325X which have better FP8 MoE support in current tooling.

5. **Monitor Issues**:
   - vLLM SM120 FP8 MoE: https://github.com/vllm-project/vllm/issues/32109
   - SGLang Blackwell tracker: https://github.com/sgl-project/sglang/issues/5338
   - SGLang NVFP4 MoE: https://github.com/sgl-project/sglang/issues/13639
   - SGLang NVIDIA Q1 2026 Roadmap: https://github.com/sgl-project/sglang/issues/17130

---

## Research: Ongoing Fixes and Patches (Jan 2026)

Extensive research across 200+ sources revealed several promising fixes in development.

### Priority 1: SGLang PR #16975 - SM120 MXFP4 Support (MOST PROMISING)

**URL**: https://github.com/sgl-project/sglang/pull/16975
**Status**: Open, awaiting code owner review
**Author**: amittell

**What it fixes**:
- Uses `StridedLayout` instead of TMA block layout for SM120
- Sets `is_persistent=False` and `num_stages=1` (reduces shared memory from 147KB to fit in 101KB)
- Auto-detects SM120 and selects `triton_kernel` backend

**Tested Results on RTX PRO 6000**:
- GPT-OSS-120B at 4K context: **151 tok/s**
- GPT-OSS-120B at 131K context: **57 tok/s**

**How to build and use**:
```bash
# Clone the PR branch
git clone -b sm120-mxfp4-support https://github.com/amittell/sglang.git
cd sglang
pip install -e "python[all]"

# Run with MXFP4 model
python3 -m sglang.launch_server \
  --model-path <MXFP4-quantized-model> \
  --tp 8 \
  --trust-remote-code \
  --port 30000
```

**Caveat**: This PR targets MXFP4 quantization. For FP8 (compressed-tensors), additional patches may be needed.

---

### Priority 2: vLLM PR #31089 - MXFP4 Triton Backend on SM120

**URL**: https://github.com/vllm-project/vllm/pull/31089
**Status**: Open, awaiting review

**Same approach as SGLang PR**:
- Add SM120 to `triton_kernels_supported` condition
- Use `StridedLayout` to avoid persistent kernel requirement
- SM120-specific: `is_persistent=False`, `num_stages=1`

**Test results**: Model loads, ~160 tok/s (slower than Marlin backend at 201 tok/s)

---

### Priority 3: Generate Custom MoE Kernel Configs

The Triton shared memory issue can be worked around by generating hardware-specific kernel configs:

```bash
cd sglang/benchmark/kernels/fused_moe_triton
python tuning_fused_moe_triton.py \
  --model /path/to/GLM-4.7-FP8 \
  --tp-size 8 \
  --dtype fp8_w8a8 \
  --tune
```

This generates configs with reduced `num_stages` and smaller block sizes that fit within 101KB.

Place generated config in:
```
sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/
E=161,N=192,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Server_Edition,dtype=fp8_w8a8,per_channel_quant=True.json
```

---

### Priority 4: TensorRT-LLM

**Status**: Partial SM120 support, known issues

**Pros**:
- Best single-stream performance (243 tok/s vs 231 for SGLang on H100)
- GLM-4 MoE supported in v1.2+

**Cons**:
- Poor high-batch scaling (1,943 tok/s vs 4,742 for vLLM at 100 concurrent)
- Known SM120 issues: `Unsupported SM version for FP8 block scaling GEMM`
- `TRTLLMGenFusedMoE does not support sm120` error reported

**When to use**: If single-stream latency is critical and batch size is low.

---

### Priority 5: FlashInfer with Padding Workaround

**PR #2261** (MERGED): FP8 GEMM fix for SM120/SM121 via dimension padding

```python
# Pad N and K dimensions to 128-element boundaries
# Trade-off: ~60% memory overhead for small N values
```

May help when combined with other fixes.

---

## Performance Comparison: Framework Benchmarks

Based on Clarifai benchmarks (GPT-OSS-120B on 2x H100):

| Metric | vLLM | SGLang | TensorRT-LLM |
|--------|------|--------|--------------|
| **Single Request Throughput** | 187 tok/s | 231 tok/s | **243 tok/s** |
| **High Batch (100 concurrent)** | **4,742 tok/s** | 3,222 tok/s | 1,943 tok/s |
| **Time to First Token (1 req)** | **0.053s** | 0.125s | 0.177s |
| **Per-Token Latency (1 req)** | 0.005s | **0.004s** | **0.004s** |

**Recommendations by use case**:
- **Fast single stream**: TensorRT-LLM (if SM120 works) or SGLang
- **Fast high batch**: vLLM > SGLang >> TensorRT-LLM
- **SM120 today**: SGLang with PR #16975

**Note**: On B200 (datacenter Blackwell SM100), TensorRT-LLM outperforms all others due to deeper hardware optimization. The poor SM120 performance is specific to consumer/workstation Blackwell.

---

## Action Plan: Ordered by Likelihood of Success

### Step 1: Build SGLang from PR #16975 (Highest Priority)
```bash
git clone -b sm120-mxfp4-support https://github.com/amittell/sglang.git
cd sglang && pip install -e "python[all]"
```
- Test with MXFP4 model first
- If FP8 still fails, try Step 2

### Step 2: Generate Custom MoE Kernel Configs
```bash
python tuning_fused_moe_triton.py --model /path/to/model --dtype fp8_w8a8 --tune
```
- Creates SM120-compatible configs with reduced shared memory usage

### Step 3: Try vLLM with TRITON_MLA Backend
```bash
VLLM_USE_V1=1 VLLM_ATTENTION_BACKEND=TRITON_MLA vllm serve zai-org/GLM-4.7-FP8 --tensor-parallel-size 8
```
- Confirmed working for GLM-4.7-Flash on RTX PRO 6000 by Reddit users

### Step 4: TensorRT-LLM (If Single-Stream is Priority)
- Build with CUDA 12.8+, PyTorch nightly
- May need BF16 fallback instead of FP8 for MoE layers

### Step 5: llama.cpp with GGUF (Fallback)
- Slower (~157 tok/s generation) but stable
- Flash Attention fix merged, works on SM120

---

## Key Links for Tracking

| Resource | URL |
|----------|-----|
| SGLang SM120 PR | https://github.com/sgl-project/sglang/pull/16975 |
| vLLM SM120 PR | https://github.com/vllm-project/vllm/pull/31089 |
| Triton PTX Fix | https://github.com/triton-lang/triton/pull/8498 |
| FlashInfer FP8 Fix | https://github.com/flashinfer-ai/flashinfer/pull/2261 |
| SGLang Blackwell Tracker | https://github.com/sgl-project/sglang/issues/5338 |
| vLLM Blackwell RFC | https://github.com/vllm-project/vllm/issues/18153 |
| SGLang NVIDIA Roadmap | https://github.com/sgl-project/sglang/issues/17130 |

---

## Software Versions

- SGLang: PR #16975 branch (sm120-mxfp4-support)
- flashinfer_python: 0.6.1
- Python: 3.12
- CUDA: 12.x (Blackwell compatible)
- Triton: 3.5.1

## Summary

**GLM-4.7-FP8 is now fully operational on 8x RTX PRO 6000 Blackwell (SM120)** using:
1. SGLang PR #16975 for SM120 kernel compatibility
2. Custom-tuned MoE kernel configs for optimal performance

### Configuration Comparison

| Configuration | Peak Throughput | Best For |
|--------------|-----------------|----------|
| **TP8** | 4,500 tok/s @ 512 conc | Long generations (500+ tokens) |
| **TP4 PP2** | 5,154 tok/s @ 384 conc | Short generations (100 tokens) |

### TP8 vs TP4 PP2 Detailed Results

**Short generations (100 tokens):**
| Concurrency | TP8 | TP4 PP2 |
|-------------|-----|---------|
| 128 | 2,120 tok/s | 1,808 tok/s |
| 256 | 3,170 tok/s | 3,306 tok/s |
| 384 | 3,520 tok/s | **5,154 tok/s** |
| 512 | 4,180 tok/s | 4,568 tok/s |

**Long generations (500 tokens):**
| Concurrency | TP8 | TP4 PP2 |
|-------------|-----|---------|
| 256 | 3,656 tok/s | 2,185 tok/s |
| 384 | 4,221 tok/s | 3,008 tok/s |
| 512 | **4,469 tok/s** | 3,986 tok/s |

**Recommendation**:
- **TP4 PP2**: Use for short-context workloads (chatbots, Q&A, code completion)
- **TP8**: Use for long-context workloads (document processing, long-form generation)

### Artifacts Saved

MoE kernel configs saved to `artifacts/moe-configs/`:
- `tp8_config.json` - E=161,N=192 for TP8 configuration
- `tp4_config.json` - E=161,N=384 for TP4 configuration

---

# Kimi K2.5 Deployment Attempt (Jan 29, 2026)

## Model Details

- **Model**: Kimi K2.5 (1T total params, 32B active - MoE architecture)
- **Architecture**: DeepSeek V3-like with 384 experts, 8+1 active per token
- **Context**: 256K tokens
- **Vision**: MoonViT (multimodal, supports images/video)
- **Native Quantization**: INT4 (QAT - Quantization Aware Training)
- **HuggingFace**: `moonshotai/Kimi-K2.5`

### Model Configuration (from config.json)
```json
{
  "n_routed_experts": 384,      // vs GLM's 161
  "num_experts_per_tok": 8,     // + 1 shared expert
  "moe_intermediate_size": 2048,
  "hidden_size": 7168,
  "num_hidden_layers": 61,
  "quantization": {
    "num_bits": 4,
    "strategy": "group",
    "group_size": 32,
    "symmetric": true,
    "format": "pack-quantized",
    "quant_method": "compressed-tensors"
  },
  "ignore": ["lm_head", "re:.*self_attn.*", "re:.*shared_experts.*", "re:.*mlp\\.(gate|up|gate_up|down)_proj.*"]
}
```

**Important**: Only routed MoE expert weights are INT4 quantized. Attention and shared experts remain BF16!

## Download

Model downloaded successfully (~557GB, 64 safetensor shards):
```bash
huggingface-cli download moonshotai/Kimi-K2.5 --local-dir /workspace/models/Kimi-K2.5
```
Download time: ~8-10 minutes on fast network.

## Deployment Attempt 1: SGLang Main Branch (FAILED)

**Error**: Same SM120 shared memory issue as GLM-4.7
```
triton.runtime.errors.OutOfResources: out of resource: shared memory, 
Required: 106496, Hardware limit: 101376. Reducing block sizes or `num_stages` may help.
```

The Triton attention extend kernel requires 106KB shared memory, but SM120 only provides 101KB.

**Command used**:
```bash
SGLANG_DISABLE_CUDNN_CHECK=1 python3 -m sglang.launch_server \
  --model-path /workspace/models/Kimi-K2.5 \
  --tp 8 \
  --trust-remote-code \
  --disable-cuda-graph \
  --tool-call-parser kimi_k2 \
  --reasoning-parser kimi_k2 \
  --port 30000
```

**Model loaded successfully**:
- Memory per GPU: ~72.33 GB weights + 0.30 GB KV cache
- Quantization: `CompressedTensorsWNA16MarlinMoEMethod` (INT4 via Marlin kernels)
- Server started and accepted connections
- **Crashed on first inference request** due to Triton shared memory exhaustion

## Deployment Attempt 2: SGLang PR #16975 Branch (FAILED - Incompatible)

The PR branch has SM120 fixes but doesn't have K2.5 model class:
- K2.5 was released Jan 27, 2026
- PR branch was created earlier and lacks `KimiK25ForConditionalGeneration`
- Copying model files from main branch causes compatibility issues

**Error**:
```
AttributeError: 'KimiK25VisionConfig' object has no attribute 'hidden_size'. Did you mean: 'mm_hidden_size'?
```

The vision model code requires newer transformers dependencies.

## Deployment Attempt 3: NVFP4 (nvidia/Kimi-K2-Thinking-NVFP4) - PARTIAL SUCCESS

Note: This is **Kimi K2-Thinking** (text-only reasoning model), not K2.5 (multimodal).

**Command**:
```bash
SGLANG_DISABLE_CUDNN_CHECK=1 python3 -m sglang.launch_server \
  --model-path nvidia/Kimi-K2-Thinking-NVFP4 \
  --tp 8 \
  --trust-remote-code \
  --disable-cuda-graph \
  --quantization modelopt_fp4 \
  --port 30000
```

**Model Loading**: SUCCESSFUL
- 119 safetensor shards loaded in ~6.5 minutes
- Memory per GPU: 73.87 GB weights
- KV Cache: 5.90 GB per GPU (180,191 tokens capacity)
- Available after load: 12.25 GB per GPU
- Total VRAM used: ~80GB per GPU

**Server Startup**: SUCCESSFUL
- Server started and responded to health checks
- Model: `DeepseekV3ForCausalLM` (uses same architecture as DeepSeek V3)
- KV cache dtype: `torch.float8_e4m3fn`

**Simple Inference**: WORKING
```bash
curl -s http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "nvidia/Kimi-K2-Thinking-NVFP4", 
       "messages": [{"role": "user", "content": "What is 2+2? Answer briefly."}], 
       "max_tokens": 50}'
```

**Response**:
```json
{
  "choices": [{
    "message": {
      "content": " <think> The user asks \"What is 2+2?\" ... 2 + 2 = 4. </think> 4"
    }
  }]
}
```

The model generates coherent responses with thinking tags!

**Benchmark Under Load**: FAILED
```
triton.runtime.errors.OutOfResources: out of resource: shared memory, 
Required: 106496, Hardware limit: 101376
```

When running benchmark with 512-token prompts, the Triton attention extend kernel crashes with the same shared memory issue. Short prompts (decode-only) work because they use a different code path.

**Key Insight**: The NVFP4 quantization uses cuDNN FP4 GEMM for matrix multiplications (not Triton), so the linear layers work. However, the **attention kernels still use Triton** and hit SM120 shared memory limits on prefill/extend operations with longer sequences.

**Log saved**: `artifacts/logs/sglang-k2-nvfp4.log`

## Key Differences: Kimi K2.5 vs GLM-4.7

| Aspect | GLM-4.7 | Kimi K2.5 |
|--------|---------|-----------|
| Total Params | 355B | 1T |
| Active Params | 32B | 32B |
| **Experts** | 161 | **384** |
| Active Experts | 4 | 8+1 shared |
| Quantization | FP8 | INT4 (QAT) |
| Vision | No | Yes (MoonViT) |
| MoE config needed | E=161,N=192 | E=384,N=??? |

## Required Fixes for K2.5 on SM120

1. **Merge SM120 fixes into SGLang main** - PR #16975 needs to be merged or K2.5 support needs to be backported

2. **Generate MoE kernel configs for E=384** - New configs needed since K2.5 has 384 experts vs GLM's 161:
```bash
cd sglang/benchmark/kernels/fused_moe_triton
python3 tuning_fused_moe_triton.py \
  --model /workspace/models/Kimi-K2.5 \
  --tp-size 8 \
  --dtype int4  # or compressed-tensors format
  --tune
```

3. **Fix Triton attention kernel shared memory** - The extend_attention kernel also hits SM120 limits (106KB > 101KB)

## Workarounds to Try

1. **Wait for SGLang/vLLM SM120 support** - Active development ongoing

2. **Use Kimi K2-Instruct instead of K2.5** - Text-only variant may work with PR #16975

3. **Try GGUF via llama.cpp** - `unsloth/Kimi-K2.5-GGUF` available (~240GB for Q4_K_M)

4. **Request K2.5 support in PR #16975** - File issue asking for multimodal model support

## Deployment Attempt 4: K2-Thinking NVFP4 with PR #16975 - FAILED

Tested whether PR #16975's SM120 fixes would help NVFP4 model.

**Result**: Same crash!
```
triton.runtime.errors.OutOfResources: out of resource: shared memory, 
Required: 106496, Hardware limit: 101376
```

**Analysis**: PR #16975 fixes only apply to specific kernel configurations (MXFP4/FP8 MoE kernels). The NVFP4 model uses different code paths:
- Linear layers: cuDNN FP4 GEMM (works)
- Attention: Triton kernels (crashes on prefill)

The attention kernel crash is in a different code path than what PR #16975 fixes.

## Deployment Attempt 5: K2-Thinking NVFP4 with extend_attention.py SM120 Patch - SUCCESS!

Applied a custom patch to `extend_attention.py` to add SM120-specific block sizes.

### The Fix

**File**: `sglang/srt/layers/attention/triton_ops/extend_attention.py`

**Problem**: The code checked `CUDA_CAPABILITY[0] >= 9` which matched SM120 (12.x), but used Hopper-sized blocks (128, 64) that require too much shared memory.

**Solution**: Add SM120-specific case before the SM90+ check:

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

### Benchmark Results (512-token prompts)

```
============ Serving Benchmark Result ============
Successful requests:                     16        
Total input tokens:                      4072      
Total generated tokens:                  2284      
Output token throughput (tok/s):         110.86    
Total token throughput (tok/s):          308.49    
Mean TTFT (ms):                          909.00    
Mean TPOT (ms):                          77.40     
==================================================
```

### Higher Concurrency Benchmark (256-token prompts, 64 requests)

```
============ Serving Benchmark Result ============
Successful requests:                     64        
Request throughput (req/s):              4.88      
Output token throughput (tok/s):         284.68    
Total token throughput (tok/s):          984.04    
Mean TTFT (ms):                          251.39    
Mean TPOT (ms):                          120.50    
==================================================
```

### Patch File Saved

`artifacts/patches/sm120-extend-attention.patch`

## Current Status Summary

| Model | Quantization | SGLang Version | Status | Notes |
|-------|-------------|----------------|--------|-------|
| Kimi K2.5 | INT4 | Main | FAILED | Triton attention shared memory (106KB > 101KB) |
| Kimi K2.5 | INT4 | PR #16975 | FAILED | Model class incompatibility |
| Kimi K2.5 | INT4 | **Main + SM120 patch** | **WORKING** | Full inference working! |
| Kimi K2-Thinking | NVFP4 | Main | **PARTIAL** | Works for short prompts (~50 tokens), fails on prefill |
| Kimi K2-Thinking | NVFP4 | **Main + SM120 patch** | **WORKING** | Full inference with 512+ token prompts |

### K2.5 INT4 Benchmark Results (with SM120 patch)

**Throughput Scaling by Concurrency** (256 input, 100 output tokens):

| Concurrency | Output Throughput | Total Throughput | Mean TTFT | Mean TPOT |
|-------------|-------------------|------------------|-----------|-----------|
| 32          | 171 tok/s         | 712 tok/s        | 363ms     | 96ms      |
| 64          | 214 tok/s         | 864 tok/s        | 2,026ms   | 84ms      |
| 128         | 258 tok/s         | 884 tok/s        | 8,330ms   | 85ms      |
| 256         | 267 tok/s         | 935 tok/s        | 19,010ms  | 84ms      |
| 500         | 270 tok/s         | 985 tok/s        | 38,647ms  | 88ms      |
| 1000        | 281 tok/s         | **985 tok/s**    | 85,590ms  | 91ms      |

**Note**: K2.5 INT4 has limited KV cache (4,578 tokens) due to higher memory usage from BF16 attention weights, which limits throughput scaling.

### K2-Thinking NVFP4 Benchmark Results (with SM120 patch)

**Throughput Scaling by Concurrency** (256 input, 100 output tokens):

| Concurrency | Output Throughput | Total Throughput | Mean TTFT | Mean TPOT |
|-------------|-------------------|------------------|-----------|-----------|
| 128         | 524 tok/s         | 1,795 tok/s      | 1,324ms   | 160ms     |
| 256         | 867 tok/s         | 3,039 tok/s      | 827ms     | 207ms     |
| 512         | 1,192 tok/s       | 4,345 tok/s      | 1,954ms   | 401ms     |
| 1024        | 1,610 tok/s       | **5,655 tok/s**  | 3,560ms   | 687ms     |
| 2048        | 1,650 tok/s       | **5,816 tok/s**  | 12,406ms  | 576ms     |

**Peak Output Throughput**: 4,941 tok/s (burst) at 1024 concurrency
**Peak Total Throughput**: ~5,800 tok/s sustained

K2-Thinking NVFP4 achieves ~6x higher throughput than K2.5 INT4 due to larger KV cache (180K vs 4.5K tokens) from FP4 quantization reducing memory footprint.

### Performance Comparison: K2.5 INT4 vs K2-Thinking NVFP4

| Metric | K2.5 INT4 | K2-Thinking NVFP4 |
|--------|-----------|-------------------|
| Peak Total Throughput | 985 tok/s | **5,816 tok/s** |
| KV Cache Size | 4,578 tokens | 180,191 tokens |
| VRAM per GPU (weights) | 72.33 GB | 73.87 GB |
| Model Type | Multimodal (vision) | Text-only (reasoning) |
| Quantization | INT4 (Marlin) | NVFP4 (cuDNN FP4) |

### Root Cause Analysis

**Why K2-Thinking NVFP4 partially works but K2.5 INT4 doesn't:**

1. **NVFP4**: Uses cuDNN FP4 GEMM for linear layers (bypasses Triton). Only hits shared memory limit on attention extend/prefill kernels.

2. **INT4 (Marlin)**: Uses Triton for both MoE and attention kernels. Hits shared memory limit on first inference.

3. **Both fail under load**: The Triton attention extend kernel (`106KB > 101KB`) crashes when processing sequences longer than ~100 tokens.

### The Fix Applied

The `extend_attention.py` patch adds SM120 (compute capability 12.x) as a special case with smaller block sizes:
- Uses (64, 64), (32, 64), or (32, 32) blocks depending on head dimension
- Similar to sm86/sm89 Ampere which also has ~100KB shared memory limit
- Reduces shared memory usage to fit within SM120's 101KB limit

### Why PR #16975 Alone Wasn't Enough

PR #16975 patches the **MoE fused expert kernels** with SM120 support, but the **attention extend kernel** is in a separate file and wasn't patched. Our fix addresses the attention kernel specifically.

## Next Steps

1. **Submit upstream PR**: The extend_attention.py SM120 fix should be contributed to SGLang main

2. **Combine with PR #16975**: For comprehensive SM120 support, both MoE and attention fixes are needed

3. **Tune MoE kernel configs for K2.5**: Generate E=384 expert configs for optimal performance

## Artifacts

### Patches
- `artifacts/patches/sm120-extend-attention.patch` - SM120 attention kernel fix

### Logs
- `artifacts/logs/sglang-k2-nvfp4.log` - K2-Thinking NVFP4 initial test
- `artifacts/logs/sglang-k2-nvfp4-pr16975.log` - K2-Thinking with PR #16975
- `artifacts/logs/sglang-k2-nvfp4-sm120fix.log` - K2-Thinking with SM120 fix (SUCCESS)
- `artifacts/logs/sglang-k25-int4.log` - K2.5 INT4 initial test
- `artifacts/logs/sglang-k25-int4-sm120fix.log` - K2.5 INT4 with SM120 fix (SUCCESS)

### Research
- `artifacts/sm120-attention-fix-research.md` - Comprehensive SM120 fix research summary
