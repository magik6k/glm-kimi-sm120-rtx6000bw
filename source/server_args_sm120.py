    is_blackwell_supported,
    is_cuda,
    is_flashinfer_available,
    is_hip,
    is_hopper_with_cuda_12_3,
    is_no_spec_infer_or_topk_one,
    is_npu,
    is_port_available,
    is_remote_url,
    is_sm90_supported,
    is_sm100_supported,
    is_sm120_supported,
    is_triton_kernels_available,
    is_valid_ipv6_address,
    json_list_type,
    nullable_str,
    parse_connector_type,
    wait_port_available,
    xpu_has_xmx_support,
)
from sglang.srt.utils.hf_transformers_utils import check_gguf_file
from sglang.utils import is_in_ci

logger = logging.getLogger(__name__)

# Define constants
DEFAULT_UVICORN_ACCESS_LOG_EXCLUDE_PREFIXES = ()
SAMPLING_BACKEND_CHOICES = {"flashinfer", "pytorch", "ascend"}
LOAD_FORMAT_CHOICES = [
    "auto",
    "pt",
    "safetensors",
    "npcache",
    "dummy",
    "sharded_state",
    "gguf",
    "bitsandbytes",
    "layered",
    "flash_rl",
    "remote",
    "remote_instance",
    "fastsafetensors",
--
                elif is_blackwell_supported() and is_mxfp4_quant_format:
                    self.moe_runner_backend = "flashinfer_mxfp4"
                    logger.warning(
                        "Detected SM100 and MXFP4 quantization format for GPT-OSS model, enabling FlashInfer MXFP4 MOE kernel."
                    )
                elif self.ep_size == 1 and is_triton_kernels_available():
                    self.moe_runner_backend = "triton_kernel"
