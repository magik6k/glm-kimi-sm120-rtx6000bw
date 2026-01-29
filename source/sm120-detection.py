is_blackwell_supported = is_blackwell = lru_cache(maxsize=1)(
    partial(
        _check_cuda_device_version,
        device_capability_majors=[10, 12],
        cuda_version=(12, 8),
    )
)
is_sm120_supported = lru_cache(maxsize=1)(
    partial(
        _check_cuda_device_version, device_capability_majors=[12], cuda_version=(12, 8)
    )
)
is_sm100_supported = lru_cache(maxsize=1)(
    partial(
        _check_cuda_device_version, device_capability_majors=[10], cuda_version=(12, 8)
    )
)
is_sm90_supported = lru_cache(maxsize=1)(
    partial(
        _check_cuda_device_version, device_capability_majors=[9], cuda_version=(12, 3)
    )
)


try:
    import sgl_kernel  # noqa: F401

    is_intel_amx_backend_available = hasattr(
