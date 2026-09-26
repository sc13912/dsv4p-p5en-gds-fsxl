# DeepSeek-V4-Pro-0813 weight loading from FSx for Lustre over GPUDirect Storage.
# Additive layer on the upstream vLLM image: nothing is removed or downgraded.
# pip delta vs base = instanttensor only.
# Base: AWS Deep Learning Container for vLLM (ECR Public) - same upstream vLLM 0.28.0 release
# build (VLLM_IMAGE_TAG=vllm/vllm-openai:v0.28.0), re-published by AWS into ECR and adding
# EFA 1.47.0 + aws-ofi-nccl. Digest-pinned for reproducibility.
FROM public.ecr.aws/deep-learning-containers/vllm:0.28.0-gpu-py312-cu130-ubuntu24.04-ec2-v1.1@sha256:c2b40ed282bebe21e4a78eb0b3a232c3b29266b0b6f8dabf48e4e07550948bde

LABEL org.opencontainers.image.title="dsv4p-vllm-gds"

ENV DEBIAN_FRONTEND=noninteractive

# gds-tools only. The nvidia-gds meta-package pulls nvidia-fs-dkms, which builds
# a kernel module; nvidia-fs belongs to the host.
RUN apt-get update && apt-get install -y --no-install-recommends \
        gds-tools-13-0 numactl pciutils \
    && rm -rf /var/lib/apt/lists/*

RUN set -eux; \
    for b in gdscheck gdsio gdsio_verify gds_stats gdscp; do \
        ln -sf "/usr/local/cuda-13.0/gds/tools/$b" "/usr/local/bin/$b"; \
    done

# The base ships only libcufile.so.0. Loaders probe ctypes.CDLL("libcufile.so").
RUN set -eux; \
    L="$(dirname "$(find /usr/local/lib -name 'libcufile.so.0' | head -1)")"; \
    ln -sf "$L/libcufile.so.0" "$L/libcufile.so"; \
    echo "$L" > /etc/ld.so.conf.d/000-cufile-wheel.conf; \
    ldconfig; \
    python3 -c "import ctypes; ctypes.CDLL('libcufile.so')"

# --no-deps: a plain install re-resolves torch and downgrades nvidia-nccl-cu13.
# --break-system-packages: Ubuntu 24.04 marks the system Python PEP 668
# "externally managed", so both pip and uv refuse without it. This installs into
# /usr/local/lib/python3.12/dist-packages - the same interpreter vLLM lives in.
RUN pip install --no-cache-dir --no-deps --break-system-packages instanttensor==0.1.9 \
    && python3 -c "from instanttensor import Backend; print([b.name for b in Backend])"

# The "execution" block is NOT optional. Omit it and cuFile defaults to
# parallel_io=false / max_io_threads=0 / max_request_parallelism=0 — no I/O
# threadpool at all, which costs ~1.4x on this workload (total load 69.6 s vs 44.6 s).
# per_buffer_cache_size_kb also defaults to 1024; 16384 is what the measured
# configuration used.
RUN printf '%s\n' \
    '{' \
    '  "logging": { "level": "ERROR" },' \
    '  "profile": { "nvtx": false, "cufile_stats": 3 },' \
    '  "properties": {' \
    '    "allow_compat_mode":             false,' \
    '    "use_poll_mode":                 false,' \
    '    "max_direct_io_size_kb":         16384,' \
    '    "max_device_cache_size_kb":      131072,' \
    '    "per_buffer_cache_size_kb":      16384,' \
    '    "max_device_pinned_mem_size_kb": 33554432' \
    '  },' \
    '  "execution": {' \
    '    "max_io_threads":           4,' \
    '    "parallel_io":              true,' \
    '    "min_io_threshold_size_kb": 8192,' \
    '    "max_request_parallelism":  4' \
    '  },' \
    '  "fs": { "lustre": { "posix_gds_min_kb": 0 } }' \
    '}' > /etc/cufile.json

ENV INSTANTTENSOR_BACKEND=CUFILE \
    INSTANTTENSOR_IO_DEPTH=64 \
    PYTHONUNBUFFERED=1

ENTRYPOINT []
CMD ["/bin/bash"]
