# Triton (vLLM backend) + Python-backend embedding/reranker models, for the RTX 5090.
# Serves: text_embedding (BAAI/bge-base-en-v1.5), cross_reranker (BAAI/bge-reranker-base),
#         nemotron (nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 via vLLM backend).
FROM nvcr.io/nvidia/tritonserver:26.07-vllm-python-py3

# Some NVIDIA base images mark the system Python as externally managed.
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# sentence-transformers for the two Python-backend models.
# The base image ships NVIDIA's own pre-release torch build (2.13.0a0+...nv26.07), which pip's
# resolver cannot match against sentence-transformers' "torch>=2.2" requirement. So:
#   1) install sentence-transformers WITHOUT dependencies (torch + transformers 5.x are already here),
#   2) add its remaining small deps, pinning numpy so vLLM's numpy is never replaced.
RUN set -eux; \
    NP_V=$(python3 -c "import numpy; print(numpy.__version__)"); \
    printf "numpy==%s\n" "$NP_V" > /tmp/constraints.txt; \
    pip install --no-cache-dir --no-deps "sentence-transformers==6.2.0"; \
    pip install --no-cache-dir -c /tmp/constraints.txt scikit-learn scipy Pillow; \
    python3 -c "import torch, numpy, transformers, sentence_transformers, vllm; print('torch', torch.__version__, '| numpy', numpy.__version__, '| transformers', transformers.__version__, '| sentence-transformers', sentence_transformers.__version__, '| vllm', vllm.__version__)"
