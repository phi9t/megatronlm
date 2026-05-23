ARG BASE_IMAGE=verlai/verl:vllm011.dev7
FROM ${BASE_IMAGE}

RUN uv pip install --system --extra-index-url https://pypi.nvidia.com \
    nvidia-modelopt==0.44.0rc5 \
    "transformers>=5.8.1,<5.9.0" \
    "peft>=0.18.1"
