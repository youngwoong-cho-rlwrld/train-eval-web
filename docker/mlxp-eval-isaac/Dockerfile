# syntax=docker/dockerfile:1.7
FROM mlxp.kr.ncr.ntruss.com/rlwrld-gpu-base:latest

SHELL ["/bin/bash", "-lc"]

ARG ISAACLAB_REPO=https://github.com/isaac-sim/IsaacLab.git
ARG ISAACLAB_REF=64ecea24f51bd008d2ae75dc2a233db5db515a71
ARG RLWRLD_ISAAC_REPO=git@github.com:RLWRLD/rlwrld_isaac.git
ARG RLWRLD_ISAAC_REF=torque_dev
ARG GROOT_REPO=git@github.com:RLWRLD/gr00t.git
ARG GROOT_REF=seungcheol/torque_dev3

ENV DEBIAN_FRONTEND=noninteractive
ENV OMNI_KIT_ACCEPT_EULA=Y
ENV NO_ALBUMENTATIONS_UPDATE=1
ENV TOKENIZERS_PARALLELISM=false
ENV RLWRLD_ROOT=/workspace
ENV ISAACLAB_DIR=/workspace/IsaacLab
ENV ISAAC_DIR=/workspace/rlwrld_isaac
ENV GROOT_DIR=/workspace/gr00t

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    git \
    git-lfs \
    libasound2 \
    libatk-bridge2.0-0 \
    libatk1.0-0 \
    libcups2 \
    libdbus-1-3 \
    libegl1 \
    libgl1 \
    libglib2.0-0 \
    libgtk-3-0 \
    libice6 \
    libnss3 \
    libsm6 \
    libx11-6 \
    libxcb1 \
    libxcomposite1 \
    libxcursor1 \
    libxext6 \
    libxi6 \
    libxkbcommon0 \
    libxrandr2 \
    libxrender1 \
    libxtst6 \
    mesa-vulkan-drivers \
    openssh-client \
    pkg-config \
    python3-dev \
    unzip \
    vulkan-tools \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /workspace

RUN mkdir -p /root/.ssh \
    && ssh-keyscan github.com >> /root/.ssh/known_hosts

RUN git clone "${ISAACLAB_REPO}" "${ISAACLAB_DIR}" \
    && cd "${ISAACLAB_DIR}" \
    && git checkout "${ISAACLAB_REF}"

RUN --mount=type=ssh git clone "${RLWRLD_ISAAC_REPO}" "${ISAAC_DIR}" \
    && cd "${ISAAC_DIR}" \
    && git checkout "${RLWRLD_ISAAC_REF}" \
    && uv venv --python 3.11 \
    && source .venv/bin/activate \
    && uv pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128 \
    && uv pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com \
    && uv pip install cmake==3.22.1 setuptools \
    && "${ISAACLAB_DIR}/isaaclab.sh" -i \
    && uv pip install --no-build-isolation flatdict==4.0.1 \
    && uv pip install -e "source/allex_sim[remote]"

RUN --mount=type=ssh GIT_LFS_SKIP_SMUDGE=1 git clone "${GROOT_REPO}" "${GROOT_DIR}" \
    && cd "${GROOT_DIR}" \
    && git checkout "${GROOT_REF}" \
    && uv venv --python 3.10 \
    && source .venv/bin/activate \
    && GIT_LFS_SKIP_SMUDGE=1 uv pip install -e ".[base,allex]" \
    && uv pip install flash-attn==2.7.4.post1 --no-build-isolation \
    && uv pip install "pydantic==2.10.6" "numpydantic==1.6.7"

RUN source "${ISAAC_DIR}/.venv/bin/activate" \
    && uv pip install --no-build-isolation -e "${ISAACLAB_DIR}/source/isaaclab"

RUN "${ISAAC_DIR}/.venv/bin/python" -c 'import importlib.util; [importlib.util.find_spec(m) or (_ for _ in ()).throw(RuntimeError(f"missing {m}")) for m in ["isaaclab", "isaaclab_tasks", "isaacsim", "allex_sim"]]; print("isaac env imports ok")' \
    && "${GROOT_DIR}/.venv/bin/python" -c 'import importlib.util; [importlib.util.find_spec(m) or (_ for _ in ()).throw(RuntimeError(f"missing {m}")) for m in ["torch", "transformers", "wandb"]]; print("gr00t env imports ok")'

RUN apt-get update && apt-get install -y --no-install-recommends \
    iproute2 \
    libglu1-mesa \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
CMD ["/bin/bash"]
