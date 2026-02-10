ARG UBUNTU_VERSION=22.04
ARG CUDA_MAJOR_VERSION=11.8.0
ARG PYTORCH_VERSION=2.0.1
ARG TORCHVISION_VERSION=0.15.2

FROM nvidia/cuda:${CUDA_MAJOR_VERSION}-runtime-ubuntu${UBUNTU_VERSION}

# propagate build args
ARG CUDA_MAJOR_VERSION
ARG PYTORCH_VERSION
ARG TORCHVISION_VERSION

ARG USER_UID=1001
ARG USER_GID=1001

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

USER root

RUN groupadd --gid ${USER_GID} user \
    && useradd -m --no-log-init --uid ${USER_UID} --gid ${USER_GID} user

RUN mkdir /input /output \
    && chown user:user /input /output

WORKDIR /home/user
ENV PATH="/home/user/.local/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    vim \
    screen \
    zip \
    unzip \
    git \
    openssh-server \
    python3-pip \
    python3-dev \
    python-is-python3 \
    python3-venv \
    libglib2.0-0 \
    libgl1 \
    libjpeg-turbo8 \
    libopenjp2-7 \
    libopenslide0 \
    libpng16-16 \
    libtiff5 \
    libtiff-dev \
    zlib1g-dev \
    && mkdir /var/run/sshd \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# install ASAP
ARG ASAP_URL=https://github.com/computationalpathologygroup/ASAP/releases/download/ASAP-2.2-(Nightly)/ASAP-2.2-Ubuntu2204.deb
RUN apt-get update && curl -L ${ASAP_URL} -o /tmp/ASAP.deb && apt-get install --assume-yes /tmp/ASAP.deb && \
    SITE_PACKAGES=`python3 -c "import sysconfig; print(sysconfig.get_paths()['purelib'])"` && \
    printf "/opt/ASAP/bin/\n" > "${SITE_PACKAGES}/asap.pth" && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt/app

RUN python -m pip install --upgrade "pip<25.3" setuptools wheel \
    && python -m pip install --upgrade pip-tools \
    && rm -rf /home/user/.cache/pip

COPY --chown=user:user requirements.in /opt/app/requirements.in
RUN CUDA_IDENTIFIER_PYTORCH=$(echo "cu${CUDA_MAJOR_VERSION}" | sed "s|\.||g" | cut -c1-5) \
    && sed -i -e "s|%PYTORCH_VERSION%|${PYTORCH_VERSION}|g" requirements.in \
    && sed -i -e "s|%TORCHVISION_VERSION%|${TORCHVISION_VERSION}|g" requirements.in \
    && python -m piptools compile requirements.in --verbose \
      --index-url https://pypi.org/simple \
      --extra-index-url https://download.pytorch.org/whl/${CUDA_IDENTIFIER_PYTORCH} \
    && python -m piptools sync \
    && rm -rf ~/.cache/pip*

EXPOSE 22 8888

RUN chown -R user:user /opt
RUN chmod -R 775 /opt

# switch to user
USER user