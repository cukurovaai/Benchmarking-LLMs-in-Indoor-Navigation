# Base image with NVIDIA CUDA support
FROM nvidia/cudagl:10.1-devel-ubuntu18.04

# Add NVIDIA GPG keys and repositories
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub && \
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub && \
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda.list && \
    echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    git-lfs \
    curl \
    vim \
    ca-certificates \
    libjpeg-dev \
    libpng-dev \
    libglfw3-dev \
    libglm-dev \
    libx11-dev \
    libomp-dev \
    libegl1-mesa-dev \
    libgl1-mesa-glx \
    pkg-config \
    wget \
    zip \
    libxi-dev \
    libxcursor-dev \
    libxinerama-dev \
    libxrandr-dev \
    x11-apps \
    x11-xserver-utils \
    unzip && \
    rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN curl -L -o ~/miniconda.sh -O https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -k && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda install numpy pyyaml scipy ipython mkl mkl-include -y && \
    /opt/conda/bin/conda clean -ya

# Add conda to PATH
ENV PATH /opt/conda/bin:$PATH

# Install CMake
RUN wget https://github.com/Kitware/CMake/releases/download/v3.14.0/cmake-3.14.0-Linux-x86_64.sh && \
    mkdir /opt/cmake && \
    sh ./cmake-3.14.0-Linux-x86_64.sh --prefix=/opt/cmake --skip-license && \
    ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake && \
    rm cmake-3.14.0-Linux-x86_64.sh

# Create conda environment
RUN conda create -n bench_llm_nav python=3.8 -y

# Clone vlmaps repository and set up
RUN git clone https://github.com/cukurovaai/Benchmarking-LLMs-in-Indoor-Navigation.git /Bench_LLM_Nav
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate bench_llm_nav && conda install jupyter -y && cd /Bench_LLM_Nav && python -m pip install --upgrade pip==24.0 && bash install.bash && pip install -e ."

# Copy the Python script that downloads the episodes into the Docker image
COPY ./download_mp.py /Bench_LLM_Nav/dataset/download_mp.py

# Download Matterport3D data to the 'scans' and 'tasks' folders within the repository
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate bench_llm_nav && \
    python /Bench_LLM_Nav/dataset/download_mp.py -o /Bench_LLM_Nav/tasks --task habitat && \
    unzip /Bench_LLM_Nav/tasks/v1/tasks/mp3d_habitat.zip -d /Bench_LLM_Nav/tasks && \
    rm -r /Bench_LLM_Nav/tasks/v1 && \
    mkdir /Bench_LLM_Nav/drive"

# Download the LSeg checkpoint
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate bench_llm_nav && \
    python /Bench_LLM_Nav/application/lseg_download.py"

RUN echo "source /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo 'export PYTHONPATH="${PYTHONPATH}:/vlmaps/"' >> ~/.bashrc && \
    echo 'export OPENAI_KEY="<your copied key>"' >> ~/.bashrc && \
    echo "conda activate bench_llm_nav" >> ~/.bashrc

RUN /bin/bash -c "mkdir /Bench_LLM_Nav/drive/vlmaps_dataset"

# Set the working directory
WORKDIR /Bench_LLM_Nav
