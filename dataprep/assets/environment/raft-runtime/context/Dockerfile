FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest

WORKDIR /
ENV CONDA_PREFIX=/azureml-envs/raft-runtime
ENV CONDA_DEFAULT_ENV=$CONDA_PREFIX
ENV RAFT_DIR="/var/code/"

RUN apt-get -y update && apt-get -y upgrade && apt-get -y install libmagickwand-dev

# Clone Gorilla to {RAFT_DIR}
RUN git clone --no-checkout --depth=1 --filter=tree:0 https://github.com/cedricvidal/gorilla.git --branch raft-distillation-recipe --single-branch ${RAFT_DIR} && cd ${RAFT_DIR} && git sparse-checkout set --no-cone raft && git checkout && git pull

# # Create conda environment
COPY conda.yaml .
RUN conda env create -p $CONDA_PREFIX -f conda.yaml && \
    rm conda.yaml && \
    conda run -p $CONDA_PREFIX pip cache purge && \
    conda clean -a -y

# clean conda and pip caches
RUN rm -rf ~/.cache/pip

# Prepend path to AzureML conda environment
ENV PATH=$CONDA_PREFIX/bin:$PATH

RUN  pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
