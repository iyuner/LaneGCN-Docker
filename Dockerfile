FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

# Common shell utils
RUN apt-get update && \
    apt-get install -y git sudo wget htop python3-pip tmux

# Miniconda
# RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
#     sh miniconda.sh -b -p $HOME/miniconda && \
#     eval "$(/root/miniconda/bin/conda shell.bash hook)" &&\
#     conda init && \
#     rm miniconda.sh

# Miniconda (UPDATE: It it better not isntalled in the root dir)
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh

# Add conda to the path
ENV PATH /root/miniconda3/bin:$PATH

COPY . /var/lanegcn-docker
EXPOSE 80
# 
WORKDIR /var/lanegcn-docker
# ENTRYPOINT /bin/bash

# Add conda to the path
ENV PATH /root/miniconda3/bin:$PATH

RUN conda create --quiet -y --name lanegcn python=3.7
# RUN conda init bash

# Add conda installation dir to PATH (instead of doing 'conda activate')
ENV PATH /opt/conda/envs/lanegcn/bin:$PATH

RUN sudo apt install -y libopenmpi-dev

RUN pip install mpi4py
RUN sudo apt install -y libgl1-mesa-glx

# RUN source activate lanegcn
RUN conda install pytorch==1.5.1 torchvision cudatoolkit=10.2 -c pytorch # pytorch=1.5.1 when the code is release

# install argoverse api
RUN pip install  git+https://github.com/argoai/argoverse-api.git

# install others dependancy
RUN pip install scikit-image IPython tqdm ipdb


# install horovod with GPU support, this may take a while
RUN HOROVOD_GPU_OPERATIONS=NCCL pip install horovod==0.19.4

# if you have only SINGLE GPU, install for code-compatibility
RUN pip install horovod

RUN python -c "print('here!')"

# From now on, use bash
SHELL ["bash", "-c"]


# found out... all dependencies are installed in the /root/miniconda3/bin/pip
# so not in the env-lanegcn