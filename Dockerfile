# read the doc: https://huggingface.co/docs/hub/spaces-sdks-docker
# you will also find guides on how best to write your Dockerfile
FROM tensorflow/tensorflow:2.4.2-gpu

# fix for broken keys in Ubuntu-18.04
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub

# install Python 3.7
RUN apt-get update && apt-get install -y --no-install-recommends software-properties-common \
    libsm6 libxext6 libxrender-dev curl \
    && rm -rf /var/lib/apt/lists/*

RUN echo "**** Installing Python ****" && \
    add-apt-repository ppa:deadsnakes/ppa &&  \
    apt-get install -y build-essential python3.7 python3.7-dev python3-pip && \
    curl -O https://bootstrap.pypa.io/get-pip.py && \
    python3.7 get-pip.py && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /code

RUN apt-get update -y
RUN apt install git --fix-missing -y

# Download pretrained parenchyma model
RUN wget "https://github.com/andreped/livermask/releases/download/trained-models-v1/model.h5"
COPY ./model.h5 /code/model.h5

# install dependencies
COPY ./demo/requirements.txt /code/demo/requirements.txt
RUN python3.7 -m pip install --no-cache-dir --upgrade -r /code/demo/requirements.txt

# resolve issue with tf==2.4 and gradio dependency collision issue
RUN python3.7 -m pip install --force-reinstall typing_extensions==4.0.0

# Set up a new user named "user" with user ID 1000
RUN useradd -m -u 1000 user

# Switch to the "user" user
USER user

# Set home to the user's home directory
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set the working directory to the user's home directory
WORKDIR $HOME/app

# Copy the current directory contents into the container at $HOME/app setting the owner to the user
COPY --chown=user . $HOME/app

CMD ["python3.7", "demo/app.py"]
