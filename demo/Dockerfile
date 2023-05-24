# read the doc: https://huggingface.co/docs/hub/spaces-sdks-docker
# you will also find guides on how best to write your Dockerfile

FROM python:3.7

WORKDIR /code

# install dependencies
COPY ./requirements.txt /code/requirements.txt
RUN python3 -m pip install --no-cache-dir --upgrade -r /code/requirements.txt

# resolve issue with tf==2.4 and gradio versioning issue
RUN python3 -m pip install --force-reinstall typing_extensions==4.0.0

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

CMD ["python3", "app.py"]
