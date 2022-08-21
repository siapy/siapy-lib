# escape=`
# https://towardsdatascience.com/complete-tutorial-on-building-images-using-docker-1f2be49ea8a6

# base image
ARG UBUNTU_VERSION=18.04
FROM ubuntu:$UBUNTU_VERSION

# disable interactive
ENV DEBIAN_FRONTEND noninteractive
# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# update ubuntu and install utilities
RUN apt update --fix-missing &&  `
    apt install -y --no-install-recommends ca-certificates git sudo curl && `
    apt clean

# set working directory
WORKDIR /app

# install miniconda and dependencies
ARG MINICONDA_DOWNLOAD_LINK=https://repo.anaconda.com/miniconda/Miniconda3-py38_4.12.0-Linux-x86_64.sh

# allow us to run conda from anywhere in the file system
ENV PATH /app/miniconda3/bin:$PATH
RUN curl $MINICONDA_DOWNLOAD_LINK --create-dirs -o Miniconda.sh && `
    bash Miniconda.sh -b -p ./miniconda3 && `
    rm Miniconda.sh && `
    conda init && `
    conda update -y --all && `
    python -m pip install --upgrade pip setuptools wheel

# for matplotlib plots
RUN apt-get update -y && `
    apt-get install -y libx11-dev && `
    apt-get install -y python3-tk

# install open-cv
RUN apt-get update && `
    apt-get install -y python3-opencv && `
    pip install opencv-python

# Install pip requirements
COPY ./requirements.txt /app
RUN pip install --no-cache-dir -r requirements.txt && `
    rm requirements.txt

# copy scripts to the folder
COPY ./siapy/ /app/siapy
COPY ./main.py /app


# add user with sudo privileges
# disable passord prompt and add alias for easier program handling
RUN adduser --disabled-password --gecos '' user && `
    adduser user sudo && `
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && `
    echo 'run() { python3 main.py program=$1; }' >> /home/user/.bashrc

USER user
RUN sudo apt-get update
