# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.8-slim

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# set working directory
WORKDIR /app

# for matplotlib plots
RUN apt-get update -y
RUN apt-get install -y libx11-dev
RUN apt-get install -y python3-tk

# install open-cv
RUN apt-get update && apt-get install -y python3-opencv
RUN pip install opencv-python

# Install pip requirements
COPY ./requirements.txt /app
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# copy scripts to the folder
COPY ./siapy/ /app/siapy
COPY ./main.py /app

# Creates a non-root user with an explicit UID and adds permission to access the /app folder
RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /app
USER appuser

# During debugging, this entry point will be overridden
CMD ["/bin/bash"]
