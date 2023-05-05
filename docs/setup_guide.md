# Setup Guide

Currently, we provide two options for application use: 1.) `local setup` for direct access to the source code or 2.) `docker setup` for a containerized version of the application. Note that the use may be simplified by utilizing the second approach.

For using either of the options first clone the repository:

```sh
git clone https://github.com/Agricultural-institute/siapy.git
```

## Local setup

To get a local copy up and running you need to follow the steps below.

### Requirements

* Python >= `3.8`
* Packages included in `requirements.txt` file
* Anaconda for easy installation (not necessary)

### Install dependencies

Create and activate a virtual environment:

```sh
conda create -n siapy-env python=3.9 anaconda
conda activate siapy-env
```

Install packages into the virtual environment:

```sh
cd siapy
python -m pip install -r requirements.txt
```

If using the application on windows machine, you need to install Microsoft C++ Build Tools from <https://visualstudio.microsoft.com/visual-cpp-build-tools/>
. Check the box that you also want to install windows SDK.

If inpoly was not installed, run:
```
pip show inpoly
pip install inpoly
```

Also you need to install additional dependencies for hydra:
```
pip install hydra-core --upgrade
pip install hydra_colorlog --upgrade
```

You can use the application locally by running:

```sh
python3 main.py COMMANDS
```

## Setup using docker

If you would like to use a containerized version of the application, you should follow the steps below.

1. Install [docker](https://docs.docker.com/engine/install/)

2. Install [xserver](https://sourceforge.net/projects/xming/)

   After installation, open it in the background. Note that the server is only needed for windows machines.

3. Go to the project's root and make `run.sh` file executable:

```sh
cd siapy
chmod u+x run.sh

```

4. Run the application by using the command:

```sh
./run.sh

```

Command, if executed appropriately, will open the shell inside the container.
