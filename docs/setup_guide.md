Setup Guide
==================

Currently, we provide two options for application use: 1.) `local setup` for direct access to the source code or 2.) `docker setup` for a containerized version of the application. Note that the use may be simplified by utilizing the second approach.

For using either of the options first clone the repository:
```sh
$ git clone https://github.com/Agricultural-institute/SiaPy.git
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
$ conda create -n siapy-env python=3.8
$ conda activate siapy-env
```

Install packages into the virtual environment:
```sh
$ cd SiaPy
$ python -m pip install -r requirements.txt
```

You can use the application locally by running:
```
$ python3 main.py COMMANDS
```

## Setup using docker

If you would like to use a containerized version of the application, you should follow the steps below.

1. Install [docker](https://docs.docker.com/engine/install/)
2. Install [xserver](https://sourceforge.net/projects/xming/)

    After installation, open it in the background. Note that the server is only needed for windows machines.

3. Go to the project's root and make `run.sh` file executable:

    ```sh
    cd Siapy
    chmod u+x run.sh
    ```

4. Run the application by using the command:

    ```sh
    ./run.sh
    ```
    Command, if executed appropriately, will open the shell inside the container.





