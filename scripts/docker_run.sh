#!/bin/bash
# get network address of computer
IP_ADDRS=$(ipconfig.exe | grep 'Ethernet' -A4 | cut -d":" -f 2 | tail -n1 | sed -e 's/\s*//g')
# set display variable
DISPLAY=$IP_ADDRS:0.0
# grab data directory path from configs file
DATA_DIR_PATH=$(cat ../configs/data_loader/data_loader.yaml | grep data_dir_path | cut --complement -d ":" -f 1)
# replace section of path to set wsl form of path
# DATA_DIR_PATH=${DATA_DIR_PATH/"C:"/"/mnt/c"}
# set data directory mapping
DATA_DIR_MAP=$DATA_DIR_PATH:/app/data
# set serverX mapping (not necessary on windows)
SERVER_MAP=/tmp/.X11-unix:/tmp/.X11-unix:rw
# set outputs mapping
OUT_MAP=$(pwd)/../outputs:/app/outputs
# set configs mapping
CONFIG_MAP=$(pwd)/../configs:/app/configs
# run docker container
docker run --rm --name siapy-app -e DISPLAY=$DISPLAY -v $SERVER_MAP -v $OUT_MAP -v $CONFIG_MAP -v $DATA_DIR_MAP -it siapy-api
