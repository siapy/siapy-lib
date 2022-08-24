#!/bin/bash
# create output folders and set permissions
mkdir -pm 0777 outputs
# get network address of computer
IP_ADDRS=$(ipconfig.exe | grep 'Ethernet' -A4 | cut -d":" -f 2 | tail -n1 | sed -e 's/\s*//g')
# set display variable
export DISPLAY=$IP_ADDRS:0.0
# grab data directory path from configs file
DATA_DIR_PATH=$(cat ./configs/data_loader/data_loader.yaml | grep data_dir_path | cut --complement -d ":" -f 1)
# set data directory mapping
DATA_DIR_MAP=$DATA_DIR_PATH:/app/data
# set serverX mapping (not necessary on windows)
SERVER_MAP=/tmp/.X11-unix:/tmp/.X11-unix:rw
# run docker-compose to start
docker build --rm -t siapy-api .
docker-compose run --rm -v $DATA_DIR_MAP -v $SERVER_MAP --name siapy-main siapy-api
