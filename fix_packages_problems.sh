#!/bin/bash

apt update
apt install firefox code -y
pip install waymo-open-dataset-tf-2-6-0
pip install keras==2.6.*
pip install tensorflow==2.6.*