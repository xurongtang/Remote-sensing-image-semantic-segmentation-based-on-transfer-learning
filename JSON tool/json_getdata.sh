#!/bin/bash

# clear
rm /home/rton/pyproj/ieee_building_extract/MyProj/data/train/label/*
rm /home/rton/pyproj/ieee_building_extract/MyProj/data/val/label/*
# run
python -u json_label_trans.py
# clear
rm /home/rton/pyproj/ieee_building_extract/Pytorch-UNet/data/masks/*
# copy
cd /home/rton/pyproj/ieee_building_extract/Pytorch-UNet/data/masks
cp /home/rton/pyproj/ieee_building_extract/MyProj/data/train/label/*.npy ./
cp /home/rton/pyproj/ieee_building_extract/MyProj/data/val/label/*.npy ./