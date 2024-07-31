#!/bin/bash

. activate base
conda activate Rope
python Mope.py -s data/face/sample.png -t benchmark/target-1080p.mp4 -o data/output -d 0

