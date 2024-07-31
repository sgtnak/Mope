#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

conda activate Rope
python Mope.py -s data/faces/qzx2.jpg -t data/videos/zhurui_half1_2min.mp4 -o data/output
