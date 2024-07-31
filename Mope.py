#!/usr/bin/env python3

# Rope/Mope.py (placed in the same directory as Rope.py)

import os
import argparse

from rope import Mock

parser = argparse.ArgumentParser(
    prog='Rope (No GUI)',
    description='Rope with GUI removed'
)

parser.add_argument('-s', '--source', type=str, default="data/faces/qzx2.jpg", help="Source face image, used to swap the target video")
parser.add_argument('-t', '--target', type=str, default="data/videos/zhurui_half1_2min.mp4", help="Target video to be swapped")
parser.add_argument('-o', '--output', type=str, default="data/output", help="Folder to output swapped video")
parser.add_argument('-p', '--params', type=str, default="saved_parameters.json", help="JSON file to load parameters from")
parser.add_argument('-b', '--begin_from', type=int, default=0, help="Which frame to start swapping from")
parser.add_argument('-d', '--device', type=str, default='0', help="Set CUDA visible device")

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)   # or whatever

if __name__ == "__main__":
    Mock.run(args.source, args.target, args.output, args.begin_from, args.params)