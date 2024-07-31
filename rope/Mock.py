# #!/usr/bin/env python3
# Rope/rope/Mock.py (place in the same directory as Coordinator.py)

import time
import torch
from torchvision import transforms
import json

import rope.VideoManager as VM
import rope.Models as Models
from rope.external.clipseg import CLIPDensePredT
from rope.FaceHelper import FaceHelper

from rope.Logger import get_logger

logger = get_logger()

resize_delay = 1
mem_delay = 1

def load_clip_model():
    # https://github.com/timojl/clipseg
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_session = CLIPDensePredT(version='ViT-B/16', reduce_dim=64, complex_trans_conv=True)
    # clip_session = CLIPDensePredTMasked(version='ViT-B/16', reduce_dim=64)
    clip_session.eval()
    clip_session.load_state_dict(torch.load('./models/rd64-uni-refined.pth'), strict=False) 
    clip_session.to(device)    
    return clip_session 

def step():
    if vm.get_action_length() > 0:
        action.append(vm.get_action())
    
    if len(action) > 0:
        if action[0][0] == "stop_play":
            return False, vm.current_frame
        action.pop(0)
    
    vm.process()
    return True, vm.current_frame
    
    
def run(source_image, target_video, save_path, start_at, params_path):
    global vm, action, frame, r_frame, resize_delay, mem_delay

    with open(params_path, 'r') as f:
        parameters = json.load(f)

    logger.info(
        f"Rope configs: \n"
        f"- Source Image: \033[93m{source_image}\033[0m ;\n"
        f"- target_video: \033[93m{target_video}\033[0m ;\n"
        f"- Save Path: \033[93m{save_path}\033[0m ;\n"
        f"- Begin from frame \033[93m[{start_at}]\033[0m ;\n"
        f"Loading parameters from \033[93m{params_path}\033[0m, notably:\n" + 
        (f"- Using \033[93m{parameters['RestorerTypeTextSel']}\033[0m;\n" if parameters['RestorerSwitch'] else "") +
        f"- Using number of Threads \033[95m<< {parameters['ThreadsSlider']} >>\033[0m.\n" + 
        f"- (read JSON file for more detail)"
    )

    models = Models.Models()
    vm = VM.VideoManager(models)

    action = []
    frame = []
    r_frame = []    

    vm.parameters = parameters
    vm.control = {
        "AudioButton": False, 
        "SwapFacesButton": True, 
        "MaskViewButton": False
    }

    vm.load_target_video(target_video)
    vm.saved_video_path = save_path

    sample_frame = vm.get_requested_frame()[0]
    face_helper = FaceHelper(models, sample_frame, source_image, parameters)
    vm.assign_found_faces(face_helper.target_faces)

    vm.current_frame = start_at
    vm.play_video("record")

    frame = start_at
    
    interrupts = 0
    while True:
        try:
            flag, frame = step()
            if not flag:
                break
            step()
            time.sleep(0.1)
            logger.info(f"Current frame: {frame} / {vm.video_frame_total}", end='\r')
        except KeyboardInterrupt:
            vm.play = False
            logger.warn(f"Attempted to stop recording at {frame} / {vm.video_frame_total}")
            interrupts += 1
        if interrupts >= 3:
            logger.info("Forced exit")
            vm.play_video("stop")
            exit()
        elif interrupts >= 2:
            logger.warn("Warning: ^C once more will force program to exit.")
    


