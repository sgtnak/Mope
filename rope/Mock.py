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

from tqdm import tqdm

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
    
    
def run(source_image, target_video, save_path, start_at, params_path, logger_path="log.txt"):
    global vm, action, frame, r_frame, resize_delay, mem_delay

    with open(params_path, 'r') as f:
        parameters = json.load(f)

    logger = get_logger(logger_path)
    logger.info(
        f"Rope configs: \n"
        f"- Source Image: {source_image};\n"
        f"- target_video: {target_video} ;\n"
        f"- Save Path: {save_path} ;\n"
        f"- Begin from frame [{start_at}] ;\n"
        f"Loading parameters from {params_path}, notably:\n" + 
        (f"- Using {parameters['RestorerTypeTextSel']};\n" if parameters['RestorerSwitch'] else "") +
        f"- Using number of Threads << {parameters['ThreadsSlider']} >>.\n" + 
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

    models.load_models()

    vm.current_frame = start_at
    vm.play_video("record")

    frame = start_at
    last_frame = 0
    
    interrupts = 0
    start_time = time.time()
    last_output = time.time()
    with tqdm(total=vm.video_frame_total, ncols=120) as pbar:
        while True:
            try:
                flag, frame = step()
                if not flag:
                    break
                step()
                time.sleep(0.01)
                if time.time() - last_output > 1:
                    # logger.info(f"[{int(time.time() - start_time)}s] Current frame: {frame} / {vm.video_frame_total}")
                    logger.info(pbar.__str__())
                    pbar.update(frame - last_frame)
                    last_frame = frame
                    last_output = time.time()
            except KeyboardInterrupt:
                vm.play = False
                logger.warn(f"Warning: Attempted to stop recording at {frame} / {vm.video_frame_total}")
                interrupts += 1
            
            if interrupts >= 4:
                logger.info("Forced exit")
                vm.play_video("stop")
                exit()
            elif interrupts == 2:
                logger.warn("Warning: ^C once more will force program to exit.")
                interrupts += 1
    


