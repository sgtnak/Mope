import numpy as np
import torch
import cv2
import os
import mimetypes

class FaceHelper:
    def __init__(self, models, video_image, source_face, parameters):
        self.models = models
        self.video_image = video_image
        self.parameters = parameters

        self.target_faces = []
        self.target_face = {    
                            "TKButton":                 [],
                            "ButtonState":              "off",
                            "Image":                    [],
                            "Embedding":                [],
                            "SourceFaceAssignments":    [],
                            "EmbeddingNumber":          0,       #used for adding additional found faces
                            'AssignedEmbedding':        [],     #the currently assigned source embedding, including averaged ones
                            }
        
        self.source_face = source_face
        self.source_embedding = None
        self.prepare()

    def prepare(self):
        self.load_input_faces()
        self.find_faces()
        self.assign_face()

    def load_input_faces(self):
        self.source_faces = []

        # Next Load images
        
        file = self.source_face
        bad_image = False
        try:
            file_type = mimetypes.guess_type(file)[0][:5]
        except:
            print('Unrecognized file type:', file)
            bad_image = True
        if bad_image:
            raise ValueError("Bad image input:" + file)
            
        # Its an image
        if file_type == 'image':
            img = cv2.imread(file)

            if img is not None:
                img = torch.from_numpy(img.astype('uint8')).to('cuda')

                pad_scale = 0.2
                padded_width = int(img.size()[1]*(1.+pad_scale))
                padded_height = int(img.size()[0]*(1.+pad_scale))

                padding = torch.zeros((padded_height, padded_width, 3), dtype=torch.uint8, device='cuda:0')

                width_start = int(img.size()[1]*pad_scale/2)
                width_end = width_start+int(img.size()[1])
                height_start = int(img.size()[0]*pad_scale/2)
                height_end = height_start+int(img.size()[0])

                padding[height_start:height_end, width_start:width_end,  :] = img
                img = padding

                img = img.permute(2,0,1)
                try:
                    kpss = self.models.run_detect(img, max_num=1)[0] # Just one face here
                except IndexError:
                    print('Image cropped too close:', file)
                else:
                    face_emb, cropped_image = self.models.run_recognize(img, kpss)
                    crop = cv2.cvtColor(cropped_image.cpu().numpy(), cv2.COLOR_BGR2RGB)
                    crop = cv2.resize(crop, (85, 85))

                    # new_source_face = self.source_face.copy()
                    # self.source_faces.append(new_source_face)

                    # self.source_faces[-1]["Embedding"] = face_emb
                    # self.source_faces[-1]["ButtonState"] = False
                    # self.source_faces[-1]["file"] = file
                    self.source_embedding = face_emb

            else:
                print('Bad file', file)
                raise ValueError('Bad file' + file)
        else:
            raise ValueError('Not an image:' + file)

        torch.cuda.empty_cache()
    
    def find_faces(self):
        try:
            img = torch.from_numpy(self.video_image).to('cuda')
            img = img.permute(2,0,1)
            kpss = self.models.run_detect(img, max_num=50)

            ret = []
            for face_kps in kpss:

                face_emb, cropped_img = self.models.run_recognize(img, face_kps)
                ret.append([face_kps, face_emb, cropped_img])

        except Exception:
            print(" No media selected")

        else:
            # Find all faces and add to target_faces[]
            if ret:
                # Apply threshold tolerence
                threshhold = self.parameters["ThresholdSlider"]

                # if self.parameters["ThresholdState"]:
                    # threshhold = 0.0

                # Loop thgouh all faces in video frame
                for face in ret:
                    found = False

                    # Check if this face has already been found
                    for emb in self.target_faces:
                        if self.findCosineDistance(emb['Embedding'], face[1]) >= threshhold:
                            found = True
                            break

                    # If we dont find any existing simularities, it means that this is a new face and should be added to our found faces
                    if not found:
                        crop = cv2.resize(face[2].cpu().numpy(), (82, 82))

                        new_target_face = self.target_face.copy()
                        self.target_faces.append(new_target_face)
                        last_index = len(self.target_faces)-1

                        self.target_faces[last_index]["ButtonState"] = False
                        self.target_faces[last_index]["Embedding"] = face[1]
                        self.target_faces[last_index]["EmbeddingNumber"] = 1

    def assign_face(self):
        for i in range(len(self.target_faces)):
            self.target_faces[i]["AssignedEmbedding"] = self.source_embedding
            self.target_faces[i]["SourceFaceAssignments"] = [0]
    
    
    def findCosineDistance(self, vector1, vector2):
        cos_dist = 1 - np.dot(vector1, vector2)/(np.linalg.norm(vector1)*np.linalg.norm(vector2)) # 2..0
        return 100-cos_dist*50