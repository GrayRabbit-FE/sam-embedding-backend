import numpy as np
import cv2
from segment_anything import sam_model_registry, SamPredictor
import torch
import time


# check torch version and if cuda is available
print(torch.__version__)
print(torch.cuda.is_available())
# checkpoint = "sam_vit_b_01ec64.pth"
print(time.time())
# set large model related configs
checkpoint = "notebooks/sam_vit_h_4b8939.pth"
model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint=checkpoint)
sam.to(device='cuda')
predictor = SamPredictor(sam)

image_name = 'r.jpg'
image = cv2.imread('notebooks/images/' + image_name)
predictor.set_image(image)
image_embedding = predictor.get_image_embedding().cpu().numpy()
np.save(image_name + ".npy", image_embedding)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)