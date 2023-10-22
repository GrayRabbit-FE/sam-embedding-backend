import base64
import numpy as np
import cv2
from segment_anything import sam_model_registry, SamPredictor
import torch
import time
from flask import Flask, request

# check torch version and if cuda is available
print(torch.__version__)
print(torch.cuda.is_available())
# checkpoint = "sam_vit_b_01ec64.pth"
print(time.time())
# set large model related configs
checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint=checkpoint)
sam.to(device='cuda')
predictor = SamPredictor(sam)

app = Flask(__name__)


def base64_to_image(base64_code):
    img_data = base64.b64decode(base64_code)
    img_array = np.fromstring(img_data, np.uint8)
    img = cv2.imdecode(img_array, cv2.COLOR_RGB2BGR)
    return img


@app.route('/embedding', methods=["GET", "POST"])
def index():
    print('request received')
    if (request.method != 'POST'):
        return 'Only support POST method'

    content_type = request.headers.get('Content-Type')
    if (content_type == 'application/json'):
        print('start calculate embedding')
        base64data = request.get_json()['imgurl']
        image = base64_to_image(base64data)
        predictor.set_image(image)
        image_embedding = predictor.get_image_embedding().cpu().numpy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print(time.time())
        return image_embedding
    else:
        return 'Content-Type not supported!'


app.run(port=2020, host="0.0.0.0", debug=True)
print('server starts working')
