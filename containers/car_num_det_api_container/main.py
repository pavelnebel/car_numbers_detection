import os

from detection import detection_recognition

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

import warnings
import numpy
import numpy as np
import cv2
import base64

warnings.filterwarnings('ignore')


class PredictRequest(BaseModel):
    user: str
    image: str
    image_name: str


def image_bytes_to_str(im_path):
    with open(im_path, mode='rb') as file:
        image_bytes = file.read()
    image_str = base64.encodebytes(image_bytes).decode('utf-8')
    return image_str


app = FastAPI()

paths_nomer = './model1_nomer.tflite'
paths_resnet = './model_resnet.tflite'


@app.get("/")
def index():
    return {
        "message: Index!"
    }


@app.post("/image")
def get_image(json_input: PredictRequest):
    #  deserialization
    image_bytes = base64.b64decode(json_input.image)

    # image preprocessing
    file_bytes: numpy.ndarray = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    input_image_cv: numpy.ndarray = cv2.imdecode(file_bytes, 1)

    # receiving prediction
    detection_img: numpy.ndarray
    number, detection_img = detection_recognition(input_image_cv, paths_nomer, paths_resnet)
    # number: ['C086CC61']

    # saving tagged image
    if not os.path.exists('users_detections'): os.makedirs('users_detections')
    save_path = f'users_detections/{json_input.image_name}'
    cv2.imwrite(save_path, detection_img)

    # serialization
    json_out = {}
    json_out['tagged_image'] = image_bytes_to_str(save_path)
    json_out['detected_number'] = number

    return json_out


if __name__ == '__main__':
    uvicorn.run(
        "main:app",
        port=6688,
        # reload=True,
    )
