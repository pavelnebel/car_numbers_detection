from detection import detection_recognition
import glob
import time
import streamlit as st
import warnings
import numpy as np
import cv2
warnings.filterwarnings('ignore')


#uploaded_images= glob.glob('./Object_Detection/*')
paths_nomer= './model1_nomer.tflite'
paths_resnet= './model_resnet.tflite'

st.set_page_config(page_title="Распознавание номеров", layout="wide", page_icon="random")
st.header('Сервис по распознаванию автомобильных номеров')

uploaded_images = st.file_uploader("Choose image", type=["jpg", "png", "jpeg"])

if uploaded_images is not None: # run only when user uploads video

    file_bytes = np.asarray(bytearray(uploaded_images.read()), dtype=np.uint8)
    uploaded_images = cv2.imdecode(file_bytes, 1)

    # Блок распознавания
    exec_time = time.time()
    number, detection_img = detection_recognition(uploaded_images, paths_nomer, paths_resnet)

    # st.markdown(
    #     number,
    #     unsafe_allow_html=True,
    # )
    st.header(f'Распознанный номер: {number[0]}')

    st.image(detection_img, channels="BGR")