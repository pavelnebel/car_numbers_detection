from detection import detection_recognition
import glob
import time
import streamlit as st
import warnings
import numpy as np
import cv2
import base64
from pathlib import Path
from PIL import Image
warnings.filterwarnings('ignore')

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)

    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str

    st.markdown(page_bg_img, unsafe_allow_html=True)
    return

# Конфигурирование страницы
im = Image.open(Path.cwd()/'APP_icon'/'Иконка.png')
st.set_page_config(page_title="Распознавание", layout="wide", page_icon=im)

# Устанавливаем фон
set_png_as_page_bg(Path.cwd()/'APP_bg'/'Bg.jpg')

#uploaded_images= glob.glob('./Object_Detection/*')
paths_nomer= './model1_nomer.tflite'
paths_resnet= './model_resnet.tflite'

#st.set_page_config(page_title="Распознавание номеров", layout="wide", page_icon="random")

st.header('Сервис по распознаванию автомобильных номеров')

url = 'https://t.me/pavelnebel'
full_ref = f'<a href="{url}" style="color: #0d0aab">by FriendlyDev</a>'
st.markdown(f"<h2 style='font-size: 20px; text-align: right; color: black;'>{full_ref}</h2>", unsafe_allow_html=True)


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

else:
    st.header('preview')

    path_gif1 = 'preview/preview.jpg'
    file1_ = open(path_gif1, "rb")
    contents1 = file1_.read()

    path_gif2 = 'preview/preview_result.jpg'
    file2_ = open(path_gif2, "rb")
    contents2 = file2_.read()

    data_url1 = base64.b64encode(contents1).decode("utf-8")
    data_url2 = base64.b64encode(contents2).decode("utf-8")
    file1_.close()
    file2_.close()


    st.markdown(
        f'<img src="data:image/gif;base64,{data_url1}" alt="cat gif">',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<img src="data:image/gif;base64,{data_url2}" alt="cat gif">',
        unsafe_allow_html=True,
    )