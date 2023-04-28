from detection import detection_recognition
import glob
import time
import streamlit as st
import warnings
warnings.filterwarnings('ignore')


tich_file= glob.glob('./Object_Detection/*')
paths_nomer= './model1_nomer.tflite'
paths_resnet= './model_resnet.tflite'


st.set_page_config(page_title="Распознавание номеров", layout="wide", page_icon="random")
st.header('Сервис по распознаванию автомобильных номеров')

uploaded_images = st.file_uploader("Choose video", type=["jpg", "png", "jpeg"])

if uploaded_images is not None: # run only when user uploads video

    vid = uploaded_images.name

    # Блок распознавания
    exec_time = time.time()
    number, detection_img = detection_recognition(tich_file, paths_nomer, paths_resnet)

    # st.markdown(
    #     number,
    #     unsafe_allow_html=True,
    # )
    st.header(f'Распознанный номер: {number[0]}')

    st.image(uploaded_images, channels="BGR")