import os
import cv2

import numpy as np
from skimage.feature import canny
from skimage.transform import hough_line, hough_line_peaks
from skimage.transform import rotate
from skimage.color import rgb2gray

# import matplotlib
# from matplotlib import pyplot as plt
# from matplotlib import cm
# import matplotlib.gridspec as gridspec

import itertools
# import glob
import tensorflow as tf
import requests, io


def detection_recognition(upload_images, paths_nomer, paths_resnet):
    # img_name1=upload_images
    # image = cv2.imread(img_name1)
    #
    # image0 = cv2.imread(img_name1, 1)
    image_height, image_width, _ = upload_images.shape
    image = cv2.resize(upload_images, (1024, 1024))
    image = image.astype(np.float32)

    # paths = './model_resnet.tflite'
    interpreter = tf.lite.Interpreter(model_path=paths_resnet)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    X_data1 = np.float32(image.reshape(1, 1024, 1024, 3))
    input_index = (interpreter.get_input_details()[0]['index'])
    interpreter.set_tensor(input_details[0]['index'], X_data1)
    interpreter.invoke()
    detection = interpreter.get_tensor(output_details[0]['index'])
    net_out_value2 = interpreter.get_tensor(output_details[1]['index'])
    net_out_value3 = interpreter.get_tensor(output_details[2]['index'])
    net_out_value4 = interpreter.get_tensor(output_details[3]['index'])
    img = upload_images

    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Converts from one colour space to the other
    img3 = img[:, :, :]

    box_x = int(detection[0, 0, 0] * image_height)
    box_y = int(detection[0, 0, 1] * image_width)
    box_width = int(detection[0, 0, 2] * image_height)
    box_height = int(detection[0, 0, 3] * image_width)

    cv2.rectangle(img2, (box_y, box_x), (box_height, box_width), (230, 230, 21), thickness=5)

    net_out_value3

    image = upload_images[box_x:box_width, box_y:box_height, :]
    img2 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image_detection_car_num = img2

    grayscale = rgb2gray(image)
    edges = canny(grayscale, sigma=3.0)
    out, angles, distances = hough_line(edges)
    h, theta, d = out, angles, distances
    angle_step = 0.5 * np.diff(theta).mean()
    d_step = 0.5 * np.diff(d).mean()
    bounds = [np.rad2deg(theta[0] - angle_step),
              np.rad2deg(theta[-1] + angle_step),
              d[-1] + d_step, d[0] - d_step]

    _, angles_peaks, _ = hough_line_peaks(out, angles, distances, num_peaks=20)
    angle = np.mean(np.rad2deg(angles_peaks))
    angle

    if 0 <= angle <= 90:
        rot_angle = angle - 90
    elif -45 <= angle < 0:
        rot_angle = angle - 90
    elif -90 <= angle < -45:
        rot_angle = 90 + angle
    else:
        rot_angle = 0
    if abs(rot_angle) > 20:
        rot_angle = 0

    rotated = rotate(image, rot_angle, resize=True) * 255
    rotated = rotated.astype(np.uint8)

    rotated1 = rotated[:, :, :]
    if rotated.shape[1] / rotated.shape[0] < 2:
        minus = np.abs(int(np.sin(np.radians(rot_angle)) * rotated.shape[0]))
        rotated1 = rotated[minus:-minus, :, :]

    lab = cv2.cvtColor(rotated1, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    letters = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'E', 'H', 'K', 'M', 'O', 'P', 'T', 'X',
               'Y']

    # For a real OCR application, this should be beam search with a dictionary
    # and language model.  For this example, best path is sufficient.

    def decode_batch(out):
        ret = []
        for j in range(out.shape[0]):
            out_best = list(np.argmax(out[j, 2:], 1))
            out_best = [k for k, g in itertools.groupby(out_best)]
            outstr = ''
            for c in out_best:
                if c < len(letters):
                    outstr += letters[c]
            ret.append(outstr)
        return ret

    # Загружаем  обученную tflite модель CNN-LSTM-CTC для распознования символов
    # paths='./model1_nomer.tflite'
    interpreter = tf.lite.Interpreter(model_path=paths_nomer)
    interpreter.allocate_tensors()
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    img = final
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (128, 64))
    img = img.astype(np.float32)
    img /= 255

    img1 = img.T
    img1.shape
    X_data1 = np.float32(img1.reshape(1, 128, 64, 1))
    input_index = (interpreter.get_input_details()[0]['index'])
    interpreter.set_tensor(input_details[0]['index'], X_data1)

    interpreter.invoke()

    net_out_value = interpreter.get_tensor(output_details[0]['index'])
    pred_texts = decode_batch(net_out_value)
    pred_texts
    # ['A999AA99']
    # print(pred_texts)

    # fig = plt.figure(figsize=(10, 10))
    # outer = gridspec.GridSpec(2, 1, wspace=10, hspace=0.1)
    # ax1 = plt.Subplot(fig, outer[0])
    # fig.add_subplot(ax1)
    # ax2 = plt.Subplot(fig, outer[1])
    # fig.add_subplot(ax2)
    # #print('Predicted:', pred_texts[0])
    # img = X_data1[0,:, :, 0].T
    # ax1.set_title('Input img')
    # ax1.imshow(img, cmap='gray')
    # ax1.set_xticks([])
    # ax1.set_yticks([])
    # ax2.set_title('Acrtivations')
    # ax2.imshow(net_out_value[0].T, cmap='binary', interpolation='nearest')
    # ax2.set_yticks(list(range(len(letters) + 1)))
    # ax2.set_yticklabels(letters + ['blank'])
    # ax2.grid(False)
    # for h in np.arange(-0.5, len(letters) + 1 + 0.5, 1):
    #     ax2.axhline(h, linestyle='-', color='k', alpha=0.5, linewidth=1)

    #   for i in range(len(upload_images)):
    # img_name1=upload_images
    # path = img_name1
    # image0 = cv2.imread(img_name1, 1)
    # image_height, image_width, _ = image0.shape
    image = cv2.resize(upload_images, (1024, 1024))
    image = image.astype(np.float32)
    # paths='./model_resnet.tflite'
    interpreter = tf.lite.Interpreter(model_path=paths_resnet)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    X_data1 = np.float32(image.reshape(1, 1024, 1024, 3))
    input_index = (interpreter.get_input_details()[0]['index'])
    interpreter.set_tensor(input_details[0]['index'], X_data1)

    #
    interpreter.invoke()

    detection = interpreter.get_tensor(output_details[0]['index'])
    net_out_value2 = interpreter.get_tensor(output_details[1]['index'])
    net_out_value3 = interpreter.get_tensor(output_details[2]['index'])
    net_out_value4 = interpreter.get_tensor(output_details[3]['index'])
    img = upload_images
    razmer = img.shape

    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Converts from one colour space to the other
    img3 = img[:, :, :]

    box_x = int(detection[0, 0, 0] * image_height)
    box_y = int(detection[0, 0, 1] * image_width)
    box_width = int(detection[0, 0, 2] * image_height)
    box_height = int(detection[0, 0, 3] * image_width)
    if np.min(detection[0, 0, :]) >= 0:
        cv2.rectangle(img2, (box_y, box_x), (box_height, box_width), (230, 230, 21), thickness=5)

        # plt.imshow(img2)
        # plt.xticks([]), plt.yticks([])  # Hides the graph ticks and x / y axis
        # plt.show()
        image = img3[box_x:box_width, box_y:box_height, :]
        grayscale = rgb2gray(image)
        edges = canny(grayscale, sigma=3.0)
        out, angles, distances = hough_line(edges)
        _, angles_peaks, _ = hough_line_peaks(out, angles, distances, num_peaks=20)
        angle = np.mean(np.rad2deg(angles_peaks))

        if 0 <= angle <= 90:
            rot_angle = angle - 90
        elif -45 <= angle < 0:
            rot_angle = angle - 90
        elif -90 <= angle < -45:
            rot_angle = 90 + angle
        else:
            rot_angle = 0
        if abs(rot_angle) > 20:
            rot_angle = 0

        rotated = rotate(image, rot_angle, resize=True) * 255
        rotated = rotated.astype(np.uint8)
        rotated1 = rotated[:, :, :]
        minus = np.abs(int(np.sin(np.radians(rot_angle)) * rotated.shape[0]))
        if rotated.shape[1] / rotated.shape[0] < 2 and minus > 10:
            rotated1 = rotated[minus:-minus, :, :]
        lab = cv2.cvtColor(rotated1, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        # paths='./model1_nomer.tflite'
        interpreter = tf.lite.Interpreter(model_path=paths_nomer)
        interpreter.allocate_tensors()
        # Get input and output tensors.
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        #         img =rotated1
        img = final  # лучше работает при плохом освещении

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (128, 64))
        img = img.astype(np.float32)
        img /= 255

        img1 = img.T
        img1.shape
        X_data1 = np.float32(img1.reshape(1, 128, 64, 1))
        input_index = (interpreter.get_input_details()[0]['index'])
        interpreter.set_tensor(input_details[0]['index'], X_data1)

        interpreter.invoke()

        net_out_value = interpreter.get_tensor(output_details[0]['index'])
        pred_texts = decode_batch(net_out_value)
    return (pred_texts, image_detection_car_num)
