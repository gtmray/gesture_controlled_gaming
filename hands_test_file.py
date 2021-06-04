import cv2
import numpy as np
from object_detector import ObjectDetection
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

model = 'models\\model2.h5'
model = load_model(model)


def preprocess_img(img_for_pred):
    img_width, img_height = 200, 300
    img_pred = cv2.resize(img_for_pred, (img_height, img_width))

    img_new = np.zeros((img_width, img_height, 3))  # Converting channel 1 image to channel 3
    img_new[:, :, 0] = img_pred
    img_new[:, :, 1] = img_pred
    img_new[:, :, 2] = img_pred

    img_pred = np.reshape(img_new, [1, img_width, img_height, 3])  # Reshaping to shape that keras accepts

    return img_pred

p1_hand = (0, 0)
p2_hand = (200, 300)
lower_hsv_hand = (0, 56, 0)
# lower_hsv_hand = 0

cap = cv2.VideoCapture(1)
obj = ObjectDetection(hands_or_blue=2, p1=p1_hand, p2=p2_hand, lower_hsv=lower_hsv_hand)

while True:
    _, img = cap.read()
    img = cv2.flip(img, 1)
    img_pre, img = obj.preprocess(img)
    cv2.imshow('Dataset  pre', img_pre)
    img_pre = preprocess_img(img_pre)
    result = model.predict_classes(img_pre)
    if result == 0:
        text = "Up"
    elif result == 1:
        text = "Down"
    elif result == 2:
        text = "Left"
    else:
        text = "Right"

    cv2.putText(img, text, (300, 90), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow('Dataset testing', img)

    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break

cv2.destroyAllWindows()
