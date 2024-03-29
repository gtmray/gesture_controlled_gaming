import cv2
import numpy as np
from object_detector import ObjectDetection
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
import pydirectinput as pd

model = 'models\\model_new_2.h5'
model = load_model(model)
pd.PAUSE = 0.001  # Default is 0.1 so to avoid lagging


def preprocess_img(img_for_pred):
    img_width, img_height = 200, 300
    img_pred = cv2.resize(img_for_pred, (img_height, img_width))

    img_new = np.zeros((img_width, img_height, 3))  # Converting channel 1 image to channel 3
    img_new[:, :, 0] = img_pred
    img_new[:, :, 1] = img_pred
    img_new[:, :, 2] = img_pred

    img_pred = np.reshape(img_new, [1, img_width, img_height, 3])  # Reshaping to shape that keras accepts
    img_pred = preprocess_input(img_pred)
    return img_pred


cap = cv2.VideoCapture(1)

p1_blue = (300, 0)
p2_blue = (640, 200)

p1_hand = (0, 0)
p2_hand = (200, 300)

# lower_hsv_blue = (97, 114, 43)  # blue
lower_hsv_blue = (38, 37, 116)  # 6: 02 pm
# lower_hsv_hand = (30, 0, 143)  # hands2
lower_hsv_hand = (0, 55, 0)  # 5:31 pm
# lower_hsv_hand = 0

obj_blue = ObjectDetection(draw_contour=1, hands_or_blue=1, p1=p1_blue, p2=p2_blue, lower_hsv=lower_hsv_blue)
obj_hand = ObjectDetection(draw_contour=1, hands_or_blue=0, p1=p1_hand, p2=p2_hand, lower_hsv=lower_hsv_hand)


while True:
    _, img = cap.read()
    #img = cv2.resize(img, (800, 700))
    img = cv2.flip(img, 2)
    img_pre_hand, img_hand = obj_hand.preprocess(img)  # Get preprocessed image
    img_contour_hand, _ = obj_hand.contours(img_pre_hand, img_hand)  # Get image with contours

    img_pre_blue, img_blue = obj_blue.preprocess(img)
    img_contour_blue, kill = obj_blue.contours(img_pre_blue, img_blue)

    # Prediction changes only starts from here
    img_pre = preprocess_img(img_pre_hand)  # Converting to keras form of input
    result = model.predict_classes(img_pre)

    if result == 0:
        text = "Up"
        pd.press('Up')
    elif result == 1:
        text = "Down"
        pd.press('Down')
    elif result == 2:
        text = "Left"
        pd.press('Left')
    else:
        text = "Right"
        pd.press('Right')

    if kill:
        text = "Kill"
        pd.press('Space')

    cv2.putText(img_contour_blue, text, (300, 90), cv2.FONT_HERSHEY_SIMPLEX, 3, (150, 75, 94), 2, cv2.LINE_AA)
    cv2.imshow('Hands', img_pre_hand)  # ROI hands image
    cv2.imshow('Blue', img_pre_blue)
    cv2.imshow('Image', img_contour_blue)

    k = cv2.waitKey(1) & 0xff
    if k == 27:  # If backspace break
        break

cv2.destroyAllWindows()
