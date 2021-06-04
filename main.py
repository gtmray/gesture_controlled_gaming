import cv2
import numpy as np
from object_detector import ObjectDetection

cap = cv2.VideoCapture(1)

p1_blue = (300, 0)
p2_blue = (640, 200)

p1_hand = (0, 0)
p2_hand = (200, 500)

lower_hsv_blue = (97, 114, 43)  # blue
lower_hsv_hand = (30, 0, 143)  # hands2

obj_blue = ObjectDetection(draw_contour=1, hands_or_blue=1, p1=p1_blue, p2=p2_blue, lower_hsv=lower_hsv_blue)
obj_hand = ObjectDetection(draw_contour=1, hands_or_blue=0, p1=p1_hand, p2=p2_hand, lower_hsv=lower_hsv_hand)


while True:
    _, img = cap.read()
    img = cv2.flip(img, 2)
    img_pre_hand, img_hand = obj_hand.preprocess(img)
    img_contour_hand = obj_hand.contours(img_pre_hand, img_hand)

    img_pre_blue, img_blue = obj_blue.preprocess(img)
    img_contour_blue = obj_blue.contours(img_pre_blue, img_blue)

    cv2.imshow('Image Blue', img_contour_blue)

    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break

cv2.destroyAllWindows()
