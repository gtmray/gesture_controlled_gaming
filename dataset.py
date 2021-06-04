import cv2
from object_detector import ObjectDetection

p1_hand = (0, 0)
p2_hand = (200, 300)
lower_hsv_hand = (0, 36, 0)
file_directory = 'Data\\Right\\'
name = 'HANDS'
i = 0
limit = 999

cap = cv2.VideoCapture(1)
obj = ObjectDetection(hands_or_blue=2, p1=p1_hand, p2=p2_hand, lower_hsv=lower_hsv_hand)


while True:
    _, img = cap.read()
    img = cv2.flip(img, 1)
    img_pre, img = obj.preprocess(img)
    cv2.imshow('Dataset testing', img)
    cv2.imshow('Dataset  pre', img_pre)

    filename = file_directory + name + str(i) + '.jpg'
    # cv2.imwrite(filename, img_pre)
    if i == limit:
        break
    i += 1

    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break

cv2.destroyAllWindows()
