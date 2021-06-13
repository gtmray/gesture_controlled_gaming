import cv2
import numpy as np


class ObjectDetection:
    def __init__(self, hands_or_blue, p1, p2, lower_hsv=0, draw_contour=0):
        self.p1 = p1
        self.p2 = p2
        self.draw_contour = draw_contour
        self.hands_or_blue = hands_or_blue
        self.lower_hsv = lower_hsv

        cv2.namedWindow("Adjusting value", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Adjusting value", (400, 400))
        cv2.createTrackbar("Threshold", "Adjusting value", 0, 255, self.nothing)

        cv2.createTrackbar("LH", "Adjusting value", 0, 179, self.nothing)
        cv2.createTrackbar("LS", "Adjusting value", 0, 255, self.nothing)
        cv2.createTrackbar("LV", "Adjusting value", 0, 255, self.nothing)
        cv2.createTrackbar("HH", "Adjusting value", 179, 179, self.nothing)
        cv2.createTrackbar("HS", "Adjusting value", 255, 255, self.nothing)
        cv2.createTrackbar("HV", "Adjusting value", 255, 255, self.nothing)


    def nothing(self, x):
        pass

    def preprocess(self, img):

        cv2.rectangle(img, self.p1, self.p2, (255, 0, 0), 1)
        roi = img[self.p1[1]:self.p2[1], self.p1[0]:self.p2[0]]

        hsv_img = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        lh = cv2.getTrackbarPos("LH", "Adjusting value")
        ls = cv2.getTrackbarPos("LS", "Adjusting value")
        lv = cv2.getTrackbarPos("LV", "Adjusting value")
        hh = cv2.getTrackbarPos("HH", "Adjusting value")
        hs = cv2.getTrackbarPos("HS", "Adjusting value")
        hv = cv2.getTrackbarPos("HV", "Adjusting value")

        higher_hsv = (hh, hs, hv)

        if self.lower_hsv == 0:
            l_hsv = (lh, ls, lv)
        else:
            l_hsv = self.lower_hsv

        masked = cv2.inRange(hsv_img, l_hsv, higher_hsv)  # Make required color white and background black
        blurred = cv2.GaussianBlur(masked, (5, 5), 0)

        #color_filtered = cv2.bitwise_and(roi, roi, mask=blurred)  # Get the original image containing only required color and others black
        #inverting = cv2.bitwise_not(blurred)  # Inverting masked white -> black and black -> white

        dilate = cv2.dilate(blurred, (3, 3), iterations=6)
        kernel = np.ones((5, 5), np.uint8)
        erosion = cv2.erode(dilate, kernel)

        # morpho = cv2.morphologyEx(dilate, cv2.MORPH_OPEN, kernel)
        # morpho = cv2.morphologyEx(morpho, cv2.MORPH_CLOSE, kernel)

        return erosion, img


    def contours(self, erosion, img):

        kill = False
        th = cv2.getTrackbarPos("Threshold", "Adjusting value")
        _, thresh = cv2.threshold(erosion, th, 255, cv2.THRESH_BINARY)

        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        try:
            cm = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)[0]  # Sort by area
            cm = cm + [self.p1[0], 0]  # Translating to left
            area = cv2.contourArea(cm)

            if area > 3000:  # To avoid the boundary from getting drawn
                kill = True
                hull = cv2.convexHull(cm)

                if self.draw_contour == 3:
                    cv2.drawContours(img, [cm], -1, (0, 255, 0), 3)  # Drawing contours in largest area
                    cv2.drawContours(img, [hull], -1, (0, 0, 255), 2)

                    # For convexity defects
                    hull = cv2.convexHull(cm, returnPoints=False)
                    defects = cv2.convexityDefects(cm, hull)
                    for i in range(defects.shape[0]):
                        p, q, r, s = defects[i, 0]
                        start = tuple(cm[p][0])
                        end = tuple(cm[q][0])
                        dip = tuple(cm[r][0])
                        cv2.line(img, start, end, [255, 255, 255], 5)
                        cv2.circle(img, dip, 5, [0, 0, 255], -1)
                elif self.draw_contour == 2:
                    cv2.drawContours(img, [cm], -1, (0, 255, 0), 3)  # Drawing contours in largest area
                    cv2.drawContours(img, [hull], -1, (0, 0, 255), 2)
                elif self.draw_contour == 1:
                    cv2.drawContours(img, [cm], -1, (0, 255, 0), 3)  # Drawing contours in largest area
                elif self.draw_contour == 0:
                    pass
                else:
                    print("Invalid contour selected")
        except:
            pass  # Do nothing if contours not found
        return img, kill

    def __str__(self):
        return "Object detection module detects hands gesture as well as object!"






