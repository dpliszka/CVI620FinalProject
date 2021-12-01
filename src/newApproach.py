import cv2
import numpy as np
import copy
import math
import time

# Starting points for ROI
cap_region_x_begin = 0
cap_region_y_end = 0.5
# cap_region_y_end=1

# Parameter default values
threshold = 128  # Binary threshold
blurValue = 41  # GaussianBlur parameter
bgSubThreshold = 50
learningRate = 0
devMode = 0  # 0 = ON

# Initializing Variables
startCounting = False
bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)
prev_frame_time = 0  # For FPS
new_frame_time = 0

# add
finger_tip_x = None
finger_tip_y = None
background_img = None
hasBackground = False
key_pressed = []


# Methods ###############################################################################
def removeBG(frame):
    fgmask = bgModel.apply(frame, learningRate=learningRate)
    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    return res


def calculateFingers(res, drawing):
    # add
    global finger_tip_x, finger_tip_y
    finger_tip_x = []
    finger_tip_y = []

    # Convexity Defect
    hull = cv2.convexHull(res, returnPoints=False)
    if len(hull) > 3:
        defects = cv2.convexityDefects(res, hull)
        if type(defects) != type(None):
            cnt = 0
            for i in range(defects.shape[0]):  # Calculating the Angle
                s, e, f, d = defects[i][0]
                start = tuple(res[s][0])
                end = tuple(res[e][0])
                far = tuple(res[f][0])
                a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  # COS
                if angle <= math.pi / 2:  # if angle <90, it's a finger

                    # add
                    cv2.circle(drawing, start, 8, [255, 255, 255], -1)
                    cv2.circle(drawing, end, 8, [200, 200, 200], -1)

                    if cnt:
                        finger_tip_x.pop()
                        finger_tip_y.pop()

                    finger_tip_x.append(start[0])
                    finger_tip_y.append(start[1])
                    finger_tip_x.append(end[0])
                    finger_tip_y.append(end[1])
                    # finger_tip_x.append(start[0])
                    # finger_tip_y.append(start[1])
                    # if i == defects.shape[0]:
                    #     finger_tip_x.append(end[0])
                    #     finger_tip_y.append(end[1])

                    # original
                    cnt += 1
                    cv2.circle(drawing, far, 8, [211, 84, 0], -1)

            return True, cnt
    return False, 0


# End of Methods ########################################################################

# Main
video = cv2.VideoCapture(0)
video.set(10, 200)

while video.isOpened():
    ret, frame = video.read()
    frame = cv2.bilateralFilter(frame, 5, 50, 100)  # smoothing filter
    frame = cv2.flip(frame, 1)  # Flip the webcam horizontally

    # Calculating FPS
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    fps = int(fps)
    fps = str(fps)

    # Show FPS
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)

    # Need to invert the rectangle to take the lower half portion of the screen
    cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 0),
                  (frame.shape[1], int(cap_region_y_end * frame.shape[0])), (255, 0, 0), 2)
    if devMode == 0:
        cv2.imshow('original', frame)

    #  Applying hand mask
    img = removeBG(frame)
    img = img[0:int(cap_region_y_end * frame.shape[0]),
          int(cap_region_x_begin * frame.shape[1]):frame.shape[1]]  # clip the ROI
    if devMode == 0:
        cv2.imshow('mask', img)

    # Convert Image for Contours
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)
    ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)

    # add
    key_pressed = []
    if not hasBackground:
        background_img = gray
        hasBackground = True

    # Get Coutours (not the best)
    thresh1 = copy.deepcopy(thresh)
    contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    length = len(contours)
    maxArea = -1
    if length > 0:
        for i in range(length):  # Find largest countour based on area
            temp = contours[i]
            area = cv2.contourArea(temp)
            if area > maxArea:
                maxArea = area
                ci = i

        res = contours[ci]
        hull = cv2.convexHull(res)
        drawing = np.zeros(img.shape, np.uint8)
        cv2.drawContours(drawing, [res], 0, (0, 255, 0), 2)
        cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 3)

        # Counting fingers is off by one
        isFinishCal, cnt = calculateFingers(res, drawing)

        # add
        for i in range(len(finger_tip_x)):
            print((finger_tip_x[i], finger_tip_y[i]))
            cv2.circle(drawing, (finger_tip_x[i], finger_tip_y[i]), 10, [255, 255, 0], -1)

        optical_flow_method = cv2.calcOpticalFlowFarneback
        params = [0.5, 3, 15, 3, 5, 1.2, 0]
        flow = optical_flow_method(background_img, gray, None, *params)
        for i in range(len(finger_tip_x)):
            flowat = flow[finger_tip_y[i], finger_tip_x[i]]
            cv2.line(frame, (finger_tip_x[i], finger_tip_y[i]),
                     (finger_tip_x[i] + round(flowat[1]), finger_tip_y[i] + round(flowat[0])), (255, 0, 0))
            angle = math.atan2(finger_tip_y[i] - (finger_tip_y[i] + round(flowat[0])),
                               finger_tip_x[i] - (finger_tip_x[i] + round(flowat[1]))) * 180 / math.pi
            print("angle: ", angle)

            if angle < -80 and angle > -100:
                key_pressed.append((finger_tip_x[i], finger_tip_y[i]))

        cv2.imshow('optical flow', frame)
        print("key pressed", key_pressed)

        if startCounting is True:
            if isFinishCal is True:
                print(cnt)
    if devMode == 0:
        cv2.imshow('Contours', drawing)

        # Waiting to kill program
    k = cv2.waitKey(10)
    if k == 27:  # ESC key
        video.release()
        cv2.destroyAllWindows()
        break
    elif k == ord('c'):
        startCounting = True
        print('Starting the counter')