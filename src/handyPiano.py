# Imports 
import cv2
import numpy as np
import math
import time
import mediapipe as mp

# Initializing MediaPipe Hands
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Initializing Video Capture
cap = cv2.VideoCapture(0)
image_width  = cap.get(3)   
image_hight = cap.get(4)

# Initializing Variables 
x_pos,y_pos = None,None
count = 0 
f_vec = [] 

# Console 
print("Press Q to quit")

# Hand Detection 
with mp_hands.Hands(
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5) as hands: # Setting sensitivity
    while cap.isOpened():
        success,image = cap.read()
        if not success:
            print("Failed to Load")
            continue
            
        image.flags.writeable = True 
        image = cv2.flip(image,1)
        image1 = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        results = hands.process(image1)
        
        # If hand detected, draw on screen
        if results.multi_hand_landmarks: 
          for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
            ## Getting only INDEX finger coordinates
            x_pos_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x 
            y_pos_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
            x_pos_dip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].x 
            y_pos_dip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y
            x_pos_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].x 
            y_pos_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y
            f_vec = [x_pos_tip,y_pos_tip,x_pos_dip,y_pos_dip,x_pos_pip,y_pos_pip]
            x_pos = x_pos_tip * image_width
            y_pos = y_pos_tip * image_hight

        # Display the hands
        cv2.imshow('Hands', image)
        
        # Listening for 'q' key to quit
        if cv2.waitKey(33)==ord('q'):
            print('thank you for using pyano !')
            break
    
    # Release and Destroy Screen
    cap.release()
    cv2.destroyAllWindows()
