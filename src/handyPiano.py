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

# Console 
print("Press Q to quit")

# Hand Detection 
with mp_hands.Hands(
    model_complexity = 0
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
            
            thumb_tip_z = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].z
            index_tip_z = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].z
            middle_tip_z = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].z
            ring_tip_z = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].z
            pinky_tip_z = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].z
                
            if ring_tip_z > 0.04: 
                print("ring pressed")
            elif pinky_tip_z > 0.03: 
                print("pinky pressed")
            elif middle_tip_z > 0.03: 
                print("middle pressed")
            elif index_tip_z > 0.01: 
                print("index pressed")
            elif thumb_tip_z > 0.04: 
                print("thumb pressed")
            else:
                print("No")

        # Display the hands
        cv2.imshow('Hands', image)
        
        # Listening for 'q' key to quit
        if cv2.waitKey(33)==ord('q'):
            print('thank you for using pyano !')
            break
    
    # Release and Destroy Screen
    cap.release()
    cv2.destroyAllWindows()
