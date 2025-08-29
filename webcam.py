import os
import pickle
from pathlib import Path
import numpy as np 
import cv2
import mediapipe as mp 
from torchvision import transforms

import torch.nn as nn 
import torch

from models.face_detector import FaceDetector
from models.liveness import LwFLNeT

face_detection = FaceDetector()
liveness_detector = LwFLNeT()

liveness_detector.load_state_dict(torch.load("model.pth", map_location='cuda'))

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    ret, frame = cap.read()  
    if not ret:
        print("Error: Failed to capture frame.")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_obj = face_detection(frame_rgb)

    if not face_obj:
        print('face not found')
        continue

    face_arr = face_obj['face_arr']

    if face_arr.shape[0] == 0 or face_arr.shape[1] == 0: 
        print('w or h is 0')
        continue
    
    x1, y1, x2, y2 = map(int, face_obj['bbox'])
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    face_arr = face_obj['face_arr']
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),  
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),             
        # ...
    ])   
    face_transform = transform(face_arr).unsqueeze(0)
    result = liveness_detector(face_transform)

    prob = nn.functional.softmax(result, dim=1)[0][1]
    print(prob)
    # jogar softmax e pegar o [0] chance de ser real

    cv2.imshow('Camera Feed', frame)  

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

face_detection.close()
