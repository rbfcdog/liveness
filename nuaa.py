import torch.nn as nn 
import torch
from pathlib import Path
from torchvision import transforms 
import cv2
from torch.utils.data import DataLoader, TensorDataset
import numpy as np 
import seaborn as sns 
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt 

from models.liveness import LwFLNeT
from models.face_detector import FaceDetector

import random


def get_nuaa(n_faces=5000):
    root = Path(__file__).resolve().parent

    face_detector = FaceDetector()

    X_train, X_test, y_train, y_test = [], [], [], []

    for t in ('train', 'test'):
        for c in ('client', 'imposter'):
            txt_path = client_train_path = root / 'data' / 'Detectedface' / f'{c}_{t}_face.txt'
            if c == 'client':
                faces_path = root / 'data' / 'Detectedface' / 'ClientFace'
            else:
                faces_path = root / 'data' / 'Detectedface' / 'ImposterFace'
            
            with open(txt_path, "r") as f:
                image_paths = [line.strip().replace("\\", "/").split()[0] for line in f.readlines()] 

            random.shuffle(image_paths)

            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((224, 224)),  
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])        # ...
            ])    

            for i, path in enumerate(image_paths):
                if i >= n_faces:
                    break

                print(path)
                img = cv2.imread(faces_path / path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                face_obj = face_detector(img)

                if not face_obj:
                    print('face not found')
                    continue

                face_arr = face_obj['face_arr']

                if face_arr.shape[0] == 0 or face_arr.shape[1] == 0: 
                    print('w or h is 0')
                    continue

                img_transformed = transform(face_arr)

                class_img = 1 if c == 'client' else 0

                if t == 'train':
                    X_train.append(img_transformed)
                    y_train.append(class_img)
                else:
                    X_test.append(img_transformed)
                    y_test.append(class_img)

    
    X = X_train + X_test
    y = y_train + y_test

    # print(X)
    # print(y)

    X = torch.stack(X)  # Convert X to a tensor
    y = torch.tensor(y)  # Convert y to a tensor
    # dataloader n tanka td de uma vez nn 

    # indices = torch.randperm(X.size(0))

    # X = X[indices]
    # y = y[indices]

    # X_train = torch.stack(X_train) 
    # y_train = torch.tensor(y_train) 
    # X_test = torch.stack(X_test)  
    # y_test = torch.tensor(y_test)

    # train_dataset = TensorDataset(X_train, y_train)
    # test_dataset = TensorDataset(X_test, y_test)

    dataset = TensorDataset(X, y)

    face_detector.close()

    return dataset