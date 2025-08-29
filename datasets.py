from torch.utils.data import Dataset
from pathlib import Path
import cv2 
from models.face_detector import FaceDetector
from torchvision import transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch

def transform_img(img):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),  
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])        # ...
    ])    

    # LBP e outro metodo de normalizacao de imagem (ver notebook)
    
    return transform(img)

class CelebA_Spoof(Dataset):
    def __init__(self, mode='train', n_faces=40000):
        root = Path(__file__).resolve().parent

        self.n_faces = n_faces 
        self.detector = FaceDetector()

        self.db_path = root / 'CelebA_Spoof_/CelebA_Spoof'
        train_label_path = self.db_path / 'metas' / 'intra_test' / 'train_label.txt'
        test_label_path = self.db_path / 'metas' / 'intra_test' / 'test_label.txt'

        if mode == 'train':
            label_path = train_label_path
        elif mode == 'test':
            label_path = test_label_path
        else:
            raise ValueError(f"Invalid mode '{mode}'")
            
        self.image_db = self._prepare_faces(label_path)

    def _prepare_faces(self, label_path):
        with open(label_path, 'r') as f:
            lines = f.readlines()

        image_paths = [line.split()[0] for line in lines] 
        # randomize (fzr igual a label)

        faces = []
        labels = []

        all_labels = [int(line.split()[1]) for line in lines]

        for i, image_path in enumerate(image_paths): 
            if i == self.n_faces:
                break 

            # botar limite de quantas imagens pega (passa classe)
            print(i, image_path)
            full_img_raw = cv2.imread(self.db_path / image_path)
            full_img = cv2.cvtColor(full_img_raw, cv2.COLOR_BGR2RGB)
            face_obj = self.detector(full_img)

            if not face_obj:
                print('face not found')
                continue

            face_arr = face_obj['face_arr']

            if face_arr.shape[0] == 0 or face_arr.shape[1] == 0: 
                print('w or h is 0')
                continue

            transformed_img = transform_img(face_arr)

            # save = np.array(transformed_img.permute(1,2,0))   
            # save = (save * 255).astype(np.uint8)
            # save = cv2.cvtColor(save, cv2.COLOR_RGB2BGR)
            # # print(save)  
            # cv2.imwrite(f'temp/{i}.png', save)

            faces.append(transformed_img)
            labels.append(all_labels[i])

        labels = torch.tensor(labels)

        image_db = list(zip(faces, labels))
        # print(image_db)

        return image_db 
            
    def __len__(self):
        return len(self.image_db)

    def __getitem__(self, idx):
        face, label = self.image_db[idx]

        return face, label

    def close(self): 
        self.detector.close()
