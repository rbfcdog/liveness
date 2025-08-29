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
from datasets import CelebA_Spoof
from nuaa import get_nuaa


train_dataset = CelebA_Spoof(mode='train', n_faces=10000) 
print(len(train_dataset))
# considerar q vai ser menos pq umas n reconhece cara, fazer que nao tenha interseccao
test_dataset = CelebA_Spoof(mode='test', n_faces=2000)

cross_dataset = get_nuaa(n_faces=400)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=4)

cross_loader = DataLoader(cross_dataset, batch_size=32, shuffle=True, num_workers=4)

epoch_n = 10

device = "cuda" if torch.cuda.is_available() else "cpu"
model = LwFLNeT()
model.to(device)

print(device)

class_weights_tensor = torch.tensor([4, 4/3], dtype=torch.float32).to(device)
# aproximacao, ver oversampling dps 

# model.load_state_dict(torch.load("model.pth"))

criterion = nn.CrossEntropyLoss(reduction='mean', weight=class_weights_tensor)
lr = 0.0001
optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)


total_steps = len(train_loader)
print_every = 10 

for e in range(epoch_n):
    model.train()
    running_loss = 0
    epoch_loss = 0 
    y_train_t = []
    y_train_pred_t = []
    
    for i, (X, y) in enumerate(train_loader):
        optim.zero_grad()

        X = X.to(device)
        y = y.to(device)

        y_pred = model(X)

        # y_pred = y_pred.to(device)

        loss = criterion(y_pred, y)
        loss.backward()
        optim.step()

        running_loss += loss.item()
        epoch_loss += running_loss

        # Store predictions & ground truth for confusion matrix
        y_train_t.append(y)
        y_train_pred_t.append(y_pred)

        if (i+1) % print_every == 0:
            print(f'{e+1}/{epoch_n} /// {i+1}/{total_steps} - {running_loss/print_every}')
            running_loss = 0

    # Compute confusion matrix for training
    y_train_t = torch.cat(y_train_t, dim=0)
    y_train_pred_t = torch.cat(y_train_pred_t, dim=0)

    y_train_p = torch.argmax(y_train_pred_t, dim=1) 

    yc_train = y_train_t.detach().cpu()
    y_pc_train = y_train_p.detach().cpu()

    cm_train = confusion_matrix(yc_train, y_pc_train)

    tn_train, fp_train, fn_train, tp_train = cm_train.ravel()

    FAR_train = fp_train / (fp_train + tn_train)
    FRR_train = fn_train / (fn_train + tp_train)

    hter_train = (FAR_train + FRR_train) * 0.5

    print(f"train loss/hter for epoch {e}: {epoch_loss/len(train_loader)} and {hter_train}")

    # Save confusion matrix for training
    sns.heatmap(cm_train, annot=True)
    plt.title(f'Training Confusion Matrix - Epoch {e}')
    plt.savefig(f'temp/train_{e}.png')
    plt.close()

    # =================== EVALUATION ===================
    model.eval()
    with torch.no_grad():   
        y_t = []
        y_pred_t = []
        loss_total = 0
        steps_total = len(test_loader)

        for X, y in test_loader:
            X = X.to(device)
            y = y.to(device)

            y_pred = model(X)

            y_pred = y_pred.to(device)


            y_t.append(y)
            y_pred_t.append(y_pred)

            loss = criterion(y_pred, y)

            loss_total += loss.item()

        y_t = torch.cat(y_t, dim=0)
        y_pred_t = torch.cat(y_pred_t, dim=0)

        y_p = torch.argmax(y_pred_t, dim=1) 

        yc = y_t.detach().cpu()
        y_pc = y_p.detach().cpu()

        cm = confusion_matrix(yc, y_pc)

        tn, fp, fn, tp = cm.ravel()

        FAR = fp / (fp + tn)
        FRR = fn / (fn + tp)

        hter = (FAR + FRR) * 0.5

        print(f"test loss/hter for epoch {e}: {loss_total/steps_total} and {hter}")

        sns.heatmap(cm, annot=True)
        plt.title(f'Test Confusion Matrix - Epoch {e}')
        plt.savefig(f'temp/test_{e}.png')
        plt.close()


        # =================== CROSS ===================
        y_t = []
        y_pred_t = []
        loss_total = 0
        steps_total = len(test_loader)

        for X, y in cross_loader:
            X = X.to(device)
            y = y.to(device)

            y_pred = model(X)

            y_pred = y_pred.to(device)


            y_t.append(y)
            y_pred_t.append(y_pred)

            loss = criterion(y_pred, y)

            loss_total += loss.item()

        y_t = torch.cat(y_t, dim=0)
        y_pred_t = torch.cat(y_pred_t, dim=0)

        y_p = torch.argmax(y_pred_t, dim=1) 

        yc = y_t.detach().cpu()
        y_pc = y_p.detach().cpu()

        cm = confusion_matrix(yc, y_pc)

        tn, fp, fn, tp = cm.ravel()

        FAR = fp / (fp + tn)
        FRR = fn / (fn + tp)

        hter = (FAR + FRR) * 0.5

        print(f"cross loss/hter for epoch {e}: {loss_total/steps_total} and {hter}")

        sns.heatmap(cm, annot=True)
        plt.title(f'TCross Confusion Matrix - Epoch {e}')
        plt.savefig(f'temp/cross_{e}.png')
        plt.close()



torch.save(model.state_dict(), "model.pth")