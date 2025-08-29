import torch 
import torch.nn as nn 

class LwFLNeT(nn.Module):
    def __init__(self):
        super(LwFLNeT, self).__init__()

        resnet = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=True)

        self.resnet_layers = nn.Sequential(*list(resnet.children())[:5])  
        # (batch_sz, 256, 56, 56)

        self.stream1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=128*5*5, out_features=128),
            nn.ReLU(),
            nn.Dropout(p=0.4), 
            nn.Linear(in_features=128, out_features=256),
            nn.ReLU(),
            nn.Dropout(p=0.2), 
        )

        self.stream2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=64*12*12, out_features=128),
            nn.ReLU(),
            nn.Dropout(p=0.4), 
            nn.Linear(in_features=128, out_features=256),
            nn.ReLU(),
        )

        self.final = nn.Sequential(
            nn.Linear(in_features=256, out_features=2),
            # nn.Softmax(dim=1)
        )

    def forward(self, x):

        x_res = self.resnet_layers(x)

        a = self.stream1(x_res)
        b = self.stream2(x_res)
        out = self.final(a + b)

        return out
