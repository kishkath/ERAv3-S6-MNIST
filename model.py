import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary


## Model

class NetArch(nn.Module):
    def __init__(self):
        super(NetArch, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 10, 3, padding=1),
            nn.BatchNorm2d(10),
            nn.ReLU())
        # out_features = 28 + 2 -3 + 1 = 28
        # Jin = 1, S = 1, Jout = Jin*S = 1
        # RF = R_in + (K-1)*Jin = 1 + (3-1)* 1 = 3

        self.conv2 = nn.Sequential(nn.Conv2d(10, 20, 3),
                                   nn.BatchNorm2d(20),
                                   nn.ReLU())
        # out_features = 28 - 3 + 1 = 26
        # Jin = 1, S = 1, Jout = Jin*S = 1
        # Rf = R_In + (K-1)*Jin = 5

        self.max_Pool = nn.MaxPool2d(2, 2)
        # out_features = (26-2)/2 + 1 = 13
        # Jin = 1, S = 2, Jout = Jin*S = 2
        # Rf = R_In + (K-1)*Jin = 5 + (2-1)*1 = 6

        self.conv3 = nn.Sequential(nn.Conv2d(20, 20, 3),
                                   nn.BatchNorm2d(20),
                                   nn.ReLU())
        # out_features = (12-3)/1 + 1 =11
        # Jin = 2, S = 1, Jout = Jin*S = 2
        # Rf = R_In + (K-1)*Jin = 6 + (3-1)*2 = 6 + 4 = 10

        self.conv4 = nn.Sequential(nn.Conv2d(20, 20, 3),
                                   nn.BatchNorm2d(20),
                                   nn.ReLU())
        # out_features = (11-3)/1 + 1 = 9
        # Jin = 2, S = 1, Jout = 2
        # Rf = R_in + (K-1)*Jin = 10 + (3-1)*2 = 14

        self.conv5 = nn.Sequential(nn.Conv2d(20, 16, 3),
                                   nn.BatchNorm2d(16),
                                   nn.ReLU())
        # out_features = (9-3) + 1 = 7
        # Jin = 2, S = 1, Jout = 2
        # Rf_in = Rin + (K-1)*Jin = 14 + (3-1)*2 = 18

        self.conv6 = nn.Sequential(nn.Conv2d(16, 16, 3),
                                   nn.BatchNorm2d(16),
                                   nn.ReLU())
        # out_features = (7-3) + 1 = 5
        # Jin = 2, S = 1, Jout = 2
        # RF_out = Rin + (K-1)*J_In = 18 + (3-1)*2 = 22

        self.conv7 = nn.Sequential(nn.Conv2d(16, 10, 3),
                                   nn.BatchNorm2d(10),
                                   nn.ReLU())
        # out_features = 5 + (3-1) = 3
        # Jin = 2, S = 1, Jout = 2
        # RF_Out = 22 + (3-1)*2 =26

        self.gap = nn.AvgPool2d(3)
        # out_features = (3-3) + 1 = 1
        # Jin = 2, S = 1, Jout = 2
        # Rf_out = R_in + (K-1)*Jin = 26 + (3-1)*2 = 30
        self.dropout = nn.Dropout2d(0.01)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.max_Pool(x)
        x = self.conv3(x)
        x = self.dropout(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.dropout(x)
        x = self.conv6(x)
        x = self.dropout(x)
        x = self.conv7(x)
        x = self.gap(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

def return_summary(device,INPUT_SIZE):
    model = NetArch().to(device)
    return summary(model, input_size=INPUT_SIZE)