#---- Define the model ---- #
import torch.nn as nn
from torchvision import models
import torch
import torch.nn.functional as F
from torch.utils import data


#---- Define the model ---- #

class Net(nn.Module):
    def __init__(self):

        """Load the pretrained model and replace last top layers"""

        super(Net, self).__init__()


        pre= models.inception_v3(pretrained=True)
        # Freeze model weights
        for param in pre.parameters():
            param.requires_grad = False

        n_inputs = pre.fc.in_features
        n_inputsaux =  pre.AuxLogits.fc.in_features

        pre.fc = nn.Sequential(
            nn.Linear(n_inputs, 2000),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.BatchNorm1d(2000),
        )

        pre.AuxLogits.fc =  nn.Linear(n_inputsaux, 6)

        self.firstlayers = pre

        self.fc1 = nn.Sequential(
            nn.Linear(2000, 1500),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.BatchNorm1d(1500),
            nn.Linear(1500, 1500),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.BatchNorm1d(1500),
            nn.Linear(1500, 1000),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1000),
            nn.Linear(1000, 1000),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1000),
            nn.Linear(1000, 1000),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1000),
            nn.Linear(1000, 1000),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1000),
            nn.Linear(1000, 1000),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1000),
            nn.Linear(1000, 1000),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1000),
            nn.Linear(1000, 800),
            nn.ReLU(),
            nn.BatchNorm1d(800),
            nn.Linear(800, 800),
            nn.ReLU(),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(800, 600),
            nn.ReLU(),
            nn.BatchNorm1d(600),
            nn.Linear(600, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.BatchNorm1d(300),
            nn.Linear(300, 200),
            nn.ReLU(),
            nn.BatchNorm1d(200),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 6),
            nn.Sigmoid()
        )

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=20),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=15, padding=1),
            nn.ReLU(True),
            nn.Conv2d(10, 10, kernel_size=15, padding=1),
            nn.ReLU(True),
            nn.Conv2d(10, 10, kernel_size=15, padding=1),
            nn.ReLU(True),
            nn.Conv2d(10, 10, kernel_size=10, padding=1),
            nn.ReLU(True),
            nn.Conv2d(10, 10, kernel_size=8, padding=1),
            nn.ReLU(True),
            nn.Conv2d(10, 10, kernel_size=8, padding=1),
            nn.ReLU(True),
            nn.Conv2d(10, 10, kernel_size=8, padding=1),
            nn.ReLU(True),
            nn.Conv2d(10, 10, kernel_size=8, padding=1),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(10, 10, kernel_size=8, padding=1),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(10, 10, kernel_size=5, padding=1),
            nn.ReLU(True),
            nn.Conv2d(10, 10, kernel_size=5, padding=1),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(10, 10, kernel_size=5, padding=1),
            nn.ReLU(True),
            nn.Conv2d(10, 10, kernel_size=5, padding=1),
            nn.ReLU(True),
            nn.Conv2d(10, 10, kernel_size=5),
            nn.ReLU(True)

        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 15 * 15, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

        self.phase = 0

    def get_theta(self):
        return self.theta

    # Spatial transformer network forward function
    def stn(self, x):

        xs = self.localization(x)
        xs = xs.view(-1, 10 * 15 * 15)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        self.theta = theta

        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False)

        return x

    def forward(self, x):
        # transform the input
        x = self.stn(x)  ### Localization Network
        if self.phase == 0:  ## Training Phase
            x,y = self.firstlayers(x)  ### Pretrainned Model
        else:  ## Evaluation Stage
            x = self.firstlayers(x)
        x = self.fc1(x)  ## Fine Tunning Layer 1
        x = self.fc2(x)  ## Fine Tunning Layer 2
        return x


