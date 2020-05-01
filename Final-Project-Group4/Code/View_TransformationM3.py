import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torchvision import models
from PIL import Image
from torchvision import transforms, datasets
from torch.utils import data
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pydicom
import torch.nn.functional as F
import torchvision
import random

import FPModel3

IMAGE_DIR = '/home/ubuntu/dataproject/rsna-intracranial-hemorrhage-detection/stage_2_train/'
FILE_DIR = '/home/ubuntu/dataproject/rsna-intracranial-hemorrhage-detection/'
IMAGE_SIZE = 299

model = FPModel3.Net()

device = torch.device("cpu")
BATCH_SIZE = 2

#---------------  Rading the images -----------------------#
x= 'ID_bcb617c26.dcm'
x_1 = 'ID_f6ad96507.dcm'


file = IMAGE_DIR + x
data = pydicom.read_file(file)
img = data.pixel_array

R = cv2.resize(np.array(img), (IMAGE_SIZE, IMAGE_SIZE))
R = R.reshape(IMAGE_SIZE, IMAGE_SIZE)
R = R.astype('float16')

img2 = np.zeros((R.shape[0], R.shape[1], 3))
img2[:, :, 0] = R  # same value in each channel

X = torch.FloatTensor(img2)
X = X.permute(2, 0, 1)


file = IMAGE_DIR + x_1
data = pydicom.read_file(file)
img = data.pixel_array

R = cv2.resize(np.array(img), (IMAGE_SIZE, IMAGE_SIZE))
R = R.reshape(IMAGE_SIZE, IMAGE_SIZE)
R = R.astype('float16')

img2 = np.zeros((R.shape[0], R.shape[1], 3))
img2[:, :, 0] = R  # same value in each channel

XT = torch.FloatTensor(img2)
XT = XT.permute(2, 0, 1)

XT = torch.stack((X,XT), dim=0)

data = XT

model = model.to(device)

model.load_state_dict(torch.load('model_finalprojectF.pt', map_location=device))
model.eval()

input_tensor = data.cpu()
output_tensor = model.stn(data).cpu()
theta = model.get_theta()

a = input_tensor[0][0]
grid = F.affine_grid(theta, input_tensor.size(),align_corners=False )
x = F.grid_sample(input_tensor, grid, align_corners=False)

b = x[0][0]
print(b.shape)
input_np = a.numpy()
c = b.detach().to(torch.device('cpu')).numpy()


fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('Transformation')
ax1.imshow(a, cmap= plt.cm.bone)
ax2.imshow(c, cmap = plt.cm.bone)
print(theta)
plt.show()
