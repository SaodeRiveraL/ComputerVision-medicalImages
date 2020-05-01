import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
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
from torchvision import models
import os

#-----------------------------------------------------
# Test Process CPU process
#-----------------------------------------------------

device = torch.device("cpu")

#-----------------------------------------------------
# Image Dir & File dir
#-----------------------------------------------------

IMAGE_DIR = '/home/ubuntu/dataproject/rsna-intracranial-hemorrhage-detection/stage_2_train/'
FILE_DIR = '/home/ubuntu/dataproject/rsna-intracranial-hemorrhage-detection/'

#-----------------------------------------------------
# General parameters
#-----------------------------------------------------

BATCH_SIZE = 10
METRICS_DIR = os.getcwd() + '/model4metrics/'
IMAGE_SIZE = 299

## -------------------------------------------------
## We create the dataloaders to feed the network
## The information in the train and test tensors(cpu)
## are taken in batches and tranformed to gpu
## because this they are transformed in batches the dataloder
## ---------------------------------------------------

### Helper functions they obtain the different....
### From Kaggle Notebook

def window_image(img, window_center, window_width, intercept, slope):
 img = (img * slope + intercept)
 img_min = window_center - window_width // 2
 img_max = window_center + window_width // 2
 img[img < img_min] = img_min
 img[img > img_max] = img_max
 return img


def get_first_of_dicom_field_as_int(x):
    # get x[0] as in int is x is a 'pydicom.multival.MultiValue', otherwise get int(x)
    if type(x) == pydicom.multival.MultiValue:
        return int(x[0])
    else:
        return int(x)

def get_windowing(data):
    dicom_fields = [data[('0028', '1050')].value,  # window center
                    data[('0028', '1051')].value,  # window width
                    data[('0028', '1052')].value,  # intercept
                    data[('0028', '1053')].value]  # slope
    return [get_first_of_dicom_field_as_int(x) for x in dicom_fields]


### -----------Definition of Dataset -------- ###
class Dataset(data.Dataset):
    '''
    From : https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
    '''
    def __init__(self, list_IDs, label):
        #Initialization'
        self.label = label
        self.list_IDs = list_IDs

    def __len__(self):
        #Denotes the total number of samples'
        return len(self.list_IDs)


    def __getitem__(self, index):
        #Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label

        if self.label == 1:

            y = y_test[ID]
            y = torch.FloatTensor(y)

            file = IMAGE_DIR + X_test[ID]
            data = pydicom.read_file(file)
            img = np.array(data.pixel_array)

        else :
            y = y_test[ID]
            y = torch.FloatTensor(y)

            file = IMAGE_DIR + X_test[ID]
            data = pydicom.read_file(file)
            img = np.array(data.pixel_array)


        window_center, window_width, intercept, slope = get_windowing(data)

        img1 = window_image(img, 40, 80, intercept, slope)
        img2 = window_image(img, 80, 200, intercept, slope)

        # Original Image

        R0= cv2.resize(img,(IMAGE_SIZE, IMAGE_SIZE))
        R0 = R0.reshape(IMAGE_SIZE,IMAGE_SIZE)
        R0 = R0.astype('float16')

        # Brain Window

        R1= cv2.resize(img,(IMAGE_SIZE, IMAGE_SIZE))
        R1 = R1.reshape(IMAGE_SIZE,IMAGE_SIZE)
        R1 = R1.astype('float16')

        # Subdural Window

        R2 = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        R2 = R2.reshape(IMAGE_SIZE, IMAGE_SIZE)
        R2 = R2.astype('float16')

        img3 = np.zeros((R0.shape[0], R0.shape[1], 3))
        img3[:, :, 0] = R0  # Original Image
        img3[:, :, 1] = R1  # Brain Window
        img3[:, :, 2] = R2  # Subdural Window

        X = torch.FloatTensor(img3)
        X = X.permute(2,0,1)

        return X, y


#---- Define the model ---- #

import FPModel3

model = FPModel3.Net().to(device)
model.fc2 = nn.Sequential(
            nn.Linear(800, 800),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.BatchNorm1d(800),
            nn.Linear(800, 600),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.BatchNorm1d(600),
            nn.Linear(600,600 ),
            nn.ReLU(),
            nn.BatchNorm1d(600),
            nn.Linear(600, 300),
            nn.ReLU(),
            nn.Linear(300, 200),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Linear(100, 6),
            nn.Sigmoid()
        )

model.load_state_dict(torch.load('model_finalprojectF_2.pt', map_location=device))

## Evaluation
model.phase = 1


#----- Define the data Loader  --- #

#---------------------- Parameters for the data loader --------------------------------

#---------------- Preapare Data Set -----------------------------#

images = pd.read_csv(METRICS_DIR+ 'testImages.csv' )
labels = pd.read_csv(METRICS_DIR+ 'testLabels.csv')

X_test = np.array(images)
y_test = np.array(labels)

X_test = X_test.reshape(X_test.shape[0])

print(X_test.shape, y_test.shape)

params = {'batch_size': BATCH_SIZE,
          'shuffle': False}

test_ids = list([ i for i in range(len(X_test))])


# Datasets
partition = {
    'test' : test_ids
}

# Data Loaders d

labels = 1

test_set = Dataset(partition['test'], labels)
test_generator = data.DataLoader(test_set, **params)
criterion = nn.BCELoss()

def test():
    with torch.no_grad():
        model.eval()
        test_loss = 0
        correct = 0
        test_accuracy = 0
        test_loss_item = list([])
        pred_labels_per_hist = list([])
        pred_labels_hist = list([])
        target_hist = list([])
        batch_idx = 1
        for  data, target in test_generator:

            data, target = data.to(device), target.to(device)

            output = model(data)

            # sum up batch loss

            loss = criterion(output, target)

            test_loss += loss.item()

            pred_labels_per = output.detach().to(torch.device('cpu')).numpy()

            pred_labels = np.where(pred_labels_per > 0.51, 1, 0)

            test_accuracy_item = 100 * accuracy_score(target.cpu().numpy(), pred_labels)

            test_loss_item.append([batch_idx,loss.item(), test_accuracy_item ])


            if len(pred_labels_per_hist) == 0:
                pred_labels_per_hist = pred_labels_per
            else:
                pred_labels_per_hist = np.vstack([pred_labels_per_hist,pred_labels_per])

            if len(pred_labels_hist)==0:
                pred_labels_hist = pred_labels
            else:
                pred_labels_hist = np.vstack([pred_labels_hist, pred_labels])

            if len(target_hist)==0:
                target_hist = target.cpu().numpy()
            else:
                target_hist = np.vstack([target_hist, target.cpu().numpy()])

            batch_idx += 1
            test_accuracy += test_accuracy_item

        test_loss /= (len(test_generator.dataset)/BATCH_SIZE)
        test_accuracy /= (len(test_generator.dataset)/BATCH_SIZE)

        test_auc = roc_auc_score(target_hist, pred_labels_per_hist)
        test_f1 = f1_score(target_hist, pred_labels_hist, average='weighted')

        print('\nTest set: Loss: {:.6f} -- Accuracy : {:.6f} -- AOC : {:.6f} -- -- f1 : {:.6f}\n'
              .format(test_loss, test_accuracy, test_auc, test_f1))

        pd.DataFrame(test_loss_item, columns = ['batch','loss','acc']).to_csv(METRICS_DIR+'test_loss_hist_item4.csv', index =False)
        pd.DataFrame(pred_labels_per_hist, columns = ['0','1','2','3','4','5']).to_csv(METRICS_DIR+'pred_labels_per4_test.csv', index = False)
        pd.DataFrame(pred_labels_hist, columns = ['0','1','2','3','4','5']).to_csv(METRICS_DIR+'pred_labels4_test.csv', index = False)
        pd.DataFrame(target_hist, columns = ['0','1','2','3','4','5']).to_csv(METRICS_DIR+'target_labels4_test.csv', index = False)

        return test_loss, test_accuracy, test_auc, test_f1

# --------- We proceed to calculate the model ----------- #

# --------- We proceed to calculate the model ----------- #
# --------- History of metrics -------------------------- #

test_loss_hist = list([])
test_loss_hist_item = list([])

test_loss_item = []

test_loss, test_acc, test_aoc, test_f1 = test()

test_loss_hist.append([test_loss, test_acc, test_aoc, test_f1])

pdtest_loss_hist = pd.DataFrame(test_loss_hist, columns = ['test_loss','test_acc', 'test_auc', 'test_f1'])

pdtest_loss_hist.to_csv(METRICS_DIR+'test_loss4.csv',index=False)

