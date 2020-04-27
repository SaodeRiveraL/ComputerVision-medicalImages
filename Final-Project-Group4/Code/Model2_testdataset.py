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
METRICS_DIR = os.getcwd() + '/model2metrics/'
IMAGE_SIZE = 224

## -------------------------------------------------
## We create the dataloaders to feed the network
## The information in the train and test tensors(cpu)
## are taken in batches and tranformed to gpu
## because this they are transformed in batches the dataloder
## ---------------------------------------------------

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

        file = IMAGE_DIR + X_test[ID]
        data = pydicom.read_file(file)
        img = np.array(data.pixel_array)


        R= cv2.resize(img,(IMAGE_SIZE, IMAGE_SIZE))
        R = R.reshape(IMAGE_SIZE,IMAGE_SIZE)
        R = R.astype('float16')

        img2 = np.zeros((R.shape[0], R.shape[1], 3))
        img2[:, :, 0] = R  # same value in each channel

        X = torch.FloatTensor(img2)
        X = X.permute(2,0,1)
        y = y_test[ID]
        y = torch.FloatTensor(y)

        return X, y

#---- Define the model ---- #

import FPModel1

model = FPModel1.Net().to(device)

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

model.load_state_dict(torch.load('model2_finalproject.pt', map_location=device))


#----- Define the data Loader  --- #

#---------------------- Parameters for the data loader --------------------------------

#---------------- Preapare Data Set -----------------------------#
n_sample = 80

test_labels = ['any','epidural','intraparenchymal', 'intraventricular','subarachnoid','subdural']
list_of_images = pd.read_csv('labels_plus_1_annotation.csv')
list_of_any = pd.read_csv('labels_0_any.csv')

## Creating a dataset for each class from the dataset that only has one annotation x1

any = list_of_any.sample(n_sample*2)
label_any = [ 0 for _  in range(len(any))]
epidural = list_of_images[list_of_images['epidural'] == 1].sample(n_sample)
label_epidural = [ 1 for _  in range(len(epidural))]
intraparenchymal = list_of_images[list_of_images['intraparenchymal'] == 1].sample(n_sample)
label_intraparenchymal = [ 2 for _  in range(len(intraparenchymal))]
intraventricular = list_of_images[list_of_images['intraventricular'] == 1].sample(n_sample)
label_intraventricular = [ 3 for _ in  range(len(intraventricular))]
subarachnoid = list_of_images[list_of_images['subarachnoid'] == 1].sample(n_sample)
label_subarachnoid = [4 for _ in range(len(subarachnoid))]
subdural = list_of_images[list_of_images['subdural'] == 1].sample(n_sample)
label_subdural = [5 for _ in range(len(subdural))]

sampleimages = pd.concat([any,epidural,intraparenchymal,intraventricular,subarachnoid,subdural])
samplelabels = np.hstack([label_any, label_epidural,label_intraparenchymal,label_intraventricular,label_subarachnoid,label_subdural])

images = np.array(sampleimages['Img']+'.dcm')
#labels = samplelabels.reshape(-1,1)
labels = np.array(sampleimages.loc[:,test_labels])
print(images.shape, labels.shape)

X_test = images
y_test = labels
BATCH_SIZE = 20

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
        print('\nTest set: Average loss: {:.6f} -- Average Accuracy : {:.6f}\n'
              .format(test_loss, test_accuracy))

        pd.DataFrame(test_loss_item, columns = ['batch','loss','acc']).to_csv(METRICS_DIR+'test_loss_hist_item2.csv', index =False)
        pd.DataFrame(pred_labels_per_hist, columns = test_labels).to_csv(METRICS_DIR+'pred_labels_per2.csv', index = False)
        pd.DataFrame(pred_labels_hist, columns = test_labels).to_csv(METRICS_DIR+'pred_labels2.csv', index = False)
        pd.DataFrame(target_hist, columns = test_labels).to_csv(METRICS_DIR+'target_labels2.csv', index = False)

        return test_loss, test_accuracy

# --------- We proceed to calculate the model ----------- #

# --------- We proceed to calculate the model ----------- #
# --------- History of metrics -------------------------- #

test_loss_hist = list([])
test_loss_hist_item = list([])
test_loss_item = []

test_loss, test_acc = test()

test_loss_hist.append([test_loss, test_acc])

pdtest_loss_hist = pd.DataFrame(test_loss_hist, columns = ['test_loss','test_acc'])
pdtest_loss_hist.to_csv(METRICS_DIR+'test_loss2.csv',index=False)

