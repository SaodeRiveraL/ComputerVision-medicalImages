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
# We check if avalilability of a gpu procesor
#-----------------------------------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
## uses less gpu memory
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
        if self.label == 1:

            file = IMAGE_DIR + X_train[ID]
            data = pydicom.read_file(file)
            img = np.array(data.pixel_array)


            R= cv2.resize(img,(IMAGE_SIZE, IMAGE_SIZE))
            R = R.reshape(IMAGE_SIZE,IMAGE_SIZE)
            R = R.astype('float16')

            img2 = np.zeros((R.shape[0], R.shape[1], 3))
            img2[:, :, 0] = R  # same value in each channel
            #img2[:, :, 1] = R
            #img2[:, :, 2] = R

            X = torch.FloatTensor(img2)
            X = X.permute(2,0,1)
            y = y_train[ID]
            y = torch.FloatTensor(y)

        else:
            file = IMAGE_DIR + X_test[ID]
            data = pydicom.read_file(file)
            img = np.array(data.pixel_array)

            R= cv2.resize(img,(IMAGE_SIZE, IMAGE_SIZE))
            R = R.reshape(IMAGE_SIZE,IMAGE_SIZE)
            R = R.astype('float16')

            img2 = np.zeros((R.shape[0], R.shape[1], 3))
            img2[:, :, 0] = R  # same value in each channel
            #img2[:, :, 1] = R
            #img2[:, :, 2] = R

            X = torch.FloatTensor(img2)
            X = X.permute(2, 0, 1)
            y = y_train[ID]
            y = torch.FloatTensor(y)


        return X, y

#---- Define the model ---- #

class Net(nn.Module):
    def __init__(self):

        """Load the pretrained model and replace last top layers"""

        super(Net, self).__init__()


        pre= models.vgg16(pretrained=True)
        # Freeze model weights
        for param in pre.parameters():
            param.requires_grad = False

        n_inputs = pre.classifier[6].in_features

        pre.classifier[6] = nn.Sequential(
            nn.Linear(n_inputs, 2000),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.BatchNorm1d(2000)
        )

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
            nn.Linear(800, 500),
            nn.ReLU(),
            nn.BatchNorm1d(500),
            nn.Linear(500, 300),
            nn.ReLU(),
            nn.Linear(300, 300),
            nn.ReLU(),
            nn.Linear(300, 100),
            nn.ReLU(),
            nn.Linear(100, 6),
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
            nn.Linear(10 * 6 * 6, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def get_theta(self):
        return self.theta

    # Spatial transformer network forward function
    def stn(self, x):

        xs = self.localization(x)
        xs = xs.view(-1, 10 * 6 * 6)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        self.theta = theta

        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False)

        return x

    def forward(self, x):
        # transform the input
        x = self.stn(x)  ### Localization Network
        x = self.firstlayers(x)  ### Pretrainned Model
        x = self.fc1(x)  ## Fine Tunning Layer 1
        x = self.fc2(x)  ## Fine Tunning Layer 2
        return x


#---------------- Preapare Data Set -----------------------------#
n_sample = 500

train_labels = ['any','epidural','intraparenchymal', 'intraventricular','subarachnoid','subdural']
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
labels = np.array(sampleimages.loc[:,train_labels])
print(images.shape, labels.shape)


#------------------------------- Divide de data for training and validation -------------------------###
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.25, random_state=0)

#------------------------------- Create the Data Loaders -------------------------------------------####

BATCH_SIZE = 15
MAX_EPOCHS = 20
PRINT_LOSS_EVERY = 1
LR = 0.001
MAX_NO_IMPROVEMENT = 7

#---------------------- Parameters for the data loader --------------------------------

params = {'batch_size': BATCH_SIZE,
          'shuffle': False}

train_ids = list([ i for i in range(len(X_train))])
test_ids = list([ i for i in range(len(X_test))])


# Datasets
partition = {
    'train' : train_ids,
    'valid' : test_ids
}

# Data Loaders d

labels = 1

training_set = Dataset(partition['train'], labels)
training_generator = data.DataLoader(training_set, **params)

labels = 0

validation_set = Dataset(partition['valid'], labels)
validation_generator = data.DataLoader(validation_set, **params)


model = Net().to(device)
model.load_state_dict(torch.load('model_finalproject.pt', map_location=device))
for param in model.parameters():
    param.requires_grad = False

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

model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=0, verbose=True)


def train(epoch):

    model.train()
    train_loss = 0
    cont = 0
    train_loss_item = list([])
    for batch_idx, (data, target) in enumerate(training_generator):

        data, target = data.to(device), target.to(device)

        # Here is the conversion to star with a linnear
        # We need a CNN so this will not be necessary

        optimizer.zero_grad()

        output=model(data)

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss+= loss.item()
        cont +=1
        train_loss_item.append([epoch, batch_idx, loss.item()])

    avg_train_loss = train_loss / cont

    print('Train Epoch: {} \tLoss: {:.6f}'.format( epoch, avg_train_loss))

    pd.DataFrame(train_loss_item, columns=['epoch', 'batch', 'loss']).to_csv(
        METRICS_DIR + 'train_loss_hist_item2_' + str(epoch) + '.csv')

    return avg_train_loss

def test(epoch):
    with torch.no_grad():
        model.eval()
        test_loss = 0
        correct = 0
        test_accuracy = 0
        valid_loss_item = list([])
        batch_idx = 1
        for  data, target in validation_generator:

            data, target = data.to(device), target.to(device)

            output = model(data)

            # sum up batch loss

            loss = criterion(output, target)

            test_loss += loss.item()

            pred_labels = output.detach().to(torch.device('cpu'))
            pred_labels = np.where(pred_labels > 0.51, 1, 0)
            test_accuracy_item = 100 * accuracy_score(target.cpu().numpy(), pred_labels)

            valid_loss_item.append([epoch, batch_idx,loss.item(), test_accuracy_item ])

            #data = pd.DataFrame([epoch, batch_idx , loss.item(), test_accuracy_item ])
            #valid_loss_item = pd.concat([valid_loss_item, data])
            batch_idx += 1
            test_accuracy += test_accuracy_item

        test_loss /= (len(validation_generator.dataset)/BATCH_SIZE)
        test_accuracy /= (len(validation_generator.dataset)/BATCH_SIZE)
        print('\nTest set: Average loss: {:.6f} -- Average Accuracy : {:.6f}\n'
              .format(test_loss, test_accuracy))

        pd.DataFrame(valid_loss_item, columns = ['epoch', 'batch','loss','acc']).to_csv(METRICS_DIR+'valid_loss_hist_item2_'+str(epoch)+'.csv')

        return test_loss, test_accuracy, valid_loss_item


# --------- We proceed to calculate the model ----------- #
# --------- History of metrics -------------------------- #

train_loss_hist = list([])
valid_loss_hist = list([])
train_loss_hist_item = list([])
valid_loss_hist_item = list([])

last_valid_loss = 100
cont =1
best_epoch = 0

# --------- History of metrics -------------------------- #

train_loss_hist = list([])
valid_loss_hist = list([])
train_loss_hist_item = list([])
valid_loss_hist_item = list([])

last_valid_loss = 100
cont =1
best_epoch = 0

for epoch in range(MAX_EPOCHS):

    train_loss_item = []
    valid_loss_item = []

    train_loss= train(epoch)
    valid_loss, valid_acc, valid_loss_item =test(epoch)

    scheduler.step(valid_loss)

    train_loss_hist.append([epoch, train_loss])
    valid_loss_hist.append([epoch, valid_loss, valid_acc])

    #train_loss_hist_item = pd.concat([train_loss_hist_item,train_loss_item])
    #valid_loss_hist_item = pd.concat([valid_loss_hist_item,valid_loss_item])

    if valid_loss < last_valid_loss:

        filepath = 'model2_finalproject.pt'
        torch.save(model.state_dict(), filepath)
        last_valid_loss = valid_loss
        cont = 0
        best_epoch = epoch

    else:
        cont += 1

    if cont> MAX_NO_IMPROVEMENT:
        print('Early Stopping')
        print('Best Epoch', best_epoch)
        break

pdtrain_loss_hist = pd.DataFrame(train_loss_hist, columns = ['epoch','train_loss'])
pdvalid_loss_hist = pd.DataFrame(valid_loss_hist, columns = ['epoch','valid_loss','valid_acc'])

pdtrain_loss_hist.to_csv(METRICS_DIR+'train_loss_hist2.csv',index=False)
pdvalid_loss_hist.to_csv(METRICS_DIR+'valid_loss_hist2.csv',index=False)
