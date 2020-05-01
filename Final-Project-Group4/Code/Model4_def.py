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
# We check if avalilability of a gpu procesor
#-----------------------------------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#-----------------------------------------------------
# Image Dir & File dir
#-----------------------------------------------------

IMAGE_DIR = '/home/ubuntu/dataproject/rsna-intracranial-hemorrhage-detection/stage_2_train/'
FILE_DIR = '/home/ubuntu/dataproject/rsna-intracranial-hemorrhage-detection/'

## ----- General Parameters ------------------------- ###

BATCH_SIZE = 50
MAX_EPOCHS = 20
PRINT_LOSS_EVERY = 1
LR = 0.001
EARLY_STOPPING = 1   # 0 No early stoppint 1 Yes early stopping
MAX_NO_IMPROVEMENT = 7
IMAGE_SIZE = 299
METRICS_DIR = os.getcwd() + '/model4metrics/'


## -------------------------------------------------
## We create the dataloaders to feed the network
## The information in the train and test tensors(cpu)
## are taken in batches and tranformed to gpu
## because this they are transformed in batches the dataloder
## uses less gpu memory
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

            y = y_train[ID]
            y = torch.FloatTensor(y)

            file = IMAGE_DIR + X_train[ID]
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


#---------------- Preapare Data Set -----------------------------#

n_sample = 3000

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


#------------------------------- Divide de data for training validation and test -------------------------###
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.35, random_state=0)
## We devide the test set into testing for validation and testing
## The testing will be saved to use in te Model3_testdataset.py
## Only labels will be saved
X_test, X_test2, y_test, y_test2 = train_test_split(X_test, y_test, test_size=0.50, random_state=0)
## We save X_test2 and y_test2

pd.DataFrame(X_test2, columns =['Images']).to_csv(METRICS_DIR+"testImages.csv", index=False)
pd.DataFrame(y_test2, columns =['0','1','2','3','4','5']).to_csv(METRICS_DIR+"testLabels.csv", index =False)


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


### Creation of the model ###
model = Net().to(device)

### Loading the Weigths of previous modesl ####

model.load_state_dict(torch.load('model_finalprojectF.pt', map_location=device))
for param in model.parameters():
    param.requires_grad = False


### Substitution of last layer for a new layer to train multilabels classification ###

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
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.BCELoss()

scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=0, verbose=True)

###
### ----------------  train and test ------------------------ ###
###

def train(epoch):

    model.train()
    train_loss = 0
    cont = 0
    train_loss_item = list([])

    pred_labels_per_hist = list([])
    target_hist = list([])

    model.phase = 0
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

        pred_labels_per = output.detach().to(torch.device('cpu')).numpy()

        if len(pred_labels_per_hist) == 0:
            pred_labels_per_hist = pred_labels_per
        else:
            pred_labels_per_hist = np.vstack([pred_labels_per_hist, pred_labels_per])

        if len(target_hist) == 0:
            target_hist = target.cpu().numpy()
        else:
            target_hist = np.vstack([target_hist, target.cpu().numpy()])

    avg_train_loss = train_loss / cont

    ## Saving labels and metrics for the epoch

    pd.DataFrame(train_loss_item, columns=['epoch', 'batch', 'loss']).to_csv(
        METRICS_DIR + 'train_loss_hist_item4_' + str(epoch) + '.csv')

    pd.DataFrame(target_hist, columns=['0', '1', '2', '3', '4', '5']).to_csv(
        METRICS_DIR + 'target_train_loss_'+ str(epoch)+'.csv',index=False)

    pd.DataFrame(pred_labels_per_hist, columns=['0', '1', '2', '3', '4', '5']).to_csv(
        METRICS_DIR + 'pred_labels_train_loss_'+ str(epoch)+'.csv', index=False)

    return avg_train_loss

def test(epoch):
    with torch.no_grad():
        model.eval()
        test_loss = 0
        correct = 0
        test_accuracy = 0
        valid_loss_item = list([])
        batch_idx = 1
        pred_labels_per_hist = list([])
        pred_labels_hist = list([])
        target_hist = list([])
        model.phase = 1
        for  data, target in validation_generator:

            data, target = data.to(device), target.to(device)

            output = model(data)

            # sum up batch loss

            loss = criterion(output, target)

            test_loss += loss.item()


            pred_labels_per = output.detach().to(torch.device('cpu')).numpy()
            pred_labels = np.where(pred_labels_per > 0.50, 1, 0)

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


            test_accuracy_item = 100 * accuracy_score(target.cpu().numpy(), pred_labels)

            valid_loss_item.append([epoch, batch_idx,loss.item(), test_accuracy_item ])

            batch_idx += 1
            test_accuracy += test_accuracy_item

        test_loss /= (len(validation_generator.dataset)/BATCH_SIZE)
        avg_test_accuray = test_accuracy / (len(validation_generator.dataset)/BATCH_SIZE)

        ## Saving labels and metrics for the epoch

        pd.DataFrame(valid_loss_item, columns = ['epoch', 'batch','loss','acc']).to_csv(METRICS_DIR+'valid_loss_hist_item4_'+str(epoch)+'.csv', index =False)
        pd.DataFrame(target_hist, columns = ['0','1','2','3','4','5']).to_csv(METRICS_DIR+'target_valid_loss_'+ str(epoch)+'.csv', index =False)
        pd.DataFrame(pred_labels_per_hist, columns = ['0','1','2','3','4','5']).to_csv(METRICS_DIR+'pred_labels_valid_loss_'+ str(epoch)+'.csv', index =False)

        return test_loss, avg_test_accuray


# --------- We proceed to calculate the model ----------- #
# --------- History of metrics -------------------------- #

train_loss_hist = list([])
valid_loss_hist = list([])
train_loss_hist_item = list([])
valid_loss_hist_item = list([])

last_valid_loss = 100
last_valid_acc = -100
last_valid_auc = -1
last_train_loss = 100
cont =1
best_epoch = 0

for epoch in range(MAX_EPOCHS):

    train_loss_item = []
    valid_loss_item = []

    train_loss= train(epoch)

    pred_labels_per  = pd.read_csv(METRICS_DIR+'pred_labels_train_loss_'+ str(epoch)+'.csv')
    target_labels = pd.read_csv(METRICS_DIR+'target_train_loss_'+ str(epoch)+'.csv')
    pred_labels = np.where(pred_labels_per > 0.50, 1, 0)

    xauc_t = roc_auc_score(target_labels,pred_labels_per)
    xf1_t =  f1_score(target_labels, pred_labels,  average = 'weighted')

    xaccuracy_t = 100 * accuracy_score(target_labels, pred_labels)

    print('Train set: Avg. loss: {:.6f} -- Accuracy : {:.6f} -- AUC : {:.6f} -- F1 : {:.6f} \n'
          .format(train_loss, xaccuracy_t, xauc_t, xf1_t))

    valid_loss, valid_acc =test(epoch)

    scheduler.step(train_loss)

    pred_labels_per  = pd.read_csv(METRICS_DIR+'pred_labels_valid_loss_'+str(epoch)+'.csv')
    target_labels = pd.read_csv(METRICS_DIR+'target_valid_loss_'+str(epoch)+'.csv')
    pred_labels = np.where(pred_labels_per > 0.50, 1, 0)

    xauc = roc_auc_score(target_labels,pred_labels_per)
    xf1 =  f1_score(target_labels, pred_labels,  average = 'weighted')

    xaccuracy = 100 * accuracy_score(target_labels, pred_labels)

    print('Test set: Avg. loss: {:.6f} -- Accuracy : {:.6f} -- AUC : {:.6f} -- F1 : {:.6f} \n'
          .format(valid_loss, xaccuracy, xauc, xf1))

    train_loss_hist.append([epoch, train_loss, xaccuracy_t, xauc_t, xaccuracy_t])
    valid_loss_hist.append([epoch, valid_loss, xaccuracy, xauc, xf1, xaccuracy])

    if train_loss < last_train_loss:

        filepath = 'model_finalprojectF_2.pt'
        torch.save(model.state_dict(), filepath)
        last_train_loss = train_loss
        cont = 0
        best_epoch = epoch
        best_train_loss = train_loss
        best_valid_loss = valid_loss

    else:
        cont += 1

    if EARLY_STOPPING==1:
        if cont> MAX_NO_IMPROVEMENT:
            print('Early Stopping')
            print('Best Epoch', best_epoch)
            break

print('Best Epoch', best_epoch)

## ---------------------------------
## Saving the main metrics
## ---------------------------------
pdfinal_results = pd.DataFrame([best_epoch, best_train_loss, best_valid_loss], columns=['Data'])
pdtrain_loss_hist = pd.DataFrame(train_loss_hist, columns = ['epoch','train_loss','train_acc','train_auc','train_f1'])
pdvalid_loss_hist = pd.DataFrame(valid_loss_hist, columns = ['epoch','valid_loss','valid_acc','valid_auc','valid_f1','valid_sumacc'])

pdtrain_loss_hist.to_csv(METRICS_DIR+'train_loss_hist4.csv',index=False)
pdvalid_loss_hist.to_csv(METRICS_DIR+'valid_loss_hist4.csv',index=False)
pdfinal_results.to_csv(METRICS_DIR+'best_epoch.csv',index=False)