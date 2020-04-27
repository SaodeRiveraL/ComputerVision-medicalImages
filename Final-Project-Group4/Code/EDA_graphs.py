import pandas as pd
import numpy as np
import matplotlib.pylab as plt
#import imageio
import scipy.ndimage as ndi

df = pd.read_csv('labels_train_images.csv')
x1 = pd.read_csv('labels_1_annotation.csv')
x2 = pd.read_csv('labels_0_annotation.csv')
x3 = pd.read_csv('labels_plus_1_annotation.csv')
x4 = pd.read_csv('labels_1_any.csv')
x5 = pd.read_csv('labels_0_any.csv')

### --- A dataset for each one ---- ###

epidural = x1[x1['epidural'] == 1]
intraparenchymal = x1[x1['intraparenchymal'] == 1]
intraventricular = x1[x1['intraventricular'] == 1]
subarachnoid = x1[x1['subarachnoid'] ==1 ]
subdural = x1[x1['subdural']==1]

print(epidural.head())