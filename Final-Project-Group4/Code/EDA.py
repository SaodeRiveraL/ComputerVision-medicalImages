import pandas as pd
import numpy as np
import matplotlib.pylab as plt
#import imageio
import scipy.ndimage as ndi

## Configuring the the varialbe file to read the dataset with the annotations

path_images = "/home/ubuntu/dataproject/rsna-intracranial-hemorrhage-detection/stage_1_train_images/"
path_files = "/home/ubuntu/dataproject/rsna-intracranial-hemorrhage-detection/"
file = path_files + 'stage_2_train.csv'

# Reading the file
df_img = pd.read_csv(file)

# Spliting the information
x= pd.Series(df_img.ID.str.split('_'))

# Creating the arrays to create the new dataset

a= pd.DataFrame ( [ 'ID_'+ x[i][1] for i in range(len(x)) ] )
b= pd.DataFrame ([ x[i][2] for i in range(len(x)) ])

# Create the new dataset with three colunes, Image, Type and Value

df_img_inter_l = pd.concat([a,b, df_img.Label] ,axis= 1)
df_img_inter_l.columns = ['Img', 'Type', 'Value']

# Transpose the Dataset to create the one record for each image.

cu  = df_img_inter_l.pivot_table( index= 'Img' ,columns = 'Type', values= 'Value'  ).reset_index()
print(cu.head(), len(cu))

# Creating the summaries by type to plot the information

result = df_img_inter_l[df_img_inter_l['Value'] > 0].groupby('Type')['Value'].sum()
result.columns = ['Type','Value']
X= result.index
y= result
labels = X
print(X,y)

# Filtering the information to plot the images with 0 annotations, 1 annotation and more than 1 annotation
# and the totals of records by category to displayed in the plot

x1= pd.DataFrame(cu[cu['epidural'] + cu['intraparenchymal'] + cu['intraventricular']+ cu['subarachnoid']+ cu['subdural']  == 1])
x2= cu[cu['epidural'] + cu['intraparenchymal'] + cu['intraventricular']+ cu['subarachnoid']+ cu['subdural']  == 0]
x3= cu[cu['epidural'] + cu['intraparenchymal'] + cu['intraventricular']+ cu['subarachnoid']+ cu['subdural']  > 0]
x4 = cu[cu['any']==0]
x5 = cu[cu['any']>0]
totals= [len(x1) , len(x2), len(x3), len(x4), len(x5)]
print(totals)

x1.to_csv('labels_1_annotation.csv',index=False)
x2.to_csv('labels_0_annotation.csv',index=False)
x3.to_csv('labels_plus_1_annotation.csv',index=False)
x4.to_csv('labels_0_any.csv', index=False)
x5.to_csv('labels_1_any.csv', index=False)
df_img_inter_l.to_csv('labels_train_images.csv', index = False)
cu.to_csv('labels_train_images_pt.csv', index=False)
print(cu.head())
print(x1.head())