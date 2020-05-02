# Final-Project-Group4 - Code

The project is a composed by two models, the first model is training on set of a sigle label per type of tumor, the second model uses the weights of the first model and trains a two more layers for a multilabel classification. Thus the order to "run" the scripts is as follows

1. **Labels_pr.py:** This script creates the labels from the kaggle dataset
2. **Model3_def.py:** This script trains the first model, in my AWS instance each epoch took approximately 0.5 hours. This scripts does the training and the validation.
3. **Model3_testdataset.py:** This scripts uses a test dataset that was saved in the previous step and performs the validation of the model.
4. **FPModel3.py:** Contains the definicion of the first model, this script has to be in the directory because is used in the validation process.
5. **Model4_der.py:** This scripts uses the weights of the previous model, so It cannot be executed unless the previous model had already generated a weights. In my AWS instance each epoch took approximately 20 mintus. This scripts does the training and the validation.
6. **Model4_testdataset.py:** This scripts uses a test dataset that was saved in the previous step and performs the validation of the model. This script requires FPModel3py.
7. **View_TransformationM3.py:** This script is a demo of how tp get theta matrix from **Model3** and how to used to transformed an image. This demos needs at least two images to work. The script already has the images, but any two images from the datasets will do the work.

## Special considerations

Each script has the following variables at the beginning 

IMAGE_DIR = '/home/ubuntu/dataproject/rsna-intracranial-hemorrhage-detection/stage_2_train/'

FILE_DIR = '/home/ubuntu/dataproject/rsna-intracranial-hemorrhage-detection/'

IMAGE_DIR contains the path to the image dataset
FILE_DIR contains the path to the file tage_2_train.csv  which holds the labels for each image (type of tumor)

The link to the dataset is 
https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/data

This dataset belongs to a kaggle competititon so it should be dowloaded with Kaggle credentials.

Besides the scripts files two additional directories are uplodaded : model3metrics and model4metrics. These directories will contain all the metrics generated during the model training and leter they will be used in Model3_metrics.py and Model4_metrics to generate the results of the model.



