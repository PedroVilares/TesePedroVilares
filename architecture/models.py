import tensorflow as tf
import os
import pandas as pd
import numpy as np
import generate_data as generator
import utils
from keras import metrics,optimizers,losses
from keras.callbacks import Callback
from keras.models import Sequential
from keras.layers import Flatten,Add,BatchNormalization,Dense,Conv2D,MaxPooling2D,Dropout,GlobalAveragePooling2D,Input,concatenate

def classify_patches_transfer(model):

    patient_folder = "patients/"
    patient_paths = os.listdir(patient_folder)
    not_folders = []
    for image in patient_paths:
        if 'g' in image:
            not_folders.append(image)
    for image in not_folders:
        patient_paths.remove(image)
    patient_paths = utils.sort_paths(patient_paths)

    for patient in patient_paths:

        patch_dataframe = pd.read_csv('patients/'+patient+'/classification_data.csv')
        df = pd.DataFrame({'paths':patch_dataframe['Patches Paths'],'labels':'Test'})
        generated_data = generator.generate(df,input_size=300)
        predictions = model.predict(generated_data,verbose=1)
        predictions_image=[]
        for i in range(len(predictions)):
            predictions_image.append(predictions[i][0])
        
        patch_dataframe['Classifications'] = predictions_image
        patch_dataframe.to_csv('patients/'+patient+'/classification_data.csv',index=False)
        print('Successfully classified '+patient+'! \n')

def classify_patches_from_scratch(model):

    patient_folder = "patients/"
    patient_paths = os.listdir(patient_folder)
    not_folders = []
    for image in patient_paths:
        if 'g' in image:
            not_folders.append(image)
    for image in not_folders:
        patient_paths.remove(image)
    patient_paths = utils.sort_paths(patient_paths)

    for patient in patient_paths:

        patch_dataframe = pd.read_csv('patients/'+patient+'/classification_data.csv')
        df = pd.DataFrame({'paths':patch_dataframe['Patches Paths'],'labels':'Test'})
        generated_data = generator.generate(df,input_size=300)
        ypred=[]
        for i in range(len(generated_data)):
            for n in range(len(generated_data[i][0])):
                test_image = generated_data[i][0][n]

                red = test_image[:,:,2].copy()
                blue = test_image[:,:,0].copy()

                test_image[:,:,0] = red
                test_image[:,:,2] = blue

                test_image = np.expand_dims( test_image, axis=0 )
                test_image = test_image/255.0

                result = model.predict(test_image)[0]
                ypred.append(result)
        predictions_image = []
        for i in ypred:
            if i[0] > i[1]:
                predictions_image.append(i[0])
            else:
                predictions_image.append(i[1])
        
        patch_dataframe['Classifications'] = predictions_image
        patch_dataframe.to_csv('patients/'+patient+'/classification_data.csv',index=False)
        print('Successfully classified '+patient+'! \n')
        