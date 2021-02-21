import tensorflow as tf
import os
import pandas as pd
import generate_data as generator
import utils
from keras import metrics,optimizers,losses
from keras.callbacks import Callback
from keras.models import Sequential
from keras.layers import Flatten,Add,BatchNormalization,Dense,Conv2D,MaxPooling2D,Dropout,GlobalAveragePooling2D,Input,concatenate

def classify_cases(model):

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
        predictions = model.predict(generated_data)
        predictions_image=[]
        for i in range(len(predictions)):
            predictions_image.append(predictions[i][0])
        
        patch_dataframe['Classifications'] = predictions_image
        patch_dataframe.to_csv('patients/'+patient+'/classification_data.csv',index=False)
        print('Successfully classified '+patient+'! \n')
        