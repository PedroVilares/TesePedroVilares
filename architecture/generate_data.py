import numpy as np
import pydicom as dicom
import imageio as io
import matplotlib.pyplot as plt
from tensorflow import keras
from keras_preprocessing.image import ImageDataGenerator

def generate(dataframe,input_size):
    """
    Generates batches of data using Image Data Generator
    """

    data_generator = ImageDataGenerator()

    data = data_generator.flow_from_dataframe(
        dataframe= dataframe,
        x_col= "paths",
        y_col= "labels",
        batch_size= 128,
        target_size= (input_size,input_size),
        color_mode= "rgb",
        class_mode= "categorical",
        shuffle=False)
        
    return data
