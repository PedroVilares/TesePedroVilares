import numpy as np
import pydicom as dicom
import imageio as io
import matplotlib.pyplot as plt
from tensorflow import keras
import preprocessing
from keras_preprocessing.image import ImageDataGenerator

def generate(dataframe,image_height,image_width,batch_size):
    """
    Generates batches of data using Image Data Generator
    """

    data_generator = ImageDataGenerator(preprocessing_function=preprocessing.preprocessing_mammography)

    data = data_generator.flow_from_dataframe(
        dataframe= dataframe,
        x_col= "paths",
        y_col= "labels",
        batch_size= batch_size,
        target_size= (image_height,image_width),
        color_mode= "grayscale",
        class_mode= "categorical")
    return data

def convert(tif_paths):
    """
    Converts TIF images to TIFF and returns TIFF paths
    """
    tiff_paths = []
    for image_path in tif_paths:
        #tif_image = io.imread(image_path)
        #tif_array = np.asarray(tif_image)
        #image_type = image_path.split('/')
        #if 'R' in image_type[3]:
        #    tif_array = np.fliplr(tif_array)
        save_path = image_path[:len(image_path)-3] + 'tiff'
        tiff_paths.append(save_path)
        #plt.imsave(fname=save_path,arr=tif_array)
        #print('Saved '+save_path)
    
    return tiff_paths