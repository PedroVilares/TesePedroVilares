import numpy as np
import pandas as pd
import pydicom as dicom
import imageio as io
import matplotlib.pyplot as plt
from skimage.measure import regionprops
from skimage.filters import threshold_yen

def preprocessing_mammography(image_array):
    mean = np.mean(image_array)
    std = np.std(image_array)
    preprocessed = (image_array - mean)/std
    return preprocessed

def crop_background(tiff_paths):

    for image_path in tiff_paths:
        #max_col = 0
        tiff_image = io.imread(image_path)
        tiff_array = np.asarray(tiff_image)
        height = tiff_array.shape[0]
        length = tiff_array.shape[1]
        #for p in range(300,tiff_array.shape[1]):
        #    column = tiff_array[:,p,1]
        #    if sum(column) == height:
        #        max_col = p
        #        break
        #image_type = image_path.split('/')
        #if 'R' in image_type[3]:
        #    tiff_array = np.fliplr(tiff_array)
        cropped_array = tiff_array[:height-50,:length-400]
        cropped_array = cropped_array.copy(order='C')
        save_path = image_path[:len(image_path)-4] + 'tiff'
        plt.imsave(fname=save_path,arr=cropped_array)
        print('Saved '+save_path)

def bounds(tiff_paths):

    for image_path in tiff_paths:
        #max_col = 0
        tiff_image = io.imread(image_path)
        tiff_array = np.asarray(tiff_image)
        threshold = threshold_yen(tiff_array)
        binarized_array = (tiff_array > threshold).astype(np.int_)
        a=regionprops(binarized_array)
        bounding_box = a[0].bbox
        print(tiff_array.shape)
        print(bounding_box)

