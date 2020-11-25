import numpy as np
import pandas as pd
import pydicom as dicom
import matplotlib.pyplot as plt

def preprocessing_mammography(image_array):
    mean = np.mean(image_array)
    std = np.std(image_array)
    preprocessed = (image_array - mean)/std

    return preprocessed



