from operator import index
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import random
import imageio as io
import pydicom as dicom
from keras_preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Flatten,Dense,Conv2D,MaxPooling2D,Dropout
from scipy.stats import entropy,mode,skew,kurtosis
from skimage.measure import regionprops
from skimage.filters import threshold_otsu
import scipy
from skimage.transform import resize
import keras

def raw_mammogram(mammogram_path):
    """Dado um path para uma mamografia, a função abre a imagem e devolve-a como array

    Args:
        mammogram_path (string): path para mamografia

    Returns:
        nd.array: imagem convertida para array
    """
    raw_image = io.imread(mammogram_path)
    raw_array = np.asarray(raw_image[:,:,1])

    return raw_array

def binarize_breast_region(raw_mammogram_array):
    """Dado um array de uma mamografia, converte o array RGBA num array binário (1 - tecido mamário; 0 - backround) pelo método de Yen

    Args:
        raw_mammogram_array (ndarray): array RGBA

    Returns:
        ndarray: array binário
    """

    threshold = threshold_otsu(raw_mammogram_array)
    binarized_array = (raw_mammogram_array > threshold).astype(np.int_)
    binarized_filled= scipy.ndimage.binary_fill_holes(binarized_array).astype(int)

    return binarized_filled

def fix_cbis_path(dataFile,column):

    """
    #Fix broken image paths because .csv paths don't match real paths
    """

    new_paths = []
    n=0
    for path in list(dataFile[column]):
        lost_path = path.split('/')
        lost_path.pop()
        lost_path.append('1-1.dcm')
        string_path= "d:/CBIS-DDSM Full/"+lost_path[0]
        for i in lost_path[1:len(lost_path)]:
            string_path = string_path+'/'+i
            
        new_paths.append(string_path)
        dataFile.loc[n,column] = new_paths[n]
        n += 1
        
    return dataFile

def fix_bcdrN_path(dataFile,column):

    new_paths = []
    n=0
    for path in list(dataFile[column]):
        lost_path = path.split('/')
        if lost_path[1][0] == ' ':
            lost_path[1]= lost_path[1][1:]
        string_path= 'd:/BCDR/BCDR-DN01_dataset'+'/'+lost_path[0][1:]
        for i in lost_path[1:len(lost_path)]:
            string_path = string_path+'/'+i
        new_paths.append(string_path)
        n += 1
    dataFile['image_path'] = new_paths   
    return dataFile

def fix_bcdr1_path(dataFile,column):

    new_paths = []
    n=0
    for path in list(dataFile[column]):
        lost_path = path.split('/')
        if lost_path[1][0] == ' ':
            lost_path[1]= lost_path[1][1:]
        string_path= 'd:/BCDR/BCDR-D01_dataset'+'/'+lost_path[0][1:]
        for i in lost_path[1:len(lost_path)]:
            string_path = string_path+'/'+i
        new_paths.append(string_path)
        n += 1
    dataFile['image_path'] = new_paths   
    return dataFile

def fix_bcdr2_path(dataFile,column):

    new_paths = []
    n=0
    for path in list(dataFile[column]):
        lost_path = path.split('/')
        if lost_path[1][0] == ' ':
            lost_path[1]= lost_path[1][1:]
        string_path= 'd:/BCDR/BCDR-D02_dataset'+'/'+lost_path[0][1:]
        for i in lost_path[1:len(lost_path)]:
            string_path = string_path+'/'+i
        new_paths.append(string_path)
        n += 1
    dataFile['image_path'] = new_paths   
    return dataFile

def fix_inbreast_path(dataFile,column):
    new_patients = []
    for patient in list(dataFile[column]):
        patient_n= 'd:/INBreast/AllDICOMs/'+str(patient)
        new_patients.append(patient_n)
    dataFile[column] = new_patients
    return dataFile

def fix_view(dataFile,column):

    n=0
    n_view=0
    for view in list(dataFile[column]):
        if view == ' LO':
            n_view = 4
        elif view == ' LCC':
            n_view = 2
        elif view == ' RO':
            n_view = 3
        elif view == ' RCC':
            n_view = 1
        dataFile.loc[n,column] = n_view
        n += 1
        
    return dataFile

def fix_view_back(dataFile,column):

    n=0
    n_view=0
    for view in list(dataFile[column]):
        if view == 4:
            n_view = 'LO'
        elif view == 2:
            n_view = 'LCC'
        elif view == 3:
            n_view = 'RO'
        elif view == 1:
            n_view = 'RCC'
        dataFile.loc[n,column] = n_view
        n += 1
        
    return dataFile

def lesion_findings(csv):
    
    lesions = []
    for i in csv.index:
        l = ''
        mass = csv.loc[i,:]['mammography_nodule']
        micro = csv.loc[i,:]['mammography_calcification']
        calc = csv.loc[i,:]['mammography_microcalcification']
        if mass == 1:
            l = l+'mass'
        if micro == 1 or calc == 1:
            if l == '':
                l = l+'calc'
            else:
                l = l+'+calc'
        lesions.append(l)
    return lesions

def merge_csv(feature_dataframe,image_dataframe):
    for feature_row in range(len(feature_dataframe)):
        for real_row in range(len(image_dataframe)):
            if feature_dataframe.loc[feature_row,'patient_id'] == image_dataframe.loc[real_row,'patient_id']:
                if feature_dataframe.loc[feature_row,'study_id']== image_dataframe.loc[real_row,'study_id']:
                    if feature_dataframe.loc[feature_row,'image_view'] == image_dataframe.loc[real_row,'image_type_name']:
                        feature_dataframe.loc[feature_row,'image_filename'] = image_dataframe.loc[real_row,'image_filename']

    return feature_dataframe

def image_mover(df,folder_name):

    image_paths = list(df['image_path'])
    image_views = list(df['image_view'])
    new_image_paths = []
    n=0
    for path in image_paths:

        image_type = path.split('/')[-1].split('.')[-1]
        if 'R' in image_views[n]:
            flip = True
        else:
            flip = False
        if image_type == 'dcm':
            dicom_images(folder_name,path,n,flip)
        else:
            raw_images(folder_name,path,n,flip)
        new_image_paths.append(n)
        print('Image',n,'saved!')
        n +=1
    
    index_df = pd.DataFrame({'Image':new_image_paths,'Original':image_paths})
    csv_name = folder_name.split('/')[2] + '_' + folder_name.split('/')[3] + '.csv'
    index_df.to_csv('image_data/csvs/'+csv_name,index=False)

def raw_images(save_path,image_path,n,flip):
    try:
        raw_image = io.imread(image_path)
        raw_array = np.asarray(raw_image)
        if flip:
            raw_array = np.fliplr(raw_array)
        save_path = save_path+ '/' + str(n) + '.bmp'
        plt.imsave(fname=save_path,arr=raw_array,cmap='gray')
    except OSError:
        print("Couldn't find image {}".format(image_path))

def dicom_images(save_path,image_path,n,flip):
    try:
        dicom_image = dicom.read_file(image_path)
        dicom_array = dicom_image.pixel_array
        if flip:
            dicom_array = np.fliplr(dicom_array)
        clahe = cv2.createCLAHE(clipLimit =4.0, tileGridSize=(8,8))
        cl_img = clahe.apply(dicom_array)
        save_path = save_path+ '/' + str(n) + '.bmp'
        plt.imsave(fname=save_path,arr=cl_img,cmap='gray')
    except OSError:
        print("Couldn't find image {}".format(image_path))

def mammary_features(image_folder,save_name):

    l = os.listdir(image_folder)
    p = [str(n) for n in range(len(l))]
    ps= []
    for i in p:
        a = image_folder+i+'.bmp'
        ps.append(a)

    average_list = []
    median_list = []
    std_list = []
    entropy_list = []
    mode_list = []
    skewness_list = []
    kurtosis_list = []

    for image_path in ps:

        raw_mammogram_array = raw_mammogram(image_path)
        arr = raw_mammogram_array.flatten()
        arr = arr[arr > 1]
        #plt.hist(x=arr,bins=100,rwidth=0.9,histtype='bar')
        avg = np.average(arr)
        median = np.median(arr)
        std = np.std(arr)
        _,counts = np.unique(arr, return_counts=True)
        e = entropy(counts, base=None)
        m = mode(arr)[0][0]
        s= skew(arr)
        k = kurtosis(arr)

        average_list.append(float("{:.5f}".format(avg)))
        median_list.append(median)
        std_list.append(float("{:.5f}".format(std)))
        entropy_list.append(float("{:.5f}".format(e)))
        mode_list.append(m)
        skewness_list.append(float("{:.5f}".format(s)))
        kurtosis_list.append(float("{:.5f}".format(k)))

    df = pd.DataFrame({
        'I_Mean':average_list,
        'I_Std':std_list,
        'I_Median':median_list,
        'I_Mode':mode_list,
        'T_Entropy':entropy_list,
        'I_Skewness':skewness_list,
        'I_Kurtosis':kurtosis_list})
    if 'normal' in image_folder:
        df['Ground Truth'] = 0
    else:
        df['Ground Truth'] = 1
    name = 'numerical_data/' + save_name
    df.to_csv(name,index=False)

def downsample(original_folder,downsample_folder,size1,size2):
    
    l = os.listdir(original_folder)
    p = [str(n) for n in range(len(l))]
    ps= []
    for i in p:
        a = original_folder+i+'.bmp'
        ps.append(a)
    
    n=0
    for image_path in ps:

        raw_mammogram_array = raw_mammogram(image_path)
        raw_mammogram_array_cropped = raw_mammogram_array[:,:raw_mammogram_array.shape[1]-600]
        downsampled_image = resize(
            image= raw_mammogram_array_cropped,
            output_shape= (size1,size2),
            preserve_range= True,
            anti_aliasing= True)
        save_path = downsample_folder + str(n) + '.bmp'
        plt.imsave(fname=save_path,arr=downsampled_image,cmap='gray')
        print(n)
        n += 1
        
def generator_transfer(size1,size2,aug,s):

    if aug:
        data_generator = ImageDataGenerator(
            rescale= 1./255,
            zoom_range= 0.15,
            rotation_range= 15
            )
    else:
        data_generator = ImageDataGenerator(rescale= 1./255)

    training_dir = 'D:/Full Mammographies/image_data/downsampled/training/'
    validation_dir = 'D:/Full Mammographies/image_data/downsampled/validation/'

    t_data = data_generator.flow_from_directory(
        directory = training_dir,
        batch_size= 12,
        target_size= (size1,size2),
        class_mode= "binary",
        shuffle = s)
    
    v_data_generator = ImageDataGenerator(rescale= 1./255)
    v_data = v_data_generator.flow_from_directory(
        directory = validation_dir,
        batch_size= 12,
        target_size= (size1,size2),
        class_mode= "binary",
        shuffle = s)
        
    return t_data,v_data

def generator_scratch(size1,size2,aug,s):

    if aug:
        data_generator = ImageDataGenerator(
            rescale= 1./255,
            zoom_range= 0.15,
            rotation_range= 15
            )
    else:
        data_generator = ImageDataGenerator(rescale= 1./255)

    l = os.listdir('D:/Full Mammographies/image_data/downsampled/training/normal')
    p = [str(n) for n in range(len(l))]
    ps= []
    for n in p:
        a = 'D:/Full Mammographies/image_data/downsampled/training/normal/'+n+'.bmp'
        ps.append(a)
    training_df_norm= pd.DataFrame({'paths':ps,'labels':'Normal'})

    l = os.listdir('D:/Full Mammographies/image_data/downsampled/training/suspicious')
    p = [str(n) for n in range(len(l))]
    ps= []
    for n in p:
        a = 'D:/Full Mammographies/image_data/downsampled/training/suspicious/'+n+'.bmp'
        ps.append(a)
    training_df_sus= pd.DataFrame({'paths':ps[:170],'labels':'Suspicious'})

    l = os.listdir('D:/Full Mammographies/image_data/downsampled/validation/normal')
    p = [str(n) for n in range(len(l))]
    ps= []
    for n in p:
        a = 'D:/Full Mammographies/image_data/downsampled/validation/normal/'+n+'.bmp'
        ps.append(a)
    validation_df_norm= pd.DataFrame({'paths':ps,'labels':'Normal'})

    l = os.listdir('D:/Full Mammographies/image_data/downsampled/validation/suspicious')
    p = [str(n) for n in range(len(l))]
    ps= []
    for n in p:
        a = 'D:/Full Mammographies/image_data/downsampled/validation/suspicious/'+n+'.bmp'
        ps.append(a)
    validation_df_sus= pd.DataFrame({'paths':ps[:70],'labels':'Suspicious'})

    training_df = pd.concat([training_df_sus,training_df_norm],ignore_index=True)
    validation_df = pd.concat([validation_df_norm,validation_df_sus],ignore_index=True)

    t_data = data_generator.flow_from_dataframe(
        dataframe= training_df,
        x_col='paths',
        y_col='labels',
        batch_size= 12,
        color_mode='grayscale',
        target_size= (size1,size2),
        class_mode= "binary",
        shuffle = s)
    
    v_data_generator = ImageDataGenerator(rescale= 1./255)
    v_data = v_data_generator.flow_from_dataframe(
        dataframe= validation_df,
        x_col='paths',
        y_col='labels',
        batch_size= 12,
        color_mode='grayscale',
        target_size= (size1,size2),
        class_mode= "binary",
        shuffle = s)
        
    return t_data,v_data

def generator_negatives(size1,size2,aug,s):

    if aug:
        data_generator = ImageDataGenerator(
            rescale= 1./255,
            zoom_range= 0.15,
            rotation_range= 15
            )
    else:
        data_generator = ImageDataGenerator(rescale= 1./255)

    l = os.listdir('D:/Full Mammographies/image_data/negatives/training/normal')
    p = [str(n) for n in range(len(l))]
    ps= []
    for n in p:
        a = 'D:/Full Mammographies/image_data/negatives/training/normal/'+n+'.bmp'
        ps.append(a)
    training_df_norm= pd.DataFrame({'paths':ps,'labels':'Normal'})

    l = os.listdir('D:/Full Mammographies/image_data/negatives/training/suspicious')
    p = [str(n) for n in range(len(l))]
    ps= []
    for n in p:
        a = 'D:/Full Mammographies/image_data/negatives/training/suspicious/'+n+'.bmp'
        ps.append(a)
    training_df_sus= pd.DataFrame({'paths':ps,'labels':'Suspicious'})

    l = os.listdir('D:/Full Mammographies/image_data/negatives/validation/normal')
    p = [str(n) for n in range(len(l))]
    ps= []
    for n in p:
        a = 'D:/Full Mammographies/image_data/negatives/validation/normal/'+n+'.bmp'
        ps.append(a)
    validation_df_norm= pd.DataFrame({'paths':ps,'labels':'Normal'})

    l = os.listdir('D:/Full Mammographies/image_data/negatives/validation/suspicious')
    p = [str(n) for n in range(len(l))]
    ps= []
    for n in p:
        a = 'D:/Full Mammographies/image_data/negatives/validation/suspicious/'+n+'.bmp'
        ps.append(a)
    validation_df_sus= pd.DataFrame({'paths':ps,'labels':'Suspicious'})

    training_df = pd.concat([training_df_sus,training_df_norm],ignore_index=True)
    validation_df = pd.concat([validation_df_norm,validation_df_sus],ignore_index=True)

    t_data = data_generator.flow_from_dataframe(
        dataframe= training_df,
        x_col='paths',
        y_col='labels',
        batch_size= 12,
        target_size= (size1,size2),
        class_mode= "binary",
        shuffle = s)
    
    v_data_generator = ImageDataGenerator(rescale= 1./255)
    v_data = v_data_generator.flow_from_dataframe(
        dataframe= validation_df,
        x_col='paths',
        y_col='labels',
        batch_size= 12,
        target_size= (size1,size2),
        class_mode= "binary",
        shuffle = s)
        
    return t_data,v_data

def generator_q1(size1,size2,view,s):

    data_generator = ImageDataGenerator(rescale= 1./255)

    training_dir = 'D:/Full Mammographies/image_data/quadrant_1/' + view + '/training/'
    validation_dir = 'D:/Full Mammographies/image_data/quadrant_1/' + view + '/validation/'

    t_data = data_generator.flow_from_directory(
        directory = training_dir,
        batch_size= 12,
        target_size= (size1,size2),
        class_mode= "binary",
        shuffle = s)
    
    v_data_generator = ImageDataGenerator(rescale= 1./255)
    v_data = v_data_generator.flow_from_directory(
        directory = validation_dir,
        batch_size= 12,
        target_size= (size1,size2),
        class_mode= "binary",
        shuffle = s)
        
    return t_data,v_data

def generator_q2(size1,size2,view,s):

    data_generator = ImageDataGenerator(rescale= 1./255)

    training_dir = 'D:/Full Mammographies/image_data/quadrant_2/' + view + '/training/'
    validation_dir = 'D:/Full Mammographies/image_data/quadrant_2/' + view + '/validation/'

    t_data = data_generator.flow_from_directory(
        directory = training_dir,
        batch_size= 12,
        target_size= (size1,size2),
        class_mode= "binary",
        shuffle = s)
    
    v_data_generator = ImageDataGenerator(rescale= 1./255)
    v_data = v_data_generator.flow_from_directory(
        directory = validation_dir,
        batch_size= 12,
        target_size= (size1,size2),
        class_mode= "binary",
        shuffle = s)
        
    return t_data,v_data

def generator_q3(size1,size2,view,s):

    data_generator = ImageDataGenerator(rescale= 1./255)

    training_dir = 'D:/Full Mammographies/image_data/quadrant_3/' + view + '/training/'
    validation_dir = 'D:/Full Mammographies/image_data/quadrant_3/' + view + '/validation/'

    t_data = data_generator.flow_from_directory(
        directory = training_dir,
        batch_size= 12,
        target_size= (size1,size2),
        class_mode= "binary",
        shuffle = s)
    
    v_data_generator = ImageDataGenerator(rescale= 1./255)
    v_data = v_data_generator.flow_from_directory(
        directory = validation_dir,
        batch_size= 12,
        target_size= (size1,size2),
        class_mode= "binary",
        shuffle = s)
        
    return t_data,v_data

def generator_patches(dataframe,input_size):
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
        class_mode= None,
        shuffle=False)
        
    return data

def augment(size1,size2,directory):

    data_generator = ImageDataGenerator(
            zoom_range= 0.25,
            rotation_range= 25,
            )

    l = os.listdir(directory)
    p = []
    for n in l:
        if '_' in n:
            continue
        p.append(n.split('.')[0])
        p.sort()
    ps= []
    for n in p:
        a = directory + n + '.bmp'
        ps.append(a)
    df= pd.DataFrame({'paths':ps,'labels':'Normal'})

    data = data_generator.flow_from_dataframe(
        dataframe= df,
        x_col='paths',
        y_col='labels',
        batch_size= 12,
        target_size= (size1,size2),
        class_mode= None,
        save_to_dir = directory,
        save_format = 'bmp',
        shuffle = False)

    return data

def create_trans_model(size1,size2):

    resnet_model = keras.applications.DenseNet201(include_top=False,weights='imagenet',input_shape=(size1,size2,3))
    resnet_model.trainable=False
    model = keras.models.Sequential()
    model.add(resnet_model)
    model.add(keras.layers.Flatten())
    model.add(Dense(1024,activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(512,activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(124,activation='relu'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    return model

def create_fs_model(size1,size2):

    model = Sequential()

    #first block
    model.add(Conv2D(16,kernel_size=(3, 3),strides=(2,2),input_shape=(size1,size2,3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

    #second block
    model.add(Conv2D(32,kernel_size=(3, 3),strides=(2, 2),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

    #third block
    model.add(Conv2D(64,kernel_size=(3, 3),strides=(1, 1),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2, 2)))

    #Fully Connected Layer
    model.add(Flatten())
    model.add(Dense(1024,activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(512,activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(124,activation='relu'))
    #model.add(Dropout(0.2))

    #Output Layer
    model.add(Dense(1,activation='sigmoid'))

    return model

def copy_negatives(original_folder,downsample):

    l = os.listdir(original_folder)
    p = [str(n) for n in range(len(l))]
    ps= []
    for i in p:
        a = original_folder+i+'.bmp'
        ps.append(a)
    
    l = original_folder.split('/')
    #if downsample:
    #    end_folder = l[0] +'/negatives_downsampled/'+l[2] + '/' +l[3]
    #else:
    #    end_folder = l[0] +'/negatives/'+l[2] + '/' +l[3]
    end_folder = 'image_data/negatives/' + l[2] + '/' +l[3]

    n=0
    for image_path in ps:
        raw_mammogram_array = raw_mammogram(image_path)
        negative_array = 255 - raw_mammogram_array
        negative_array_cropped = negative_array[:,:negative_array.shape[1]-600]
        downsampled_image = resize(
            image= negative_array_cropped,
            output_shape= (1500,1100),
            preserve_range= True,
            anti_aliasing= True)
        save_path = end_folder + '/' + str(n) + '.bmp'
        plt.imsave(fname=save_path,arr=negative_array,cmap='gray')
        n+=1

def copy_quadrants(original_folder):

    l = os.listdir(original_folder)
    p = [str(n) for n in range(len(l))]
    ps= []
    for i in p:
        a = original_folder+i+'.bmp'
        ps.append(a)
    
    p = original_folder.split('/')
    
    n=0
    for image_path in ps:
        raw_mammogram_array = raw_mammogram(image_path)
        b= binarize_breast_region(raw_mammogram_array)
        if b[height_min,width_min+50] == 1:
            c=regionprops(b[50:,:])
            bounding_box = c[0].bbox

            height_min = bounding_box[0]
            height_max = bounding_box[2]
            width_min = bounding_box[1]
            width_max = bounding_box[3]

            height = height_max-height_min
            inc = height/3

            q1 = raw_mammogram_array[height_min:round(height_min+inc)+100,width_min:width_max]
            q2 = raw_mammogram_array[round(height_min+inc)-100:round(height_min+2*inc)+100,width_min:width_max]
            q3 = raw_mammogram_array[round(height_min+2*inc)-100:height_max,width_min:width_max]

            save_path1 = p[0] + '/quadrant_1/MLO/' + p[2] + '/' + p[3] + '/' + str(n) + '.bmp'
            save_path2 = p[0] + '/quadrant_2/MLO/' + p[2] + '/' + p[3] + '/' + str(n) + '.bmp'
            save_path3 = p[0] + '/quadrant_3/MLO/' + p[2] + '/' + p[3] + '/' + str(n) + '.bmp'            

        else:

            c=regionprops(b)
            bounding_box = c[0].bbox

            height_min = bounding_box[0]
            height_max = bounding_box[2]
            width_min = bounding_box[1]
            width_max = bounding_box[3]

            height = height_max-height_min
            inc = height/3

            q1 = raw_mammogram_array[height_min:round(height_min+inc)+100,width_min:width_max]
            q2 = raw_mammogram_array[round(height_min+inc)-100:round(height_min+2*inc)+100,width_min:width_max]
            q3 = raw_mammogram_array[round(height_min+2*inc)-100:height_max,width_min:width_max]

            save_path1 = p[0] + '/quadrant_1/CC/' + p[2] + '/' + p[3] + '/' + str(n) + '.bmp'
            save_path2 = p[0] + '/quadrant_2/CC/' + p[2] + '/' + p[3] + '/' + str(n) + '.bmp'
            save_path3 = p[0] + '/quadrant_3/CC/' + p[2] + '/' + p[3] + '/' + str(n) + '.bmp'

        plt.imsave(fname=save_path1,arr=q1,cmap='gray')
        plt.imsave(fname=save_path2,arr=q2,cmap='gray')
        plt.imsave(fname=save_path3,arr=q3,cmap='gray')
        n+=1

def classify_mammograms(model,dataframe,comp):

    _,validation_gen = generator_transfer(1500,1100,False,False)
    predictions = model.predict(validation_gen,verbose=1)
    predictions_image=[]
    for i in range(len(predictions)):
        predictions_image.append(float("{:.3f}".format(predictions[i][1])))
    dataframe['Classifications'] = predictions_image
    if comp:
        dataframe.to_csv('numerical_data/classification_data_comp.csv',index=False)
    else:
        dataframe.to_csv('numerical_data/classification_data.csv',index=False)
    
def sistematic_patch_corners(binarized_array,patch_size,background_percentage,overlapping):

    """Função calcula o retângulo de operação da mamografia e centros possíveis dos patches de forma sistemática. Dependendo do patch size e da background percentage,
    devolve os vértices dos patches

    Args:
        binarized_array (ndarray): mamografia binarizada
        patch_size (int): dimensão do patch
        background_percentage (float): percentagem mínima de foreground permitida
    """
    a=regionprops(binarized_array)
    bounding_box = a[0].bbox
    height_min = bounding_box[0]
    width_min = bounding_box[1]
    height_max = bounding_box[2]
    width_max= bounding_box[3]

    #identificar as MLO
    if height_min < 250:
        muscle = binarized_array[0:250,:]
        a=regionprops(muscle)
        bounding_box = a[0].bbox
        if width_max > binarized_array.shape[1]-100:
            #Right
            if bounding_box[3] - bounding_box[1] > 150:
                height_min = 350
                width_max = width_max - 100
        else:
            #Left
            if bounding_box[3] > 150:
                height_min = 350
                width_min = width_min + 70
    else:
        if width_min < 500:
            width_max = width_max - 150
            height_max = height_max - 150
            height_min = height_min + 150
        else:
            width_min = width_min + 150
            height_max = height_max - 150
            height_min = height_min + 150

    overlapping_percentage = overlapping
    overlap = overlapping_percentage*patch_size
    window_size1=height_min
    window_size2=height_min+patch_size
    window_size3=width_min
    window_size4=width_min+patch_size
    n_patches_h = int(np.floor((width_max-width_min))/(patch_size-overlap))
    n_patches_v = int(np.floor((height_max-height_min))/(patch_size-overlap))
    patch_vertexes = []
    for i in range(n_patches_v):
        for n in range(n_patches_h):
            patch_vertex = (int(window_size1),int(window_size2),int(window_size3),int(window_size4))
            b_percentage = backround_calculator(binarized_array,patch_vertex)
            if b_percentage > background_percentage:
                patch_vertexes.append(patch_vertex)

            window_size3 = window_size3 + (patch_size-overlap)
            window_size4 = window_size4 + (patch_size-overlap)
            n+=1
        window_size3 = width_min
        window_size4 = width_min+patch_size
        window_size1 = window_size1 + (patch_size-overlap)
        window_size2 = window_size2 + (patch_size-overlap)
        i+=1

    return patch_vertexes

def random_patch_corners(binarized_array,patch_size,background_percentage):
    """Função calcula o retângulo de operação da mamografia e centros possíveis dos patches. Dependendo do patch size e da background percentage,
    devolve os vértices dos patches

    Args:
        binarized_array (ndarray): array binário
        patch_size (int): largura de um patch
        background_percentage (float): percentagem de backgroung permitido num patch

    Returns:
        list: lista de tuplos, em que cada tuplo tem quatro vértices de um patch
    """
        
    a=regionprops(binarized_array)
    bounding_box = a[0].bbox
    height_min = bounding_box[0]
    width_min = bounding_box[1]
    height_max = bounding_box[2]
    width_max= bounding_box[3]

    #identificar as MLO
    if height_min < 250:
        muscle = binarized_array[0:250,:]
        a=regionprops(muscle)
        bounding_box = a[0].bbox
        if width_max > binarized_array.shape[1]-100:
            #Right
            if bounding_box[3] - bounding_box[1] > 150:
                height_min = 350
                width_max = width_max - 100
        else:
            #Left
            if bounding_box[3] > 150:
                height_min = 350
                width_min = width_min + 70
    else:
        if width_min < 500:
            width_max = width_max - 150
            height_max = height_max - 150
            height_min = height_min + 150
        else:
            width_min = width_min + 150
            height_max = height_max - 150
            height_min = height_min + 150

    n_patches = int(np.floor(((width_max-width_min) * (height_max-height_min))/(patch_size*patch_size)))
    patch_vertexes = []
    for i in range(round(2*n_patches)): #Qual deve ser o range?
        x_center = random.randint(width_min+(patch_size/2)+100,width_max-(patch_size/2))
        y_center = random.randint(height_min+(patch_size/2)+100,height_max-(patch_size/2)-100)
        if binarized_array[y_center,x_center] == 1:
            if binarized_array[int(y_center+(patch_size/2)),x_center] == 1 and binarized_array[int(y_center-(patch_size/2)),x_center]==1:
                if binarized_array[y_center,int(x_center+patch_size/2)] == 1 and binarized_array[y_center,int(x_center-patch_size/2)]==1:
                    patch_vertex = (int(y_center-(patch_size/2)),int(y_center+(patch_size/2)),int(x_center-patch_size/2),int(x_center+patch_size/2))
                    b_percentage = backround_calculator(binarized_array,patch_vertex)
                    if b_percentage > background_percentage:
                        patch_vertexes.append(patch_vertex)

    return patch_vertexes

def backround_calculator(binary_mammogram,patch_vertexes):
    """Função calcula a percentagem de background num patch

    Args:
        binary_mammogram (ndarray): array de mamografia binária
        patch_vertex (lista): lista com vértices dos patches

    Returns:
        float: percentagem de background num patch
    """
    patch = binary_mammogram[patch_vertexes[0]:patch_vertexes[1],patch_vertexes[2]:patch_vertexes[3]]
    a = sum(sum(patch))
    b = patch.shape[0]*patch.shape[0]

    background_percentage = a/b

    return background_percentage  

def patches_by_image(image_path,patch_size,overlap):

    n=0
    try:
        folder = 'image_data/patches/' + image_path.split('/')[2] + '/' + image_path.split('/')[3] + '/' + image_path.split('/')[4].split('.')[0]
        os.mkdir(folder)
        print('Successfully created '+folder+' folder!')
    except OSError:
        print("Patch folder already exists!")

    raw_mammogram_array = raw_mammogram(image_path)
    binarized_array = binarize_breast_region(raw_mammogram_array)
    patches_vertexes_1 = sistematic_patch_corners(binarized_array,patch_size,0.99,overlap)
    patches_vertexes_2 = random_patch_corners(binarized_array,patch_size,0.99)
    patches_vertexes = patches_vertexes_1 + patches_vertexes_2

    for vertexes in patches_vertexes:
        patch = raw_mammogram_array[vertexes[0]:vertexes[1],vertexes[2]:vertexes[3]]
        filename = folder+'/'+str(n)+'.bmp'
        plt.imsave(filename,patch,cmap='gray')
        #print('Patch',n,'saved!')
        n+=1
    print('Saved',n,'patches from '+image_path+' image! \n')  

def save_patches_by_image(original_folder,patch_size,overlap):

    l = os.listdir(original_folder)
    p = [str(n) for n in range(len(l))]
    ps= []
    for i in p:
        a = original_folder+i+'.bmp'
        ps.append(a)

    for image_path in ps:

        patches_by_image(image_path,patch_size,overlap)

def classify_patches(original_folder):

    l = os.listdir(original_folder)
    p = [str(n) for n in range(len(l))]
    ps= []
    for i in p:
        a = original_folder+i+'.bmp'
        ps.append(a)
    model = keras.models.load_model("D:/Lesion-Based Patches/vgg19_trained")
    for image_path in ps:
        print('Image ',image_path.split('/')[4].split('.')[0])
        folder = 'image_data/patches/' + image_path.split('/')[2] + '/' + image_path.split('/')[3] + '/' + image_path.split('/')[4].split('.')[0]
        l = os.listdir(folder)
        patch_numbers = [str(n) for n in range(len(l))]
        patches= []
        for i in patch_numbers:
            a = folder+'/'+i+'.bmp'
            patches.append(a)
        df = pd.DataFrame({'paths':patches,'labels':'Test'})
        generated_data = generator_patches(df,input_size=300)
        predictions = model.predict(generated_data,verbose=1)
        predictions_image=[]
        for i in range(len(predictions)):
            predictions_image.append(predictions[i][0])
        
        classifications = pd.DataFrame({'Classifications':predictions_image})
        classifications.to_csv(folder+'/classifications.csv',index=False)

def add_to_features_csv(original_folder,features_df_path,threshold):

    l = os.listdir(original_folder)
    p = [str(n) for n in range(len(l))]
    ps= []
    for i in p:
        a = original_folder+i+'.bmp'
        ps.append(a)
    
    #features_df = pd.read_csv(features_df_path)
    #ratios = []
    #for image_path in ps:
    #    positive_patches = 0
    #    folder = 'image_data/patches/' + image_path.split('/')[2] + '/' + image_path.split('/')[3] + '/' + image_path.split('/')[4].split('.')[0]
    #    csv_path = folder + '/classifications.csv'
    #    csv = pd.read_csv(csv_path)
    #    classifications_list = list(csv['Classifications'])
    #    for i in classifications_list:
    #        if i > threshold:
    #            positive_patches += 1
    #    ratio = float("{:.2f}".format(positive_patches/len(classifications_list)))
    #    ratios.append(ratio)
    
    #features_df['Ratio'] = ratios
    #features_df.to_csv(features_df_path,index=False)

    #features_df1 = pd.read_csv('numerical_data/features_training_normal.csv')
    #features_df2 = pd.read_csv('numerical_data/features_training_suspicious.csv')
    features_df3 = pd.read_csv('numerical_data/features_validation_normal.csv')
    features_df4 = pd.read_csv('numerical_data/features_validation_suspicious.csv')
    #ratio1 = features_df1['Classifications']
    #ratio2 = features_df2['Classifications']
    ratio3 = list(features_df3['Ratio'])
    ratio4 = list(features_df4['Ratio'])

    #ratios = ratio1+ratio2+ratio3+ratio4
    ratios = ratio3+ratio4
    meta_df = pd.read_csv('numerical_data/classification_data_comp.csv')
    meta_df['Ratios'] = ratios
    meta_df.to_csv('numerical_data/classification_data_comp.csv',index=False)
