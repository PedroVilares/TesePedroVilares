import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import imageio as io
import pydicom as dicom
from keras_preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Flatten,Add,BatchNormalization,Dense,Conv2D,MaxPooling2D,Dropout,Input,concatenate
from scipy.stats import entropy,mode,skew,kurtosis
from skimage.measure import regionprops
from skimage.filters import threshold_otsu
import scipy
from skimage.transform import resize
from tensorflow import keras


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
    csv_name = folder_name.split('/')[1] + '_' + folder_name.split('/')[2] + '.csv'
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
        save_path = save_path+ '/' + str(n) + '.bmp'
        plt.imsave(fname=save_path,arr=dicom_array,cmap='gray')
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

    name = 'numerical_data/' + save_name + '.csv'
    df.to_csv(name,index=False)

def crop_background():

    l = os.listdir('D:/Full Mammographies/image_data/raw/training/normal')
    p = [str(n) for n in range(len(l))]
    ps= []
    for n in p:
        a = 'D:/Full Mammographies/image_data/raw/training/normal/'+n+'.bmp'
        ps.append(a)

    for image_path in ps:    
        raw_mammogram_array = raw_mammogram(image_path)
        shape = raw_mammogram_array.shape[1]
        threshold = threshold_otsu(raw_mammogram_array)
        binarized_array = (raw_mammogram_array > threshold).astype(np.int_)
        binarized_filled= scipy.ndimage.binary_fill_holes(binarized_array).astype(int)
        a=regionprops(binarized_filled)
        bounding_box = a[0].bbox
        print(shape-bounding_box[3])

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
        downsampled_image = resize(
            image= raw_mammogram_array,
            output_shape= (size1,size2),
            preserve_range= True,
            anti_aliasing= True)
        save_path = downsample_folder + str(n) + '.bmp'
        plt.imsave(fname=save_path,arr=downsampled_image[:,:downsampled_image.shape[1]-200],cmap='gray')
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

def create_trans_model(size1,size2):
    resnet_model = keras.applications.DenseNet201(include_top=False,weights='imagenet',input_shape=(size1,size2,3))
    resnet_model.trainable=False
    model = keras.models.Sequential()
    model.add(resnet_model)
    model.add(keras.layers.Flatten())
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

    model.add(Flatten())
    #Fully Connected Layer
    model.add(Dense(1024,activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(512,activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(124,activation='relu'))
    #model.add(Dropout(0.2))

    #Output Layer
    model.add(Dense(1,activation='sigmoid'))

    return model