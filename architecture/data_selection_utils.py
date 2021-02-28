import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import imageio as io
import pydicom as dicom

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
        new_paths_df = pd.DataFrame({'path':new_paths})
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

def merge_csv(feature_dataframe,image_dataframe):
    for feature_row in range(len(feature_dataframe)):
        for real_row in range(len(image_dataframe)):
            if feature_dataframe.loc[feature_row,'patient_id'] == image_dataframe.loc[real_row,'patient_id']:
                if feature_dataframe.loc[feature_row,'study_id']== image_dataframe.loc[real_row,'study_id']:
                    if feature_dataframe.loc[feature_row,'image_view'] == image_dataframe.loc[real_row,'image_type_name']:
                        feature_dataframe.loc[feature_row,'image_filename'] = image_dataframe.loc[real_row,'image_filename']

    return feature_dataframe

def all_in(candidates, sequence):
    for element in sequence:
        if element not in candidates:
            return False
    return True

def image_mover(df,p_dict):
    image_type = list(df['image_path'])[0].split('/')[-1].split('.')[-1]
    num = len(os.listdir('D:/Architecture/patients/'))
    num+=1
    patients = []
    labels = []
    image_views= []
    x_centers = []
    y_centers = []
    densities = []
    image_paths = []
    for patient in p_dict.keys():
        patient_lines = df[df['patient_id'] == patient]
        patient_name = 'patient_'+str(num)
        image_type = list(patient_lines['image_path'])[0].split('/')[-1].split('.')[-1]
        label = list(patient_lines['label'])[1]
        folder_name = 'D:/Architecture/patients/'+patient_name
        try:
            os.mkdir(folder_name)
        except OSError:
            num+=1
            continue
        views_copied = []
        i=1
        for n in patient_lines.index:
            image_view = patient_lines['image_view'][n]
            if image_view in views_copied:
                image_view = patient_lines['image_view'][n]+str(i)
                i+=1
            image_path = patient_lines['image_path'][n]
            if image_type == 'dcm':
                dicom_images(folder_name,image_path,image_view)
            else:
                raw_images(folder_name,image_path,image_view)
            views_copied.append(image_view)
            patients.append(patient_name)
            labels.append(label)
            x_centers.append(patient_lines['x_center'][n])
            y_centers.append(patient_lines['y_center'][n])
            image_views.append(image_view)
            densities.append(patient_lines['density'][n])
            image_paths.append(image_path)
        print('Patient',patient,'saved!')
        num+=1
    return patients,image_views,labels,x_centers,y_centers,densities,image_paths

def raw_images(save_path,image_path,image_view):
    try:
        raw_image = io.imread(image_path)
        raw_array = np.asarray(raw_image)
        save_path = save_path+ '/' + image_view + '.bmp'
        plt.imsave(fname=save_path,arr=raw_array)
    except OSError:
        print("Couldn't find image {}".format(image_path))

def dicom_images(save_path,image_path,image_view):
    try:
        dicom_image = dicom.read_file(image_path)
        dicom_array = dicom_image.pixel_array
        save_path = save_path+ '/' + image_view + '.bmp'
        plt.imsave(fname=save_path,arr=dicom_array)
    except OSError:
        print("Couldn't find image {}".format(image_path))