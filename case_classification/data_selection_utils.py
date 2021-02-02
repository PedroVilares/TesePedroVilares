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

def fix_bcdr_path(dataFile,column):

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

def image_mover(df,p_dict):
    image_type = list(df['image_path'])[0].split('/')[-1].split('.')[-1]
    num = len(os.listdir('D:/Architecture/patients/'))
    num+=1
    patient_dict = {}
    a= 1
    for patient in p_dict.keys():
        print(a)
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
        i=1
        for n in patient_lines.index:
            if i<5:
                image_view = patient_lines['image_view'][n]
            else:
                image_view = patient_lines['image_view'][n]+'1'
            image_path = patient_lines['image_path'][n]
            if image_type == 'dcm':
                dicom_images(folder_name,image_path,image_view)
            else:
                raw_images(folder_name,image_path,image_view)
            i+=1
        print('Patient',patient,'saved!')
        num+=1
        patient_dict[patient_name] = label
        a+=1
    return patient_dict

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