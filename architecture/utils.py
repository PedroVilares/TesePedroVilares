import os
import pandas as pd

def remove_csv(file_list):
    for file in file_list:
        if '.' in file:
            file_list.remove(file)
    return file_list

def sort_paths(file_list):
    remove_csv(file_list)
    patient_numbers = []
    for patient in file_list:
        a = patient.split("_")
        patient_numbers.append(int(a[1]))
    patient_numbers.sort()
    patient_paths = list("patient_"+str(i) for i in patient_numbers)

    return patient_paths

def integrity_check():
    print('Checking patients folder...')
    patients_folder = 'D:/Architecture/patients/'
    patient_list = os.listdir(patients_folder)
    patient_list = remove_csv(patient_list)
    patient_number = len(patient_list)
    print('--> {} patients detected!'.format(patient_number))
    
    print('Checking mammographies...')    
    low_imgs = 0
    low_img_patients = []
    num_imgs = []
    for patient in patient_list:
        image_paths = os.listdir(patients_folder+patient)
        not_images = []
        for image in image_paths:
            if 't' in image:
                not_images.append(image)
        for image in not_images:
            image_paths.remove(image)
        
        a = len(image_paths)
        if a<2:
            low_imgs += 1
            low_img_patients.append(patient)
        num_imgs.append(a)
    print('--> {} patients with less than 2 images detected!'.format(low_imgs))
    if len(low_img_patients) != 0:
        for p in low_img_patients:
            print(p)
    avg_imgs = sum(num_imgs)/len(num_imgs)
    print('--> Average of {:.2f} images per patient!'.format(avg_imgs))
    
            
            
            
            
