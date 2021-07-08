import pandas as pd
import numpy as np
import os

import automatic_patching as patcher
import utils
import matplotlib.pyplot as plt
from scipy.stats import entropy,mode,skew,kurtosis

def dataframe_builder_case(image_report,case_report):

    patient_paths = os.listdir("patients/")
    patient_number = len(patient_paths)-1
    
    patient_list = []
    gt_list = []
    density_list = []
    age_list = []
    classification_list = []
    lo = []
    lcc = []
    ro = []
    rcc = []
    lo2 = []
    lcc2 = []
    ro2 = []
    rcc2 = []

    lo_s = []
    lcc_s = []
    ro_s = []
    rcc_s = []
    lo2_s = []
    lcc2_s = []
    ro2_s = []
    rcc2_s = []

    lo_a = []
    lcc_a = []
    ro_a = []
    rcc_a = []
    lo2_a = []
    lcc2_a = []
    ro2_a = []
    rcc2_a = []

    lists = [lo,lcc,ro,rcc,lo2,lcc2,ro2,rcc2,lo_s,lcc_s,ro_s,rcc_s,lo2_s,lcc2_s,ro2_s,rcc2_s,lo_a,lcc_a,ro_a,rcc_a,lo2_a,lcc2_a,ro2_a,rcc2_a]

    for n in range(patient_number):
        p = 'Patient '+str(n+1)
        patient_df = image_report[image_report['Patient'] == p]
        gt = list(patient_df['Ground Truth'])[0]
        if gt == 'Normal':
            gt_list.append(0)
        else:
            gt_list.append(1)
        density = list(patient_df['Density'])[0]
        age = list(patient_df['Age'])[0]
        classification = list(case_report[case_report['Patient'] == p]['Classification'])[0]
        if classification == 'Normal':
            classification_list.append(0)
        else:
            classification_list.append(1)
        patient_list.append(p)
        density_list.append(density)
        age_list.append(age)

        for i in patient_df.index:
            image_view = patient_df['Image View'][i]
            num_patches = patient_df['Patches'][i]
            sus_patches = patient_df['Suspicious'][i]
            area_factor = patient_df['Area Factor'][i]
            if len(image_view) < 4:
                if 'L' in image_view:
                    if 'O' in image_view:
                        lo.append(num_patches)
                        lo_s.append(sus_patches)
                        lo_a.append(area_factor)
                    else:
                        lcc.append(num_patches)
                        lcc_s.append(sus_patches)
                        lcc_a.append(area_factor)
                else:
                    if 'O' in image_view:
                        ro.append(num_patches)
                        ro_s.append(sus_patches)
                        ro_a.append(area_factor)
                    else:
                        rcc.append(num_patches)
                        rcc_s.append(sus_patches)
                        rcc_a.append(area_factor)
            else:
                if 'L' in image_view:
                    if 'O' in image_view:
                        lo2.append(num_patches)
                        lo2_s.append(sus_patches)
                        lo2_a.append(area_factor)
                    else:
                        lcc2.append(num_patches)
                        lcc2_s.append(sus_patches)
                        lcc2_a.append(area_factor)
                else:
                    if 'O' in image_view:
                        ro2.append(num_patches)
                        ro2_s.append(sus_patches)
                        ro2_a.append(area_factor)
                    else:
                        rcc2.append(num_patches)
                        rcc2_s.append(sus_patches)
                        rcc2_a.append(area_factor)
        
        for l in lists:
            if len(l) == n:
                l.append(0)

    for l in lists:
        print(len(l))
    master_df = pd.DataFrame({'Patient': patient_list,'Age':age_list,'Ground Truth': gt_list,'LO':lo[:144],'LCC':lcc,'RO':ro,'RCC':rcc,'LO2':lo2,'LCC2':lcc2,'RO2':ro2,'RCC2':rcc2,'LO_Suspicious':lo_s[:144],'LCC_Suspicious':lcc_s,'RO_Suspicious':ro_s,'RCC_Suspicious':rcc_s,'LO2_Suspicious':lo2_s,'LCC2_Suspicious':lcc2_s,'RO2_Suspicious':ro2_s,'RCC2_Suspicious':rcc2_s,'LO_Area/Patch':lo_a[:144],'LCC_Area/Patch':lcc_a,'RO_Area/Patch':ro_a,'RCC_Area/Patch':rcc_a,'LO2_Area/Patch':lo2_a,'LCC2_Area/Patch':lcc2_a,'RO2_Area/Patch':ro2_a,'RCC2_Area/Patch':rcc2_a,'Classification':classification_list})
    master_df.to_csv('D:/Architecture/results tf/metaclassifier_case_report.csv',index=False)
    return

def age_finder(image_paths):

    dn01 = pd.read_csv('D:/BCDR/BCDR-DN01_dataset/bcdr_dn01_img.csv')
    d01 = pd.read_csv('D:/BCDR/BCDR-D01_dataset/bcdr_d01_img.csv')
    d02 = pd.read_csv('D:/BCDR/BCDR-D02_dataset/bcdr_d02_img.csv')

    age_list = []
    for path in image_paths:
        if 'INBreast' in image_paths:
            age_list.append('NaN')
        elif 'DN01' in image_paths:
            number = path.split('/')[2].split('_')[1]
            df = dn01[dn01['patient_id'] == number]
            ages = list(df['age'])
            age_list.append(ages[0])
        elif 'D01' in image_paths:
            number = path.split('/')[2].split('_')[1]
            df = d01[d01['patient_id'] == number]
            ages = list(df['age'])
            age_list.append(ages[0])
        elif 'D02' in image_paths:
            number = path.split('/')[2].split('_')[1]
            df = d02[d02['patient_id'] == number]
            ages = list(df['age'])
            age_list.append(ages[0])
            
    return age_list

def mammary_features(image_report,transfer):

    patient_folder = "patients/"
    patient_paths = os.listdir(patient_folder)
    not_folders = []
    for patient in patient_paths:
        if 'g' in patient:
            not_folders.append(patient)
    for patient in not_folders:
        patient_paths.remove(patient)
    patient_paths = utils.sort_paths(patient_paths)

    average_list = []
    median_list = []
    std_list = []
    entropy_list = []
    mode_list = []
    skewness_list = []
    kurtosis_list = []

    for patient in patient_paths:
        print(patient)

        patient_folder = 'patients/'+patient
        image_paths = os.listdir(patient_folder)
        not_images = []
        for image in image_paths:
            if 'b' not in image:
                not_images.append(image)
        for image in not_images:
            image_paths.remove(image)

        for image in image_paths:
            full_path = patient_folder +'/'+ image
            raw_mammogram_array = patcher.raw_mammogram(full_path)
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

    image_report['I_Mean'] = average_list
    image_report['I_Std'] = std_list
    image_report['I_Median'] = median_list
    image_report['I_Mode'] = mode_list
    image_report['T_Entropy'] = entropy_list
    image_report['I_Skewness'] = skewness_list
    image_report['I_Kurtosis'] = kurtosis_list

    if transfer:
        image_report.to_csv('results tf/metaclassifier_report.csv',index=False)
    else:
        image_report.to_csv('results f-s/metaclassifier_report.csv',index=False)
    

    