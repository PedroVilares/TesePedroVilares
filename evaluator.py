import os
from sklearn.metrics import roc_curve,roc_auc_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import imageio as io

def calculate_threshold_predictions(case_dict,thresh):
    predictions = []
    for _,i_classifications in case_dict.items():
        ratios = []
        for _,p_predictions in i_classifications.items():
            num_patches = len(p_predictions)
            positive_patches = 0
            for pred in p_predictions:
                if pred > 0.5:
                    positive_patches += 1
            ratio = positive_patches/num_patches
            ratios.append(ratio)
        avg_ratio = round(average_ratio(ratios),2)
        if avg_ratio > thresh:
            predictions.append(1)
        else:
            predictions.append(0)

    return predictions

def calculate_case_predictions(case_dict,ground_truth,p_list,thresh_min,thresh_max):
    predictions = []
    g_truth = []
    patient_list = []
    n=0
    for _,i_classifications in case_dict.items():
        ratios = []
        for _,p_predictions in i_classifications.items():
            num_patches = len(p_predictions)
            positive_patches = 0
            for pred in p_predictions:
                if pred > 0.5:
                    positive_patches += 1
            ratio = positive_patches/num_patches
            ratios.append(ratio)
        avg_ratio = round(average_ratio(ratios),2)
        if avg_ratio < thresh_min:
            predictions.append(0)
            g_truth.append(ground_truth[n])
            patient_list.append(p_list[n])
        elif avg_ratio > thresh_max:
            predictions.append(1)
            g_truth.append(ground_truth[n])
            patient_list.append(p_list[n])
        n+=1

    return predictions,g_truth,patient_list

def evaluate_predictions(patient_list,ground_truth,predictions):

    prediction_df = pd.DataFrame({'patient':patient_list,'ground truth':ground_truth,'predictions':predictions})
    auc = roc_auc_score(ground_truth,predictions)
    fpr,tpr,_= roc_curve(ground_truth,predictions)
    plt.plot(fpr,tpr)

    return prediction_df,auc

def report_by_case(cases_classifications,ground_truth_csv):
    txt_file = open("Classification Report by case.txt","w")
    n=1
    master_df = pd.DataFrame()
    df_list = [df for df in ground_truth_csv.groupby('patient')]
    ground_truth = []
    density = []
    for df in df_list:
        ground_truth.append(list(df[1]['label'])[0])
        density.append( list(df[1]['density'])[0])

    patient_list = []
    patches_list = []
    sus_patches_list = []
    density_list = []
    gt_list = []

    for patient,p_predictions in cases_classifications.items():

        gt = ground_truth[n-1]
        print('Patient {} | {} | Density: {}'.format(n,gt,density[n-1]))
        txt_file.write('\n Patient {} | {} | Density: {}\n'.format(n,gt,density[n-1]))
        patient_list.append('Patient {}'.format(n))
        gt_list.append(gt)
        density_list.append(density[n-1])
    
        num_patches = len(p_predictions)
        positive_patches = 0
        for pred in p_predictions:
            if pred > 0.95:
                positive_patches += 1
        ratio = positive_patches/num_patches
        print(patient +': Patches {} | Positive Patches {} | Ratio {:.2f}'.format(num_patches,positive_patches,ratio))
        txt_file.write(patient +': Patches {} | Positive Patches {} | Ratio {:.2f} \n'.format(num_patches,positive_patches,ratio))

        patches_list.append(num_patches)
        sus_patches_list.append(positive_patches)

        txt_file.write('Ratio: {} \n'.format(round(ratio,2)))
        print('\n')
        n+=1

    master_df = pd.DataFrame({'Patient': patient_list,'Ground Truth': gt_list,'Density': density_list,'Patches': patches_list,'Suspicious': sus_patches_list})
    master_df.to_csv('D:/Architecture/classification_report_by_case.csv',index=False)
    txt_file.close()

def report_by_image(cases_classifications,ground_truth_csv):
    txt_file = open("Classification Report by image.txt","w")
    n=1

    ground_truth = list(ground_truth_csv['label'])
    density = list(ground_truth_csv['density'])

    patient_list = []
    patches_list = []
    sus_patches_list = []
    image_views_list = []
    density_list = []
    gt_list = []

    for _,i_classifications in cases_classifications.items():

        print('Patient {} | {} | Density: {}'.format(n,ground_truth[n-1],density[n-1]))
        txt_file.write('\n Patient {} | {} | Density: {}\n'.format(n,ground_truth[n-1],density[n-1]))

        ratios = []
        for image_view,p_predictions in i_classifications.items():
            num_patches = len(p_predictions)
            positive_patches = 0
            for pred in p_predictions:
                if pred > 0.5:
                    positive_patches += 1
            ratio = positive_patches/num_patches
            ratios.append(ratio)
            print(image_view[1:] +': Patches {} | Positive Patches {} | Ratio {:.2f}'.format(num_patches,positive_patches,ratio))
            txt_file.write(image_view[1:] +': Patches {} | Positive Patches {} | Ratio {:.2f} \n'.format(num_patches,positive_patches,ratio))

            patient_list.append('Patient {}'.format(n))
            gt_list.append(ground_truth[n-1])
            density_list.append(density[n-1])
            patches_list.append(num_patches)
            sus_patches_list.append(positive_patches)
            image_views_list.append(image_view)

        print('Average Ratio: ',round(average_ratio(ratios),2))
        txt_file.write('Average Ratio: {} \n'.format(round(average_ratio(ratios),2)))
        print('\n')
        n+=1
        
    master_df = pd.DataFrame({'Patient': patient_list,'Ground Truth': gt_list,'Density': density_list,'Image View':image_views_list,'Patches': patches_list,'Suspicious': sus_patches_list})    
    master_df.to_csv('D:/Architecture/classification_report_by_image.csv',index=False)
    txt_file.close()

def average_ratio(ratio_list):
    return sum(ratio_list)/len(ratio_list)

def make_hist(predictions,patient_number,gt):

    n,bins,patches = plt.hist(x=predictions,bins=10,alpha=0.7,rwidth=0.8)
    plt.title('Patient {} | {}'.format(patient_number,gt))
    plt.xlabel('Predictions')
    plt.ylabel('Frequency')
    plt.show()
    return

def show_positive_patches_case(predictions,patch_folder):
    p = os.listdir(patch_folder)
    pa = []
    positive_patches = 0
    n = 0
    for pred in predictions:
        if pred > 0.95:
            positive_patches += 1
            pa.append(patch_folder+p[n])
        n+=1
    if positive_patches == 0:
        print('No patches classified as Suspicious')
        return

    if positive_patches < 4:
        cols = positive_patches
    else:
        cols = 4
    rows = np.ceil(len(pa[:30])/cols).astype(np.int_)
    f,s = plt.subplots(rows,cols,figsize=(20,20))
    i=0
    for path in pa[:30]:
        patch = io.imread(path)
        patch_number = path.split('/')[3].split('.')[0]
        if rows == 1:
            s[i%cols].imshow(patch,cmap='gray')
            s[i%cols].set_title(patch_number)
        else:
            s[i//cols,i%cols].imshow(patch,cmap='gray')
            s[i//cols,i%cols].set_title(patch_number)
        i+=1
    return

def show_positive_patches_image(predictions_dict, patient_folder):

    for image_view,i_classifications in predictions_dict.items():

        image_type = image_view.split('.')[0]
        patch_folder = patient_folder+'patches_'+image_type+'/'
        p = os.listdir(patch_folder)
        pa = []
        positive_patches = 0
        n = 0
    
        for pred in i_classifications:
            if pred > 0.95:
                positive_patches += 1
                pa.append(patch_folder+p[n])
            n+=1
        if positive_patches == 0:
            print('No patches classified as Suspicious in'+image_view)
            return
        if positive_patches < 4:
            cols = positive_patches
        else:
            cols = 4
        rows = np.ceil(len(pa)/cols).astype(np.int_)
        f,s = plt.subplots(rows,cols,figsize=(10,10))
        i=0
        for path in pa:
            patch = io.imread(path)
            patch_number = path.split('/')[3].split('.')[0]
            if rows == 1:
                s[i%cols].imshow(patch,cmap='gray')
                s[i%cols].set_title(image_type+patch_number)
            else:
                s[i//cols,i%cols].imshow(patch,cmap='gray')
                s[i%cols].set_title(image_type+patch_number)
            i+=1
    return

