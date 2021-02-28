import os
from sklearn.metrics import roc_curve,roc_auc_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import imageio as io
import utils
import automatic_patching as patcher
from sklearn.metrics import confusion_matrix


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

def evaluate_predictions(transfer):

    if transfer:
        report = pd.read_csv('D:/Architecture/results tf/classification_report_by_case.csv')
    else:
        report = pd.read_csv('D:/Architecture/results f-s/classification_report_by_case.csv')

    ground_truth_categorical = list(report['Ground Truth'])
    ground_truth = []
    for i in ground_truth_categorical:
        if i == 'Normal':
            ground_truth.append(0)
        else:
            ground_truth.append(1)
    
    classifications = []
    classifications_categorical = list(report['Classification'])
    for i in classifications_categorical:
        if i == 'Normal':
            classifications.append(0)
        else:
            classifications.append(1)

    df = pd.DataFrame({'Ground Truth':ground_truth,'Predictions':classifications})
    auc = roc_auc_score(ground_truth,classifications)
    fpr,tpr,_= roc_curve(ground_truth,classifications)
    c_matrix = confusion_matrix(ground_truth,classifications,labels=[0,1])
    plt.plot(fpr,tpr)

    return auc, df, c_matrix

def report_by_case(threshold,transfer):

    if transfer:
        txt_file = open("results tf/Classification Report by case.txt","w")
    else:
        txt_file = open("results f-s/Classification Report by case.txt","w")
    n=1

    ground_truth_csv = pd.read_csv('patients/patient_gt.csv')
    density = list(ground_truth_csv['density'])

    patient_list = []
    patches_list = []
    sus_patches_list = []
    density_list = []
    gt_list = []

    patient_folder = "patients/"
    patient_paths = os.listdir(patient_folder)
    not_folders = []
    for image in patient_paths:
        if 'g' in image:
            not_folders.append(image)
    for image in not_folders:
        patient_paths.remove(image)
    patient_paths = utils.sort_paths(patient_paths)

    for patient in patient_paths:

        patient_df = ground_truth_csv[ground_truth_csv['patient'] == patient].reset_index()
        ground_truth = patient_df['label'][0]
        density = patient_df['density'][0]
        patch_dataframe = pd.read_csv('patients/'+patient+'/classification_data.csv')
        predictions = patch_dataframe['Classifications']
        num_patches = len(predictions)
        positive_patches = 0
        for pred in predictions:
            if pred > threshold:
                positive_patches += 1
        txt_file.write('Patches {} | Positive Patches {} \n'.format(num_patches,positive_patches))

        patient_list.append('Patient {}'.format(n))
        gt_list.append(ground_truth)
        density_list.append(density)
        patches_list.append(num_patches)
        sus_patches_list.append(positive_patches)

        n += 1

    master_df = pd.DataFrame({'Patient': patient_list,'Ground Truth': gt_list,'Density': density_list,'Patches': patches_list,'Suspicious': sus_patches_list})    
    if transfer:
        master_df.to_csv('D:/Architecture/results tf/classification_report_by_case.csv',index=False)
    else:
        master_df.to_csv('D:/Architecture/results f-s/classification_report_by_case.csv',index=False)
    txt_file.close()

def report_by_image(threshold,transfer):

    if transfer:
        txt_file = open("results tf/Classification Report by image.txt","w")
    else:
        txt_file = open("results f-s/Classification Report by image.txt","w")
    n=1

    ground_truth_csv = pd.read_csv('patients/patient_gt.csv')
    ground_truth = list(ground_truth_csv['label'])
    density = list(ground_truth_csv['density'])

    patient_list = []
    patches_list = []
    sus_patches_list = []
    image_views_list = []
    density_list = []
    gt_list = []

    patient_folder = "patients/"
    patient_paths = os.listdir(patient_folder)
    not_folders = []
    for image in patient_paths:
        if 'g' in image:
            not_folders.append(image)
    for image in not_folders:
        patient_paths.remove(image)
    patient_paths = utils.sort_paths(patient_paths)

    i= 0
    for patient in patient_paths:
        patch_dataframe = pd.read_csv('patients/'+patient+'/classification_data.csv')
        df_list = [df for df in patch_dataframe.groupby('Image View')]
        txt_file.write('\n Patient {} | {} | Density: {}\n'.format(n,ground_truth[n-1],density[n-1]))

        for df_tuple in df_list:
            image_view = df_tuple[0]
            predictions = df_tuple[1]['Classifications']
            num_patches = len(predictions)
            positive_patches = 0
            for pred in predictions:
                if pred > threshold:
                    positive_patches += 1
            txt_file.write(image_view +': Patches {} | Positive Patches {} \n'.format(num_patches,positive_patches))

            patient_list.append('Patient {}'.format(n))
            gt_list.append(ground_truth[i])
            density_list.append(density[i])
            patches_list.append(num_patches)
            sus_patches_list.append(positive_patches)
            image_views_list.append(image_view)
            i += 1
        n+=1

    master_df = pd.DataFrame({'Patient': patient_list,'Ground Truth': gt_list,'Density': density_list,'Image View':image_views_list,'Patches': patches_list,'Suspicious': sus_patches_list})    
    if transfer:
        master_df.to_csv('D:/Architecture/results tf/classification_report_by_image.csv',index=False)
    else:
        master_df.to_csv('D:/Architecture/results f-s/classification_report_by_image.csv',index=False)
    txt_file.close()

def average_ratio(ratio_list):
    return sum(ratio_list)/len(ratio_list)

def make_hist(patient_number):

    patient = 'patient_'+str(patient_number)
    patch_dataframe = pd.read_csv('patients/'+patient+'/classification_data.csv')
    predictions = patch_dataframe['Classifications']

    ground_truth_csv = pd.read_csv('patients/patient_gt.csv')
    patient_df = ground_truth_csv[ground_truth_csv['patient'] == patient].reset_index()
    gt = patient_df['label'][0]

    plt.hist(x=predictions,bins=10,alpha=0.7,rwidth=0.8)
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

def patient_report(patient_number,transfer):

    patient = 'Patient '+str(patient_number)

    if transfer:
        report_csv = pd.read_csv('D:/Architecture/results tf/classification_report_by_image.csv')
    else:
        report_csv = pd.read_csv('D:/Architecture/results f-s/classification_report_by_image.csv')
        
    patient_csv = report_csv[report_csv['Patient'] == patient]

    return patient_csv

def classify_images(auc,transfer):

    if transfer:
        report = pd.read_csv('D:/Architecture/results tf/classification_report_by_image.csv')
    else:
        report = pd.read_csv('D:/Architecture/results f-s/classification_report_by_image.csv')
    
    patient_folder = "patients/"
    patient_paths = os.listdir(patient_folder)
    not_folders = []
    for image in patient_paths:
        if 'g' in image:
            not_folders.append(image)
    for image in not_folders:
        patient_paths.remove(image)
    patient_paths = utils.sort_paths(patient_paths)

    classification = []
    for patient in patient_paths:
        number = patient.split('_')[1]
        patient = 'Patient '+str(number)
        patient_df = report[report['Patient'] == patient].reset_index()
        for i in patient_df.index:
            total_patches = patient_df['Patches'][i]
            sus_patches = patient_df['Suspicious'][i]
            allowed_sus = np.floor(total_patches*(1-auc))
            if sus_patches <= allowed_sus:
                classification.append('Normal')
            else:
                classification.append('Suspicious')

    report['Classification'] = classification

    if transfer:
        report.to_csv('D:/Architecture/results tf/classification_report_by_image.csv',index=False)
    else:
        report.to_csv('D:/Architecture/results f-s/classification_report_by_image.csv',index=False)

def classify_cases(transfer):

    if transfer:
        report = pd.read_csv('D:/Architecture/results tf/classification_report_by_image.csv')
    else:
        report = pd.read_csv('D:/Architecture/results f-s/classification_report_by_image.csv')
        
    patient_folder = "patients/"
    patient_paths = os.listdir(patient_folder)
    not_folders = []
    for image in patient_paths:
        if 'g' in image:
            not_folders.append(image)
    for image in not_folders:
        patient_paths.remove(image)
    patient_paths = utils.sort_paths(patient_paths)

    classifications = []
    for patient in patient_paths:
        number = patient.split('_')[1]
        patient = 'Patient '+str(number)
        patient_df = report[report['Patient'] == patient].reset_index()
        image_classifications = list(patient_df['Classification'])
        if len(image_classifications) <= 2:
            if 'Suspicious' in image_classifications:
                classifications.append('Suspicious')
            else:
                classifications.append('Normal')

        elif len(image_classifications) > 2 and len(image_classifications) <= 4:
            i=0
            for c in image_classifications:
                if c == 'Suspicious':
                    i+=1
            if i >= 2:
                classifications.append('Suspicious')
            if i < 2:
                classifications.append('Normal')

        elif len(image_classifications) > 4 and len(image_classifications) <= 6:
            i=0
            for c in image_classifications:
                if c == 'Suspicious':
                    i+=1
            if i >= 3:
                classifications.append('Suspicious')
            if i < 3:
                classifications.append('Normal')

        elif len(image_classifications) > 6 :
            i=0
            for c in image_classifications:
                if c == 'Suspicious':
                    i+=1
            if i >= 4:
                classifications.append('Suspicious')
            if i < 4:
                classifications.append('Normal')

    if transfer:
        case_report = pd.read_csv('D:/Architecture/results tf/classification_report_by_case.csv')
        case_report['Classification'] = classifications
        case_report.to_csv('D:/Architecture/results tf/classification_report_by_case.csv',index=False)
    else:
        case_report = pd.read_csv('D:/Architecture/results f-s/classification_report_by_case.csv')
        case_report['Classification'] = classifications
        case_report.to_csv('D:/Architecture/results tf/classification_report_by_case.csv',index=False)

def classify_cases_bypass(auc,transfer):
    if transfer:
        report = pd.read_csv('D:/Architecture/results tf/classification_report_by_case.csv')
    else:
        report = pd.read_csv('D:/Architecture/results f-s/classification_report_by_case.csv')
    patient_folder = "patients/"
    patient_paths = os.listdir(patient_folder)
    not_folders = []
    for image in patient_paths:
        if 'g' in image:
            not_folders.append(image)
    for image in not_folders:
        patient_paths.remove(image)
    patient_paths = utils.sort_paths(patient_paths)

    classifications = []
    for patient in patient_paths:
        number = patient.split('_')[1]
        patient = 'Patient '+str(number)
        patient_df = report[report['Patient'] == patient].reset_index()
        for i in patient_df.index:
            total_patches = patient_df['Patches'][i]
            sus_patches = patient_df['Suspicious'][i]
            allowed_sus = np.ceil(total_patches*(1-auc))
            if sus_patches <= allowed_sus:
                classifications.append('Normal')
            else:
                classifications.append('Suspicious')

    report['Classification'] = classifications
    report.to_csv('D:/Architecture/results tf/classification_report_by_case.csv',index=False)
    if transfer:
        report.to_csv('D:/Architecture/results tf/classification_report_by_case.csv',index=False)
    else:
        report.to_csv('D:/Architecture/results f-s/classification_report_by_case.csv',index=False)

def overlay_percentage(patch_size,transfer,threshold):

    side_patch = int(patch_size/2)

    patient_folder = "patients/"
    patient_paths = os.listdir(patient_folder)
    not_folders = []
    for image in patient_paths:
        if 'g' in image:
            not_folders.append(image)
    for image in not_folders:
        patient_paths.remove(image)
    patient_paths = utils.sort_paths(patient_paths)

    ground_truth_csv = pd.read_csv('patients/patient_gt.csv')
    overlays = []
    for patient in patient_paths:

        patient_df = ground_truth_csv[ground_truth_csv['patient'] == patient]
        patient_label = list(patient_df['label'])[0]

        if patient_label == 'Suspicious':

            patch_dataframe = pd.read_csv('patients/'+patient+'/classification_data.csv')

            patient_folder = 'patients/'+patient+'/'
            image_paths = os.listdir(patient_folder)

            not_images = []
            for image in image_paths:
                if 't' in image:
                    not_images.append(image)
            for image in not_images:
                image_paths.remove(image)

            for image in image_paths:
                overlay_img = []
                image_view = image.split('.')[0]
                path = patient_folder+image
                mammogram = patcher.raw_mammogram(path)
                df = patient_df[patient_df['image view'] == image_view].reset_index()

                xGT = int(round(mammogram.shape[1]*df['x_center'][0]))
                yGT = int(round(mammogram.shape[0]*df['y_center'][0]))
            
                patch_df = patch_dataframe[patch_dataframe['Image View'] == image_view].reset_index()
                
                sus_vertexes = []
                for i in patch_df.index:
                    pred = patch_df['Classifications'][i]
                    if pred > threshold:
                        v = patch_df['Patches Vertexes'][i]
                        patch_vertexes = tuple([int(i) for i in v[1:-1].split(',')])
                        sus_vertexes.append(patch_vertexes)
                
                if len(sus_vertexes) == 0:
                    overlay_img.append(0)

                for v in sus_vertexes:
                    xC = int(v[2]+side_patch)
                    yC = int(v[0]+side_patch)

                    if yC < yGT and xC < xGT:
                        if (yC+side_patch) > (yGT-side_patch) and (xC+side_patch) > (xGT-side_patch):
                            a = (yC+side_patch)-(yGT-side_patch)
                            b = (xC+side_patch)-(xGT-side_patch)
                            #percentage = (a*b)/(patch_size*patch_size)
                            #overlay_img.append(percentage)
                            overlay_img.append(1)
                        else:
                            overlay_img.append(0)

                    elif yC < yGT and xC > xGT:
                        if (yC+side_patch) > (yGT-side_patch) and (xC-side_patch) < (xGT+side_patch):
                            a = (yC+side_patch)-(yGT-side_patch)
                            b = (xGT+side_patch)-(xC-side_patch)
                            #percentage = (a*b)/(patch_size*patch_size)
                            #overlay_img.append(percentage)
                            overlay_img.append(1)
                        else:
                            overlay_img.append(0)

                    elif yC > yGT and xC > xGT:
                        if (yC-side_patch) < (yGT+side_patch) and (xC-side_patch) < (xGT+side_patch):
                            a = (yGT+side_patch)-(yC-side_patch)
                            b = (xGT+side_patch)-(xC-side_patch)
                            #percentage = (a*b)/(patch_size*patch_size)
                            #overlay_img.append(percentage)
                            overlay_img.append(1)
                        else:
                            overlay_img.append(0)

                    elif yC > yGT and xC < xGT:
                        if (yC-side_patch) < (yGT+side_patch) and (xC+side_patch) > (xGT-side_patch):
                            a = (yGT+side_patch)-(yC-side_patch)
                            b = (xC+side_patch)-(xGT-side_patch)
                            #percentage = (a*b)/(patch_size*patch_size)
                            #overlay_img.append(percentage)
                            overlay_img.append(1)
                        else:
                            overlay_img.append(0)

                    else:
                        overlay_img.append(0)

                    #overlay_img.append(percentage)

                overlays.append(float("{:.2f}".format(sum(overlay_img)/len(overlay_img))))
        else:
            for i in patient_df.index:
                overlays.append(' ')
    
    if transfer:
        report = pd.read_csv('D:/Architecture/results tf/classification_report_by_image.csv')
        report['Overlay'] = overlays
        report.to_csv('D:/Architecture/results tf/classification_report_by_image.csv',index=False)

    else:
        report = pd.read_csv('D:/Architecture/results f-s/classification_report_by_image.csv')
        report['Overlay'] = overlays
        report.to_csv('D:/Architecture/results f-s/classification_report_by_image.csv',index=False)