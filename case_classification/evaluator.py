from sklearn.metrics import roc_curve,roc_auc_score
import matplotlib.pyplot as plt
import pandas as pd

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

def report(cases_classifications,ground_truth):
    txt_file = open("Classification Report.txt","w")
    n=1
    for _,i_classifications in cases_classifications.items():
        if ground_truth[n-1] == 0:
            gt = 'Normal'
        else:
            gt = 'Suspicious'
        print('Patient {} | {}'.format(n,gt))
        txt_file.write('Patient {} | {} \n'.format(n,gt))
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
            txt_file.write(image_view +': Patches {} | Positive Patches {} | Ratio {:.2f} \n'.format(num_patches,positive_patches,ratio))
        print('Average Ratio: ',round(average_ratio(ratios),2))
        txt_file.write('Average Ratio: {} \n'.format(round(average_ratio(ratios),2)))
        print('\n')
        n+=1
    txt_file.close()

def average_ratio(ratio_list):
    return sum(ratio_list)/len(ratio_list)