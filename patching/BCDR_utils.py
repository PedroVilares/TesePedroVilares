import pandas as pd

def image_classifications(features_dataframe,images_dataframe):
    """
    Remove space from patient name
    """

    features_patient_id= features_dataframe['patient_id']
    features_classification= features_dataframe['classification']
    patient_id_classification_dict=dict()
    i=0
    for patient_id in features_patient_id:
        if patient_id in list(patient_id_classification_dict.keys()):
            i+=1
        else:
            patient_id_classification_dict[patient_id] = features_classification[i]
            i+=1
    
    images_classification = []
    images_patient_id= images_dataframe['patient_id']
    for patient_id in images_patient_id:
        images_classification.append(patient_id_classification_dict[patient_id])

    return images_classification

def fix_path(path_list):

    new_paths = []
    n=0
    for path in path_list:
        lost_path = path.split('/')
        if lost_path[1][0] == ' ':
            lost_path[1]= lost_path[1][1:]
        string_path= lost_path[0]
        for i in lost_path[1:len(lost_path)]:
            string_path = string_path+'/'+i
        new_paths.append(string_path)
        n += 1
        
    return new_paths

def dataframe_by_view(dataframe,view):
    """
    Builds dataframe with images taken from view.
    View must be 'CC' or 'O
    """
    image_paths = dataframe['paths']
    image_labels = dataframe['labels']
    view_paths = []
    view_labels = []
    i=0
    for path in image_paths:
        if view in path:
            view_paths.append(path)
            view_labels.append(image_labels[i])
        i+=1
    view_dataframe = pd.DataFrame({'paths':view_paths,'labels':view_labels})    

    return view_dataframe