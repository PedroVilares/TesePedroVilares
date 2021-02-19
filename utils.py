import os
import pandas as pd

def remove_csv(file_list):
    for file in file_list:
        if file[:2] == 'gt':
            file_list.remove(file)

def sort_paths(file_list):
    remove_csv(file_list)
    patient_numbers = []
    for patient in file_list:
        a = patient.split("_")
        patient_numbers.append(int(a[1]))
    patient_numbers.sort()
    patient_paths = list("patient_"+str(i) for i in patient_numbers)

    return patient_paths
