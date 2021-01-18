import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import imageio as io

def patching_lesions(dataframe,tiff_paths):
    save_paths = []
    unsaved_patches = []
    for i in range(len(dataframe['image_filename'])):
        image_path = tiff_paths[i]
        full_mammo = np.asarray(io.imread(tiff_paths[i]))
        image_type = image_path.split('/')
        if 'R' in image_type[3]:
            full_mammo = np.fliplr(full_mammo)
        x_center = int(round(full_mammo.shape[1]*dataframe.loc[i,'s_x_center_mass']))
        y_center = int(round(full_mammo.shape[0]*dataframe.loc[i,'s_y_center_mass']))
        area_lesion = dataframe.loc[i,'s_area']
        side_lesion = int(round(np.sqrt(area_lesion)/2)+25)
        if side_lesion > x_center:
            x_center = side_lesion
        patch = full_mammo[y_center-side_lesion:y_center+side_lesion,x_center-side_lesion:x_center+side_lesion]
        patch = patch.copy(order='C')
        save_path = tiff_paths[i][:len(tiff_paths[i])-5]+'-p'+'.tiff'
        save_paths.append(save_path)
        plt.imsave(fname=save_path,arr=patch)
        print('Patch',i,'saved!')
        
    return save_paths,unsaved_patches

def patch_paths(dataframe,tiff_paths):
    save_paths = []
    for i in range(len(dataframe['image_filename'])):
        image_path = tiff_paths[i]
        save_path = image_path[:len(image_path)-5]+'-p'+'.tiff'
        save_paths.append(save_path)
        
    return save_paths