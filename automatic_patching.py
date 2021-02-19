import numpy as np
import imageio as io
import matplotlib.pyplot as plt
import os
import random
from skimage.filters import threshold_yen,threshold_otsu
import scipy
from skimage.measure import regionprops

def raw_mammogram(mammogram_path):
    """Dado um path para uma mamografia .tiff, a função abre a imagem e devolve-a como array

    Args:
        mammogram_path (string): path para mamografia tiff

    Returns:
        nd.array: imagem tiff convertida para array
    """
    raw_tif_image = io.imread(mammogram_path)
    raw_tif_array = np.asarray(raw_tif_image[:,:,1])
    filtered_tif_array = preprocessing_mammography(raw_tif_array)
    return filtered_tif_array

def binarize_breast_region(raw_mammogram_array):
    """Dado um array de uma mamografia, converte o array RGBA num array binário (1 - tecido mamário; 0 - backround) pelo método de Yen

    Args:
        raw_mammogram_array (ndarray): array RGBA

    Returns:
        ndarray: array binário
    """

    threshold = threshold_otsu(raw_mammogram_array)
    #threshold = threshold_yen(raw_mammogram_array)
    binarized_array = (raw_mammogram_array > threshold).astype(np.int_)
    binarized_filled= scipy.ndimage.binary_fill_holes(binarized_array).astype(int)

    return binarized_filled

def sistematic_patch_corners(binarized_array,patch_size,background_percentage,overlapping):
    """Função calcula o retângulo de operação da mamografia e centros possíveis dos patches de forma sistemática. Dependendo do patch size e da background percentage,
    devolve os vértices dos patches

    Args:
        binarized_array (ndarray): mamografia binarizada
        patch_size (int): dimensão do patch
        background_percentage (float): percentagem mínima de foreground permitida
    """
    a=regionprops(binarized_array)
    bounding_box = a[0].bbox
    height_min = bounding_box[0]
    width_min = bounding_box[1]
    height_max = bounding_box[2]
    width_max= bounding_box[3]

    #identificar as MLO
    if height_min < 250:
        muscle = binarized_array[0:250,:]
        a=regionprops(muscle)
        bounding_box = a[0].bbox
        if width_max > binarized_array.shape[1]-100:
            #Right
            if bounding_box[3] - bounding_box[1] > 150:
                height_min = 350
                width_max = width_max - 100
        else:
            #Left
            if bounding_box[3] > 150:
                height_min = 350
                width_min = width_min + 70

    overlapping_percentage = overlapping
    overlap = overlapping_percentage*patch_size
    window_size1=height_min
    window_size2=height_min+patch_size
    window_size3=width_min
    window_size4=width_min+patch_size
    n_patches_h = int(np.floor((width_max-width_min))/(patch_size-overlap))
    n_patches_v = int(np.floor((height_max-height_min))/(patch_size-overlap))
    patch_vertexes = []
    for i in range(n_patches_v):
        for n in range(n_patches_h):
            patch_vertex = (int(window_size1),int(window_size2),int(window_size3),int(window_size4))
            b_percentage = backround_calculator(binarized_array,patch_vertex)
            if b_percentage > background_percentage:
                patch_vertexes.append(patch_vertex)
            window_size3 = window_size3 + (patch_size-overlap)
            window_size4 = window_size4 + (patch_size-overlap)
            n+=1
        window_size3 = width_min
        window_size4 = width_min+patch_size
        window_size1 = window_size1 + (patch_size-overlap)
        window_size2 = window_size2 + (patch_size-overlap)
        i+=1

    return patch_vertexes

def random_patch_corners(binarized_array,patch_size,background_percentage):
    """Função calcula o retângulo de operação da mamografia e centros possíveis dos patches. Dependendo do patch size e da background percentage,
    devolve os vértices dos patches

    Args:
        binarized_array (ndarray): array binário
        patch_size (int): largura de um patch
        background_percentage (float): percentagem de backgroung permitido num patch

    Returns:
        list: lista de tuplos, em que cada tuplo tem quatro vértices de um patch
    """
        
    a=regionprops(binarized_array)
    bounding_box = a[0].bbox
    height_min = bounding_box[0]
    width_min = bounding_box[1]
    height_max = bounding_box[2]
    width_max= bounding_box[3]

    #identificar as MLO
    if height_min < 250:
        muscle = binarized_array[0:250,:]
        a=regionprops(muscle)
        bounding_box = a[0].bbox
        if width_max > binarized_array.shape[1]-100:
            #Right
            if bounding_box[3] - bounding_box[1] > 150:
                height_min = 350
                width_max = width_max - 100
        else:
            #Left
            if bounding_box[3] > 150:
                height_min = 350
                width_min = width_min + 70

    n_patches = int(np.floor(((width_max-width_min) * (height_max-height_min))/(patch_size*patch_size)))
    patch_vertexes = []
    for i in range(round(4*n_patches)): #Qual deve ser o range?
        x_center = random.randint(width_min+(patch_size/2)+50,width_max-(patch_size/2)-50)
        y_center = random.randint(height_min+(patch_size/2)+50,height_max-(patch_size/2)-50)
        if binarized_array[y_center,x_center] == 1:
            if binarized_array[int(y_center+(patch_size/2)),x_center] == 1 and binarized_array[int(y_center-(patch_size/2)),x_center]==1:
                if binarized_array[y_center,int(x_center+patch_size/2)] == 1 and binarized_array[y_center,int(x_center-patch_size/2)]==1:
                    patch_vertex = (int(y_center-(patch_size/2)),int(y_center+(patch_size/2)),int(x_center-patch_size/2),int(x_center+patch_size/2))
                    b_percentage = backround_calculator(binarized_array,patch_vertex)
                    if b_percentage > background_percentage:
                        patch_vertexes.append(patch_vertex)

    return patch_vertexes

def backround_calculator(binary_mammogram,patch_vertexes):
    """Função calcula a percentagem de background num patch

    Args:
        binary_mammogram (ndarray): array de mamografia binária
        patch_vertex (lista): lista com vértices dos patches

    Returns:
        float: percentagem de background num patch
    """
    patch = binary_mammogram[patch_vertexes[0]:patch_vertexes[1],patch_vertexes[2]:patch_vertexes[3]]
    a = sum(sum(patch))
    b = patch.shape[0]*patch.shape[0]

    background_percentage = a/b

    return background_percentage  
    
def save_patches_by_image(image_path,patch_size,overlap):
    n=0
    save_filenames = []
    image_type = image_path.split('/')
    image_type_1 = image_type[2].split('.')
    image_view = image_type_1[0]
    try:
        folder = image_type[0]+'/'+image_type[1]+'/'+'patches_'+image_view
        os.mkdir(folder)
        print('Successfully created '+'patches_'+image_view+' folder!')
    except OSError:
        print("Patch folder already exists!")

    raw_mammogram_array = raw_mammogram(image_path)
    
    binarized_array = binarize_breast_region(raw_mammogram_array)
    patches_vertexes_1 = sistematic_patch_corners(binarized_array,patch_size,0.97,overlap)
    patches_vertexes_2 = random_patch_corners(binarized_array,patch_size,0.97)
    patches_vertexes = patches_vertexes_1 + patches_vertexes_2

    for vertexes in patches_vertexes:
        patch = raw_mammogram_array[vertexes[0]:vertexes[1],vertexes[2]:vertexes[3]]
        filename = folder+'/'+str(n)+'.bmp'
        plt.imsave(filename,patch,cmap='gray')
        #print('Patch',n,'saved!')
        save_filenames.append(filename)
        n+=1
    print('Saved',n,'patches from '+image_type_1[0]+' image!')  

    return save_filenames

def save_patches_by_case(patient_path,patch_size,overlap):

    patient_path_list = patient_path.split('/')
    patient_name = patient_path_list[1]
    save_filenames = []
    try:
        folder = patient_path+'/'+'patches/'
        os.mkdir(folder)
        print('Successfully created patches folder!')
    except OSError:
        print("Patch folder already exists!")

    image_list = os.listdir(patient_path)
    not_images = []
    for image in image_list:
        if 't' in image:
            not_images.append(image)
    for image in not_images:
        image_list.remove(image)
    n=0
    for image in image_list:

        raw_mammogram_array = raw_mammogram(patient_path+image)
        binarized_array = binarize_breast_region(raw_mammogram_array)
        patches_vertexes_1 = sistematic_patch_corners(binarized_array,patch_size,0.97,overlap)
        patches_vertexes_2 = random_patch_corners(binarized_array,patch_size,0.97)
        patches_vertexes = patches_vertexes_1 + patches_vertexes_2

        for vertexes in patches_vertexes:
            patch = raw_mammogram_array[vertexes[0]:vertexes[1],vertexes[2]:vertexes[3]]
            filename = folder+'/'+str(n)+'.bmp'
            plt.imsave(filename,patch,cmap='gray')
            save_filenames.append(filename)
            n+=1
    print('Saved',n,'patches from '+str(patient_name))  

    return save_filenames    

def only_paths(image_path):

    save_filenames = []
    image_type = image_path.split('/')
    image_type_1 = image_type[2].split('.')
    image_view = image_type_1[0]
    folder = image_type[0]+'/'+image_type[1]+'/'+'patches_'+image_view
    images = os.listdir(folder)
    for image in images:            
        filename = folder+'/'+image
        save_filenames.append(filename)

    return save_filenames

def show_binarized(patient_folder):

    image_paths = os.listdir(patient_folder)
    not_images = []
    for image in image_paths:
        if 't' in image:
            not_images.append(image)
    for image in not_images:
        image_paths.remove(image)

    if len(image_paths) < 4:
        cols = 2
    else:
        cols = 4
    rows = np.ceil(len(image_paths)/cols).astype(np.int_)
    f,s = plt.subplots(rows,cols,figsize=(20,20))
    i=0
    for image in image_paths:
        image_view = image.split('.')[0]
        path = patient_folder+image
        mammogram = raw_mammogram(path)
        binarized_array = binarize_breast_region(mammogram)
        if rows == 1:
            s[i%cols].imshow(binarized_array,cmap='gray')
            s[i%cols].set_title(image_view)
        else:
            s[i//cols,i%cols].imshow(binarized_array,cmap='gray')
            s[i//cols,i%cols].set_title(image_view)
        i+=1
    return

def show_mammograms(patient_folder):

    image_paths = os.listdir(patient_folder)
    not_images = []
    for image in image_paths:
        if 't' in image:
            not_images.append(image)
    for image in not_images:
        image_paths.remove(image)

    if len(image_paths) < 4:
        cols = 2
    else:
        cols = 4
    rows = np.ceil(len(image_paths)/cols).astype(np.int_)
    f,s = plt.subplots(rows,cols,figsize=(20,20))
    i=0
    for image in image_paths:
        image_view = image.split('.')[0]
        path = patient_folder+image
        mammogram = raw_mammogram(path)
        if rows == 1:
            s[i%cols].imshow(mammogram,cmap='gray')
            s[i%cols].set_title(image_view)
        else:
            s[i//cols,i%cols].imshow(mammogram,cmap='gray')
            s[i//cols,i%cols].set_title(image_view)
        i+=1
    return

def patches_number(patient_folder,patch_size,overlap):

    image_paths = os.listdir(patient_folder)
    not_images = []
    for image in image_paths:
        if 't' in image:
            not_images.append(image)
    for image in not_images:
        image_paths.remove(image)

    p_number = []
    for image in image_paths:
        image_view = image.split('.')[0]
        path = patient_folder+image
        mammogram = raw_mammogram(path)
        binarized_array = binarize_breast_region(mammogram)
        patches_vertexes_1 = sistematic_patch_corners(binarized_array,patch_size,0.95,overlap)
        patches_vertexes_2 = random_patch_corners(binarized_array,patch_size,0.95)
        patches_vertexes = patches_vertexes_1 + patches_vertexes_2
        p_number.append(len(patches_vertexes))
        print('Patches extracted from '+image_view+':',len(patches_vertexes))

    print('Total:',sum(p_number))
    return

def show_sides(patient_folder,patch_size,overlap):
    """Função que representa os centros dos patches numa mamografia 

    Args:
        patch_vertexes (list): lista com os vértices dos patches
        raw_mammogram_array (ndarray): array RGBA de uma mamografia

    Returns:
        ndarray: array RGBA de uma mamografia com o centro dos patches marcados
    """

    image_paths = os.listdir(patient_folder)
    not_images = []
    for image in image_paths:
        if 't' in image:
            not_images.append(image)
    for image in not_images:
        image_paths.remove(image)

    if len(image_paths) < 4:
        cols = 2
    else:
        cols = 4
    rows = np.ceil(len(image_paths)/cols).astype(np.int_)
    f,s = plt.subplots(rows,cols,figsize=(20,20))
    i=0
    for image in image_paths:
        image_view = image.split('.')[0]
        path = patient_folder+image
        mammogram = raw_mammogram(path)
        binarized_array = binarize_breast_region(mammogram)
        patches_vertexes_1 = sistematic_patch_corners(binarized_array,patch_size,0.95,overlap)
        patches_vertexes_2 = random_patch_corners(binarized_array,patch_size,0.95)
        patches_vertexes = patches_vertexes_1 + patches_vertexes_2
        line_side = 15
        for vertex in patches_vertexes:
            mammogram[vertex[0]:vertex[1],vertex[3]-line_side:vertex[3]+line_side] = 0 
            mammogram[vertex[0]-line_side:vertex[0]+line_side,vertex[2]:vertex[3]] = 0 
            mammogram[vertex[0]:vertex[1],vertex[2]-line_side:vertex[2]+line_side] = 0 
            mammogram[vertex[1]-line_side:vertex[1]+line_side,vertex[2]:vertex[3]] = 0

        if rows == 1:
            s[i%cols].imshow(mammogram,cmap='gray')
            s[i%cols].set_title(image_view)
        else:
            s[i//cols,i%cols].imshow(mammogram,cmap='gray')
            s[i//cols,i%cols].set_title(image_view)
        i+=1

    return

def show_centers(patient_folder,patch_size,overlap):
    """Função que representa os centros dos patches numa mamografia 

    Args:
        patch_vertexes (list): lista com os vértices dos patches
        raw_mammogram_array (ndarray): array RGBA de uma mamografia

    Returns:
        ndarray: array RGBA de uma mamografia com o centro dos patches marcados
    """
    image_paths = os.listdir(patient_folder)
    not_images = []
    for image in image_paths:
        if 't' in image:
            not_images.append(image)
    for image in not_images:
        image_paths.remove(image)

    if len(image_paths) < 4:
        cols = 2
    else:
        cols = 4
    rows = np.ceil(len(image_paths)/cols).astype(np.int_)
    f,s = plt.subplots(rows,cols,figsize=(20,20))
    i=0
    half_side = int(patch_size/2)
    for image in image_paths:
        image_view = image.split('.')[0]
        path = patient_folder+image
        mammogram = raw_mammogram(path)
        binarized_array = binarize_breast_region(mammogram)
        patches_vertexes_1 = sistematic_patch_corners(binarized_array,patch_size,0.95,overlap)
        patches_vertexes_2 = random_patch_corners(binarized_array,patch_size,0.95)
        patches_vertexes = patches_vertexes_1 + patches_vertexes_2
        n = 25
        center_side = int(half_side-n)
        for vertex in patches_vertexes:
            mammogram[vertex[0]+center_side:vertex[1]-center_side,vertex[2]+center_side:vertex[3]-center_side] = 0
        
        if rows == 1:
            s[i%cols].imshow(mammogram,cmap='gray')
            s[i%cols].set_title(image_view)
        else:
            s[i//cols,i%cols].imshow(mammogram,cmap='gray')
            s[i//cols,i%cols].set_title(image_view)
        i+=1

    return   

def show_positive_patches_on_mammography(patient_folder,patch_size,overlap,predictions,patient_df):
    image_paths = os.listdir(patient_folder)
    not_images = []
    for image in image_paths:
        if 't' in image:
            not_images.append(image)
    for image in not_images:
        image_paths.remove(image)
    
    positive_patches = []
    n=0
    for pred in predictions:
        if pred > 0.95:
            positive_patches.append(n)
        n+=1
    if positive_patches == 0:
        print('No patches classified as Suspicious')
        return
    
    if len(image_paths) < 4:
        cols = 2
    else:
        cols = 4
    rows = np.ceil(len(image_paths)/cols).astype(np.int_)
    f,s = plt.subplots(rows,cols,figsize=(20,20))
    vertexes = dict()
    index = 0
    patient_label = list(patient_df['label'])[0]
    for image in image_paths:
        t=[]
        r_ns = []
        image_view = image.split('.')[0]
        path = patient_folder+image
        mammogram = raw_mammogram(path)
        binarized_array = binarize_breast_region(mammogram)
        patches_vertexes_1 = sistematic_patch_corners(binarized_array,patch_size,0.95,overlap)
        patches_vertexes_2 = random_patch_corners(binarized_array,patch_size,0.95)
        patches_vertexes = patches_vertexes_1 + patches_vertexes_2
        line_side = 15
        a = len(patches_vertexes)
        for n in positive_patches:
            ni= n - index
            if a > ni: 
                r_ns.append(n)
                t.append(patches_vertexes[ni])
        vertexes[image] = t        
        for n in r_ns:
            positive_patches.remove(n)
        index += len(patches_vertexes)    
        
    i=0
    for image in image_paths:
        r_ns = []
        image_view = image.split('.')[0]
        path = patient_folder+image
        mammogram = raw_mammogram(path)
        binarized_array = binarize_breast_region(mammogram)
        patches_vertexes = sistematic_patch_corners(binarized_array,patch_size,0.90,overlap)
        if patient_label == 'Suspicious':
            df_line = patient_df.loc[i,:]
            x_center = int(round(mammogram.shape[0]*df_line['x_center']))
            y_center = int(round(mammogram.shape[1]*df_line['y_center']))
            line_side = 15
            half_side = int(patch_size/2)
            mammogram[y_center-half_side:y_center+half_side,x_center-half_side-line_side:x_center-half_side+line_side] = 1
            mammogram[y_center+half_side-line_side:y_center+half_side+line_side,x_center-half_side:x_center+half_side] = 1
            mammogram[y_center-half_side:y_center+half_side,x_center+half_side-line_side:x_center+half_side+line_side] = 1
            mammogram[y_center-half_side-line_side:y_center-half_side+line_side,x_center-half_side:x_center+half_side] = 1
        line_side = 15
        for im,v in vertexes.items():
            if im == image:
                for vertex in v:
                    mammogram[vertex[0]:vertex[1],vertex[3]-line_side:vertex[3]+line_side] = 0 
                    mammogram[vertex[0]-line_side:vertex[0]+line_side,vertex[2]:vertex[3]] = 0 
                    mammogram[vertex[0]:vertex[1],vertex[2]-line_side:vertex[2]+line_side] = 0 
                    mammogram[vertex[1]-line_side:vertex[1]+line_side,vertex[2]:vertex[3]] = 0
        if rows == 1:
            s[i%cols].imshow(mammogram,cmap='gray')
            s[i%cols].set_title(image_view)
        else:
            s[i//cols,i%cols].imshow(mammogram,cmap='gray')
            s[i//cols,i%cols].set_title(image_view)
        i+=1

        del vertexes[image]


    
    return

def preprocessing_mammography(image_array):
    mean = np.mean(image_array)
    std = np.std(image_array)
    preprocessed = (image_array - mean)/std
    return preprocessed

