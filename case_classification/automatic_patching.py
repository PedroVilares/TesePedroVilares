import numpy as np
import imageio as io
import matplotlib.pyplot as plt
import os
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
    binarized_array = (raw_mammogram_array > threshold).astype(np.int_)
    binarized_filled= scipy.ndimage.binary_fill_holes(binarized_array).astype(int)

    return binarized_filled

def sistematic_patch_corners(binarized_array,patch_size,background_percentage):
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

    overlapping_percentage = 0.2
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
    
def save_patches(image_path):
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
    patches_vertexes = sistematic_patch_corners(binarized_array,350,0.95)
    for vertexes in patches_vertexes:
        patch = raw_mammogram_array[vertexes[0]:vertexes[1],vertexes[2]:vertexes[3]]
        filename = folder+'/'+str(n)+'.bmp'
        plt.imsave(filename,patch,cmap='gray')
        #print('Patch',n,'saved!')
        save_filenames.append(filename)
        n+=1
    print('Saved',n,'patches from '+image_type_1[0]+' image!')  

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

def preprocessing_mammography(image_array):
    mean = np.mean(image_array)
    std = np.std(image_array)
    preprocessed = (image_array - mean)/std
    return preprocessed

