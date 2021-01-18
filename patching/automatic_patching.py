import numpy as np
import imageio as io
import matplotlib.pyplot as plt
import random
from skimage.filters import threshold_yen
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
    mean = np.mean(raw_tif_array)
    std = np.std(raw_tif_array)
    filtered_tif_array = (raw_tif_array-mean)/std
    return filtered_tif_array

def binarize_breast_region(raw_mammogram_array):
    """Dado um array de uma mamografia, converte o array RGBA num array binário (1 - tecido mamário; 0 - backround) pelo método de Yen

    Args:
        raw_mammogram_array (ndarray): array RGBA

    Returns:
        ndarray: array binário
    """

    threshold = threshold_yen(raw_mammogram_array)
    binarized_array = (raw_mammogram_array > threshold).astype(np.int_)

    return binarized_array

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

    n_patches = int(np.floor(((width_max-width_min) * (height_max-height_min))/(patch_size*patch_size)))
    patch_vertexes = []
    for i in range(round(4*n_patches)): #Qual deve ser o range?
        x_center = random.randint(width_min+(patch_size/2),width_max-(patch_size/2))
        y_center = random.randint(height_min+(patch_size/2),height_max-(patch_size/2))
        if binarized_array[y_center,x_center] == 1:
            if binarized_array[int(y_center+(patch_size/2)),x_center] == 1 and binarized_array[int(y_center-(patch_size/2)),x_center]==1:
                if binarized_array[y_center,int(x_center+patch_size/2)] == 1 and binarized_array[y_center,int(x_center-patch_size/2)]==1:
                    patch_vertex = (int(y_center-(patch_size/2)),int(y_center+(patch_size/2)),int(x_center-patch_size/2),int(x_center+patch_size/2))
                    b_percentage = backround_calculator(binarized_array,patch_vertex)
                    if b_percentage > background_percentage:
                        patch_vertexes.append(patch_vertex)

    return patch_vertexes

def sistematic_patch_corners(binarized_array,patch_size,background_percentage):
    """[summary]

    Args:
        binarized_array ([type]): [description]
        patch_size ([type]): [description]
        background_percentage ([type]): [description]
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

def backround_calculator(binary_mammogram,patch_vertex):
    """Função calcula a percentagem de background num patch

    Args:
        binary_mammogram (ndarray): array de mamografia binária
        patch_vertex (lista): lista com vértices dos patches

    Returns:
        float: percentagem de background num patch
    """
    patch = binary_mammogram[patch_vertex[0]:patch_vertex[1],patch_vertex[2]:patch_vertex[3]]
    a = sum(sum(patch))
    b = patch.shape[0]*patch.shape[0]

    background_percentage = a/b

    return background_percentage

def show_centers(patch_vertexes,raw_mammogram_array):
    """Função que representa os centros dos patches numa mamografia 

    Args:
        patch_vertexes (list): lista com os vértices dos patches
        raw_mammogram_array (ndarray): array RGBA de uma mamografia

    Returns:
        ndarray: array RGBA de uma mamografia com o centro dos patches marcados
    """
    patch_side = patch_vertexes[0][1]-patch_vertexes[0][0] 
    half_side= int(patch_side/2)
    center_side = 20
    for vertex in patch_vertexes:
        raw_mammogram_array[vertex[0]+half_side-center_side:vertex[1]-half_side+center_side,vertex[2]+half_side-center_side:vertex[3]-half_side+center_side] = 0 
    return raw_mammogram_array    
    
def save_patches(path_list,patch_size,background_percentage,save_filename_folder,patching_type):

    n=0
    for path in path_list:
        raw_mammogram_array = raw_mammogram(path)
        binarized_array = binarize_breast_region(raw_mammogram_array)
        if patching_type == 'random':
            patches_vertexes = random_patch_corners(binarized_array,patch_size,background_percentage)
        elif patching_type == 'sistematic':
            patches_vertexes = sistematic_patch_corners(binarized_array,patch_size,background_percentage)
        else:
            print('Invalid patch type! Must be sistematic or random!')
            break
        for vertexes in patches_vertexes:
            patch = raw_mammogram_array[vertexes[0]:vertexes[1],vertexes[2]:vertexes[3]]
            filename = save_filename_folder+str(n)+'.jpeg'
            plt.imsave(filename,patch,cmap='gray')
            print('Patch',n,'saved!')
            n+=1

def nothing(binarized_array):
    height = binarized_array.shape[0]
    width = binarized_array.shape[1]
    width_max = 0
    width_min = 500
    height_max = 0
    height_min = 1000

    #Largura mínima e máxima do retângulo de operação
    for row in range(height):
        i=0
        n=0
        p=0
        for px in binarized_array[row,:]:
            if px > 0:
                i += 1
            else:
                if p < (height//2):
                    n+= 1
        if i > width_max:
            width_max = i
        if n < width_min:
            width_min = n
    if width_min < 500:
        width_max = width_max + width_min
    
    #Altura mínima e máxima do retângulo de operação
    for column in range(width):
        i=0
        n=0
        p=0
        for px in binarized_array[:,column]:
            p+=1
            if px > 0:
                i += 1
            else:
                if p < (height//2):
                    n +=1
        if i > height_max:
            height_max = i
        if n < height_min:
            height_min = n
    if height_min < 1000:
        height_max = height_max + height_min

    if width_min == 500:
        width_min = 0    
    print('Bounding box: ',height_min,width_min,height_max,width_max)
