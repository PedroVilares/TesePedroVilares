import numpy as np
import imageio as io
import random

def raw_mammogram(mammogram_path):
    """Dado um path para uma mamografia .tiff, a função abre a imagem e devolve-a como array

    Args:
        mammogram_path (string): path para mamografia tiff

    Returns:
        nd.array: imagem tiff convertida para array
    """
    raw_tif_image = io.imread(mammogram_path)
    raw_tif_array = np.asarray(raw_tif_image[:,:,1])
    return raw_tif_array

def binarize_breast_region(raw_mammogram_array):
    """Dado um array de uma mamografia, converte o array RGBA num array binário (1 - tecido mamário; 2 - backround)

    Args:
        raw_mammogram_array (nd.array): array RGBA

    Returns:
        nd.array: array binário
    """

    binarized_array = (raw_mammogram_array > 10).astype(np.int_)

    return binarized_array

def patch_corners(binarized_array,patch_size,background_percentage):
    """Função calcula o retângulo de operação da mamografia e centros possíveis dos patches. Dependendo do patch size e da background percentage,
    devolve os vértices dos patches

    Args:
        binarized_array (ndarray): array binário
        patch_size (int): largura de um patch
        background_percentage (float): percentagem de backgroung permitido num patch

    Returns:
        list: lista de tuplos, em que cada tuplo tem quatro vértices de um patch
    """
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
    #print(width_min,width_max,height_min,height_max)
    n_patches = int(np.floor(((width_max-width_min) * (height_max-height_min))/(patch_size*patch_size)))
    patch_vertexes = []
    for patch in range(round(2*n_patches)): #Qual deve ser o range?
        x_center = random.randint(width_min+patch_size,width_max-patch_size)
        y_center = random.randint(height_min+patch_size,height_max)
        if binarized_array[y_center,x_center] == 1:
            if binarized_array[int(y_center+(patch_size/2)),x_center] == 1 and binarized_array[int(y_center-(patch_size/2)),x_center]:
                if binarized_array[y_center,int(x_center+patch_size/2)] == 1 and binarized_array[y_center,int(x_center-patch_size/2)]:
                    patch_vertex = (int(y_center-(patch_size/2)),int(y_center+(patch_size/2)),int(x_center-patch_size/2),int(x_center+patch_size/2))
                    b_percentage = backround_calculator(binarized_array,patch_vertex)
                    if b_percentage > background_percentage:
                        patch_vertexes.append(patch_vertex)

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
    for vertex in patch_vertexes:
        raw_mammogram_array[vertex[0]+230:vertex[1]-230,vertex[2]+230:vertex[3]-230] = 1
    return raw_mammogram_array

def draw_patches(patch_coordinates,raw_image):
    
    initial_patch = raw_image

    return NotImplementedError



