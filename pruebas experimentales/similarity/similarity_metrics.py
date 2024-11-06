import numpy as np
import re

from skimage.metrics import structural_similarity as ssim

def normalize_1_1(img):
    """ Normalize an image to the range [-1, 1].

    Parameters
    ----------
    img : numpy array [H,W]
        The input image.

    Returns
    -------
    numpy array
        The normalized image, with pixel values in the range [-1, 1].
    """
    
    abs_max = np.max(np.abs(img))
    return 2*(img+abs_max) / (2*abs_max)-1
       

def cosine_dist(img1, img2):
    """ Calculate the cosine distance between two images.

    Parameters
    ----------
    img1 : numpy array [H,W]
        First input image.
    img2 : numpy array [H,W]
        Second input image.

    Returns
    -------
    float
        The cosine distance between the two images, where a value of 0 indicates identical images,
        and a value of 1 indicates completely dissimilar images.

    """
    img1_f = img1.flatten()
    img2_f = img2.flatten()
    return np.dot(img1_f, img2_f) / (np.linalg.norm(img1_f)*np.linalg.norm(img2_f))

def IoU_metric(img1, img2, threshold):
    """ Calculate the Intersection over Union (IoU) metric between two images using a symmetric threshold.
    
    Parameters
    ----------
    img1 : numpy array [H,W]
        First input image.
    img2 : numpy array [H,W]
        Second input image.
    threshold : float
        The threshold value used to determine the significant pixels for computing the intersection 
        and union.

    Returns
    -------
    float
        The combined IoU score between the two images, where a value of 1 indicates perfect overlap
        and a value of 0 indicates no overlap.
    """
    
    img1_above = img1 > threshold
    img1_below = img1 < -threshold
    img2_above = img2 > threshold
    img2_below = img2 < -threshold
    
    intersc_above = np.logical_and(img1_above, img2_above).sum()
    intersc_below = np.logical_and(img1_below, img2_below).sum()
    
    union_above = np.logical_or(img1_above, img2_above).sum()
    union_below = np.logical_or(img1_below, img2_below).sum()

    total_intersc = intersc_above + intersc_below
    total_union = union_above + union_below

    combined_iou = total_intersc / total_union if total_union != 0 else 0
    return combined_iou


def compute_similarity(fed_exps, central_exps):
    """ Compute the similarity metrics between federated and central explanations.

    This function iterates over the federated and central explanations, and computes some 
    similarity metrics for each pair of images. The metrics calculated include:
    - Mean Squared Error (MSE)
    - Structural Similarity Index (SSIM)
    - Cosine Distance
    - Intersection over Union (IoU)

    The results are organized into a dictionary, where each key corresponds to an explainer,
    and the value is a list of dictionaries containing the calculated metrics for each pair of images.

    The function performs the following steps:
    1. Normalize the images from the federated and central explanations to a range of [-1, 1].
    2. For each pair of images, calculate the MSE, SSIM, cosine distance, and IoU based on the normalized images.
    3. Store the results in a dictionary that groups them by explainer name.

    Parameters
    ----------
    fed_exps : tuple
        A tuple where the first element is a dictionary of federated explanations.
    central_exps : dict
        A dictionary of centralized system's explanations.

    Returns
    -------
    dict
        A dictionary containing similarity metrics (MSE, SSIM, Cosine Distance, IoU) for each explainer.
    """
    
    fed_exp_, _ = fed_exps
    
    results_imgs = {}
    for exp_name, fed_data, central_data in zip(fed_exp_.keys(), fed_exp_.values(), central_exps.values()):
        results_imgs[exp_name] = []
        
        for fed_d, central_d in zip(fed_data, central_data):
            results = {}
            _, img_str = fed_d[-1]
            
            label = re.search(r"label:\s*(\d+)", img_str)
            label = int(label.group(1))
            
            fed_img, _ = fed_d[label]
            central_img, _ = central_d[label]
            
            fed_img_n = normalize_1_1(fed_img)
            central_img_n = normalize_1_1(central_img)
            
            mse_result = np.mean((fed_img_n - central_img_n)**2)
            results['MSE'] = mse_result
            
            ssim_result = ssim(fed_img_n, central_img_n, data_range=2)
            results['SSIM'] = ssim_result
            
            cosine_result = cosine_dist(fed_img_n, central_img_n)
            results['COSINE'] = cosine_result
            
            iou_result = IoU_metric(fed_img_n, central_img_n, 0.5)
            results['IoU'] = iou_result
            
            results_imgs[exp_name].append(results)
            
    return results_imgs
            
            