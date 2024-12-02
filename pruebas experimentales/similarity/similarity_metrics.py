import numpy as np
import re

from skimage.metrics import structural_similarity as ssim
from scipy.stats import rankdata


def assert_imgs_size(img1,img2):
    assert img1.shape == img2.shape, "The explanations must have the same size"

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

def normalize_0_1(img):
    """ Normalize an image to the range [0, 1].

    Parameters
    ----------
    img : numpy array [H,W]
        The input image.

    Returns
    -------
    numpy array
        The normalized image, with pixel values in the range [0, 1].
    """
    abs_max = np.max(np.abs(img))
    return (img + abs_max) / (2 * abs_max)

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
    assert_imgs_size(img1,img2)
    
    img1_f = img1.flatten()
    img2_f = img2.flatten()
    return 1 - np.dot(img1_f, img2_f) / (np.linalg.norm(img1_f)*np.linalg.norm(img2_f))

def IoU_metric(img1, img2, threshold):
    """ Calculate the Intersection over Union (IoU) metric between two images using a threshold.
    
    Parameters
    ----------
    img1 : numpy array [H,W]
        First input image.
    img2 : numpy array [H,W]
        Second input image.
    threshold : float
        The threshold value used to determine the significant pixels for computing the 
        intersection and union.

    Returns
    -------
    float
        The IoU score between the two images, where a value of 1 indicates perfect overlap
        and a value of 0 indicates no overlap.
    """
    assert_imgs_size(img1,img2)
    
    percent = threshold*100
    flat_img1= img1.flatten()
    flat_img2= img2.flatten()
    
    img1_above = img1 >= np.percentile(flat_img1, percent)
    img2_above = img2 >= np.percentile(flat_img2, percent)
    
    
    intersc_above = np.logical_and(img1_above, img2_above).sum()
    union_above = np.logical_or(img1_above, img2_above).sum()

    
    pos_iou = intersc_above / union_above if union_above != 0 else 0.0
    
    return pos_iou

def pcc(img1, img2):
    """ Calculate the Pearson Correlation Coefficient (PCC) metric between two images.
    
    Parameters
    ----------
    img1 : numpy array [H,W]
        First input image.
    img2 : numpy array [H,W]
        Second input image.

    Returns
    -------
    float
        The PCC score between the two images, where a value of 1 indicates perfect positive 
        correlation, -1 indicates perfect negative correlation, and 0 indicates no correlation.
    """
    assert_imgs_size(img1,img2)
    
    mean_1, mean_2 = np.mean(img1), np.mean(img2)
    std_1, std_2 = np.std(img1), np.std(img2)
    n = img1.size
    
    pcc = np.sum((img1-mean_1) * (img2-mean_2)) / (n*std_1*std_2)
    return pcc

def srcc(img1, img2):
    """ Calculate the Spearman Rank Correlation Coefficient (SRCC) metric between two images.

    Parameters
    ----------
    img1 : numpy array [H,W]
        First input image.
    img2 : numpy array [H,W]
        Second input image.
    
    Returns
    -------
    float
        The SRCC score between the two images, where a value of 1 indicates perfect positive 
        monotonic correlation, -1 indicates perfect negative monotonic correlation, and 0 indicates
        no monotonic correlation.
    """

    assert_imgs_size(img1,img2)
    
    img1_flat = img1.flatten()
    img2_flat = img2.flatten()
    n = len(img1_flat)
    
    rank1 = rankdata(img1_flat, method='max')
    rank2 = rankdata(img2_flat, method='max')
    d = rank1 - rank2
    
    srcc_value = 1 - (6*np.sum(d**2))/(n*(n**2-1))
    return srcc_value

def compute_similarity(fed_exps, central_exps):
    """ Compute the similarity metrics between federated and central explanations.

    This function iterates over the federated and central explanations, and computes some 
    similarity metrics for each pair of images. The metrics calculated include:
    - Pearson correlation Coefficient (PCC)
    - Spearman Rank Correlation Coefficient (SRCC)
    - Structural Similarity Index (SSIM)
    - Cosine Distance
    - Intersection over Union (IoU)

    The results are organized into a dictionary, where each key corresponds to an explainer,
    and the value is a list of dictionaries containing the calculated metrics for each pair of images.

    Parameters
    ----------
    fed_exps : tuple
        A tuple where the first element is a dictionary of federated explanations.
    central_exps : dict
        A dictionary of centralized system's explanations.

    Returns
    -------
    dict
        A dictionary containing similarity metrics (PCC, SRCC, SSIM, Cosine Distance, IoU) for each explainer.
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
            
            results['PCC'] = pcc(fed_img, central_img)
            
            results['SRCC'] = srcc(fed_img, central_img)
            
            ssim_result = ssim(fed_img_n, central_img_n, data_range=2)
            results['SSIM'] = ssim_result 
            
            cosine_result = cosine_dist(normalize_0_1(fed_img), normalize_0_1(central_img_n))
            results['COSINE'] = cosine_result
            
            pos_iou_result = IoU_metric(fed_img, central_img, 0.9)
            results['IoU'] = pos_iou_result
            
            results_imgs[exp_name].append(results)
            
    return results_imgs
            
            