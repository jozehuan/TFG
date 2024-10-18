from functools import partial

from flex.pool.decorators import set_explainer, get_explanations

import lime
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm 

import torch
from torch.utils.data import DataLoader

from skimage.color import gray2rgb, rgb2gray

@set_explainer
def set_LimeImageExplainer(*args, **kwargs):
    '''Define the LimeImageExplainer in the nodes, using the decorator @set_explainer

     Args:
    -----
        flex_model (FlexModel): object storing information needed to run a Pytorch model

        kernel_width (float): kernel width for the exponential kernel
        kernel (Callable): similarity kernel that takes euclidean distances and kernel width as input and outputs weights in (0,1). If None, defaults to an exponential kernel
        feature_selection (str): feature selection method. can be 'forward_selection', 'lasso_path', 'none' or 'auto'
        random_state (int): value used to generate random numbers. If None, the random state will be initialized using the internal numpy seed.

    '''

    k_w = kwargs.get("kernel_width", 0.25)
    k = kwargs.get("kernel", None)
    f_s = kwargs.get("feature_selection", 'auto')
    r_s = kwargs.get("random_state", None)

    explainer = lime_image.LimeImageExplainer(kernel_width = k_w, kernel = k,
                                                   feature_selection = f_s,
                                                   verbose = False, random_state = r_s)
    
    return explainer


def predict_(color_img, model):
    """Convert the image to grayscale and get the model's prediction

    Args:
    -----
        color_img (Array):  RGB image (the predictor is responsible for correctly formatting the image before making a prediction)
        model (nn.Module): cassification model
    """
    
    img_g = rgb2gray(color_img)
    img_g_tensor = torch.tensor(img_g, dtype=torch.float32)
    
    #plt.imshow(img_g_tensor[0], cmap='gray')
    #plt.title("dentro de _predict")
    #plt.show()
    
    with torch.no_grad():  # Desactivar el cálculo de gradientes para la predicción
        preds = model(img_g_tensor.to('cpu'))  # Enviar la imagen al mismo dispositivo que el modelo (CPU en este caso)

    return preds

@get_explanations
def get_LimeExplanations(flex_model, *args, **kwargs):
    '''Generate explanations for the specified data, according to the explainers defined by the specified model, using the decorator @get_explanations

     Args:
    -----
        flex_model (FlexModel): object storing information needed to run a Pytorch model
        data (flex.data.Dataset): objet storing the specified data to be explained

    Note:
    -----
        The argument 'data' should be provided through *args or **kwargs when calling the function.
    '''

    exp_output = {}
    images = []

    node_data = kwargs.get("data", None) # -- Añadir error de si no se introducen datos, que no se pueda realizar o que use los suyos
    dataset = node_data.to_torchvision_dataset()
    dataloader = DataLoader(dataset, batch_size=20)

    for imgs, _ in dataloader:
        imgs = imgs.to('cpu')
        images.extend(imgs.tolist())

    classifier = partial(predict_, model = flex_model["model"])
    segmenter = SegmentationAlgorithm('slic', n_segments=50, compactness=10, sigma=0.25)

    for exp_name, exp in flex_model["explainers"].items():
        explanations = []
        for data in images:
            explanation = exp.explain_instance(gray2rgb(data), classifier_fn = classifier,
                                               top_labels=10, hide_color=0, 
                                               num_samples=10000, segmentation_fn=segmenter)
            
            explanations.append([data, explanation])

        exp_output[exp_name] = explanations

    return exp_output
