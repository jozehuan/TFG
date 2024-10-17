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
    
    k_w = kwargs.get("kernel_width", 0.25)
    k = kwargs.get("kernel", None)
    f_s = kwargs.get("feature_selection", 'auto')
    r_s = kwargs.get("random_state", None)

    explainer = lime_image.LimeImageExplainer(kernel_width = k_w, kernel = k,
                                                   feature_selection = f_s,
                                                   verbose = False, random_state = r_s)
    
    return explainer


def predict_(color_img, model):
    """Predictor:
            color_img - imagen en formato '(28, 28, 3)'
            (el predictor se encarga de dar formato correcto a la imagen antes de predecir) 
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
def get_LimeExplanations(flex_model, node_data, *args, **kwargs):

    exp_output = []
    images = []

    dataset = node_data.to_torchvision_dataset()
    dataloader = DataLoader(dataset, batch_size=1024)

    for imgs, _ in dataloader:
        imgs = imgs.to('cpu')
    
        images.extend(imgs.tolist())

    classifier = partial(predict_, model = flex_model["model"])
    segmenter = SegmentationAlgorithm('slic', n_segments=50, compactness=10, sigma=0.25)

    for i, exp in enumerate(flex_model["explainers"]):
        explanations = []
        for data in images:
            explanation = exp.explain_instance(gray2rgb(data), classifier_fn = classifier,
                                               top_labels=10, hide_color=0, 
                                               num_samples=10000, segmentation_fn=segmenter)
            
            explanations.append(explanation)

        exp_output.append(explanations)

    return exp_output
