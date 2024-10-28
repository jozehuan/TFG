from functools import partial

from flex.pool.decorators import set_explainer, get_explanations, get_SP_explanation
from flex.pool.xai import Image_SubmodularPick

import lime
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm 

from tqdm import tqdm # barra de progreso

from flex.pool.explanation import Explanation

import torch
from torch.utils.data import DataLoader

from skimage.color import gray2rgb, rgb2gray

LIME_SEGMENTATION_ALGORITHMS = ['quickshift', 'slic', 'felzenszwalb']

def ERROR_MSG_SEG_ALG_NOT_FOUND_GENERATOR(a):
    return f"Unknown {a} segmentation algorithm. Valid options are: {LIME_SEGMENTATION_ALGORITHMS}"

@set_explainer
def set_LimeImageExplainer(flex_model, *args, **kwargs):
    '''Define the LimeImageExplainer in the nodes, using the decorator @set_explainer

     Args:
    -----
        flex_model (FlexModel): object storing information needed to run a Pytorch model

        kernel_width (float, optional): kernel width for the exponential kernel
        kernel (Callable, optional): similarity kernel that takes euclidean distances and kernel width as input and outputs weights in (0,1). If None, defaults to an exponential kernel
        feature_selection (str, optional): feature selection method. can be 'forward_selection', 'lasso_path', 'none' or 'auto'
        random_state (int, optional): value used to generate random numbers. If None, the random state will be initialized using the internal numpy seed.

    '''

    k_w = kwargs.get("kernel_width", 0.25)
    k = kwargs.get("kernel", None)
    f_s = kwargs.get("feature_selection", 'auto')
    r_s = kwargs.get("random_state", None)

    # explainer instance
    explainer = lime_image.LimeImageExplainer(kernel_width = k_w, kernel = k,
                                                   feature_selection = f_s,
                                                   verbose = False, random_state = r_s)
    
    dict_result = {'explainer' : explainer} # output dict
    explain_instance_kwargs = {} 

    # explain instance params
    if (top_labels := kwargs.get('top_labels')) is not None: explain_instance_kwargs['top_labels'] = top_labels
    if (num_features := kwargs.get('num_features')) is not None: explain_instance_kwargs['num_features'] = num_features
    if (num_samples := kwargs.get('num_samples')) is not None: explain_instance_kwargs['num_samples'] = num_samples
    if (batch_size := kwargs.get('batch_size')) is not None: explain_instance_kwargs['batch_size'] = batch_size
    if (distance_metric := kwargs.get('distance_metric')) is not None: explain_instance_kwargs['distance_metric'] = distance_metric
    if (model_regressor := kwargs.get('model_regressor')) is not None: explain_instance_kwargs['model_regressor'] = model_regressor
    if (random_seed := kwargs.get('random_seed')) is not None: explain_instance_kwargs['random_seed'] = random_seed

    # if the segmentation algorithm is provided, add it to the params; if not, generate one
    if (segmentation_fn := kwargs.get('segmentation_fn')) is not None: explain_instance_kwargs['segmentation_fn'] = segmentation_fn
    else:
        if (algo_type := kwargs.get('algo_type')) is not None:
            if algo_type not in LIME_SEGMENTATION_ALGORITHMS:
                raise ValueError(f"Unknown segmentation algorithm: {algo_type}. Valid options are: {LIME_SEGMENTATION_ALGORITHMS}")
            assert algo_type in LIME_SEGMENTATION_ALGORITHMS, ERROR_MSG_SEG_ALG_NOT_FOUND_GENERATOR(algo_type) 

            segment_params = kwargs.get("segment_params", {})
            segmenter = SegmentationAlgorithm(algo_type, **segment_params)
            explain_instance_kwargs['segmentation_fn'] = segmenter
    
    dict_result['explain_instance_kwargs'] = explain_instance_kwargs
    return dict_result  # OUTPUT: {explainer : explainer , explain_instance_kwargs : {**kwargs}}


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
def get_LimeExplanations(flex_model, node_data, *args, **kwargs):
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

    data = kwargs.get("data", None) 
    data_ = data if data is not None else node_data
    dataset = data_.to_torchvision_dataset()
    dataloader = DataLoader(dataset, batch_size=20)

    for imgs, _ in dataloader:
        imgs = imgs.to('cpu')
        images.extend(imgs.tolist())

    classifier = partial(predict_, model = flex_model['model'])
    
    for exp_name, exp in flex_model['explainers'].items():
        cl_name = exp['explainer'].__class__.__name__
        if cl_name == 'LimeImageExplainer':
            explanations = []
            for data in tqdm(images, desc="Getting LIME explanations: ", mininterval=2):
                #explanation = exp['explainer'].explain_instance(gray2rgb(data), classifier_fn = classifier, **exp['explain_instance_kwargs'])
                explanation = Explanation(model = flex_model['model'], exp = exp['explainer'], data_to_explain = torch.tensor(data).unsqueeze(0),  **exp['explain_instance_kwargs'])
                explanations.append(explanation)
            exp_output[exp_name] = explanations

    return exp_output


@get_SP_explanation
def get_SP_LimeImageExplanation(flex_model, node_data, *args, **kwargs):
    
    exp_names = kwargs.get("exp_names", None)
    if exp_names is None: exp_names = list(flex_model['explanations'].keys())

    # if exp_names is a single value, convert it into a list
    if isinstance(exp_names, str) or not isinstance(exp_names, (list, tuple)):
        exp_names = [exp_names]

    classifier = partial(predict_, model = flex_model['model'])
    result_dict = {}

    for name in exp_names:
        exp = flex_model['explainers'][name]['explainer']
        data, _ = node_data[0:20].to_list()

        sp_obj = Image_SubmodularPick(exp, data, predict_fn = classifier, **kwargs, **flex_model['explainers'][name]['explain_instance_kwargs'])

        result_dict[name] = sp_obj.sp_explanations
    
    return result_dict