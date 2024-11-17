import numpy as np
import warnings

from flex.pool.decorators import set_explainer, get_explanations

from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm 

from tqdm import tqdm # barra de progreso

from flex.pool.explanation import Explanation

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

    data = kwargs.get("data", None) 
    data_ = data if data is not None else node_data
    dataset = data_.to_torchvision_dataset()
    
    for exp_name, exp in flex_model['explainers'].items():
        cl_name = exp['explainer'].__class__.__name__
        if cl_name == 'LimeImageExplainer':
            explanations = []
            for i in tqdm(range(len(dataset)), desc="Getting LIME explanations: ", mininterval=2): 
                data, label = dataset[i]
                explanation = Explanation(model = flex_model['model'], exp = exp['explainer'], id_data = i, label = label, **exp['explain_instance_kwargs'])
                
                explanations.append(explanation)
            exp_output[exp_name] = explanations

    return exp_output


def ERROR_MSG_MIN_ARG_GENERATOR(f, min_args):
    return f"The decorated function: {f.__name__} is expected to have at least {min_args} argument/s."


@get_explanations
def get_SP_LimeImageExplanation(flex_model, node_data, *args, **kwargs):
    exp_output = {}

    num_exps_desired = kwargs.get('num_exps_desired', 3)
    if (e_name := kwargs.get('explanation_name', None)) is None: True # METER AQUI ERROR

    data = kwargs.get("data", None) 
    data_ = data if data is not None else node_data
    dataset = data_.to_torchvision_dataset()

    explanations_all = flex_model['explanations'][e_name]

    explanations = {}
    map_exp = {}

    next(iter(dataset), None)

    for d, exp in enumerate (explanations_all):
        data, label = dataset[exp._id_data]
        num_labels, prediction, _ = exp.get_pred_info(data.unsqueeze(0))
        
        if prediction == exp._label:
            if label in explanations:
                explanations[label].append(exp.get_explanation(data, label))
            else:
                explanations[label] = [exp.get_explanation(data, label)]
                map_exp[label] = {}

            map_exp[label][len(explanations[label]) - 1] = d

    num_exps_per_class = [num_exps_desired] * num_labels

    try:
        exps = next(iter(explanations.values()), None)
        n_pixels = exps[0].shape[0] * exps[0].shape[1]
    except Exception:
        warnings.warn("Warning: Unable to calculate the number of pixels. Ending the function.")
        return None
    
    exp_output[f'SP-{e_name}'] = []
    
    for l , exps in tqdm(explanations.items(), desc=f"Getting SP-{e_name} explanations: ", mininterval=2):
        V = []
        num_exps_desired = min(num_exps_desired, len(exps))

        W = np.zeros((len(exps), n_pixels))
        for i, exp in enumerate(exps):
                for j, value in enumerate(exp.flatten()):
                    W[i, j] += value
                    
        importance = np.sum(abs(W), axis=0)**.5
        
        # Now run the SP-LIME greedy algorithm
        remaining_indices = set(range(len(exps)))
        num_exps_desired = min(num_exps_desired * len(exps), len(exps))

        for _ in range(num_exps_desired):
            best = 0
            best_ind = None
            current = 0
            for i in remaining_indices:
                current = np.dot(
                        (np.sum(abs(W)[V + [i]], axis=0) > 0), importance
                        )  # coverage function
                if current >= best:
                    best = current
                    best_ind = i

            _, best_label  = dataset[explanations_all[map_exp[l][best_ind]]._id_data]
            if num_exps_per_class[best_label] > 0: 
                V.append(best_ind)
                num_exps_per_class[best_label] -= 1

            remaining_indices -= {best_ind}
        
        sp_explanations = [explanations_all[map_exp[l][i]] for i in V] 
        exp_output[f'SP-{e_name}'] += sp_explanations
    return exp_output