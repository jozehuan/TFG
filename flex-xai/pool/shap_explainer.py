from functools import partial

from flex.pool.decorators import set_explainer, get_explanations, get_SP_explanation, plot_explanations
from flex.pool.explanation import Explanation

from captum.attr import DeepLiftShap, GradientShap

from tqdm import tqdm # barra de progreso

import torch
from torch.utils.data import DataLoader

from skimage.color import gray2rgb, rgb2gray

@set_explainer
def set_DeepShapExplainer(flex_model, *args, **kwargs):
    '''Define the DeepShap explainer in the nodes, using the decorator @set_explainer

     Args:
    -----
        baselines (float, optional): 
        return_convergence_delta (bool, optional): 
    '''

    # explainer instance
    explainer = DeepLiftShap(flex_model["model"])
    
    dict_result = {'explainer' : explainer} # output dict
    explain_instance_kwargs = {} 

    # explain instance params
    #  baselines=imgs, target=j, return_convergence_delta=True
    if (baselines := kwargs.get('baselines')) is not None: explain_instance_kwargs['baselines'] = baselines
    if (return_convergence_delta := kwargs.get('return_convergence_delta')) is not None: explain_instance_kwargs['return_convergence_delta'] = return_convergence_delta
    
    dict_result['explain_instance_kwargs'] = explain_instance_kwargs
    return dict_result

# Deprecated - anterior definición sin uso de la clase Explanation
'''
@get_explanations
def get_DeepShapExplanations(flex_model, node_data, *args, **kwargs):
    Generate explanations for the specified data, according to the explainers defined by the specified model, using the decorator @get_explanations

     Args:
    -----
        flex_model (FlexModel): object storing information needed to run a Pytorch model
        data (flex.data.Dataset): objet storing the specified data to be explained

    Note:
    -----
        The argument 'data' should be provided through *args or **kwargs when calling the function.
    
    exp_output = {}
    images = []

    data = kwargs.get("data", None) 
    data_ = data if data is not None else node_data
    dataset = data_.to_torchvision_dataset()
    dataloader = DataLoader(dataset, batch_size=20)

    for imgs, _ in dataloader:
        imgs = imgs.to('cpu')
        images.extend(imgs.tolist())

    num_labels = 10

    # get baselines from de data of the node
    node_data_pt =  node_data.to_torchvision_dataset()
    data_loader = DataLoader(node_data_pt, batch_size=len(node_data_pt), shuffle=False)
    baseline, _ = next(iter(data_loader)) 

    for exp_name, exp in flex_model['explainers'].items():
        if exp['explainer'].__class__.__name__ == 'DeepLiftShap':
            explanations = []
            for data in tqdm(images, desc="Getting DeepLiftShap explanations: ", mininterval=2):
                explanation = []
                for j in range(num_labels):
                    if "baselines" in exp['explain_instance_kwargs']:
                        explanation_j = exp['explainer'].attribute(torch.tensor(data).unsqueeze(0), target = j, **exp['explain_instance_kwargs'])
                    else:
                        explanation_j = exp['explainer'].attribute(torch.tensor(data).unsqueeze(0), target = j, baselines = baseline, **exp['explain_instance_kwargs'])
                    explanation.append(explanation_j)
                explanations.append(explanation)
            exp_output[exp_name] = explanations

    return exp_output
'''

@get_explanations
def get_DeepShapExplanations(flex_model, node_data, *args, **kwargs):
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

    # get baselines from de data of the node
    node_data_pt =  node_data.to_torchvision_dataset()
    data_loader = DataLoader(node_data_pt, batch_size=len(node_data_pt), shuffle=False)
    baseline, _ = next(iter(data_loader)) 

    for exp_name, exp in flex_model['explainers'].items():
        if exp['explainer'].__class__.__name__ == 'DeepLiftShap':
            explanations = []
            for data in tqdm(images, desc="Getting DeepLiftShap explanations: ", mininterval=2):
                if "baselines" in exp['explain_instance_kwargs']:
                    explanation = Explanation(model = flex_model['model'], exp = exp['explainer'], data_to_explain = torch.tensor(data).unsqueeze(0),  **exp['explain_instance_kwargs'])
                else:
                    explanation = Explanation(model = flex_model['model'], exp = exp['explainer'], data_to_explain = torch.tensor(data).unsqueeze(0), baselines = baseline, **exp['explain_instance_kwargs'])

                explanations.append(explanation)
            exp_output[exp_name] = explanations

    return exp_output

