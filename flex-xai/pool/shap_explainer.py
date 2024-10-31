from functools import partial

from flex.pool.decorators import set_explainer, get_explanations, get_SP_explanation, plot_explanations
from flex.pool.explanation import Explanation

from captum.attr import DeepLiftShap, GradientShap, KernelShap

from tqdm import tqdm # progress bar

from skimage.segmentation import slic

import torch
from torch.utils.data import DataLoader


def set_ShapExplainer(flex_model, explainer, *args, **kwargs):
    dict_result = {'explainer' : explainer} 
    explain_instance_kwargs = {} 

    if (baselines := kwargs.get('baselines')) is not None: explain_instance_kwargs['baselines'] = baselines
    if (return_conv_delta := kwargs.get('return_convergence_delta')) is not None: explain_instance_kwargs['return_convergence_delta'] = return_conv_delta
    
    if (n_samples := kwargs.get('n_samples')) is not None: explain_instance_kwargs['n_samples'] = n_samples
    if (stdevs := kwargs.get('stdevs')) is not None: explain_instance_kwargs['stdevs'] = stdevs
    if (pert_per_eval := kwargs.get('perturbations_per_eval')) is not None: explain_instance_kwargs['perturbations_per_eval'] = pert_per_eval

    dict_result['explain_instance_kwargs'] = explain_instance_kwargs
    return dict_result

@set_explainer
def set_DeepShapExplainer(flex_model, *args, **kwargs):
    '''Define the DeepShap explainer in the nodes, using the decorator @set_explainer

     Args:
    -----
        baselines (float, optional): 
        return_convergence_delta (bool, optional): 
    '''

    explainer = DeepLiftShap(flex_model["model"])
    return set_ShapExplainer(flex_model, explainer, *args, **kwargs)

@set_explainer
def set_GradientShapExplainer(flex_model, *args, **kwargs):
    '''Define the GradientShap explainer in the nodes, using the decorator @set_explainer

     Args:
    -----
        baselines (float, optional): 
        return_convergence_delta (bool, optional): 
    '''

    explainer = GradientShap(flex_model["model"])
    return set_ShapExplainer(flex_model, explainer, *args, **kwargs)

@set_explainer
def set_KernelShapExplainer(flex_model, *args, **kwargs):
    '''Define the GradientShap explainer in the nodes, using the decorator @set_explainer

     Args:
    -----
        baselines (float, optional): 
        return_convergence_delta (bool, optional): 
    '''

    explainer = KernelShap(flex_model["model"])
    return set_ShapExplainer(flex_model, explainer, *args, **kwargs)

# Deprecated - anterior definición de DeepShap sin uso de la clase Explanation
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
def get_ShapExplanations(flex_model, node_data, *args, **kwargs):
    '''Generate explanations for the specified data, according to the explainers defined by the specified model, using the decorator @get_explanations

     Args:
    -----
        flex_model (FlexModel): The node's FlexModel
        node_data (flex.data.Dataset): The node's dataset
        data (flex.data.Dataset, optional): objet storing the specified data to be explained

    Note:
    -----
        The argument 'data' should be provided through **kwargs when calling the function.
        If not provided, the entire dataset will be explained.
    '''
    exp_output = {}
    images = []
    labels = []

    data = kwargs.get("data", None) 
    data_ = data if data is not None else node_data
    dataset = data_.to_torchvision_dataset()
    dataloader = DataLoader(dataset, batch_size=20)

    for imgs, l in dataloader:
        imgs = imgs.to('cpu')
        images.extend(imgs.tolist())
        labels.extend(l.tolist())

    # get baselines from de data of the node
    node_data_pt =  node_data.to_torchvision_dataset()
    data_loader = DataLoader(node_data_pt, batch_size=len(node_data_pt), shuffle=False)
    baseline, _ = next(iter(data_loader)) 

    for exp_name, exp in flex_model['explainers'].items():
        cl_name = exp['explainer'].__class__.__name__
        if cl_name in ('DeepLiftShap', 'GradientShap'):
            explanations = []
            for data, label in tqdm(zip(images, labels), desc=f'Getting {cl_name} explanations: ', mininterval=2):
                if "baselines" in exp['explain_instance_kwargs']:
                    explanation = Explanation(model = flex_model['model'], exp = exp['explainer'], data_to_explain = torch.tensor(data).unsqueeze(0), label = label, **exp['explain_instance_kwargs'])
                else:
                    explanation = Explanation(model = flex_model['model'], exp = exp['explainer'], data_to_explain = torch.tensor(data).unsqueeze(0), label = label, baselines = baseline, **exp['explain_instance_kwargs'])

                explanations.append(explanation)
            exp_output[exp_name] = explanations
        
        if cl_name in ('KernelShap'):
            explanations = []
            for data, label in tqdm(zip(images, labels), desc=f'Getting {cl_name} explanations: ', mininterval=2):
                if "baselines" in exp['explain_instance_kwargs']:
                    explanation = Explanation(model = flex_model['model'], exp = exp['explainer'], data_to_explain = torch.tensor(data).unsqueeze(0), label = label, **exp['explain_instance_kwargs'])
                else:
                    segm = slic(torch.tensor(data).squeeze(0).detach().numpy() , n_segments=100, compactness=0.05, sigma=0.4, channel_axis=None)
                    segm_ = torch.tensor(segm).unsqueeze(0).unsqueeze(0) - 1
                    base = torch.zeros_like(torch.tensor(data).unsqueeze(0))
                    explanation = Explanation(model = flex_model['model'], exp = exp['explainer'], data_to_explain = torch.tensor(data).unsqueeze(0), label = label, baselines=base, feature_mask=segm_, **exp['explain_instance_kwargs'])
                
                explanations.append(explanation)
            exp_output[exp_name] = explanations

    return exp_output

