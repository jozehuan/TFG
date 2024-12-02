from tqdm import tqdm 
from skimage.segmentation import slic

from captum.attr import DeepLiftShap, GradientShap, KernelShap

import torch
from torch.utils.data import DataLoader

from flex.pool.decorators import set_explainer, compute_explanations
from .explanation import Shap_explanation


def set_ShapExplainer(flex_model, explainer, *args, **kwargs):
    '''Base function to define SHAP explainers
    
     Args:
    -----
        flex_model :  flex.model.FlexModel
            The node's FlexModel
        explainer : Union[DeepLiftShap, GradientShap, KernelShap]
            The Captum explanation method to be used as an explainer.
    '''
    
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
        flex_model :  flex.model.FlexModel
            The node's FlexModel
    '''

    explainer = DeepLiftShap(flex_model["model"])
    return set_ShapExplainer(flex_model, explainer, *args, **kwargs)

@set_explainer
def set_GradientShapExplainer(flex_model, *args, **kwargs):
    '''Define the GradientShap explainer in the nodes, using the decorator @set_explainer

     Args:
    -----
        flex_model :  flex.model.FlexModel
            The node's FlexModel 
    '''

    explainer = GradientShap(flex_model["model"])
    return set_ShapExplainer(flex_model, explainer, *args, **kwargs)

@set_explainer
def set_KernelShapExplainer(flex_model, *args, **kwargs):
    '''Define the GradientShap explainer in the nodes, using the decorator @set_explainer

     Args:
    -----
        flex_model :  flex.model.FlexModel
            The node's FlexModel
    '''

    explainer = KernelShap(flex_model["model"])
    return set_ShapExplainer(flex_model, explainer, *args, **kwargs)

@compute_explanations
def get_ShapExplanations(flex_model, node_data, *args, **kwargs):
    '''Generate explanations for the specified data, according to the explainers defined by
    the specified model, using the decorator @get_explanations

     Args:
    -----
        flex_model :  flex.model.FlexModel
            The node's FlexModel
        node_data : flex.data.Dataset
            The node's dataset
        data : flex.data.Dataset, optional
            store the specified data to be explained

    Note:
    -----
        The argument 'data' should be provided through **kwargs when calling the function.
        If not provided, the node's data will be explained.
    '''
    exp_output = {}

    data = kwargs.get("data", None) 
    data_ = data if data is not None else node_data
    dataset = data_.to_torchvision_dataset()

    # get baselines from de data of the node
    node_data_pt =  node_data.to_torchvision_dataset()
    data_loader = DataLoader(node_data_pt, batch_size=len(node_data_pt), shuffle=False)
    baseline, _ = next(iter(data_loader)) 

    for exp_name, exp in flex_model['explainers'].items():
        cl_name = exp['explainer'].__class__.__name__
        if cl_name in ('DeepLiftShap', 'GradientShap'):
            explanations = []
            for i in tqdm(range(len(dataset)), desc=f'Getting {cl_name} explanations: ', mininterval=2): 
                data, label = dataset[i]
                if "baselines" in exp['explain_instance_kwargs']:
                    explanation = Shap_explanation(model = flex_model['model'], exp = exp['explainer'], id_data = i, label = label, **exp['explain_instance_kwargs'])
                else:
                    explanation = Shap_explanation(model = flex_model['model'], exp = exp['explainer'], id_data = i, label = label, baselines = baseline, **exp['explain_instance_kwargs'])
                
                explanations.append(explanation)
            exp_output[exp_name] = explanations
        
        if cl_name in ('KernelShap'):
            explanations = []
            for i in tqdm(range(len(dataset)), desc=f'Getting {cl_name} explanations: ', mininterval=2): 
                data, label = dataset[i]
                
                segm = slic(data.clone().detach().squeeze(0).detach().numpy() , n_segments=100, compactness=0.05, sigma=0.4, channel_axis=None)
                segm_ = torch.tensor(segm).unsqueeze(0).unsqueeze(0) - 1
                base = torch.zeros_like(data.clone().detach().unsqueeze(0))
                explanation = Shap_explanation(model = flex_model['model'], exp = exp['explainer'], id_data = i, label = label, baselines=base, feature_mask=segm_, **exp['explain_instance_kwargs'])
                
                explanations.append(explanation)
            exp_output[exp_name] = explanations

    return exp_output

