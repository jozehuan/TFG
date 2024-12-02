import numpy as np
from functools import partial
import time

import torch

from skimage.color import gray2rgb, rgb2gray

from flex.pool.decorators import get_explanations
from flex.pool import Explanation

from config import device

def predict_(color_img, model, to_gray):
    """Convert the image to grayscale and get the model's prediction

    Params:
    -----
        color_img : numpy.array
            RGB image (the predictor is responsible for correctly formatting the image before making a prediction)
        model : nn.Module
            cassification model
    """
    if to_gray:
        gray_img = np.array([rgb2gray(img) for img in color_img])    
        img_tensor = torch.tensor(gray_img, dtype=torch.float32).unsqueeze(1)
    else:
        img_tensor = torch.tensor(color_img, dtype=torch.float32).permute(0, 3, 1, 2)

    model = model.to('cpu')
    with torch.no_grad():  
        preds = model(img_tensor.to('cpu')) 

    return preds

class Base_explanation(Explanation):
    """Base class to represent explanations."""
    
    def __init__(self, model, exp, id_data, label, **kwargs):
        """ Initializes the base explanation.

        Params
        ----------
        model : flex.model.FlexModel
            The node's FlexModel instance
        explainer
            explainer instance
        id_data : int
            Explained data ID
        label 
            data label
        """
        super().__init__(model, exp, id_data, label, **kwargs)
        
        self._time = []
        self._explanations = None 

    def get_pred_info(self, data):
        """ Get num of labels, prediction and prob. prediction of the explained data.
        
        Params
        ----------
        data : torch.Tensor
            explained data
        
        """
        self._model = self._model.to(device)
        self._model.eval()
        pred = self._model(data.to(device))
        probs = torch.softmax(pred, dim=1)[0].tolist()
        num_labels = len(probs)
        prediction = torch.argmax(pred, dim=1).item()

        return num_labels, prediction, probs
    
    def get_mean_time(self):
        """ Compute mean explanation time
        """
        if len(self._time) > 0:
            return sum(self._time) / len(self._time)
        else: return 0
    
    def generate_exps_list(self, data):
        """ Set the list that stores the explanations.
        
        Params
        ----------
        data : torch.Tensor
            explained data
        
        """
        if self._explanations is None:
            num_labels, _, _ = self.get_pred_info(data)
            self._explanations = [None] * num_labels




class Lime_explanation(Base_explanation):
    """Base class to represent LIME explanations."""
    
    def __init__(self, model, exp, id_data, label, **kwargs):
        """ Initializes LIME explanation.

        Params
        ----------
        model : flex.model.FlexModel
            The node's FlexModel instance
        explainer
            explainer instance
        id_data : int
            Explained data ID
        label 
            data label
        """
        super().__init__(model, exp, id_data, label, **kwargs)
        self._explain_instance = None   
    
    def get_explanation(self, data, label):
        """ Compute LIME explanation
        
        Params
        ----------
        data : torch.Tensor
            explained data
        label
            data label
        
        """
        
        self.generate_exps_list(data)
        # Evaluar solo cuando sea necesario
        if self._explanations[label] is None:
            data_to_explain = gray2rgb(data.squeeze(0).cpu().detach().numpy())
            classifier = partial(predict_, model=self._model, to_gray = True)
            
            if self._explain_instance is None:
                start_time = time.perf_counter()
                self._explain_instance = self._explainer.explain_instance(data_to_explain, classifier_fn=classifier,
                                                               **self._explain_kwargs)
                end_time = time.perf_counter()
                self._time.append(end_time - start_time)
                
            segments = self._explain_instance.segments
            explanation_j = np.vectorize(dict(self._explain_instance.local_exp[label]).get)(segments)
            self._explanations[label] = np.nan_to_num(explanation_j, nan=0)
        return self._explanations[label]
    
    def get_segments(self, data):
        """ Get data segmentation
        
        Params
        ----------
        data : torch.Tensor
            explained data
        """
        
        self.generate_exps_list(data)

        data_to_explain = gray2rgb(data.squeeze(0).cpu().detach().numpy())
        classifier = partial(predict_, model=self._model, to_gray = True)
        
        if self._explain_instance is None:
            self._explain_instance = self._explainer.explain_instance(data_to_explain, classifier_fn=classifier,
                                                            **self._explain_kwargs)
        
        return self._explain_instance.segments

    

class Shap_explanation(Base_explanation):
    """Base class to represent SHAP explanations."""
    
    def __init__(self, model, exp, id_data, label, **kwargs):
        """ Initializes SHAP explanation.

        Params
        ----------
        model : flex.model.FlexModel
            The node's FlexModel instance
        explainer
            explainer instance
        id_data : int
            Explained data ID
        label 
            data label
        """
        
        super().__init__(model, exp, id_data, label, **kwargs)
        self._feature_mask = None
        
        if(feature_mask := self._explain_kwargs.get('feature_mask')) is not None: 
            self._feature_mask = feature_mask
        
    def get_explanation(self, data, label):
        """ Compute SHAP explanation
        
        Params
        ----------
        data : torch.Tensor
            explained data
        label
            data label
        
        """
        
        self.generate_exps_list(data)
        # Evaluar solo cuando sea necesario
        if self._explanations[label] is None:
            start_time = time.perf_counter()
            explanation_j = self._explainer.attribute(data.unsqueeze(0), target=label, **self._explain_kwargs)
            end_time = time.perf_counter()
            self._time.append(end_time - start_time)

            self._explanations[label] = explanation_j.squeeze().detach().numpy()
        return self._explanations[label]
    
    def get_segments(self, data):
        """ Get data segmentation
        
        Params
        ----------
        data : torch.Tensor
            explained data
        """
        
        class_name = self._explainer.__class__.__name__
        
        if class_name == 'KernelShap':
            return self._feature_mask.squeeze(0).squeeze(0).numpy()
        else: 
            return None

@get_explanations
def all_explanations(exps, node_data, *args, **kwargs):
    """ Get data explanations for each label 
    
    Params
    ----------
    exps
        explainer
    node_data : flex.data.Dataset
        node data
        
    """
    data = kwargs.get("data", None) 
    data_ = data if data is not None else node_data
    dataset = data_.to_torchvision_dataset()

    exp_output = []
    exp_name = kwargs.get("name", None)

    from tqdm import tqdm
    for e in tqdm(exps, desc=f"Computing {exp_name} explanations: ", mininterval=2): 
        explanations = []

        data, label = dataset[e._id_data]
        num_labels, prediction, probs = e.get_pred_info(data.unsqueeze(0))

        for j in range(num_labels):
            explanation_j = e.get_explanation(data, label=j)
            explanations.append((explanation_j, f'{j}\n({probs[j]*100:.2f}%)'))
        
        explanations.append((-data.squeeze(0), f'label: {label}\npred: {prediction}'))
        exp_output.append(explanations)
    
    print(f'\t(mean time {exp_name} explanations: {e.get_mean_time():.8f} seconds)')
    
    return exp_output

@get_explanations
def label_explanations(exps, node_data, *args, **kwargs):
    """ Get data explanations for its label 
    
    Params
    ----------
    exps
        explainer
    node_data : flex.data.Dataset
        node data
        
    """
    data = kwargs.get("data", None) 
    data_ = data if data is not None else node_data
    dataset = data_.to_torchvision_dataset()

    exp_output = []

    for e in exps:
        explanations = []

        data, label = dataset[e._id_data]
        _, prediction, probs = e.get_pred_info(data.unsqueeze(0))

        explanation_label = e.get_explanation(data, label=label)
        explanations.append((explanation_label, f'{label}\n({probs[label]*100:.2f}%)'))

        if label is not prediction:
            explanation_pred = e.get_explanation(data, label=prediction)
            explanations.append((explanation_pred, f'{prediction}\n({probs[prediction]*100:.2f}%)'))

        explanations.append((-data.squeeze(0), f'label: {label}\npred: {prediction}'))
        exp_output.append(explanations)
    
    return exp_output

@get_explanations
def segment_explanations(exps, node_data, *args, **kwargs):
    """ Get data segmentation 
    
    Params
    ----------
    exps
        explainer
    node_data : flex.data.Dataset
        node data
        
    """
    data = kwargs.get("data", None) 
    data_ = data if data is not None else node_data
    dataset = data_.to_torchvision_dataset()

    exp_output = []

    for e in exps:
        exp_segments = []

        data, label = dataset[e._id_data]
        segments = e.get_segments(data)

        exp_segments.append((-data.squeeze(0).squeeze(0).numpy(), f'label: {label}'))

        if segments is not None:
            n_segments = np.max(segments)
            exp_segments.append((segments, f'({n_segments} segments)'))
        else:
            exp_output = None
            break

        exp_output.append(exp_segments)

    return exp_output    


@get_explanations
def get_global_mean(exps, node_data, *args, **kwargs):
    """ Get local explanations global means for each label.
    
    Params
    ----------
    exps
        explainer
    node_data : flex.data.Dataset
        node data
        
    """
    data = kwargs.get("data", None) 
    data_ = data if data is not None else node_data
    dataset = data_.to_torchvision_dataset()

    collect_exps = {}
    for e in exps:
        data, label = dataset[e._id_data]
        if label not in collect_exps:
            collect_exps[label] = []
        
        collect_exps[label].append(e.get_explanation(data, label=label))

    exp_output = []
    for exp_label, exps_ in collect_exps.items():
        exp_output.append((np.mean(exps_, axis=0), f'{exp_label}'))
    
    return exp_output