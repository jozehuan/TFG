import numpy as np
from functools import partial

import torch

from flex.pool.decorators import to_plot_explanation, centralized

from skimage.color import gray2rgb, rgb2gray

device = 'cpu'

def predict_(color_img, model, to_gray):
    """Convert the image to grayscale and get the model's prediction

    Args:
    -----
        color_img (Array):  RGB image (the predictor is responsible for correctly formatting the image before making a prediction)
        model (nn.Module): cassification model
    """
    if to_gray:
        gray_img = np.array([rgb2gray(img) for img in color_img])    
        img_tensor = torch.tensor(gray_img, dtype=torch.float32).unsqueeze(1)
    else:
        img_tensor = torch.tensor(color_img, dtype=torch.float32).permute(0, 3, 1, 2)


    # plt.imshow(img_g_tensor[0][0], cmap='gray')
    # plt.title("dentro de _predict")
    # plt.axes('off')
    # plt.show()
    model = model.to('cpu')
    with torch.no_grad():  
        preds = model(img_tensor.to('cpu'))  # Enviar la imagen al mismo dispositivo que el modelo (CPU en este caso)

    return preds

class Explanation:
    def __init__(self, model, exp, id_data, label, *args, **kwargs):
        self._model = model
        self._explainer = exp
        self._id_data = id_data
        self._explain_kwargs = kwargs
        self._label = label

        self._explain_instance = None
        self._feature_mask = None
        self._explanations = None 

    def get_pred_info(self, data):
        self._model = self._model.to(device)
        self._model.eval()
        pred = self._model(data.to(device))
        probs = torch.softmax(pred, dim=1)[0].tolist() #probs = pred[0].tolist()
        num_labels = len(probs)
        prediction = torch.argmax(pred, dim=1).item()

        return num_labels, prediction, probs
    
    def generate_exps_list(self, data):
        if self._explanations is None:
            num_labels, _, _ = self.get_pred_info(data)
            self._explanations = [None] * num_labels

    def lime_explanation(self, data, label):
        self.generate_exps_list(data)
        # Evaluar solo cuando sea necesario
        if self._explanations[label] is None:
            data_to_explain = gray2rgb(data.squeeze(0).cpu().detach().numpy())
            classifier = partial(predict_, model=self._model, to_gray = True)
            
            if self._explain_instance is None:
                self._explain_instance = self._explainer.explain_instance(data_to_explain, classifier_fn=classifier,
                                                               **self._explain_kwargs)
                
            segments = self._explain_instance.segments
            explanation_j = np.vectorize(dict(self._explain_instance.local_exp[label]).get)(segments)
            self._explanations[label] = np.nan_to_num(explanation_j, nan=0)
        return self._explanations[label]
    
    def shap_explanation(self, data, label):
        self.generate_exps_list(data)
        # Evaluar solo cuando sea necesario
        if self._explanations[label] is None:
            explanation_j = self._explainer.attribute(data.unsqueeze(0), target=label, **self._explain_kwargs)
            self._explanations[label] = explanation_j.squeeze().detach().numpy()
        return self._explanations[label]
    
    def get_explanation(self, data, label):
        class_name = self._explainer.__class__.__name__
        
        if class_name == 'LimeImageExplainer':
            return self.lime_explanation(data, label)
        
        if class_name in ('DeepLiftShap', 'GradientShap', 'KernelShap'):

            if(feature_mask := self._explain_kwargs.get('feature_mask')) is not None: 
                self._feature_mask = feature_mask
            return self.shap_explanation(data, label)
        
    def lime_segments(self, data):
        self.generate_exps_list(data)

        data_to_explain = gray2rgb(data.squeeze(0).cpu().detach().numpy())
        classifier = partial(predict_, model=self._model, to_gray = True)
        
        if self._explain_instance is None:
            self._explain_instance = self._explainer.explain_instance(data_to_explain, classifier_fn=classifier,
                                                            **self._explain_kwargs)
        
        return self._explain_instance.segments

    def get_segments(self, data):
        class_name = self._explainer.__class__.__name__
        
        if class_name == 'LimeImageExplainer':
            return self.lime_segments(data)
        elif class_name == 'KernelShap':
            return self._feature_mask.squeeze(0).squeeze(0).numpy()
        else: 
            return None

@centralized
def to_centralized(): return True

@to_plot_explanation
def all_explanations(exps, node_data, *args, **kwargs):
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
    
    return exp_output

@to_plot_explanation
def label_explanations(exps, node_data, *args, **kwargs):
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

@to_plot_explanation
def segment_explanations(exps, node_data, *args, **kwargs):
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