import numpy as np
import os
from functools import partial

import torch
from flex.pool.decorators import to_plot_explanation, centralized

from skimage.color import gray2rgb, rgb2gray

from tqdm import tqdm # progress bar

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
colors = []
for j in np.linspace(1, 0, 100):
    colors.append((30.0 / 255, 136.0 / 255, 229.0 / 255, j))
for j in np.linspace(0, 1, 100):
    colors.append((255.0 / 255, 13.0 / 255, 87.0 / 255, j))
red_transparent_blue = LinearSegmentedColormap.from_list("red_transparent_blue", colors)


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
    with torch.no_grad():  # Desactivar el cálculo de gradientes para la predicción
        preds = model(img_tensor.to('cpu'))  # Enviar la imagen al mismo dispositivo que el modelo (CPU en este caso)

    return preds

class Explanation:
    
    def __init__(self, model, exp, id_data, label, *args, **kwargs):
        self.model = model
        self.explainer = exp
        self._id_data = id_data
        self.explain_kwargs = kwargs
        self._label = label

        
        self._explanations = [None] * 10 #* self.num_labels !!!!! SOLUCIONAR

    def get_pred_info(self, data):
        self.model.eval()
        pred = self.model(data)
        probs = pred[0].tolist()
        num_labels = len(probs)
        prediction = torch.argmax(pred, dim=1).item()

        return num_labels, prediction, probs
    
    def lime_explanation(self, data, label):

        # Evaluar solo cuando sea necesario
        if self._explanations[label] is None:
            
            if data.shape[0] == 3:
                data_to_explain = np.transpose(data.cpu().detach().numpy(), (1, 2, 0)) 
                classifier = partial(predict_, model=self.model, to_gray = False)
            elif data.shape[0] == 1:
                data_to_explain = gray2rgb(data.squeeze(0).cpu().detach().numpy())
                classifier = partial(predict_, model=self.model, to_gray = True)
            else: True # CAMBIAR A ERROR

            explanation = self.explainer.explain_instance(
                data_to_explain,
                classifier_fn=classifier,
                **self.explain_kwargs
            )
            segments = explanation.segments
            explanation_j = np.vectorize(dict(explanation.local_exp[label]).get)(segments)
            self._explanations[label] = np.nan_to_num(explanation_j, nan=0)
        return self._explanations[label]
    
    def shap_explanation(self, data, label):
        # Evaluar solo cuando sea necesario
        if self._explanations[label] is None:
            explanation_j = self.explainer.attribute(data.unsqueeze(0), target=label, **self.explain_kwargs)
            self._explanations[label] = explanation_j.squeeze().detach().numpy()
        return self._explanations[label]
    
    def get_explanation(self, data, label):
        class_name = self.explainer.__class__.__name__
        
        if class_name == 'LimeImageExplainer':
            return self.lime_explanation(data, label)
        
        if class_name in ('DeepLiftShap', 'GradientShap', 'KernelShap'):
            return self.shap_explanation(data, label)

@centralized
def to_centralized(): return True

@to_plot_explanation
def to_all_heatmaps(exps, node_data, *args, **kwargs):
    data = kwargs.get("data", None) 
    data_ = data if data is not None else node_data
    dataset = data_.to_torchvision_dataset()

    exp_output = []

    for e in exps:
        explanations = []

        data, label = dataset[e._id_data]
        num_labels, prediction, probs = e.get_pred_info(data.unsqueeze(0))

        for j in range(num_labels):
            explanation_j = e.get_explanation(data, label=j)
            explanations.append((explanation_j, f'{j}\n({probs[j]*100:.2f}%)'))
        
        explanations.append((-data.squeeze(0), f'label: {label}\npred: {prediction}'))
        exp_output.append(explanations)
    
    return exp_output

        
