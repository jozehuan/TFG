import numpy as np
from functools import partial

import torch
from flex.pool.decorators import plot_explanations

from captum.attr import DeepLiftShap, GradientShap

from skimage.color import gray2rgb, rgb2gray

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
colors = []
for j in np.linspace(1, 0, 100):
    colors.append((30.0 / 255, 136.0 / 255, 229.0 / 255, j))
for j in np.linspace(0, 1, 100):
    colors.append((255.0 / 255, 13.0 / 255, 87.0 / 255, j))
red_transparent_blue = LinearSegmentedColormap.from_list("red_transparent_blue", colors)


def predict_(color_img, model):
    """Convert the image to grayscale and get the model's prediction

    Args:
    -----
        color_img (Array):  RGB image (the predictor is responsible for correctly formatting the image before making a prediction)
        model (nn.Module): cassification model
    """
    
    img_g_tensor = torch.from_numpy(rgb2gray(color_img)).unsqueeze(1)
    
    # plt.imshow(img_g_tensor[0][0], cmap='gray')
    # plt.title("dentro de _predict")
    # plt.axes('off')
    # plt.show()
    
    with torch.no_grad():  # Desactivar el cálculo de gradientes para la predicción
        preds = model(img_g_tensor.to('cpu'))  # Enviar la imagen al mismo dispositivo que el modelo (CPU en este caso)

    return preds

class Explanation:
    
    def __init__(self, model, exp, data_to_explain, *args, **kwargs):
        self.model = model
        self.explainer = exp
        self.data = data_to_explain
        self.explain_kwargs = kwargs

        self.model.eval()
        pred = self.model(data_to_explain)
        self.probs = pred[0].tolist()
        self.prediction = torch.argmax(pred, dim=1)
        
    def get_explanation(self, label):
        class_name = self.explainer.__class__.__name__
        
        if class_name == 'LimeImageExplainer':
            classifier = partial(predict_, model = self.model)
            
            explanation = self.explainer.explain_instance(gray2rgb(self.data.squeeze().numpy()), 
                                            classifier_fn = classifier, 
                                            **self.explain_kwargs)
            
            segments = explanation.segments
            explanation_label = np.vectorize(dict(explanation.local_exp[label]).get)(segments) 
            return explanation_label
        
        if class_name == 'DeepLiftShap':
            explanation_j = self.explainer.attribute(self.data, target=label, **self.explain_kwargs)
            return explanation_j.squeeze().detach().numpy()


@plot_explanations
def plot_heatmap(flex_model, node_data, *args, **kwargs):

    for _, exps in flex_model["explanations"].items():
        for e in exps:
            fig_size = np.array([11 * 0.11 * (10 + 1), 6])
            fig, ax =plt.subplots(nrows=1, ncols=(10 + 1), figsize=fig_size, squeeze=False)
    
            max_vals = [np.max(np.abs(e.get_explanation(label=k))) for k in range(10)]
            max_val = np.max(max_vals)

            axes = ax.ravel()
            
            axes[0].imshow(-e.data.squeeze(0).squeeze(0), cmap=plt.get_cmap("gray"))
            axes[0].axis("off")
            axes[0].set_title(f'pred: {e.prediction}')

            for j in range(10):
                axes[j+1].imshow(-e.data.squeeze(0).squeeze(0), cmap=plt.get_cmap("gray"), alpha=0.15, extent=(-1, e.get_explanation(label=j).shape[1], e.get_explanation(label=j).shape[0], -1))
                im = axes[j+1].imshow(e.get_explanation(label=j), cmap=red_transparent_blue, vmin=-max_vals[j], vmax=max_vals[j]) #, vmin=-max_val, vmax=max_val)
                fig.colorbar(im, ax=axes[j+1], orientation="horizontal")
                axes[j+1].axis("off")
                axes[j+1].set_title(f'{j} ({e.probs[j]*100:.2f}%)')
            
            fig.tight_layout()
            #fig.subplots_adjust(hspace=0.5)
            #cb = fig.colorbar(
            #    im, ax=np.ravel(axes).tolist(), label="SHAP value", orientation="horizontal", aspect=fig_size[0] / 0.2)
            #cb.outline.set_visible(False)
            
            plt.show()
