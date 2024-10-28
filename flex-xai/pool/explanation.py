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
        self.prediction = torch.argmax(pred, dim=1).item()
        
    def get_explanation(self, label):
        class_name = self.explainer.__class__.__name__
        
        if class_name == 'LimeImageExplainer':
            classifier = partial(predict_, model = self.model)            
            explanation = self.explainer.explain_instance(gray2rgb(self.data.squeeze(0).squeeze(0).cpu().detach().numpy()), 
                                            classifier_fn = classifier, 
                                            **self.explain_kwargs)
            
            segments = explanation.segments
            explanation_j = np.vectorize(dict(explanation.local_exp[label]).get)(segments) 
            return np.nan_to_num(explanation_j, nan=0)
        
        if class_name in ('DeepLiftShap', 'GradientShap'):
            explanation_j = self.explainer.attribute(self.data, target=label, **self.explain_kwargs)
            return explanation_j.squeeze().detach().numpy()

def adjust_values_iqr(image):
    # Calcular los cuartiles
    P5 = np.percentile(image, 5)
    P95 = np.percentile(image, 95)
    
    # Calcular el IQR
    IQR = P95 - P5
    
    # Definir los límites
    lower_bound = P5 - 1.5 * IQR
    upper_bound = P95 + 1.5 * IQR
    
    # Ajustar los valores fuera de los límites
    image_clipped = np.where(image < lower_bound, lower_bound, image)
    image_clipped = np.where(image > upper_bound, upper_bound, image)
    
    return image_clipped

@plot_explanations
def plot_heatmap(flex_model, node_data, *args, **kwargs):

    for exp_name, exps in flex_model["explanations"].items():
        for e in exps:
            num_labels = len(e.probs)

            fig_size = np.array([(num_labels + 1) * 0.13 * (num_labels + 1), 6])
            fig, ax =plt.subplots(nrows=2, ncols=(num_labels + 1), figsize=fig_size, squeeze=False)
            #fig.suptitle(f'{exp_name}')

            max_vals = [np.max(np.abs(e.get_explanation(label=k))) for k in range(num_labels)]
            max_val = max(max_vals)
            
            ax[0,0].imshow(-e.data.squeeze(0).squeeze(0), cmap=plt.get_cmap("gray"))
            ax[0,0].axis("off");  ax[1, 0].axis("off")
            ax[0,0].set_title(f'pred: {e.prediction}')

            for j in range(num_labels):
                explanation_j = adjust_values_iqr(e.get_explanation(label=j))
                max_value = np.max(np.abs(explanation_j))


                ax[0,j+1].imshow(-e.data.squeeze(0).squeeze(0), cmap=plt.get_cmap("gray"), alpha=0.15, extent=(-1, explanation_j.shape[1], explanation_j.shape[0], -1))
                im = ax[0, j+1].imshow(explanation_j, cmap=red_transparent_blue, vmin=-max_val, vmax=max_val)
                ax[0, j+1].set_title(f'{j}\n({e.probs[j]*100:.2f}%)')
                ax[0, j+1].axis("off")

                ax[1,j+1].imshow(-e.data.squeeze(0).squeeze(0), cmap=plt.get_cmap("gray"), alpha=0.15, extent=(-1, explanation_j.shape[1], explanation_j.shape[0], -1))
                im = ax[1,j+1].imshow(explanation_j, cmap=red_transparent_blue, vmin=-max_value, vmax=max_value)
                
                print(f'valor maximo para {j}: {max_vals[j]} \npercentiles: {np.percentile(e.get_explanation(label=j), [1,5,95,99])}')
                print(f'valor maximo para chiped{j}: {max_value} \npercentiles: {np.percentile(explanation_j, [1,5,95,99])}\n')

                fig.colorbar(im, ax=ax[1,j+1], orientation="horizontal")
                ax[1,j+1].axis("off")
            
            cbar_ax = fig.add_axes([0.125, 0.54, 0.775, 0.02])  # Ajusta la posición [left, bottom, width, height]
            cb = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
            cb.outline.set_visible(False)

            fig.tight_layout()
            fig.subplots_adjust(hspace=0.5)
            #cb = fig.colorbar(
            #    im, ax=np.ravel(axes).tolist(), label="SHAP value", orientation="horizontal", aspect=fig_size[0] / 0.2)
            #cb.outline.set_visible(False)
            
            plt.show()