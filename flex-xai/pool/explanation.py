import numpy as np
import os
from functools import partial

import torch
from flex.pool.decorators import plot_explanations

from captum.attr import DeepLiftShap, GradientShap

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
    
    def __init__(self, model, exp, data_to_explain, label, *args, **kwargs):
        self.model = model
        self.explainer = exp
        self.data = data_to_explain
        self.explain_kwargs = kwargs
        self._label = label

        self.model.eval()
        pred = self.model(data_to_explain)
        self.probs = pred[0].tolist()
        self.num_labels = len(self.probs)
        self.prediction = torch.argmax(pred, dim=1).item()

        self._explanations = [None] * self.num_labels
    
    def lime_explanation(self, label):
        # Evaluar solo cuando sea necesario
        if self._explanations[label] is None:
            classifier = partial(predict_, model=self.model)
            
            explanation = self.explainer.explain_instance(
                gray2rgb(self.data.squeeze(0).squeeze(0).cpu().detach().numpy()),
                classifier_fn=classifier,
                **self.explain_kwargs
            )
            segments = explanation.segments
            explanation_j = np.vectorize(dict(explanation.local_exp[label]).get)(segments)
            self._explanations[label] = np.nan_to_num(explanation_j, nan=0)
        return self._explanations[label]
    
    def shap_explanation(self, label):
        # Evaluar solo cuando sea necesario
        if self._explanations[label] is None:
            explanation_j = self.explainer.attribute(self.data, target=label, **self.explain_kwargs)
            self._explanations[label] = explanation_j.squeeze().detach().numpy()
        return self._explanations[label]
    
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
        
        if class_name in ('DeepLiftShap', 'GradientShap', 'KernelShap'):
            explanation_j = self.explainer.attribute(self.data, target=label, **self.explain_kwargs)
            return explanation_j.squeeze().detach().numpy()
        
    def get_explanation(self, label):
        class_name = self.explainer.__class__.__name__
        
        if class_name == 'LimeImageExplainer':
            return self.lime_explanation(label)
        
        if class_name in ('DeepLiftShap', 'GradientShap', 'KernelShap'):
            return self.shap_explanation(label)


@plot_explanations
def plot_heatmap(flex_model, node_data, *args, **kwargs):

    output_dir = 'images/temp'
    if (pathname := kwargs.get('pathname')) is not None: output_dir = pathname
  
    for exp_name, exps in flex_model["explanations"].items():
        cur_output_dir = output_dir + '/' + exp_name
        try:
            os.makedirs(cur_output_dir, exist_ok=True)
        except:
            True # CAMBIAR ESTO PARA MANEJO ERRORES
    
        for i, e in enumerate(tqdm(exps, desc=f'Generate heatmaps of {exp_name} explanations: ', mininterval=2)):
            num_labels = e.num_labels

            fig_size = np.array([(num_labels + 1) * 0.15 * (num_labels + 1), 8])
            fig, ax =plt.subplots(nrows=2, ncols=(num_labels + 2)//2, figsize=fig_size, squeeze=False)
            fig.suptitle(f'{exp_name}')

            ax[0,0].imshow(-e.data.squeeze(0).squeeze(0), cmap=plt.get_cmap("gray"))
            ax[0,0].axis("off");  ax[1, 0].axis("off")
            ax[0,0].set_title(f'label: {e._label}\npred: {e.prediction}')

            cur_row = 0
            col_delay = 0 
            for j in range(num_labels):
                explanation_j = e.get_explanation(label=j)

                try:
                    max_value = np.max(np.abs(explanation_j))
                except:
                    print(f'explanation: {explanation_j}')
                    continue

                if j == ((num_labels + 2)//2 - 1):
                    cur_row += 1
                    col_delay = j

                ax[cur_row, j+1-col_delay].imshow(-e.data.squeeze(0).squeeze(0), cmap=plt.get_cmap("gray"), alpha=0.15, extent=(-1, explanation_j.shape[1], explanation_j.shape[0], -1))
                im = ax[cur_row, j+1-col_delay].imshow(explanation_j, cmap=red_transparent_blue, vmin=-max_value, vmax=max_value)
                fig.colorbar(im, ax=ax[cur_row, j+1-col_delay], orientation="horizontal")
                ax[cur_row, j+1-col_delay].axis("off")
                ax[cur_row, j+1-col_delay].set_title(f'{j}\n({e.probs[j]*100:.2f}%)')

            fig.tight_layout()
            fig.subplots_adjust(hspace=0.5)
            
            #plt.show()
            output_path = os.path.join(cur_output_dir, f'heatmap_{i}.png')
            fig.savefig(output_path)
            plt.close(fig)  # Cerrar la figura para liberar memoria

'''
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def plot_heatmap(flex_model, node_data, *args, **kwargs):
    # Crear la carpeta de destino si no existe
    output_dir = 'images/model1'
    os.makedirs(output_dir, exist_ok=True)

    for exp_name, exps in flex_model["explanations"].items():
        for i, e in enumerate(tqdm(exps, desc=f'Generate heatmaps of {exp_name} explanations:', mininterval=2)):
            num_labels = e.num_labels

            fig_size = np.array([(num_labels + 1) * 0.15 * (num_labels + 1), 8])
            fig, ax = plt.subplots(nrows=2, ncols=(num_labels + 2) // 2, figsize=fig_size, squeeze=False)
            fig.suptitle(f'{exp_name}')

            # Primera imagen
            ax[0, 0].imshow(-e.data.squeeze(0).squeeze(0), cmap=plt.get_cmap("gray"))
            ax[0, 0].axis("off")
            ax[1, 0].axis("off")
            ax[0, 0].set_title(f'label: {e._label}\npred: {e.prediction}')

            cur_row = 0
            col_delay = 0 
            for j in range(num_labels):
                explanation_j = e.get_explanation(label=j)

                try:
                    max_value = np.max(np.abs(explanation_j))
                except Exception as ex:
                    print(f'Error in explanation {j}: {ex}')
                    continue

                if j == ((num_labels + 2) // 2 - 1):
                    cur_row += 1
                    col_delay = j

                # Segunda imagen
                ax[cur_row, j + 1 - col_delay].imshow(
                    -e.data.squeeze(0).squeeze(0), cmap=plt.get_cmap("gray"), alpha=0.15,
                    extent=(-1, explanation_j.shape[1], explanation_j.shape[0], -1)
                )
                im = ax[cur_row, j + 1 - col_delay].imshow(
                    explanation_j, cmap=red_transparent_blue, vmin=-max_value, vmax=max_value
                )
                fig.colorbar(im, ax=ax[cur_row, j + 1 - col_delay], orientation="horizontal")
                ax[cur_row, j + 1 - col_delay].axis("off")
                ax[cur_row, j + 1 - col_delay].set_title(f'{j}\n({e.probs[j] * 100:.2f}%)')

            fig.tight_layout()
            fig.subplots_adjust(hspace=0.5)

            # Guardar la figura en la carpeta
            output_path = os.path.join(output_dir, f'{exp_name}_heatmap_{i}.png')
            fig.savefig(output_path)
            plt.close(fig)  # Cerrar la figura para liberar memoria

'''