import numpy as np
import os
import re
from tqdm import tqdm # progress bar

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

colors = []
for j in np.linspace(1, 0, 100):
    colors.append((30.0 / 255, 136.0 / 255, 229.0 / 255, j))
for j in np.linspace(0, 1, 100):
    colors.append((255.0 / 255, 13.0 / 255, 87.0 / 255, j))
red_transparent_blue = LinearSegmentedColormap.from_list("red_transparent_blue", colors)

colors = []
for j in np.linspace(0, 1, 100):
    colors.append((255.0 / 255, 13.0 / 255, 87.0 / 255, j))
red_transparent = LinearSegmentedColormap.from_list("red_transparent", colors)

def plot_heatmaps(explainers, name, n_round : int = None):
    """ Generates and saves heatmaps for the explanations produced by the specified explainers.

    This function iterates through the explanatios of each explainer, generating heatmaps for 
    each explanation and saving them as PNG files in a designated directory.

    Parameters
    ----------
    explainers : dict
        A dictionary where keys are names of explainers and values are lists of explanation tuples.
        
    name : str
        The name of the system.
    """
    
    for exp_name, exp_list in explainers.items():
        for i, exp in enumerate(tqdm(exp_list, desc=f'Generate heatmaps of {exp_name} explanations: ', mininterval=2)):
            len_exp = len(exp)
            nrows = (len_exp-1)//6 + 1
            
            if i == 0:
                output_dir = f"images/{name}/{exp_name}/heatmaps_{len_exp-1}"
                
                try:
                    os.makedirs(output_dir, exist_ok=True)
                except Exception as e:
                    raise RuntimeError(f"Failed to create directory {output_dir}: {e}")
                
             
            fig_size = np.array([len_exp * 1.7 , 4*nrows])
            fig, ax =plt.subplots(nrows=nrows, ncols=(len_exp+(nrows-1))//nrows, figsize=fig_size, squeeze=False)
            fig.suptitle(f'{exp_name}')
            
            original_img, original_img_str = exp[-1]
            
            ax[0,0].imshow(original_img, cmap=plt.get_cmap("gray"))
            for k in range(nrows):
                ax[k,0].axis("off")
            ax[0,0].set_title(original_img_str)
            
            cur_row = 0
            col_delay = 0
            for j in range(len_exp-1):
                explanation_j, explanation_j_str = exp[j]
                
                try:
                    max_value = np.max(np.abs(explanation_j))
                except:
                    print(f'explanation: {explanation_j}')
                    continue
                        
                if j == ((len_exp + 1)//nrows - 1):
                    cur_row += 1
                    col_delay = j
                
                ax[cur_row, j+1-col_delay].imshow(original_img, cmap=plt.get_cmap("gray"), alpha=0.15, 
                                                  extent=(-1, explanation_j.shape[1], explanation_j.shape[0], -1))
                        
                im = ax[cur_row, j+1-col_delay].imshow(explanation_j, cmap=red_transparent_blue, 
                                                       vmin=-max_value, vmax=max_value)
                fig.colorbar(im, ax=ax[cur_row, j+1-col_delay], orientation="horizontal")
                ax[cur_row, j+1-col_delay].axis("off")
                ax[cur_row, j+1-col_delay].set_title(explanation_j_str)
                
            fig.tight_layout()
            fig.subplots_adjust(hspace=0.5)
            
            if n_round is None:
                output_path = os.path.join(output_dir, f'heatmap_{i}.png')
            else:
                output_path = os.path.join(output_dir, f'heatmap_{i}__round_{n_round}.png')
            fig.savefig(output_path)
            plt.close(fig)

def plot_heatmaps_compared(fed_exps, central_exps, metrics):
    """ Generates and saves compared heatmaps for federated and central explanations.
    
    This function compares heatmaps generated from federated and central models
    for a set of explanations, saving the results as PNG files.
    
    Parameters
    ----------
    fed_exps : tuple
        A tuple containing a dictionary where keys are names of explainers
          and values are lists of explanation tuples from federated learning; 
          and the name of the federated system.
    
    central_exps : dict
        A dictionary where keys are names of explainers and values are lists of
        explanation tuples from the centralized model.
    
    metrics : dict
        A dictionary where keys correspond to explanation names and values are
        dictionaries containing evaluation metrics (e.g., MSE, SSIM) for each explanation.
    """
    
    fed_explainers , name = fed_exps
    
    for exp_name, fed_data, central_data in zip(fed_explainers.keys(), fed_explainers.values(), central_exps.values()):
        
        exp_metrics = metrics[exp_name]
        
        output_dir = f"images/{name}/{exp_name}/heatmaps_compared"
        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception as e:
            raise RuntimeError(f"Failed to create directory {output_dir}: {e}")
            
        for i, (fed_d, central_d) in tqdm(enumerate(zip(fed_data, central_data)), desc=f'Generate heatmaps (compared) of {exp_name} explanations: ', mininterval=2):
            
                
            fig_size = np.array([3 * 1.7 , 6])
            fig, ax =plt.subplots(nrows=2, ncols=3, figsize=fig_size, squeeze=False)
            
            original_img, original_img_str = fed_d[-1]
            ax[0,0].imshow(original_img, cmap=plt.get_cmap("gray"))
            ax[0,0].axis("off"); ax[1,0].axis("off")
            ax[0,0].set_title(original_img_str)
            
            label = re.search(r"label:\s*(\d+)", original_img_str)
            label = int(label.group(1))
            
            fed_img, fed_img_str = fed_d[label]
            fed_max_value = np.max(np.abs(fed_img))
            ax[0,1].imshow(original_img, cmap=plt.get_cmap("gray"), alpha=0.15, 
                                              extent=(-1, fed_img.shape[1], fed_img.shape[0], -1))
            im = ax[0,1].imshow(fed_img, cmap=red_transparent_blue, 
                                                   vmin=-fed_max_value, vmax=fed_max_value)
            fig.colorbar(im, ax=ax[0,1], orientation="horizontal")
            ax[0,1].axis("off")
            ax[0,1].set_title(f'{name}: ' + fed_img_str)
            
            
            central_img, central_img_str = central_d[label]
            central_max_value = np.max(np.abs(central_img))
            ax[0,2].imshow(original_img, cmap=plt.get_cmap("gray"), alpha=0.15, 
                                              extent=(-1, central_img.shape[1], central_img.shape[0], -1))
            im = ax[0,2].imshow(central_img, cmap=red_transparent_blue, 
                                                   vmin=-central_max_value, vmax=central_max_value)
            fig.colorbar(im, ax=ax[0,2], orientation="horizontal")
            ax[0,2].axis("off")
            ax[0,2].set_title('central: ' + central_img_str)
            
            percent = 90
            flat_fed_img = fed_img.flatten()
            flat_central_img = central_img.flatten()
            
            fed_img_above = fed_img >= np.percentile(flat_fed_img, percent)
            central_img_above = central_img >= np.percentile(flat_central_img, percent)
            
            ax[1,1].imshow(original_img, cmap=plt.get_cmap("gray"), alpha=0.15, 
                                              extent=(-1, fed_img_above.shape[1], fed_img_above.shape[0], -1))
            ax[1][1].imshow(fed_img_above, cmap=red_transparent)
            ax[1][1].axis('off')  
            
            ax[1,2].imshow(original_img, cmap=plt.get_cmap("gray"), alpha=0.15, 
                                              extent=(-1, central_img_above.shape[1], central_img_above.shape[0], -1))
            ax[1][2].imshow(central_img_above, cmap=red_transparent)
            ax[1][2].axis('off') 
            
            cosine_metric = exp_metrics[i]['COSINE']
            pos_iou_metric, neg_iou_metric = exp_metrics[i]['IoU']
            mse_metric = exp_metrics[i]['MSE']
            ssim_metric = exp_metrics[i]['SSIM'] 
            
            fig.suptitle(f'({exp_name})\n MSE: {mse_metric:.4f}  cos.distance: {cosine_metric:.4f}\nSSIM: {ssim_metric:.4f}  IoU: {pos_iou_metric:.4f}, {neg_iou_metric:.4f}')
            fig.tight_layout()
            #fig.subplots_adjust(hspace=0.5)
            
            output_path = os.path.join(output_dir, f'heatmap_compared_{i}.png')
            fig.savefig(output_path)
            plt.close(fig)
