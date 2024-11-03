import numpy as np
import os
from tqdm import tqdm # progress bar

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

colors = []
for j in np.linspace(1, 0, 100):
    colors.append((30.0 / 255, 136.0 / 255, 229.0 / 255, j))
for j in np.linspace(0, 1, 100):
    colors.append((255.0 / 255, 13.0 / 255, 87.0 / 255, j))
red_transparent_blue = LinearSegmentedColormap.from_list("red_transparent_blue", colors)

def plot_heatmaps(explanations, name):
    for exp_name, exp_list in explanations.items():
        output_dir = f"images/{name}/{exp_name}/heatmaps"
        
        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception as e:
            raise RuntimeError(f"Failed to create directory {output_dir}: {e}")
        
        
        for i, exp in enumerate(tqdm(exp_list, desc=f'Generate heatmaps of {exp_name} explanations: ', mininterval=2)):
            len_exp = len(exp)
             
            fig_size = np.array([len_exp * 0.15 * len_exp, 8])
            fig, ax =plt.subplots(nrows=2, ncols=(len_exp + 1)//2, figsize=fig_size, squeeze=False)
            fig.suptitle(f'{exp_name}')
            
            original_img, original_img_str = exp[-1]
            
            ax[0,0].imshow(original_img, cmap=plt.get_cmap("gray"))
            ax[0,0].axis("off");  ax[1, 0].axis("off")
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
                        
                if j == ((len_exp + 1)//2 - 1):
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
            
            output_path = os.path.join(output_dir, f'heatmap_{i}.png')
            fig.savefig(output_path)
            plt.close(fig)