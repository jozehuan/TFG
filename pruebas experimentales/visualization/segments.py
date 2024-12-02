import numpy as np
import os
from tqdm import tqdm # progress bar

import matplotlib.pyplot as plt

def plot_segments(explainers, name, n_round : int = None):
    """ Generates and saves images of the segmentatios produced by the specified explainers.

    This function iterates through the explanatios of each explainer, generating images for 
    each explanation and saving them as PNG files in a designated directory.

    Parameters
    ----------
    explainers : dict
        A dictionary where keys are names of explainers and values are lists of explanation tuples.
        
    name : str
        The name of the system.
    """
    
    for exp_name, segment_list in explainers.items():
        
        output_dir = f"images/{name}/{exp_name}/segments"
        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception as e:
            raise RuntimeError(f"Failed to create directory {output_dir}: {e}")
            
        for i, segments in enumerate(tqdm(segment_list, desc=f'Generate segments visualization of {exp_name} explanations: ', mininterval=2)):
            fig_size = np.array([3.5 , 4])
            fig, ax =plt.subplots(nrows=1, ncols=2, figsize=fig_size, squeeze=False)
            fig.suptitle(f'{exp_name}')
            
            original_img, original_img_str = segments[0]
            ax[0,0].imshow(original_img, cmap=plt.get_cmap("gray"))
            ax[0,0].axis("off")
            ax[0,0].set_title(original_img_str)
            
            segment_img, segment_img_str = segments[1]
            from skimage.color import label2rgb
            overlay = label2rgb(segment_img, image=original_img, alpha=0.5, image_alpha=0.5, bg_label=0, kind='overlay')
            ax[0,1].imshow(overlay)
            ax[0,1].axis("off")
            ax[0,1].set_title(segment_img_str)
            
            fig.tight_layout()
            if n_round is None:
                output_path = os.path.join(output_dir, f'segmentation_{i}.png')
            else:
                output_path = os.path.join(output_dir, f'segmentation_{i}__round_{n_round}.png')
                
            fig.savefig(output_path)
            plt.close(fig)

