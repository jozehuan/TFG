import numpy as np
import os

import matplotlib.pyplot as plt

def barplot(class_counter, name, ncols : int = 4):
    """ Generates and saves barpolts for the data distribution on the federal system.

    Parameters
    ----------
    class_counter: list
        A list of tuples, where each tuple contains the node ID, the count of samples 
        for each class, and the total number of samples in the node.
    name : str
        The name of the system.
    ncols : int, optional
        Number of colums of the image. Default: 4
    """
    output_dir = f'images/{name}'
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        raise RuntimeError(f"Failed to create directory {output_dir}: {e}")
        
        
    num_classes = len(class_counter)
    colors_ = plt.cm.viridis(np.linspace(0,1,num_classes))
    
    nrows = (num_classes+ncols-1)//ncols
    fig, axs = plt.subplots(nrows, ncols, figsize=(6*ncols, 4 * nrows))
    axs = axs.flatten()
    
    for k, dict_counter, total in class_counter:
        axs[k].bar(dict_counter.keys(), dict_counter.values(), color=colors_)
        axs[k].set_xlabel('classes', fontsize=16)
        if k%ncols == 0:
            axs[k].set_ylabel('count', fontsize=16)
        axs[k].set_title(f'node {k} (total {total})', fontsize=20)
        
        v_max = max(dict_counter.values())
        v_min = min(dict_counter.values())
        axs[k].set_ylim(max(0, v_min - (v_max - v_min) / 2))
        
        axs[k].set_xticks(range(min(dict_counter.keys()), max(dict_counter.keys()) + 1))
    
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.5)
    
    output_path = os.path.join(output_dir, 'class_counter.png')
    fig.savefig(output_path)
    plt.close(fig)
    