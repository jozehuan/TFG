import numpy as np
import flex

import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from PIL import ImageOps

torch.manual_seed(42)

from captum.attr import DeepLiftShap, GradientShap, KernelShap

import matplotlib.pyplot as plt

from skimage.color import gray2rgb, rgb2gray, label2rgb

from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
from skimage.segmentation import mark_boundaries

from explanation import Explanation

from matplotlib.colors import LinearSegmentedColormap
colors = []
for j in np.linspace(1, 0, 100):
    colors.append((30.0 / 255, 136.0 / 255, 229.0 / 255, j))
for j in np.linspace(0, 1, 100):
    colors.append((255.0 / 255, 13.0 / 255, 87.0 / 255, j))
red_transparent_blue = LinearSegmentedColormap.from_list("red_transparent_blue", colors)


def plot_image(image, label, pred_label, pred_probs, attr_, cmap_, vmax = None):
    t= f'Label: {label}, Prediction: {pred_label}'
    
    fil , col = image.shape
    
    fig_size = np.array([col * 0.11 * (len(attr_) + 1), fil * 0.12 * (2 + 1)])
    fig, axes =plt.subplots(nrows=2, ncols=(len(attr_) + 1), figsize=fig_size, squeeze=False)
    
    axes[0, 0].imshow(-image, cmap=plt.get_cmap("gray"))
    axes[0, 0].set_title(t)
    axes[0, 0].axis("off"); axes[1, 0].axis("off")
    
    if vmax is None:
        max_vals = [np.max(np.abs(attr_[k])) for k in range(len(attr_))]
        max_val = np.max(max_vals)
    else:
        max_vals = [vmax for k in range(len(attr_))]
        max_val = vmax
    
    for j in range(len(attr_)):
        axes[0, j+1].imshow(-image, cmap=plt.get_cmap("gray"), alpha=0.15, extent=(-1, attr_[j].shape[1], attr_[j].shape[0], -1))
        im = axes[0, j+1].imshow(attr_[j], cmap=cmap_, vmin=-max_val, vmax=max_val)
        tit = f'{j} ({pred_probs[j]*100:.3f}%)'
        axes[0, j+1].set_title(tit)
        axes[0, j+1].axis("off")
        
    cbar_ax = fig.add_axes([0.125, 0.54, 0.775, 0.02])  # Ajusta la posición [left, bottom, width, height]
    cb = fig.colorbar(im, cax=cbar_ax, orientation="horizontal", label="SHAP value")
    cb.outline.set_visible(False)
    
    
    for j in range(len(attr_)):
        axes[1, j+1].imshow(-image, cmap=plt.get_cmap("gray"), alpha=0.15, extent=(-1, attr_[j].shape[1], attr_[j].shape[0], -1))
        im = axes[1, j+1].imshow(attr_[j], cmap=cmap_, vmin=-max_vals[j], vmax=max_vals[j]) #, vmin=-max_val, vmax=max_val)
        fig.colorbar(im, ax=axes[1, j+1], orientation="horizontal")
        tit = f'{j} ({pred_probs[j]*100:.3f}%)'
        axes[1, j+1].axis("off")
    
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.5)
    #cb = fig.colorbar(
    #    im, ax=np.ravel(axes).tolist(), label="SHAP value", orientation="horizontal", aspect=fig_size[0] / 0.2)
    #cb.outline.set_visible(False)
    plt.show()
    
    

transform_dflt = transforms.Compose([
    transforms.ToTensor(),  # Convierte las imágenes a tensores
    #transforms.Normalize((0.5,), (0.5,)), # Normaliza con la media y desviación estándar
])

dt_set_GRAY = datasets.MNIST(root='mnist', train=True, download=True, transform=transform_dflt)

lengths = [1000, 15000, 44000]
subset1, subset2, subset3 = random_split(dt_set_GRAY, lengths)

dataset = flex.data.Dataset.from_torchvision_dataset(dt_set_GRAY)
dataset1 = flex.data.Dataset.from_torchvision_dataset(subset1)
dataset2 = flex.data.Dataset.from_torchvision_dataset(subset2)

class Net_CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=5),
            nn.Dropout(),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(320, 50),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(50, 10),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, 320)
        x = self.fc_layers(x)
        return x



model = Net_CNN()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

model = model.train()
model = model.to('cpu')

train_dataset = dataset2.to_torchvision_dataset()
train_dataloader = DataLoader(train_dataset, batch_size=len(train_dataset.data))
for _ in range(10):
    for imgs, labels in train_dataloader:
        imgs, labels = imgs.to('cpu'), labels.to('cpu')
        optimizer.zero_grad()
         
        pred = model(imgs)
        loss = criterion(pred, labels)
        loss.backward()
        optimizer.step()

test_dataset = dataset1.to_torchvision_dataset()
test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset.data))

# LIME EXP:
    
explainer = lime_image.LimeImageExplainer(verbose = False)
segmenter = SegmentationAlgorithm('slic', n_segments=100, compactness=0.1, sigma=0.3)
lime_dict = {'top_labels': 10, 'hide_color': 0, 'num_samples': 1000, 'segmentation_fn':segmenter}

for imgs, labels in test_dataloader:
    for j in range(1):#range(int(labels.shape[0])):
        img_ = imgs[j]
        
        img_color = gray2rgb(img_.squeeze().numpy())
        img_return = torch.from_numpy(rgb2gray(img_color)).unsqueeze(0)
        
        e = Explanation(model = model, exp = explainer, data_to_explain = img_, **lime_dict)
        sol = e.get_explanation(int(labels[j]))
        
        vmax = np.nanmax(np.abs(sol))
        
        plt.imshow(-img_.squeeze().numpy(), cmap=plt.get_cmap("gray"), alpha=0.15)
        plt.imshow(sol, cmap=red_transparent_blue, vmin=-vmax, vmax=vmax)
        plt.title(f'{int(labels[j])}')
        plt.show()
        
        exp_deep_shap = DeepLiftShap(model)
        e2 = Explanation(model = model, exp = exp_deep_shap, data_to_explain = img_, baselines=imgs)
        soll = e2.get_explanation(int(labels[j]))
        vmax = np.nanmax(np.abs(soll))
        
        plt.imshow(-img_.squeeze().numpy(), cmap=plt.get_cmap("gray"), alpha=0.15)
        plt.imshow(soll, cmap=red_transparent_blue, vmin=-vmax, vmax=vmax)
        plt.title(f'{int(labels[j])}')
        plt.show()