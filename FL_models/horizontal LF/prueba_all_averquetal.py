import numpy as np

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


# Función para transormar la imagen en RGB
def colorize_img(img):
    img = transforms.ToPILImage()(img)  # Convertir el tensor a imagen PIL
    img_colored = ImageOps.colorize(img.convert("L"), black="black", white="white")  # Colorear
    return transforms.ToTensor()(img_colored)  # Convertir de nuevo a tensor

transform_colorize = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(colorize_img)
])

transform_dflt = transforms.Compose([
    transforms.ToTensor(),  # Convierte las imágenes a tensores
    #transforms.Normalize((0.5,), (0.5,)), # Normaliza con la media y desviación estándar
    #transforms.Lambda(lambda x: x.numpy().squeeze())
])

dt_set_GRAY = datasets.MNIST(root='mnist', train=True, download=True, transform=transform_dflt)

lengths = [1000, 5000, 54000]
subset1, subset2, sx3 = random_split(dt_set_GRAY, lengths)


cl_dataloader = DataLoader(subset2, batch_size=20)

# # Mostrar una imagen del dataset coloreada con su etiqueta
# img, label = dt_set_RGB[0]

# # Visualizar la imagen con su título correspondiente (la etiqueta)
# plt.imshow(img.permute(1, 2, 0))  # Cambiar las dimensiones para visualizar correctamente
# plt.title(f"Label: {label}")
# plt.show()


def predict_(img):
    model.eval()
    
    img_g_tensor = torch.tensor(rgb2gray(img)).unsqueeze(1)
    
    with torch.no_grad():  # Desactivar el cálculo de gradientes para la predicción
        preds = model(img_g_tensor.to('cpu')) 

    return preds

class Net(nn.Module):
    
    """
    Red neuronal básica
    """
    
    def __init__(self, num_classes=10):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=1)
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        return nn.functional.log_softmax(x, dim=1)
    
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

# CNN ------------------------------------------------------------------------------

model = Net_CNN()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

model = model.train()
model = model.to('cpu')

zk_img = None 

for _ in range(5):
    for imgs, labels in cl_dataloader:
        imgs, labels = imgs.to('cpu'), labels.to('cpu')
        optimizer.zero_grad()
         
        pred = model(imgs)
        zk_img = imgs
        loss = criterion(pred, labels)
        loss.backward()
        optimizer.step()


model.eval()

z_imgs = []; z_labels = []; z_pred = []

#Predecir las imágenes del test
test_dataloader = DataLoader(subset1, batch_size=20000)
for imgs, labels in test_dataloader:
    imgs, labels = imgs.to('cpu'), labels.to('cpu')
    
    with torch.no_grad():
        pred_ = model(imgs)
        
    pred_labels = torch.argmax(pred_, dim=1)
    z_pred.extend(pred_)
    z_labels.extend(labels.tolist())
    z_imgs.extend(imgs.tolist())

zzzq_img_rgb = gray2rgb(z_imgs[0]) # PRueba para ver tipo y dimensiones, no hace nada más

baseline_dist = torch.randn(10, 28, 28) * 0.001 + 0.5

for i in range(10): 
    image = torch.tensor(z_imgs[i]).unsqueeze(0)
    print(image.shape)
    image_ = image.squeeze().detach().numpy()  # Quitar la dimensión adicional (1) para convertir a (28, 28)
    image_rgb = gray2rgb(image_)
    
    _, pred_label = torch.max(z_pred[i], dim=0)
    pred_probs = torch.exp(z_pred[i])  
    
    from skimage.segmentation import slic
    segm = slic(np.expand_dims(image_, axis=0), n_segments=100, compactness=0.1, sigma=0.5, channel_axis=0) - 1
    segm_ = torch.tensor(segm).unsqueeze(0).unsqueeze(0)
    
    attr_ = []
    attr_g_ = []
    attr_k_ = []
    
    attr_l_ = []
    
    explainer = lime_image.LimeImageExplainer(verbose = False)
    segmenter = SegmentationAlgorithm('slic', n_segments=100, compactness=1, sigma=1)
    explanation = explainer.explain_instance(image_rgb, classifier_fn = predict_, 
                                             top_labels=10, hide_color=0, num_samples=10000, segmentation_fn=segmenter)
    segments = explanation.segments
    
    for j in range(10):
        dl_shap = DeepLiftShap(model)
        attr, d = dl_shap.attribute(image.to('cpu'), baselines=imgs, target=j, return_convergence_delta=True)
        attr_.append(attr.squeeze().detach().numpy())    # Igual, convertir a (28, 28)
        
        gr_shap = GradientShap(model)
        attr_g, d_g = gr_shap.attribute(image.to('cpu'), baselines=imgs, target=j, return_convergence_delta=True)
        attr_g_.append(attr_g.squeeze().detach().numpy())
        
        kr_shap = KernelShap(model) 
        attr_k = kr_shap.attribute(image.to('cpu'), baselines=image.to('cpu'), feature_mask=segm_, target=j, n_samples=2000, perturbations_per_eval=50)
        attr_k_.append(attr_k.squeeze().detach().numpy())
        
        attr_l = np.vectorize(dict(explanation.local_exp[j]).get)(segments)
        attr_l_.append(attr_l)
    
 
    plot_image(image_, z_labels[i], pred_label, pred_probs, attr_, red_transparent_blue)
    plot_image(image_, z_labels[i], pred_label, pred_probs, attr_g_, red_transparent_blue)
    plot_image(image_, z_labels[i], pred_label, pred_probs, attr_k_, red_transparent_blue)
    plot_image(image_, z_labels[i], pred_label, pred_probs, attr_l_, red_transparent_blue, vmax=0.2)
    
    

    
    
    
    
