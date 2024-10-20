import numpy as np

import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from PIL import ImageOps

import matplotlib.pyplot as plt

from skimage.color import gray2rgb, rgb2gray, label2rgb

from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
from skimage.segmentation import mark_boundaries

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

lengths = [30000, 30000]
subset1, subset2 = random_split(dt_set_GRAY, lengths)


cl_dataloader = DataLoader(subset2, batch_size=20)

# # Mostrar una imagen del dataset coloreada con su etiqueta
# img, label = dt_set_RGB[0]

# # Visualizar la imagen con su título correspondiente (la etiqueta)
# plt.imshow(img.permute(1, 2, 0))  # Cambiar las dimensiones para visualizar correctamente
# plt.title(f"Label: {label}")
# plt.show()


def predict_(color_img):
    """Predictor:
            color_img - imagen en formato '(28, 28, 3)'
            (el predictor se encarga de dar formato correcto a la imagen antes de predecir) 
    """
    
    img_g = rgb2gray(color_img)
    img_g_tensor = torch.tensor(img_g, dtype=torch.float32)
    
    #plt.imshow(img_g_tensor[0], cmap='gray')
    #plt.title("dentro de _predict")
    #plt.show()
    
    with torch.no_grad():  # Desactivar el cálculo de gradientes para la predicción
        preds = model(img_g_tensor.to('cpu'))  # Enviar la imagen al mismo dispositivo que el modelo (CPU en este caso)

    return preds

# class Net(nn.Module):
#     def __init__(self):
#         super().__init__()

#         self.conv_layers = nn.Sequential(
#             nn.Conv2d(1, 10, kernel_size=5),
#             nn.MaxPool2d(2),
#             nn.ReLU(),
#             nn.Conv2d(10, 20, kernel_size=5),
#             nn.Dropout(),
#             nn.MaxPool2d(2),
#             nn.ReLU(),
#         )
#         self.fc_layers = nn.Sequential(
#             nn.Linear(320, 50),
#             nn.ReLU(),
#             nn.Dropout(),
#             nn.Linear(50, 10),
#             nn.Softmax(dim=1),
#         )

#     def forward(self, x):
#         x = self.conv_layers(x)
#         x = x.view(-1, 320)
#         x = self.fc_layers(x)
#         return x

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 256)  # Primera capa oculta con 256 neuronas
        self.fc2 = nn.Linear(256, 128)       # Segunda capa oculta con 128 neuronas
        self.fc3 = nn.Linear(128, 10)  # Capa de salida con 10 clases (dígitos)

    def forward(self, x):
        # Aplana la entrada (imágenes de 1x28x28 a 784)
        x = x.view(x.size(0), -1)  # Cambia la forma a (batch_size, 784)
        x = torch.relu(self.fc1(x))  # Activa la primera capa
        x = torch.relu(self.fc2(x))  # Activa la segunda capa
        x = self.fc3(x)               # Salida (logits)
        return x
    
model = Net()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

model = model.train()
model = model.to('cpu')

for _ in range(5):
    for imgs, labels in cl_dataloader:
        imgs, labels = imgs.to('cpu'), labels.to('cpu')
        optimizer.zero_grad()
         
        pred = model(imgs)
        loss = criterion(pred, labels)
        loss.backward()
        optimizer.step()


model.eval()

z_imgs = []; z_labels = []; z_pred = []

#Predecir las imágenes del test
test_dataloader = DataLoader(subset1, batch_size=20)
for imgs, labels in test_dataloader:
    imgs, labels = imgs.to('cpu'), labels.to('cpu')
    
    with torch.no_grad():
        pred_ = model(imgs)
        
    pred_labels = torch.argmax(pred_, dim=1)
    z_pred.extend(pred_)
    z_labels.extend(labels.tolist())
    z_imgs.extend(imgs.tolist())

zzzq_img_rgb = gray2rgb(z_imgs[0]) # PRueba para ver tipo y dimensiones, no hace nada más

import shap


images, _ = zip(*[dt_set_GRAY[i] for i in range(500)])  # Ignorar las etiquetas con _
# Convertir la lista de imágenes a un tensor
images_tensor = torch.stack(images)  # (403, 1, 28, 28)

background = images_tensor[:400]  # Primeras 400 imágenes
a_to_explain = images_tensor[400:405]  # Las siguientes 3 imágenes


exp_shap = shap.DeepExplainer(model, background)

shap_values = exp_shap.shap_values(a_to_explain)

shap_numpy = list(np.transpose(shap_values, (4, 0, 2, 3, 1)))
test_numpy = np.swapaxes(np.swapaxes(a_to_explain.numpy(), 1, -1), 1, 2)

# Plotear los valores SHAP
shap.image_plot(shap_numpy, -test_numpy)