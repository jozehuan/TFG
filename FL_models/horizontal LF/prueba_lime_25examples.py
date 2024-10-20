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

dt_set_RGB = datasets.MNIST(root='mnist', train=True, download=True, transform=transform_colorize)
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
    
#     """
#     Red neuronal básica
#     """
    
#     def __init__(self, num_classes=10):
#         super().__init__()
#         self.flatten = nn.Flatten()
#         self.fc1 = nn.Linear(28 * 28, 128)
#         self.fc2 = nn.Linear(128, num_classes)

#     def forward(self, x):
#         x = self.flatten(x)
#         x = self.fc1(x)
#         x = nn.functional.relu(x)
#         x = self.fc2(x)
#         return nn.functional.log_softmax(x, dim=1)

class Net(nn.Module):
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
        
for i in range(2):
    #break
    img, label, pred = z_imgs[i], z_labels[i], z_pred[i]
    
    # Añadir un batch dimension (ya que el modelo espera batches de imágenes)
    img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    # Obtener la predicción con la mayor probabilidad y las probabilidades de cada clase
    _, pred_label = torch.max(pred, dim=0)
    pred_probs = pred.tolist()
    
    # Definir el explicador
    explainer = lime_image.LimeImageExplainer(verbose = False)
    #segmenter = SegmentationAlgorithm('quickshift', kernel_size=1, max_dist=5, ratio=0.001)
    segmenter = SegmentationAlgorithm('slic', n_segments=90, compactness=10, sigma=0.25)

    img____rgb = gray2rgb(img)
    explanation = explainer.explain_instance(gray2rgb(img), classifier_fn = predict_, 
                                             top_labels=10, hide_color=0, num_samples=1000, segmentation_fn=segmenter)
    
    
    # Obtener segmentos y cantidad
    segments = explanation.segments
    segments = np.squeeze(segments)
    n_segments = np.max(segments)

    #Obtener score de calidad de la explicación generada
    exp_fit = explanation.score
    
    img_rgb = gray2rgb(img[0])

    # Visualizar la imagen
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    ax[0].imshow(img[0], cmap='gray')
    ax[0].set_title(f"Label: {label} - Prediction: {pred_label} - Score: {pred_probs[pred_label]*100:.2f}%")
    ax[0].axis('off') 
    # Mostrar la imagen con los bordes de los superpíxeles resaltados
    ax[1].imshow(mark_boundaries(img_rgb, segments))
    ax[1].set_title(f'Superpixel Segmentation ({n_segments} segments)')
    ax[1].axis('off')  # Ocultar los ejes
    plt.tight_layout() 
    plt.show()
    
    temp, mask = explanation.get_image_and_mask(label, positive_only=True, num_features=15, hide_rest=False, min_weight = 0.01)
    temp, mask = np.squeeze(temp), np.squeeze(mask)
    fig, (ax1, ax2) = plt.subplots(1,2, figsize = (12, 6))
    ax1.imshow(label2rgb(mask,temp, bg_label = 0), interpolation = 'nearest')
    ax1.set_title('Positive Regions for {}'.format(label)) 
    ax1.axis('off')
    temp, mask = explanation.get_image_and_mask(label, positive_only=False, num_features=15, hide_rest=False, min_weight = 0.01)
    temp, mask = np.squeeze(temp), np.squeeze(mask)
    ax2.imshow(label2rgb(3-mask,temp, bg_label = 0), interpolation = 'nearest')
    ax2.set_title('Positive/Negative Regions for {}'.format(label))
    ax2.axis('off')
    plt.tight_layout()   
    plt.show()



    # positive for each class
    fig, m_axs = plt.subplots(2,5, figsize = (20,10))
    for i, c_ax in enumerate(m_axs.flatten()):
        temp, mask = explanation.get_image_and_mask(i, positive_only=True, num_features=15, hide_rest=False, min_weight = 0.01 )
        temp, mask = np.squeeze(temp), np.squeeze(mask)
        c_ax.imshow(label2rgb(mask,img[0], bg_label = 0), interpolation = 'nearest')
        c_ax.set_title(f'Positive for {i}\nScore {pred_probs[i]*100:.2f}%')
        c_ax.axis('off')


    # positive/negative for each class
    fig, m_axs = plt.subplots(2,5, figsize = (20,10))
    for i, c_ax in enumerate(m_axs.flatten()):
        temp, mask = explanation.get_image_and_mask(i, positive_only=False, num_features=15, hide_rest=False, min_weight = 0.01 )
        temp, mask = np.squeeze(temp), np.squeeze(mask)
        c_ax.imshow(label2rgb(3-mask,img[0], bg_label = 0), interpolation = 'nearest')
        c_ax.set_title(f'Pos/Neg for {i}\nScore {pred_probs[i]*100:.2f}%')
        c_ax.axis('off')




    # with bounds
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=15, hide_rest=False)
    temp, mask = np.squeeze(temp), np.squeeze(mask)
    img_boundry1 = mark_boundaries(temp, mask)

    temp2, mask2 = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=15, hide_rest=False)
    temp2, mask2 = np.squeeze(temp2), np.squeeze(mask2)
    img_boundry2 = mark_boundaries(temp2, -1*mask2)

    # Crear subgráficos para mostrar ambas imágenes
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))  # Ajusta el tamaño de la figura si es necesario

    # Mostrar la primera imagen con su título
    ax[0].imshow(img_boundry1)
    ax[0].set_title(f'Positive Regions\nExplanation Fit: {exp_fit:.5f}')
    ax[0].axis('off')  # Ocultar ejes

    # Mostrar la segunda imagen con su título
    ax[1].imshow(img_boundry2)
    ax[1].set_title(f'Positive/Negative Regions\nExplanation Fit: {exp_fit:.5f}')
    ax[1].axis('off')  # Ocultar ejes

    # Mostrar la figura
    plt.tight_layout()  # Ajusta el espacio entre subgráficos
    plt.show()





    # Mostrar los pesos de las 3 secciones más significativas (positivas)
    # Obtener los valores del explicador de la etiqueta
    local_exp_val = explanation.local_exp[explanation.top_labels[0]]
    sorted_local_exp_val = sorted(local_exp_val, key=lambda x: x[1], reverse=True)

    # Obtener la imagen y la máscara con las características positivas
    temp, mask = explanation.get_image_and_mask(label, positive_only=True, num_features=3, hide_rest=False, min_weight=0)
    temp, mask = np.squeeze(temp), np.squeeze(mask)
    unique_mask = mask * segments

    sorted_local_exp_val = sorted_local_exp_val[0:3]
    legend_label = []
    contador_aux = 1
    for i, v in sorted_local_exp_val:
        unique_mask[unique_mask == i] = contador_aux
        legend_label.append(f'{v:.4f}')
        contador_aux += 1

    colors_ = ['red', 'green', 'yellow']

    # Mostrar la imagen original con las regiones coloreadas
    plt.figure(figsize=(20, 10))
    plt.imshow(label2rgb(unique_mask, temp, bg_label=0, colors=colors_), interpolation='nearest')
    plt.title(f'Weights of positive regions for {label}', fontsize ='16')
    plt.axis('off')

    # Añadir leyenda
    # Crear líneas para la leyenda
    lines = [plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=color, markersize=10) for color in colors_]
    # Ajustar el espacio entre la imagen y la leyenda
    plt.subplots_adjust(bottom=0.05)
    # Añadir la leyenda fuera de la imagen
    plt.legend(lines, legend_label, loc='upper center', bbox_to_anchor=(0.5, -0.1), fontsize ='16', ncol=3, frameon=False)

    plt.show()

print('empieza SP')
import mySP
explainer = lime_image.LimeImageExplainer(verbose = False)
sp_obj = mySP.Image_SubmodularPick(explainer, z_imgs, predict_, sample_size=100, num_exps_desired=10, top_labels=10)

sp_v = sp_obj.V
sp_exps = sp_obj.sp_explanations

for explanation in sp_exps:
    # Obtener segmentos y cantidad
    segments = np.squeeze(explanation.segments)
    n_segments = np.max(segments)

    #Obtener score de calidad de la explicación generada
    exp_fit = explanation.score
    
    img_rgb = np.squeeze(explanation.image)
    img = np.squeeze(rgb2gray(explanation.image))
    label = pred_label = explanation.top_labels[0]

    # Visualizar la imagen
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    ax[0].imshow(img, cmap='gray')
    ax[0].set_title(f"Label: {label} - Prediction: {pred_label}")
    ax[0].axis('off') 
    # Mostrar la imagen con los bordes de los superpíxeles resaltados
    ax[1].imshow(mark_boundaries(img_rgb, segments))
    ax[1].set_title(f'Superpixel Segmentation ({n_segments} segments)')
    ax[1].axis('off')  # Ocultar los ejes
    plt.tight_layout() 
    plt.show()
    
    temp, mask = explanation.get_image_and_mask(label, positive_only=True, num_features=1000, hide_rest=False, min_weight = 0)
    temp, mask = np.squeeze(temp), np.squeeze(mask)
    fig, (ax1, ax2) = plt.subplots(1,2, figsize = (12, 6))
    ax1.imshow(label2rgb(mask,temp, bg_label = 0), interpolation = 'nearest')
    ax1.set_title('Positive Regions for {}'.format(label)) 
    ax1.axis('off')
    temp, mask = explanation.get_image_and_mask(label, positive_only=False, num_features=1000, hide_rest=False, min_weight = 0)
    temp, mask = np.squeeze(temp), np.squeeze(mask)
    ax2.imshow(label2rgb(3-mask,temp, bg_label = 0), interpolation = 'nearest')
    ax2.set_title('Positive/Negative Regions for {}'.format(label))
    ax2.axis('off')
    plt.tight_layout()   
    plt.show()



    # positive for each class
    fig, m_axs = plt.subplots(2,5, figsize = (20,10))
    for i, c_ax in enumerate(m_axs.flatten()):
        temp, mask = explanation.get_image_and_mask(i, positive_only=True, num_features=1000, hide_rest=False, min_weight = 0)
        temp, mask = np.squeeze(temp), np.squeeze(mask)
        c_ax.imshow(label2rgb(mask,img, bg_label = 0), interpolation = 'nearest')
        c_ax.set_title(f'Positive for {i}')
        c_ax.axis('off')


    # positive/negative for each class
    fig, m_axs = plt.subplots(2,5, figsize = (20,10))
    for i, c_ax in enumerate(m_axs.flatten()):
        temp, mask = explanation.get_image_and_mask(i, positive_only=False, num_features=1000, hide_rest=False, min_weight = 0 )
        temp, mask = np.squeeze(temp), np.squeeze(mask)
        c_ax.imshow(label2rgb(3-mask,img, bg_label = 0), interpolation = 'nearest')
        c_ax.set_title(f'Pos/Neg for {i}')
        c_ax.axis('off')




    # with bounds
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=15, min_weight = 0 , hide_rest=False)
    temp, mask = np.squeeze(temp), np.squeeze(mask)
    img_boundry1 = mark_boundaries(temp, mask)

    temp2, mask2 = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=15, min_weight = 0 , hide_rest=False)
    temp2, mask2 = np.squeeze(temp), np.squeeze(mask)
    img_boundry2 = mark_boundaries(temp2, -1*mask2)

    # Crear subgráficos para mostrar ambas imágenes
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))  # Ajusta el tamaño de la figura si es necesario

    # Mostrar la primera imagen con su título
    ax[0].imshow(img_boundry1)
    ax[0].set_title(f'Positive Regions\nExplanation Fit: {exp_fit:.5f}')
    ax[0].axis('off')  # Ocultar ejes

    # Mostrar la segunda imagen con su título
    ax[1].imshow(img_boundry2)
    ax[1].set_title(f'Positive/Negative Regions\nExplanation Fit: {exp_fit:.5f}')
    ax[1].axis('off')  # Ocultar ejes

    # Mostrar la figura
    plt.tight_layout()  # Ajusta el espacio entre subgráficos
    plt.show()





    # Mostrar los pesos de las 3 secciones más significativas (positivas)
    # Obtener los valores del explicador de la etiqueta
    local_exp_val = explanation.local_exp[explanation.top_labels[0]]
    sorted_local_exp_val = sorted(local_exp_val, key=lambda x: x[1], reverse=True)

    # Obtener la imagen y la máscara con las características positivas
    temp, mask = explanation.get_image_and_mask(label, positive_only=True, num_features=3, hide_rest=False, min_weight=0)
    temp, mask = np.squeeze(temp), np.squeeze(mask)
    unique_mask = mask * segments

    sorted_local_exp_val = sorted_local_exp_val[0:3]
    legend_label = []
    contador_aux = 1
    for i, v in sorted_local_exp_val:
        unique_mask[unique_mask == i] = contador_aux
        legend_label.append(f'{v:.4f}')
        contador_aux += 1

    colors_ = ['red', 'green', 'yellow']

    # Mostrar la imagen original con las regiones coloreadas
    plt.figure(figsize=(20, 10))
    plt.imshow(label2rgb(unique_mask, temp, bg_label=0, colors=colors_), interpolation='nearest')
    plt.title(f'Weights of positive regions for {label}', fontsize ='16')
    plt.axis('off')

    # Añadir leyenda
    # Crear líneas para la leyenda
    lines = [plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=color, markersize=10) for color in colors_]
    # Ajustar el espacio entre la imagen y la leyenda
    plt.subplots_adjust(bottom=0.05)
    # Añadir la leyenda fuera de la imagen
    plt.legend(lines, legend_label, loc='upper center', bbox_to_anchor=(0.5, -0.1), fontsize ='16', ncol=3, frameon=False)

    plt.show()