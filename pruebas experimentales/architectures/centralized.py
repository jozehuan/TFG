import numpy as np
import os
from functools import partial

import matplotlib.pyplot as plt
from tqdm import tqdm # progress bar

import torch
from torch.utils.data import DataLoader
from skimage.color import gray2rgb
from skimage.segmentation import slic

import captum.attr as attr

from flex.pool import predict_

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from matplotlib.colors import LinearSegmentedColormap
colors = []
for j in np.linspace(1, 0, 100):
    colors.append((30.0 / 255, 136.0 / 255, 229.0 / 255, j))
for j in np.linspace(0, 1, 100):
    colors.append((255.0 / 255, 13.0 / 255, 87.0 / 255, j))
red_transparent_blue = LinearSegmentedColormap.from_list("red_transparent_blue", colors)


class CENTRAL_System:
    
    def __init__(self,  model, criterion, optimizer_func, opt_kwargs, explainers, train_data, test_data):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer_func(self.model.parameters(), **opt_kwargs)
        
        self.data_test = test_data
        self.data_train = train_data
        self.clases = set()
        
        self.explainers = explainers
        self.explanations = {}
    
    def train(self, n_rounds: int = 10):
        
        train_dataloaders = []
        for data in self.data_train:
            train_dataloaders.append(DataLoader(data, batch_size=20))
        
        test_dataloader = DataLoader(self.data_test, batch_size=20, shuffle=True, pin_memory=False)
        
        # Mueve el modelo al dispositivo
        
        self.model = self.model.train().to(device)
        
        for i in range(n_rounds):
            for train_dataloader in train_dataloaders:
                for imgs, labels in train_dataloader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    self.optimizer.zero_grad()
                    
                    pred = self.model(imgs)
                    loss = self.criterion(pred, labels)
                    loss.backward()
                    self.optimizer.step()
                    
                    self.clases.update(labels.tolist())
                
            # Evaluación
            self.model.eval()
            test_loss = 0
            test_acc = 0
            total_count = 0
            losses = []
            with torch.no_grad():
                for data, target in test_dataloader:
                    total_count += target.size(0)
                    data, target = data.to(device), target.to(device)
                    
                    output = self.model(data)
                    losses.append(self.criterion(output, target).item())
                    pred = output.data.max(1, keepdim=True)[1]
                    test_acc += pred.eq(target.data.view_as(pred)).long().cpu().sum().item()
            
            test_loss = sum(losses) / len(losses)
            test_acc /= total_count
            
            print(f"CENTRAL SYSTEM, round {i+1}: Test acc: {test_acc:.4f}, test loss: {test_loss:.4f}")
    
    def explain(self, data = None):
        self.model.eval()
        imgs = data.to_torchvision_dataset()
        
        for exp_name, exp_dict in self.explainers.items():
            explainer = exp_dict['explainer']
            exp_kwargs = exp_dict['explain_instance_kwargs']
            class_name = explainer.__class__.__name__
            
            self.explanations[exp_name] = {}
            
            for img, l in imgs:
                self.explanations[exp_name][(img, l)] = {}
                
                if class_name == 'LimeImageExplainer':
                    classifier = partial(predict_, model=self.model.to(device), to_gray = True)
                    exp_instance = explainer.explain_instance(gray2rgb(img.squeeze(0).cpu().detach().numpy()), classifier_fn=classifier,
                                                                             **exp_kwargs)
                    segments = exp_instance.segments
                    
                    for j in self.clases:
                        explanation_j = np.vectorize(dict(exp_instance.local_exp[j]).get)(segments)
                        self.explanations[exp_name][(img, l)][j] = np.nan_to_num(explanation_j, nan=0)
                
                if class_name in ('DeepLiftShap', 'GradientShap'):
                    explainer = getattr(attr, class_name)(self.model)
                    data_loader = DataLoader(self.data_test, batch_size=len(self.data_test), shuffle=False)
                    baseline, _ = next(iter(data_loader)) 
                    
                    for j in self.clases:
                        self.explanations[exp_name][(img, l)][j] = explainer.attribute(img.unsqueeze(0), target=j, baselines = baseline, **exp_kwargs).squeeze().detach().numpy()
                
                if class_name == 'KernelShap':
                    explainer = getattr(attr, class_name)(self.model)
                    segm = slic(img.clone().detach().squeeze(0).detach().numpy() , n_segments=200, compactness=0.05, sigma=0.4, channel_axis=None)
                    segm_ = torch.tensor(segm).unsqueeze(0).unsqueeze(0) - 1
                    base = torch.zeros_like(img.clone().detach().unsqueeze(0))
                    
                    for j in self.clases:
                        self.explanations[exp_name][(img, l)][j] = explainer.attribute(img.unsqueeze(0), target=j, baselines = base, feature_mask=segm_, **exp_kwargs).squeeze().detach().numpy()
        
        
    def get_explanations(self):
        dict_output = {}
        
        for exp_name, exp_dict in self.explanations.items():
            exp_output = []
            
            for tuple_image, exp in exp_dict.items():
                data = tuple_image[0]
                label = tuple_image[1]
                
                self.model.eval()
                pred = self.model(data)
                probs = pred[0].tolist()
                prediction = torch.argmax(pred, dim=1).item()
                
                explanations = []
                for j in range(len(self.clases)):
                    explanations.append((exp[j],f'{j}\n({probs[j]*100:.2f}%)'))
                    
                explanations.append((-data.squeeze(0), f'label: {label}\npred: {prediction}'))
                exp_output.append(explanations)
         
            dict_output[exp_name] = exp_output
        
        return dict_output
    

    def plot_heatmap(self, data, output_dir = 'centralized_temp'):
        
        for exp_name, exps in self.explanations.items():
            cur_output_dir = 'images/' + output_dir + '/' + exp_name
            try:
                os.makedirs(cur_output_dir, exist_ok=True)
            except:
                True # CAMBIAR ESTO PARA MANEJO ERRORES
        
            for i, tuple_image in enumerate(tqdm(exps.keys(), desc=f'Generate heatmaps of {exp_name} explanations: ', mininterval=2)):
                expl_arrays = exps[tuple_image]
                data = tuple_image[0]
                label = tuple_image[1]
                
                self.model.eval()
                pred = self.model(data)
                probs = pred[0].tolist()
                prediction = torch.argmax(pred, dim=1).item()
                            
                fig_size = np.array([(len(self.clases) + 1) * 0.15 * (len(self.clases) + 1), 8])
                fig, ax =plt.subplots(nrows=2, ncols=(len(self.clases) + 2)//2, figsize=fig_size, squeeze=False)
                fig.suptitle(f'{exp_name}')
    
                #ax[0,0].imshow(-e.data.squeeze(0).squeeze(0), cmap=plt.get_cmap("gray"))
                if data.squeeze(0).shape[0] == 3:
                    ax[0, 0].imshow(data.squeeze(0).permute(1, 2, 0))
                else:
                    ax[0,0].imshow(-data.squeeze(0), cmap=plt.get_cmap("gray"))
                ax[0,0].axis("off");  ax[1, 0].axis("off")
                ax[0,0].set_title(f'label: {label}\npred: {prediction}')
    
                cur_row = 0
                col_delay = 0 
                for j in range(len(self.clases)):
                    explanation_j = expl_arrays[j]
    
                    try:
                        max_value = np.max(np.abs(explanation_j)) 
                    except:
                        print(f'explanation: {explanation_j}')
                        continue
    
                    if j == ((len(self.clases) + 2)//2 - 1):
                        cur_row += 1
                        col_delay = j
    
                    if data.squeeze(0).shape[0] == 3:
                        ax[cur_row, j+1-col_delay].imshow(data.squeeze(0).permute(1, 2, 0), alpha=0.15, extent=(-1, explanation_j.shape[1], explanation_j.shape[0], -1))
                    else:
                        ax[cur_row, j+1-col_delay].imshow(-data.squeeze(0), cmap=plt.get_cmap("gray"), alpha=0.15, extent=(-1, explanation_j.shape[1], explanation_j.shape[0], -1))
                    
                    
                    if explanation_j.shape[0] == 3:
                        #image_mean = explanation_j.mean(axis=0)
                        explanation_j = np.transpose(explanation_j, (1, 2, 0))
                        explanation_j_norm = (explanation_j - explanation_j.min()) / (explanation_j.max() - explanation_j.min())
                        im = ax[cur_row, j+1-col_delay].imshow(explanation_j_norm, cmap=red_transparent_blue, vmin=-max_value, vmax=max_value)
                    else:
                        im = ax[cur_row, j+1-col_delay].imshow(explanation_j, cmap=red_transparent_blue, vmin=-max_value, vmax=max_value)
                    fig.colorbar(im, ax=ax[cur_row, j+1-col_delay], orientation="horizontal")
                    ax[cur_row, j+1-col_delay].axis("off")
                    ax[cur_row, j+1-col_delay].set_title(f'{j}\n({probs[j]*100:.2f}%)')
    
                fig.tight_layout()
                fig.subplots_adjust(hspace=0.5)
                
                #plt.show()
                output_path = os.path.join(cur_output_dir, f'heatmap_{i}.png')
                fig.savefig(output_path)
                plt.close(fig)  # Cerrar la figura para liberar memoria
