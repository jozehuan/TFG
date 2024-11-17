import numpy as np
from functools import partial

from pprint import pprint
from sklearn.metrics import classification_report

import torch
from torch.utils.data import DataLoader
from skimage.color import gray2rgb
from skimage.segmentation import slic

import captum.attr as attr

from flex.pool import predict_

device = 'cpu'

from matplotlib.colors import LinearSegmentedColormap
colors = []
for j in np.linspace(1, 0, 100):
    colors.append((30.0 / 255, 136.0 / 255, 229.0 / 255, j))
for j in np.linspace(0, 1, 100):
    colors.append((255.0 / 255, 13.0 / 255, 87.0 / 255, j))
red_transparent_blue = LinearSegmentedColormap.from_list("red_transparent_blue", colors)

class CENTRAL_System:
    
    def __init__(self,  model, criterion, optimizer_func, opt_kwargs, explainers, train_data, test_data):
        """ Initializes the centralized system.
    
        Params
        ----------
        model : torch.nn.Module
            The neural network model to be trained.
        criterion : callable
            The loss function used to evaluate the model's performance during training.
        optimizer_func : callable
            The optimizer function to be used for updating the model's parameters.
        opt_kwargs : dict
            A dictionary of additional arguments to be passed to the optimizer function.
        explainers : list
            Explainability models.
        train_data : dataset
            The dataset used for training the model.
        test_data : dataset
            The dataset used for testing the model.

        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer_func(self.model.parameters(), **opt_kwargs)
        
        self.data_test = test_data
        self.data_train = train_data
        self.clases = set()
        
        self.explainers = explainers
        self.explanations = {}
    
    def train(self, n_rounds: int = 10):
        """  Main function for model training.
        
        Params
        ----------
        n_rounds : int, optional
            Number of training rounds. Default is 10.
        """
        
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
            all_preds = []
            all_targets = []
            with torch.no_grad():
                for data, target in test_dataloader:
                    total_count += target.size(0)
                    data, target = data.to(device), target.to(device)
                    
                    output = self.model(data)
                    losses.append(self.criterion(output, target).item())
                    pred = output.data.max(1, keepdim=True)[1]
                    
                    all_preds.extend(pred.cpu().numpy())
                    all_targets.extend(target.cpu().numpy())
                    
                    test_acc += pred.eq(target.data.view_as(pred)).long().cpu().sum().item()
            
            test_loss = sum(losses) / len(losses)
            test_acc /= total_count
            
            all_preds = np.array(all_preds).flatten()
            all_targets = np.array(all_targets).flatten()
            c_metrics = classification_report(all_targets, all_preds, output_dict=True, zero_division=0)
            
            print(f"CENTRAL SYSTEM, round {i+1}: Test acc: {test_acc:.4f}, test loss: {test_loss:.4f}")
            if i is (n_rounds-1): pprint(c_metrics)
            
    def explain(self, data = None):
        """ Get all the data explanations.
        
        Params
        ----------
        data : flex.data.dataset.Dataset, optional
            Data to be explained. Default None
            If None, explanations are generated for the test dataset.            
        """ 
        self.model.eval()
        if data is not None:
            imgs = data.to_torchvision_dataset()
        else:
            imgs = self.data_test
        
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
        """ Retrieves the explanations for the model's predictions.
    
        Return
        ----------
        dict
            A dictionary containing explanations for each explainer. 
            The keys are the explanation names, and the values are lists of explanations 
            for each image.
        """
        dict_output = {}
        
        for exp_name, exp_dict in self.explanations.items():
            exp_output = []
            
            for tuple_image, exp in exp_dict.items():
                data = tuple_image[0]
                label = tuple_image[1]
                
                self.model.eval()
                pred = self.model(data)
                probs = torch.softmax(pred, dim=1)[0].tolist()
                prediction = torch.argmax(pred, dim=1).item()
                
                explanations = []
                for j in range(len(self.clases)):
                    explanations.append((exp[j],f'{j}\n({probs[j]*100:.2f}%)'))
                    
                explanations.append((-data.squeeze(0), f'label: {label}\npred: {prediction}'))
                exp_output.append(explanations)
         
            dict_output[exp_name] = exp_output
        
        return dict_output