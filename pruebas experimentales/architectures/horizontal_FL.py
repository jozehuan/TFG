import numpy as np
from collections import Counter, OrderedDict

import torch
from torchvision import datasets
from torch.utils.data import DataLoader

import flex

from flex.model import FlexModel

from flex.data import Dataset

from flex.pool import FlexPool
from flex.pool import deploy_server_model_pt
from flex.pool import collect_clients_weights_pt, fed_avg
from flex.pool import set_aggregated_weights_pt

from flex.pool import set_LimeImageExplainer
from flex.pool import get_LimeExplanations, get_SP_LimeImageExplanation

from flex.pool import set_DeepShapExplainer
from flex.pool import set_GradientShapExplainer, set_KernelShapExplainer
from flex.pool import get_ShapExplanations

from flex.pool import all_explanations, label_explanations, segment_explanations
from flex.pool import to_centralized

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from enum import Enum
class archs(Enum):
    CS = ['client-server','cs']
    P2P = ['peer-to-peer','p2p']


def train(client_flex_model: FlexModel, client_data: Dataset):
    """ Trains the client-side neural network model.

    This function prepares the client data for training, initializes the model and optimizer,
    and performs one training epoch. It updates the model parameters based on the loss computed
    during training.

    Parameters
    ----------
    client_flex_model : FlexModel
        An instance of FlexModel containing the model, optimizer, and criterion to be used for training.
    client_data : Dataset
        The dataset containing training samples for the client.
    """
    
    train_dataset = client_data.to_torchvision_dataset()
    cl_dataloader = DataLoader(train_dataset, batch_size=20)
    
    model = client_flex_model["model"]
    optimizer = client_flex_model["optimizer_func"](
        model.parameters(), **client_flex_model["optimizer_kwargs"]
    )
    
    # Pasar al modelo los pesos agregados en el servidor
    agg_weights = client_flex_model.get("aggregated_weights")
    if agg_weights is not None:
        model_keys = list(model.state_dict().keys())
        agg_weights_order = OrderedDict()
        for i, tensor in enumerate(agg_weights):
            agg_weights_order[model_keys[i]] = tensor
        model.load_state_dict(agg_weights_order)
    
    model = model.train()
    model = model.to(device)
    criterion = client_flex_model["criterion"]
    
    for imgs, labels in cl_dataloader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        
        pred = model(imgs)
        loss = criterion(pred, labels)
        loss.backward()
        optimizer.step()

def evaluate_global_model(server_flex_model: FlexModel, test_data: Dataset):
    """ Evaluates the global model using the provided test dataset.

    This function sets the model to evaluation mode, computes the loss and accuracy
    on the test dataset, and returns these metrics. 

    Parameters
    ----------
    server_flex_model : FlexModel
        An instance of FlexModel containing the global model and the loss criterion.
    test_data : Dataset
        The dataset containing test samples to evaluate the model's performance.

    Returns
    -------
    test_loss : float
        The average loss over the test dataset.
    test_acc : float
        The accuracy of the model on the test dataset, represented as a fraction between 0 and 1.
    """
    
    model = server_flex_model["model"]
    model.eval()
    test_loss = 0
    test_acc = 0
    total_count = 0
    model = model.to(device)
    criterion = server_flex_model["criterion"]
    
    # get test data as a torchvision object
    test_dataset = test_data.to_torchvision_dataset()
    test_dataloader = DataLoader(test_dataset, batch_size=20, 
                                 shuffle=True, pin_memory=False)
    
    losses = []
    with torch.no_grad():
        for data, target in test_dataloader:
            total_count += target.size(0)
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            losses.append(criterion(output, target).item())
            pred = output.data.max(1, keepdim=True)[1]
            test_acc += pred.eq(target.data.view_as(pred)).long().cpu().sum().item()

    test_loss = sum(losses) / len(losses)
    test_acc /= total_count
    return test_loss, test_acc




class HFL_System:
    
    def __init__(self, name : str = 'modelname', dataset_root : str ='datasets', dataset : str = 'mnist', download : bool = True,
                 transform : callable = None, config_seed : int =0, replacement : bool = False, nodes : int = 2, 
                 n_classes: int = 2, balance_nodes : bool = True, nodes_weights: list = None, balance_factor : float = 0.25,
                 server_weight : float = 0.2, balance_classes : bool = True, alpha_inf : float = 0.4, alpha_sup : float = 0.6):
        
        """ Initializes the HFL model.

        This constructor initializes the class by loading the dataset and configuring 
        the distribution of data among the nodes. It supports various datasets and 
        offers flexibility in how the data is distributed and balanced across nodes.

        Parameters
        ----------
        name : str
            The name of the system. Default is 'modelname'.
        dataset_root : str
            The directory where the dataset is located. Default is 'datasets'.
        dataset : str
            The name of the dataset (e.g., 'mnist', 'fashion_mnist'). Default is 'mnist'
        download : bool
            Indicates whether to download the dataset. Default is True.
        transform : callable
            Transformations to apply to the dataset (e.g., normalization, augmentation).
            Default is None.
        config_seed : int
            Seed for random configuration. Default is 0.
        replacement : bool
            Indicates whether the sampling procedure used to split the centralized dataset 
            is with replacement or not. Default is False.
        nodes : int
            The number of nodes or owners. Default is 2.
        n_classes : int
            The number of classes in the dataset. Default is 2.
        balance_nodes : bool
            Indicates whether the nodes should all have the same number of instances.
            Default is True.
        nodes_weights : list
            Weights of the nodes if provided. Default is None.
        balance_factor : float
            The factor used to balance the data distribution, must be in the range (0, 1).
            Default is 0.25.
        server_weight : float
            Weight of the server in the data distribution, must be in the range (0, 1).
            Default is 0.2.
        balance_classes : bool
            Indicates whether the instances of each class in each node should be balanced.
            Default is True.
        alpha_inf : float
            The lower bound for normalization of class distributions. Default is 0.4.
        alpha_sup : float
            The upper bound for normalization of class distributions. Default is 0.6.
        """
        
        np.random.seed(config_seed)
        torch.manual_seed(config_seed)
        
        self._name = name
        self._arch = None
        
        # Carga del dataset y transformación a objeto 'flex.data.Dataset'
        if dataset.lower() == 'mnist':
            dt_train = datasets.MNIST(root=dataset_root, train=True, download=download, transform=transform)
            dt_test = datasets.MNIST(root=dataset_root, train=False, download=download, transform=transform)
            dt_all = torch.utils.data.ConcatDataset([dt_train, dt_test])
        if dataset.lower() == 'fashion_mnist':
            dt_train = datasets.FashionMNIST(root=dataset_root, train=True, download=download, transform=transform)
            dt_test = datasets.FashionMNIST(root=dataset_root, train=False, download=download, transform=transform)
            dt_all = torch.utils.data.ConcatDataset([dt_train, dt_test])
        else: True
            #Retornar error
            
        self._dataset = flex.data.Dataset.from_torchvision_dataset(dt_all)
        self._dataset_lenght = len(self._dataset)
        
        # Definición de la configuración de la partición del conjunto de datos
        self._config = flex.data.FedDatasetConfig(seed = config_seed)
        
            # Si los propietarios poseen instancias comunes o no
        self._config.replacement = replacement
            # Número de propietarios. Como no se proporcionan id propios para cada
            # nodo, los id se definirán como enteros desde el 0 hacia adelante
        self._config.n_nodes = nodes
            # Número de clases
        self._num_classes = n_classes
        
        
        # Si se quiere que los nodos no estén balanceados, hay que configurar la distribución
        if not balance_nodes:
            # En el caso que se proporcione una distribución para cada propietario
            if nodes_weights is not None and isinstance(nodes_weights, list) and len(nodes_weights) == self._config.n_nodes:
                   self._nodes_weights = nodes_weights / np.sum(nodes_weights, axis=0)
                   
            else:
                # Se determina una distribución aleatoria para cada nodo
                if(balance_factor < 1):
                    half_range = self._dataset_lenght * balance_factor / self._config.n_nodes
                    self._nodes_weights =  np.random.uniform(max(0, self._dataset_lenght / self._config.n_nodes - half_range), 
                                                             min(self._dataset_lenght, self._dataset_lenght / self._config.n_nodes + half_range), 
                                                             self._config.n_nodes - 1)
            
                    self._nodes_weights = self._nodes_weights / np.sum(self._nodes_weights, axis=0) * (1-server_weight)
                    self._nodes_weights = np.insert(self._nodes_weights, 0, server_weight)
                    self._nodes_weights = self._nodes_weights / np.sum(self._nodes_weights, axis=0)
                    
            # Se añade dicha distribución a la configuración
            nodes_w_tiping: np.typing.NDArray = self._nodes_weights
            self._config.weights = nodes_w_tiping
    
        
        # Si se quiere que, en cada nodo, se distribuya cada clase de manera desbalanceada
        if not balance_classes:
            # Se define la matriz de distribución, para cada nodo y cada clase
            self._alphas = np.random.uniform(alpha_inf, alpha_sup, [self.config.n_nodes, self._num_classes])
            self._alphas = self._alphas / np.sum(self._alphas, axis=0)
            self._alphas = self._alphas
            
            # Se añade la distribución a la configuración
            self._config.weights_per_class = self._alphas
           
            
        # Se establece la distribución de los datos en cada nodo
        self._fed_dataset = flex.data.FedDataDistribution.from_config(self._dataset, self._config)
        
    
    def class_counter(self, node_id : int = None):    
        """ Counts the number of instances per class in a specific node or across all nodes.

        This function retrieves the count of each class in a specified node or in all nodes if no node is specified. 
        The result is returned as a list of tuples containing the node index, the class counts, and the total number 
        of instances in that node. Optionally, it can also display the class distribution for each node in bar plot form.
    
        Params
        ----------
        node_id : int, optional
            The specific node to count instances for. If None, the function counts 
            instances for all nodes. Default is None.
        
        Return
        ----------
        list of tuples
            A list of tuples, each containing:
                - int: the node index,
                - dict: a dictionary with class counts (keys are class labels and values are counts),
                - int: the total count of instances in that node.
        """
        
        nodes = range(self._config.n_nodes) if node_id is None else [node_id]
        
        res = [(k, dict(sorted((c := Counter(self._fed_dataset[k].to_list()[1])).items())),
                sum(c.values())) for k in nodes]
            
        return res
    
    def set_model(self,  build : callable, arch : str = 'cs', server_id: int = 0):  
        """ Sets up the horizontal federated model.
        
        Params
        ----------
        build : callable
            Function to initialize the model.
        arch : Enum
            Type of federated architecture, specifying the setup for the federated learning structure. 
            Default is 'cs'.
        server_id : int
            Identifier of the server (if applicable). Default is 0.
        """
        
        if arch in archs.CS:
            self._flex_pool = FlexPool.client_server_pool(
                 fed_dataset=self._fed_dataset, server_id=server_id, init_func=build)
            
        elif arch in archs.P2P:
            self._flex_pool = FlexPool.p2p_pool(fed_dataset=self._fed_dataset, init_func=build)
            
        else:
            raise ValueError(f"Invalid architecture type. Must be {archs.CS} or {archs.P2P}")
            
    def train_n_rounds(self, n_rounds: int = 10, clients_per_round : int = 2):
        """ Main function for model training.
        
        Params
        ----------
        n_rounds : int, optional
            Number of training rounds. Default is 10.
        clients_per_round : int, optional
            Number of clients participating per round. Default is 2.
        """
        
        z_clients = self._flex_pool.clients
        z_servers = self._flex_pool.servers
        
        if self._arch in archs.CS:
            print(f"\nNumber of nodes in the pool: {len(self._flex_pool)}\nClient-Server architecture ({len(z_servers)} server plus {len(z_clients)} clients) \nServer ID: {list(z_servers._actors.keys())}. The server is also an aggregator.\n")
        elif self._arch in archs.P2P:
            print(f"\nNumber of nodes in the pool: {len(self._flex_pool)}\nPeer-to-Peer architecture")
        
        for i in range(n_rounds):
            print(f"\nRunning round: {i+1} of {n_rounds}")
            selected_clients_pool = self._flex_pool.clients.select(clients_per_round)
            selected_clients = selected_clients_pool.clients
            print(f"Selected clients for this round: {list(selected_clients._actors.keys())}")
            # Deploy the server model to all the clients
            self._flex_pool.servers.map(deploy_server_model_pt, self._flex_pool.clients)
            # Each selected client trains her model
            selected_clients.map(train)
            # The aggregador collects weights from the selected clients and aggregates them
            self._flex_pool.aggregators.map(collect_clients_weights_pt, selected_clients)
            self._flex_pool.aggregators.map(fed_avg)
            
            # Llegar a algún elemento del modelo general
            # w_cl2 = self._flex_pool.aggregators._models[0]['aggregated_weights']
            
            # The aggregator send its aggregated weights to the server
            self._flex_pool.aggregators.map(set_aggregated_weights_pt, self._flex_pool.servers)
            metrics = self._flex_pool.servers.map(evaluate_global_model)
            loss, acc = metrics[0]
            print(f"Server: Test acc: {acc:.4f}, test loss: {loss:.4f}")
            
    
    def set_explainers(self):
        """ Set predefinided explainers to the servers."""
        
        self._flex_pool.map(set_LimeImageExplainer, name='lime_slic', top_labels = 10, num_samples=2000, algo_type='slic', segment_params={'n_segments' : 200, 'compactness' : 0.05, 'sigma' : 0.4})
        #self._flex_pool.servers.map(set_LimeImageExplainer, name='lime_quick', top_labels = 10, num_samples=2000, algo_type='quickshift', segment_params={'kernel_size' : 1, 'max_dist' : 2, 'ratio' : 0.2, 'sigma' : 0.05})
        #self._flex_pool.servers.map(set_LimeImageExplainer, name='lime__felz', top_labels = 10, num_samples=2000, algo_type='felzenszwalb', segment_params={'scale' : 0.4, 'sigma' : 0.1, 'min_size' : 5})
        
        self._flex_pool.map(set_DeepShapExplainer, name='deepshap')
        self._flex_pool.map(set_GradientShapExplainer, name='gradshap', n_samples=1000, stdevs=0.5)
        self._flex_pool.map(set_KernelShapExplainer, name='kernelshap', n_samples=1000, perturbations_per_eval=50)
        
        
    def get_explanations(self, data = None):
        """ Get all the explanations of the servers. Also include SP-explanations.
        
        Params
        ----------
        data : flex.data.dataset.Dataset, optional
            Data to be explained. Default None
            If None, explanations are generated for the entire dataset.
            
        Returns
        ----------
        tuple
            - dict: The explanations generated by the servers.
            - str: The system's name.
            
        """ 
        
        self._flex_pool.servers.map(get_LimeExplanations, data=data)
        self._flex_pool.servers.map(get_ShapExplanations, data=data)
            
        self._flex_pool.servers.map(get_SP_LimeImageExplanation, data=data, explanation_name = 'lime_slic')
        self._flex_pool.servers.map(get_SP_LimeImageExplanation, data=data, explanation_name = 'deepshap')
        self._flex_pool.servers.map(get_SP_LimeImageExplanation, data=data, explanation_name = 'gradshap')
        self._flex_pool.servers.map(get_SP_LimeImageExplanation, data=data, explanation_name = 'kernelshap')
        
        return (self._flex_pool.servers.map(all_explanations, data=data)[0], self._name)
            
    def label_explanations(self, data = None):
        """ Get all the explanations of the servers, only for the label's image.
        
        Params
        ----------
        data : flex.data.dataset.Dataset, optional
            Explained data. Default None
            If None, explanations are generated for the entire dataset.
            
        Returns
        ----------
        tuple
            - dict: The explanations generated by the servers for the specified label's image.
            - str: The system's name.
        """
        
        return (self._flex_pool.servers.map(label_explanations, data=data)[0], self._name)
    
    def segments_lime(self, data= None):
        """ Get segments images of the LIME explainers and KernelShap
        
        Params
        ----------
        data : flex.data.dataset.Dataset, optional
            Explained data. Default None
            If None, explanations are generated for the entire dataset.
            
        Returns
        ----------
        tuple
            - dict: The segments images.
            - str: The system's name.
        """ 
        
        return (self._flex_pool.servers.map(segment_explanations, data=data)[0], self._name)
    
    def to_centalized(self):
        """ Get a centralized version of the federated dataset.
        
        The clients' data constitutes the train dataset.
        The servers' data constitutes the test dataset.
        
        Returns
        ----------
        tuple
            - model: federated system model.
            - criterion: The loss function used for training the model.
            - optimizer_func: The optimizer function to be applied to the model.
            - opt_kwargs: Additional keyword arguments for the optimizer function.
            - explainers: Explainability models.
            - train_data: Centralized training dataset, generated from clients' data.
            - test_data: Centralized test dataset, generated from servers' data.
        """ 
        
        server_info = self._flex_pool.servers.map(to_centralized)
        model, criterion, optimizer_func, opt_kwargs, explainers, server_data = server_info[0]
        test_data = server_data.to_torchvision_dataset()
        
        client_info = self._flex_pool.clients.map(to_centralized)
        train_data = []
        for cl_info in client_info:
            _, _, _, _, _, client_data = cl_info
            client_data = client_data.to_torchvision_dataset()
            train_data.append(client_data)
        
        return model, criterion, optimizer_func, opt_kwargs, explainers, train_data, test_data