import numpy as np
import copy
from collections import Counter, OrderedDict
from sklearn.metrics import classification_report

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

from explainers import set_LimeImageExplainer
from explainers import get_LimeExplanations, get_SP_LimeImageExplanation

from explainers import set_DeepShapExplainer
from explainers import set_GradientShapExplainer, set_KernelShapExplainer
from explainers import get_ShapExplanations

from explainers import all_explanations, label_explanations, segment_explanations, get_global_mean

from config import device

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
    cl_dataloader = DataLoader(train_dataset, batch_size=64)
    
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
    
    model = model.to(device)
    model = model.train()
    criterion = client_flex_model["criterion"]
    
    for imgs, labels in cl_dataloader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        
        pred = model(imgs)
        loss = criterion(pred, labels)
        loss.backward()
        optimizer.step()

def evaluate_global_model(flex_model: FlexModel, flex_data: Dataset, *args, **kwargs):
    """ Evaluates the global model using the provided test dataset.

    This function sets the model to evaluation mode, computes the loss and accuracy
    on the test dataset, and returns these metrics. 

    Parameters
    ----------
    flex_model : FlexModel
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
    
    model = flex_model["model"]
    model = model.to(device)
    model.eval()
    test_loss = 0
    test_acc = 0
    total_count = 0
    criterion = flex_model["criterion"]
    
    data_eval = kwargs.get('data_eval', flex_data.to_torchvision_dataset())
    dataloader = DataLoader(data_eval, batch_size=20, 
                                 shuffle=True, pin_memory=False)
    
    
    losses = []
    all_preds = []
    all_targets = []
    
    for data, target in dataloader:
        total_count += target.size(0)
        data, target = data.to(device), target.to(device)
        
        output = model(data)
        losses.append(criterion(output, target).item())
        pred = output.data.max(1, keepdim=True)[1]
        
        all_preds.extend(pred.cpu().numpy())
        all_targets.extend(target.cpu().numpy())
        
        test_acc += pred.eq(target.data.view_as(pred)).long().cpu().sum().item()

    test_loss = sum(losses) / len(losses)
    test_acc /= total_count
    
    all_preds = np.array(all_preds).flatten()
    all_targets = np.array(all_targets).flatten()
    c_metrics = classification_report(all_targets, all_preds, output_dict=True, zero_division=0)
    
    return test_loss, test_acc, c_metrics

def to_centralized(flex_model: FlexModel, node_data: Dataset):
    """ Obtain information from the federated system to design an equivalent centralized model.  

    Parameters
    ----------
    flex_model : FlexModel
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
    try:
        model = copy.deepcopy(flex_model["model"])
        criterion = copy.deepcopy(flex_model["criterion"])
        optimizer_func = copy.deepcopy(flex_model["optimizer_func"])
        opt_kwargs = copy.deepcopy(flex_model["optimizer_kwargs"])
        explainers = copy.deepcopy(flex_model["explainers"])
    except:
        model = criterion = optimizer_func = opt_kwargs = explainers = None

    return model, criterion, optimizer_func, opt_kwargs, explainers, node_data

class HFL_System:
    
    def __init__(self, name : str = 'modelname', dataset_root : str ='datasets', dataset : str = 'mnist', download : bool = True,
                 transform : callable = None, config_seed : int =0, replacement : bool = False, nodes : int = 2, 
                 n_classes: int = 2, balance_nodes : bool = True, nodes_weights: list = None, balance_factor : float = 0.2,
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
            Default is 0.2.
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
        self._config_seed = config_seed
        
        self._name = name
        self._arch = None
        
        # Load the dataset and transform to 'flex.data.Dataset' type
        if dataset.lower() == 'mnist':
            dt_train = datasets.MNIST(root=dataset_root, train=True, download=download, transform=transform)
            dt_test = datasets.MNIST(root=dataset_root, train=False, download=download, transform=transform)
            dt_all = dt_train #torch.utils.data.ConcatDataset([dt_train, dt_test])
        elif dataset.lower() == 'fashion_mnist':
            dt_train = datasets.FashionMNIST(root=dataset_root, train=True, download=download, transform=transform)
            dt_test = datasets.FashionMNIST(root=dataset_root, train=False, download=download, transform=transform)
            dt_all = torch.utils.data.ConcatDataset([dt_train, dt_test])
        else:
            raise ValueError(f"Unsupported dataset: {dataset}. Please choose either 'mnist' or 'fashion_mnist'.")
            
        self._dataset = flex.data.Dataset.from_torchvision_dataset(dt_all)
        self._dataset_lenght = len(self._dataset)
        
        self._data_test, self._data_val = torch.utils.data.random_split(dt_test, 
                                                                        [0.75, 0.25], 
                                                                        generator=torch.Generator().manual_seed(config_seed))
        
        
        # Definition of the federated system configuration
        self._config = flex.data.FedDatasetConfig(seed=config_seed)
        
            # Whether the owners have common instances or not
        self._config.replacement = replacement
            # Number of owners. Since no unique ids are provided for each node,
            # ids will be defined as integers starting from 0 onwards
        self._config.n_nodes = nodes
            # Number of classes
        self._num_classes = n_classes
        
        
        # If we don't want the nodes to be balanced, the distribution should be configured
        if not balance_nodes:
            # If a distribution for each owner is provided
            if nodes_weights is not None and isinstance(nodes_weights, list) and len(nodes_weights) == self._config.n_nodes:
                self._nodes_weights = nodes_weights / np.sum(nodes_weights, axis=0)
                
            else:
                # A random distribution is determined for each node
                if balance_factor < 1:
                    half_range = self._dataset_lenght * balance_factor / self._config.n_nodes
                    self._nodes_weights = np.random.uniform(max(0, self._dataset_lenght / self._config.n_nodes - half_range), 
                                                             min(self._dataset_lenght, self._dataset_lenght / self._config.n_nodes + half_range), 
                                                             self._config.n_nodes - 1)
        
                    self._nodes_weights = self._nodes_weights / np.sum(self._nodes_weights, axis=0) * (1 - server_weight)
                    self._nodes_weights = np.insert(self._nodes_weights, 0, server_weight)
                    self._nodes_weights = self._nodes_weights / np.sum(self._nodes_weights, axis=0)
                    
            # Add this distribution to the configuration
            nodes_w_tiping: np.typing.NDArray = self._nodes_weights
            self._config.weights = nodes_w_tiping
        
        
        # If we want each node to distribute classes in an unbalanced manner
        if not balance_classes:
            # Define the distribution matrix for each node and each class
            self._alphas = np.random.uniform(alpha_inf, alpha_sup, [self.config.n_nodes, self._num_classes])
            self._alphas = self._alphas / np.sum(self._alphas, axis=0)
            self._alphas = self._alphas
        
            # Add the distribution to the configuration
            self._config.weights_per_class = self._alphas
        
        
        # Set the data distribution on each node
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
    
    def set_model(self,  build : callable, arch : str = 'cs', server_id: int = 0, **kwargs):  
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
        
        if arch in archs.CS.value:
            self._arch = arch
            self._flex_pool = FlexPool.client_server_pool(
                 fed_dataset=self._fed_dataset, server_id=server_id, init_func=build, **kwargs)
            
        elif arch in archs.P2P.value:
            self._arch = arch
            self._flex_pool = FlexPool.p2p_pool(fed_dataset=self._fed_dataset, init_func=build, **kwargs)
            
        else:
            raise ValueError(f"Invalid architecture type. Must be {archs.CS} or {archs.P2P}")
            
    def cs_train_n_rounds(self, n_rounds: int = 10, clients_per_round : int = None, no_client_ids : int = None, data_to_explain = None):
        """ Main function for model training.
        
        Params
        ----------
        n_rounds : int, optional
            Number of training rounds. Default is 10.
        clients_per_round : int, optional
            Number of clients participating per round. Default is 2.
        """
        
        result = []
        
        # Choose clients to train
        selected_clients = self._flex_pool.clients
        if no_client_ids is not None:
            selected_clients = self._flex_pool.clients.select(
                criteria=lambda actor_id, _: actor_id not in no_client_ids)
            print(f"All clients selected, except: {no_client_ids}")
                
        for i in range(n_rounds):
            print(f"\nRunning round: {i+1} of {n_rounds}")
            
            if clients_per_round is not None:
                selected_clients = self._flex_pool.clients.select(clients_per_round)
                print(f"Selected clients for this round: {list(selected_clients._actors.keys())}")
            
            # Deploy the server model to all the clients
            self._flex_pool.servers.map(deploy_server_model_pt, self._flex_pool.clients)
            # Each selected client trains her model
            selected_clients.map(train)
            # The aggregador collects weights from the selected clients and aggregates them
            self._flex_pool.aggregators.map(collect_clients_weights_pt, selected_clients)
            self._flex_pool.aggregators.map(fed_avg)
            
            # The aggregator send its aggregated weights to the server
            self._flex_pool.aggregators.map(set_aggregated_weights_pt, self._flex_pool.servers)
            metrics = self._flex_pool.servers.map(evaluate_global_model, data_eval=self._data_val, seed = self._config_seed)
            loss, acc, c_metrics = metrics[0]
            print(f"Server (VALIDATION): acc: {acc:.4f}, loss: {loss:.4f}")
            #if i is (n_rounds-1): pprint(c_metrics)
            
            if data_to_explain is not None:
                result.append( (self.get_explanations(data_to_explain),
                                self.label_explanations(data_to_explain),
                                self.segments(data_to_explain) ) )
        
        metrics = self._flex_pool.servers.map(evaluate_global_model, data_eval=self._data_test, seed = self._config_seed)
        loss, acc, c_metrics = metrics[0]
        print(f"Server (TEST): acc: {acc:.4f}, loss: {loss:.4f}")
        #pprint(c_metrics)
        
        self._flex_pool.servers.map(deploy_server_model_pt, self._flex_pool.clients)
        return result  
    
    def p2p_train_n_rounds(self, n_rounds: int = 10, clients_per_round : int = None, no_client_ids : int = None, data_to_explain = None):
        """ Main function for model training.
        
        Params
        ----------
        n_rounds : int, optional
            Number of training rounds. Default is 10.
        clients_per_round : int, optional
            Number of clients participating per round. Default is 2.
        """
        
        result = []
        
        # Choose clients to train
        selected_clients = self._flex_pool.clients
        
        selected_client_to_eval = self._flex_pool.clients.select(
            criteria=lambda actor_id, _: actor_id in [0])

        for i in range(n_rounds):
            print(f"\nRunning round: {i+1} of {n_rounds}")
            
            if clients_per_round is not None:
                selected_clients = self._flex_pool.clients.select(clients_per_round)
                print(f"Selected clients for this round: {list(selected_clients._actors.keys())}")
            
            # Each selected client trains her model
            selected_clients.map(train)
            
            # All the clients (also aggregators) collects weights from the selected clients and aggregates them
            self._flex_pool.aggregators.map(collect_clients_weights_pt, selected_clients)
            self._flex_pool.aggregators.map(fed_avg)
            
            # Each client set its aggregated weights to its model
            for i in self._flex_pool.aggregators.actor_ids:
                selected_aggr = self._flex_pool.aggregators.select(
                    criteria=lambda actor_id, _: actor_id in [i])
            
                selected_aggr.map(set_aggregated_weights_pt, selected_aggr)
            
            metrics = selected_client_to_eval.map(evaluate_global_model, validation=True, seed = self._config_seed)
            loss, acc, c_metrics = metrics[0]
            print(f"Client 0 (VALIDATION): acc: {acc:.4f}, loss: {loss:.4f}")
            #if i is (n_rounds-1): pprint(c_metrics)
        
        metrics = selected_client_to_eval.map(evaluate_global_model, validation=False, seed = self._config_seed)
        loss, acc, c_metrics = metrics[0]
        print(f"Client 0 (TEST): acc: {acc:.4f}, loss: {loss:.4f}")
        #pprint(c_metrics)
        
        self._flex_pool.servers.map(deploy_server_model_pt, self._flex_pool.clients)
        return result  
    
    def train_n_rounds(self, n_rounds: int = 10, clients_per_round : int = None, no_client_ids : int = None, data_to_explain = None):
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
        
        if self._arch in archs.CS.value:
            print(f"\nNumber of nodes in the pool: {len(self._flex_pool)}\nClient-Server architecture ({len(z_servers)} server plus {len(z_clients)} clients) \nServer ID: {list(z_servers._actors.keys())}. The server is also an aggregator.\n")
            return self.cs_train_n_rounds(n_rounds, clients_per_round, no_client_ids, data_to_explain)    
        elif self._arch in archs.P2P.value:
            print(f"\nNumber of nodes in the pool: {len(self._flex_pool)}\nPeer-to-Peer architecture")
            return self.p2p_train_n_rounds(n_rounds, clients_per_round, no_client_ids, data_to_explain)
        else: 
            return None
        
    
    def evaluate_node(self, node_id : int = None):
        if node_id is None:
            metrics = self._flex_pool.servers.map(evaluate_global_model)
            return  metrics[0]
        
        else: 
            selected_client = self._flex_pool.clients.select(
                criteria=lambda actor_id, _: actor_id  in [node_id])
            metrics = selected_client.map(evaluate_global_model)
            return  metrics[0]
    
    def set_explainers(self, clients: bool = False):
        """
        Configura los explicadores predefinidos en los servidores y opcionalmente en los clientes.
        
        Args:
            clients (bool): Si es True, también asigna los explicadores a los clientes.
        """
        def configure_explainers(pool):
            """Configura los explicadores en un conjunto de servidores o clientes."""
            
            lime_slic_params = {'name': 'lime_slic', 'top_labels': 10, 'num_samples': 2000, 
                                'algo_type': 'slic', 'segment_params': {'n_segments': 100, 'compactness': 3, 'sigma': 0.4}} #{'n_segments': 200, 'compactness': 0.05, 'sigma': 0.4}}
            lime_quickshift_params = {'name': 'lime_qs', 'top_labels': 10, 'num_samples': 2000, 
                                'algo_type': 'quickshift', 'segment_params': {'kernel_size' : 1, 'max_dist' : 2, 'ratio' : 0.005} }
            deepshap_params = {'name': 'deepshap'}
            gradshap_params = {'name': 'gradshap', 'n_samples': 1000, 'stdevs': 0.5}
            kernelshap_params = {'name': 'kernelshap', 'n_samples': 1000, 'perturbations_per_eval': 50}
    
            pool.map(set_LimeImageExplainer, **lime_slic_params)
            
            pool.map(set_LimeImageExplainer, **lime_quickshift_params)
            
            pool.map(set_DeepShapExplainer, **deepshap_params)
            pool.map(set_GradientShapExplainer, **gradshap_params)
            pool.map(set_KernelShapExplainer, **kernelshap_params)
    
        configure_explainers(self._flex_pool.servers)
        if clients:
            configure_explainers(self._flex_pool.clients)
        
    def get_explanations(self, data=None, client_id: int = None, sub_pick: bool = False):
        """
        Obtiene todas las explicaciones generadas por los servidores o un cliente específico. 
        Incluye las explicaciones SP (Shared Prediction).
        
        Params
        ----------
        data : flex.data.dataset.Dataset, optional
            Conjunto de datos a explicar. Si no se especifica, se utilizan todos los datos disponibles.
            
        client_id : int, optional
            ID del cliente específico para el cual se desean obtener las explicaciones. 
            Si es None, se obtienen explicaciones de todos los servidores.
        
        Returns
        ----------
        tuple
            - dict: Las explicaciones generadas.
            - str: Nombre del sistema.
        """
        
        def assign_explanations(pool, data, sub_pick):
            """Asigna las funciones de explicación al conjunto de actores especificado."""
            pool.map(get_LimeExplanations, data=data)
            pool.map(get_ShapExplanations, data=data)
            
            if sub_pick:
                pool.map(get_SP_LimeImageExplanation, data=data, explanation_name='lime_slic', num_exps_desired=5)
                pool.map(get_SP_LimeImageExplanation, data=data, explanation_name='lime_qs', num_exps_desired=5)
        
        if client_id is None:
            # Get explanatios for the servers
            assign_explanations(self._flex_pool.servers, data, sub_pick)
            explanations = self._flex_pool.servers.map(all_explanations, data=data)[0]
        else:
            # Select clients
            client_pool = self._flex_pool.clients.select(
                criteria=lambda actor_id, _: actor_id in [client_id]
            )
            assign_explanations(client_pool, data, sub_pick)
            explanations = client_pool.map(all_explanations, data=data)[0]
    
        return explanations, self._name

            
    def label_explanations(self, data = None, client_id: int = None):
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
        
        pool = self._flex_pool.servers
        if client_id is not None:
            pool = self._flex_pool.clients.select(
                criteria=lambda actor_id, _: actor_id in [client_id]
            )
        
        return (pool.map(label_explanations, data=data)[0], self._name)
    
    def segments(self, data= None, client_id: int = None):
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
        
        pool = self._flex_pool.servers
        if client_id is not None:
            pool = self._flex_pool.clients.select(
                criteria=lambda actor_id, _: actor_id in [client_id]
            )
    
        return (pool.map(segment_explanations, data=data)[0], self._name)
    
    def to_centralized(self):
        """ Get a centralized version of the federated dataset.
        
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
        client_info = self._flex_pool.clients.map(to_centralized)
        
        test_data, val_data = self._data_test, self._data_val
        
        model, criterion, optimizer_func, opt_kwargs, explainers, server_data = server_info[0]
        
        train_data = []
        train_data.append(server_data.to_torchvision_dataset())
        
        for cl_info in client_info:
            _, _, _, _, _, client_data = cl_info
            client_data = client_data.to_torchvision_dataset()
            train_data.append(client_data)
        
        return model, criterion, optimizer_func, opt_kwargs, explainers, train_data, test_data, val_data
    
    def global_mean(self, data = None, client_id: int = None):
        """ Get SHAP global means
        
        Params
        ----------
        data : flex.data.Dataset, optional
            explained data
        
        Returns
        ----------
        tuple
            - A dict with global means for each explainer 
            - name of the system
        """ 
        pool = self._flex_pool.servers
        if client_id is not None:
            pool = self._flex_pool.clients.select(
                criteria=lambda actor_id, _: actor_id in [client_id]
            )

        return (pool.map(get_global_mean, data=data)[0], self._name)