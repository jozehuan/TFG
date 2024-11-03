import numpy as np

from collections import Counter, OrderedDict

import matplotlib.pyplot as plt

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

from flex.pool import  to_all_heatmaps, to_centralized


device = 'cpu'

def train(client_flex_model: FlexModel, client_data: Dataset):
    """Función de entramiento del modelo:
            client_flex_model - modelo definido
            client_data - datos del cliente a usar
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
    
    for _ in [0]:
        for imgs, labels in cl_dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            pred = model(imgs)
            loss = criterion(pred, labels)
            loss.backward()
            optimizer.step()


def evaluate_global_model(server_flex_model: FlexModel, test_data: Dataset):
    """Función de evaluación del modelo entrenado:
            server_flex_model - modelo definido
            test_data - datos a usar
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
        
        """
        Constructor del modelo HFL
        Inicializa la clase, cargando el dataset y configurando la distribución
        de los datos en los nodos.

        Args
        ----------
        dataset_root : str
            Nombre del conjunto de datos [default: dataset]
        download : bool
            Indica si se debe descargar el conjunto de datos de internet [default: True]
        transform : callable
            Transformaciones a aplicar al conjunto de datos.
        config_seed : int
            Semilla para la configuración aleatoria [default: 0]
        replacement : bool
            Si el procedimiento de muestreo utilizado para dividir el conjunto 
            de datos centralizado es con reemplazo o no [default: False]
        nodes : int
            Número de nodos o propietarios [default: 2]
        n_classes : int
            Número de clases [default: 2]
        balance_nodes : bool
            Indica si los nodos deben de tener todos el mismo número de instancias [default: True]
        nodes_weights : list 
            Pesos de los nodos, si se proporciona [default: None]
        balance_factor : float
            Factor de balance, debe estar en el rango (0,1) [default: 0.25]
         server_weight : float
             Peso del servidor, debe estar en el rango (0,1) [default: 0.2]
        balance_classes: bool
            Indica si la cantidad de instancias de cada clase en cada nodo debe 
            estar balanceada [default: True]
        alpha_inf: float
            Ínfimo de normalización [default: 0.4]
        alpha_sup: float
            Supremo de normalización [default: 0.6]
        """
        # -- NOTA: VER COMO GENERALIZAR DISTRIBUCIONES ALEATORIAS --
        #       NO USAR SÓLO LA UNIFORME
        
        np.random.seed(config_seed)
        torch.manual_seed(config_seed)
        
        self._name = name
        
        # Carga del dataset y transformación a objeto 'flex.data.Dataset'
        if dataset in ('mnist', 'MNIST'):
            dt_train = datasets.MNIST(root=dataset_root, train=True, download=download, transform=transform)
            dt_test = datasets.MNIST(root=dataset_root, train=False, download=download, transform=transform)
            dt_all = torch.utils.data.ConcatDataset([dt_train, dt_test])
        if dataset in ('fashion_mnist', 'Fashion_MNIST'):
            dt_train = datasets.FashionMNIST(root=dataset_root, train=True, download=download, transform=transform)
            dt_test = datasets.FashionMNIST(root=dataset_root, train=False, download=download, transform=transform)
            dt_all = torch.utils.data.ConcatDataset([dt_train, dt_test])
        if dataset in ('CIFAR10', 'cifar10'):
            dt_train = datasets.CIFAR10(root=dataset_root, train=True, download=download, transform=transform)
            dt_test = datasets.CIFAR10(root=dataset_root, train=False, download=download, transform=transform)
            dt_all = torch.utils.data.ConcatDataset([dt_train, dt_test])
            
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
        
        
    def barplot_(self, c: dict, k: int, ax):
        
        """
        Gráfico de barras
        Muestra la cantidad de instancias de cada clase para un propietario en
        específico.

        Params
        ----------
        c : dic
            Diccionario que posee las cantidades de cada clase
        k : int
            Nodo específico
        ax : matplotlib.axes.Axes
            Eje sobre el cual dibujar el gráfico
        """
        
        colores = plt.cm.viridis(np.linspace(0, 1, self._num_classes)) 
        ax.bar(c.keys(), c.values(), color=colores)
        ax.set_xlabel('Classes')
        ax.set_ylabel('Count')
        ax.set_title(f"node {k} (total {len(self._fed_dataset[k].to_list()[1])})")
        
        v_max = max(c.values()); v_min = min(c.values())
        ax.set_ylim(max(0, v_min-(v_max-v_min)/2))
        
        # Mostrar el gráfico
        ax.set_xticks(list(c.keys()))
        ax.set_xticklabels(list(c.keys()))
    
    def class_counter(self, node_id : int = None, ncols: int = 2, plot : bool = False):
        
        """
        Contador de elmentos de cada clase
        Cuenta la cantidad de cada elemento en un nodo en específico o en todos en general.
        También tiene la opción de graficar el resultado.

        Parameters
        ----------
        node_id : int
            Propietario específico [Default: None]
        plot : bool
            Si se requiere de graficar el resultado
        """
        
        res = ''
        if node_id is None:
            if plot:
                nrows = (self._config.n_nodes+ncols-1)//ncols
                fig, axs = plt.subplots(nrows, ncols, figsize=(6*ncols, 4 * nrows))
                axs = axs.flatten() 
        
            for k in range(self._config.n_nodes):
                res = res + "NODE --- " + str(k) + ":\n"
                conteo = Counter(self._fed_dataset[k].to_list()[1])
                conteo = dict(sorted(conteo.items()))
                res = res + "\t" + str(conteo) + "\n"
                res = res + "\t" + "Total: " + str(len(self._fed_dataset[k].to_list()[1])) + "\n\n"
                
                if plot: 
                    self.barplot_(conteo, k, axs[k])
            if plot:
                plt.tight_layout()
                plt.show()
                
        else:
            res = res + "NODE --- " + str(node_id) + ":\n"
            conteo = Counter(self._fed_dataset[node_id].to_list()[1])
            conteo = dict(sorted(conteo.items()))
            res = res + "\t" + str(conteo) + "\n"
            res = res + "\t" + "Total: " + str(len(self._fed_dataset[node_id].to_list()[1])) + "\n\n"
            
            
            if plot:
                fig, ax = plt.subplots(figsize=(6, 4))
                self.barplot_(conteo, node_id, ax)
                plt.tight_layout()
                plt.show()
            
        return res
    
    def set_model(self,  build : callable, arch : str = 'cs', server_id: int = 0):
        
        """
        Establece el modelo federado horizontal.

        Parameters
        ----------
        build : callable
            Diccionario que posee las cantidades de cada clase
        arch : str  # <----- cambiar a un Enum
            Tipo de arquitectura federada [Default: client-server]
        server_id : int
            Identificador del servidor (si lo hubiera) [Default: 0]
        """
        if arch in ('client-server', 'cs'):
            self._flex_pool = FlexPool.client_server_pool(
                 fed_dataset=self._fed_dataset, server_id=server_id, init_func=build)
            
        if arch in ('peer-to-peer', 'p2p'):
            self._flex_pool = FlexPool.p2p_pool(fed_dataset=self._fed_dataset, init_func=build)
            
    def train_n_rounds(self, n_rounds: int = 10, clients_per_round : int = 2):
        
        """
        Función principal para el entrenamiento del modelo.
        
        Parameters
        ----------
        n_rounds : int = 10 
            Número de rondas
        clients_per_round : int = 2
            Número de clientes por ronda
        """
        
        z_clients = self._flex_pool.clients
        z_servers = self._flex_pool.servers
        
        print(
            f"\nNumber of nodes in the pool: {len(self._flex_pool)} ({len(z_servers)} server plus {len(z_clients)} clients) \nServer ID: {list(z_servers._actors.keys())}. The server is also an aggregator.\n"
        )
        
        
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
            
    
    def set_explainers(self, *args, **kwargs):
        # self._flex_pool.clients.map(set_LimeImageExplainer, name='lime_slic', **kwargs)
        # self._flex_pool.clients.map(set_DeepShapExplainer, name='deepshap')
        # self._flex_pool.clients.map(set_GradientShapExplainer, name='gradshap', n_samples=1000, stdevs=0.5)
        # self._flex_pool.clients.map(set_KernelShapExplainer, name='kernelshap', n_samples=1000, perturbations_per_eval=50)
        
        self._flex_pool.servers.map(set_LimeImageExplainer, name='lime_slic', top_labels = 10, num_samples=2000, algo_type='slic', segment_params={'n_segments' : 200, 'compactness' : 0.05, 'sigma' : 0.4})
        #self._flex_pool.servers.map(set_LimeImageExplainer, name='lime_quick', top_labels = 10, num_samples=2000, algo_type='quickshift', segment_params={'kernel_size' : 1, 'max_dist' : 2, 'ratio' : 0.2, 'sigma' : 0.05})
        #self._flex_pool.servers.map(set_LimeImageExplainer, name='lime__felz', top_labels = 10, num_samples=2000, algo_type='felzenszwalb', segment_params={'scale' : 0.4, 'sigma' : 0.1, 'min_size' : 5})
        
        self._flex_pool.servers.map(set_DeepShapExplainer, name='deepshap')
        self._flex_pool.servers.map(set_GradientShapExplainer, name='gradshap', n_samples=1000, stdevs=0.5)
        self._flex_pool.servers.map(set_KernelShapExplainer, name='kernelshap', n_samples=1000, perturbations_per_eval=50)
        
        
    def get_explanations(self, data = None):
        # selected_clients = self._flex_pool.clients.select(1).clients 
        
        # selected_clients.map(get_LimeExplanations, data=data)
        # selected_clients.map(get_ShapExplanations, data=data)
        # selected_clients.map(plot_heatmap, pathname='images/' + self._name)
        
        self._flex_pool.servers.map(get_LimeExplanations, data=data)
        self._flex_pool.servers.map(get_ShapExplanations, data=data)
            
        #self._flex_pool.servers.map(get_SP_LimeImageExplanation, data=data, explanation_name = 'lime_slic')
        
        return (self._flex_pool.servers.map(to_all_heatmaps, data=data)[0], self._name)
            
        #self._flex_pool.servers.map(get_SP_LimeImageExplanation, data=data, explanation_name = 'lime_slic')
        
   
    def to_centalized(self):
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