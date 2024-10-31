import time

import models as nets

from architectures import HFL_model

from torchvision import transforms

device = 'cpu'


# CNN ----------------------------------------------------------------------------------------------
        
modelHFL = HFL_model(name='CS_cnn', download=True, transform=transforms.Compose([transforms.ToTensor()]), 
                     config_seed=42, replacement=False, nodes=20, n_classes=10,
                     balance_nodes=False, nodes_weights=None, balance_factor=0.25,
                     balance_classes=True, alpha_inf=0.4, alpha_sup=0.6)

# Copia profunda del modelo
#modelHFL2 = copy.deepcopy(modelHFL)

modelHFL.class_counter(ncols = 4, plot = True)
modelHFL.set_model(arch = 'cs', build = nets.build_server_CNN_model)

#print(modelHFL._flex_pool._actors)

start_time = time.time()

modelHFL.train_n_rounds(n_rounds = 5, clients_per_round = 19)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"\n\nTime: {elapsed_time:.2f} segundos")

modelHFL.set_explainers()
#modelHFL.set_explainers(name='lime_quick', algo_type='felzenszwalb', num_samples=10000)

data = modelHFL._flex_pool.servers._data.data[0]
data = data[0:10]

modelHFL.get_explanations(data) # Explicar datos dados en la entrada
#modelHFL.get_explanations() # Explicar todos los datos de los clientes

# Simple NN ----------------------------------------------------------------------------------------------

# modelHFL2 = HFL_model(name='CS_simplenn', download=True, transform=transforms.Compose([transforms.ToTensor()]), 
#                      config_seed=42, replacement=False, nodes=20, n_classes=10,
#                      balance_nodes=False, nodes_weights=None, balance_factor=0.25,
#                      balance_classes=True, alpha_inf=0.4, alpha_sup=0.6)

# modelHFL2.set_model(arch = 'cs', build = nets.build_server_Net_model)

# start_time = time.time()

# modelHFL2.train_n_rounds(n_rounds = 5, clients_per_round = 5)

# end_time = time.time()
# elapsed_time = end_time - start_time
# print(f"\n\nTime: {elapsed_time:.2f} segundos")

# modelHFL2.set_explainers()

# modelHFL2.get_explanations(data)