import time

from torchvision import transforms

import models as nets

from visualization import plot_heatmaps, plot_heatmaps_global_means, barplot

from architectures import HFL_System

"""
CASE STUDY 1: Script for the simulation and analysis of horizontal federated learning (HFL) 
systems and centralized systems (CENTRAL) using CNN models on the MNIST dataset.
"""

N_ROUNDS = 15

config = {"download": True, "dataset": "mnist", "transform": transforms.Compose([transforms.ToTensor()]),
          "config_seed": 42, "replacement": False, "nodes": 20, "n_classes": 10, "server_weight": 0.05,
          "balance_nodes": False, "nodes_weights": None, "balance_factor": 0.25, "balance_classes": True,
          "alpha_inf": 0.4, "alpha_sup": 0.6}

# ---- CNN case --------------------------------------------------------------------------------

# Generate the Client-Server federated system
hfl_cs_cnn_sys = HFL_System(name='hfl_cs_cnn', **config)

# Show the data distribution on the nodes (node 0 = server) 
barplot(hfl_cs_cnn_sys.class_counter(), 'hfl_cs_cnn', ncols=4)

# Set CNN model defined in models module
hfl_cs_cnn_sys.set_model(arch = 'cs', build = nets.build_server_CNN_model)

# Train federated model
start_time = time.time()
hfl_cs_cnn_sys.train_n_rounds(n_rounds = N_ROUNDS)
end_time = time.time()
time_train_cnn = end_time - start_time
print(f"\n\nTrain time (CNN): {time_train_cnn:.2f} seconds")

# Set the federated model's explainers
hfl_cs_cnn_sys.set_explainers(clients=True)

 # Get federated model's explanations
start_time = time.time()
hfl_cs_cnn_exps = hfl_cs_cnn_sys.get_explanations(sub_pick=True)
end_time = time.time()
time_exp_cnn = end_time - start_time
print(f"\n\nExplanation time (CNN): {time_exp_cnn:.2f} seconds")

# Plot explanations
l_exps = hfl_cs_cnn_sys.label_explanations()
plot_heatmaps({k: v for k, v in l_exps[0].items() if k.startswith("SP")}, l_exps[1])

plot_heatmaps_global_means(*hfl_cs_cnn_sys.global_mean())