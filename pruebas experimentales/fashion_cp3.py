import time

from torchvision import transforms

import models as nets

from visualization import plot_heatmaps, plot_segments, barplot

from architectures import HFL_System

"""
CASE STUDY 3: Script for the simulation and analysis of training in a horizontal
 federated learning (HFL) system, where explanations are generated at each learning
 round, using the Fashion-MNIST dataset.
"""

N_ROUNDS = 100

config = {"download": True, "dataset": "fashion_mnist", "transform": transforms.Compose([transforms.ToTensor()]),
          "config_seed": 42, "replacement": False, "nodes": 20, "n_classes": 10, "server_weight": 0.05,
          "balance_nodes": False, "nodes_weights": None, "balance_factor": 0.25, "balance_classes": True,
          "alpha_inf": 0.4, "alpha_sup": 0.6}

# ---- CNN case --------------------------------------------------------------------------------

# Generate the Client-Server federated system
hfl_cs_cnn_sys = HFL_System(name='hfl_cs_cnn', **config)

# Show the data distribution on the nodes (node 0 = server) 
barplot(hfl_cs_cnn_sys.class_counter(), 'hfl_cs_cnn', ncols=3)

# Set CNN model defined in models module
hfl_cs_cnn_sys.set_model(arch = 'cs', build = nets.build_server_CNN_model, lr=0.0025)

# Set the federated model's explainers
hfl_cs_cnn_sys.set_explainers()

# Train federated model
start_time = time.time()
exp_list = hfl_cs_cnn_sys.train_n_rounds(n_rounds = N_ROUNDS)
end_time = time.time()
time_train_cnn = end_time - start_time
print(f"\n\nTrain time (CNN): {time_train_cnn:.2f} seconds")

# Plot explanatios of each round
for i, exp_round in enumerate(exp_list):
    explanations, label_explanations, segments = exp_round
    
    plot_segments(*segments, n_round = i+1)
    plot_heatmaps(*explanations, n_round = i+1)
    plot_heatmaps(*label_explanations, n_round = i+1)