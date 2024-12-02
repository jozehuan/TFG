import time
from pprint import pprint

from torchvision import transforms

import models as nets

from visualization import plot_heatmaps, plot_segments, barplot

from architectures import HFL_System

"""
CASE STUDY 2: Script for the simulation and analysis of training in a horizontal
 federated learning (HFL) system where a client is absent during training, using
 the Fashion-MNIST dataset.
"""


N_ROUNDS = 100

config = {"download": True, "dataset": "fashion_mnist", "transform": transforms.Compose([transforms.ToTensor()]),
          "config_seed": 42, "replacement": False, "nodes": 20, "n_classes": 10, "server_weight": 0.05,
          "balance_nodes": False, "nodes_weights": None, "balance_factor": 0.25, "balance_classes": True,
          "alpha_inf": 0.4, "alpha_sup": 0.6}

# ------------------------------------------------------------------------------------

# Generate the Client-Server federated system
hfl_cs_cnn_sys = HFL_System(name='hfl_cs_cnn', **config)

# Show the data distribution on the nodes (node 0 = server) 
barplot(hfl_cs_cnn_sys.class_counter(), 'hfl_cs_cnn', ncols=3)

# Set CNN model defined in models module
hfl_cs_cnn_sys.set_model(arch = 'cs', build = nets.build_server_CNN_model)

# Train federated model
start_time = time.time()
hfl_cs_cnn_sys.train_n_rounds(n_rounds = N_ROUNDS, no_client_ids=[1])
end_time = time.time()
time_train_cnn = end_time - start_time
print(f"\n\nTrain time (CNN): {time_train_cnn:.2f} seconds")

# Evaluate absent node
metrics_cnn = hfl_cs_cnn_sys.evaluate_node(node_id=1)
loss, acc, c_metrics = metrics_cnn
print(f"Client 1: Test acc: {acc:.4f}, test loss: {loss:.4f}")
pprint(c_metrics)

# Set the federated model's explainers
hfl_cs_cnn_sys.set_explainers(clients=True)

# Get federated model's explanations
start_time = time.time()
hfl_cs_cnn_exps = hfl_cs_cnn_sys.get_explanations(client_id=1)
end_time = time.time()
time_exp_cnn = end_time - start_time
print(f"\n\nExplanation time (CNN): {time_exp_cnn:.2f} seconds")

# Plot segments
plot_segments(*hfl_cs_cnn_sys.segments(client_id=1))

# Plot explanations
plot_heatmaps(*hfl_cs_cnn_exps)
plot_heatmaps(*hfl_cs_cnn_sys.label_explanations(client_id=1))
