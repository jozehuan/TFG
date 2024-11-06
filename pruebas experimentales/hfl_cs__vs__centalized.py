import time

import torch
from torchvision import transforms

import models as nets

from visualization import plot_heatmaps, plot_heatmaps_compared, plot_segments

from architectures import HFL_System, CENTRAL_System

from similarity import compute_similarity

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = {"download": True, "dataset": "mnist", "transform": transforms.Compose([transforms.ToTensor()]),
          "config_seed": 42, "replacement": False, "nodes": 20, "n_classes": 10, "server_weight": 0.05,
          "balance_nodes": False, "nodes_weights": None, "balance_factor": 0.25, "balance_classes": True,
          "alpha_inf": 0.4, "alpha_sup": 0.6}

# ---- CNN case --------------------------------------------------------------------------------

# Generate the Client-Server federated system
hfl_cs_cnn_sys = HFL_System(name='hfl_cs_cnn', **config)

# Show the data distribution on the nodes (node 0 = server) 
hfl_cs_cnn_sys.class_counter(ncols = 4, plot = True)

# Set CNN model defined in models module
hfl_cs_cnn_sys.set_model(arch = 'cs', build = nets.build_server_CNN_model)

# Set the federated model's explainers
hfl_cs_cnn_sys.set_explainers()

# Generate the centralized system from the federated system
# (clients data = train data, server data = test data)
c_system = CENTRAL_System(*hfl_cs_cnn_sys.to_centalized())
 
n_rounds = 5
# Train federated model
start_time = time.time()
hfl_cs_cnn_sys.train_n_rounds(n_rounds = 15, clients_per_round = 19)
end_time = time.time()
time_train_cnn = end_time - start_time
print(f"\n\nTrain time (CNN): {time_train_cnn:.2f} seconds")

# Tomar datos (NO HACER PORQUE SERÁN LOS DATOS DEL SERVIDOR)
data = hfl_cs_cnn_sys._flex_pool.servers._data.data[0]
data = data[0:15]

# Get federated model's explanations
start_time = time.time()
hfl_cs_cnn_exps = hfl_cs_cnn_sys.get_explanations(data)
end_time = time.time()
time_exp_cnn = end_time - start_time
print(f"\n\nExplanation time (CNN): {time_exp_cnn:.2f} seconds")

# Plot segments:
plot_segments(*hfl_cs_cnn_sys.segments_lime(data))

# Plot explanations
plot_heatmaps(*hfl_cs_cnn_exps)
plot_heatmaps(*hfl_cs_cnn_sys.label_explanations(data))

# Train centralized system
c_system.train(n_rounds)

# Get centralized model's explanations
c_system.explain(data = data)
c_system_exps = c_system.get_explanations()

# Plot centralized explanations
plot_heatmaps(c_system_exps, 'central_cnn')

# Compute similarity metrics and compare central and federated explanations
cnn_metrics = compute_similarity(hfl_cs_cnn_exps, c_system_exps)
plot_heatmaps_compared(hfl_cs_cnn_exps, c_system_exps, cnn_metrics)


# ---- Simple NN case --------------------------------------------------------------------------------

# Generate the Client-Server federated system
hfl_cs_snn_sys = HFL_System(name='hfl_cs_snn', **config)

# Show the data distribution on the nodes (node 0 = server) 
hfl_cs_snn_sys.class_counter(ncols = 4, plot = True)

# Set CNN model defined in models module
hfl_cs_snn_sys.set_model(arch = 'cs', build = nets.build_server_Net_model)

# Set the federated model's explainers
hfl_cs_snn_sys.set_explainers()

# Generate the centralized system from the federated system
# (clients data = train data, server data = test data)
c_system2 = CENTRAL_System(*hfl_cs_snn_sys.to_centalized())

# Train federated model
start_time = time.time()
hfl_cs_snn_sys.train_n_rounds(n_rounds = 15, clients_per_round = 19)
end_time = time.time()
time_train_snn = end_time - start_time
print(f"\n\nTrain time (SNN): {time_train_snn:.2f} seconds")

# Get federated model's explanations
start_time = time.time()
hfl_cs_snn_exps = hfl_cs_snn_sys.get_explanations(data)
end_time = time.time()
time_exp_snn = end_time - start_time
print(f"\n\nExplanaton time (SNN): {time_exp_snn:.2f} seconds")

# Plot segments:
plot_segments(*hfl_cs_snn_sys.segments_lime(data))

# Plot explanations
plot_heatmaps(*hfl_cs_snn_exps)
plot_heatmaps(*hfl_cs_snn_sys.label_explanations(data))

# Train centralized system
c_system2.train(n_rounds)

# Get centralized model's explanations
c_system2.explain(data = data)
c_system2_exps = c_system2.get_explanations()

# Plot centralized explanations
plot_heatmaps(c_system2_exps, 'central_snn')

# Compute similarity metrics and compare central and federated explanations
snn_metrics = compute_similarity(hfl_cs_snn_exps, c_system2_exps)
plot_heatmaps_compared(hfl_cs_snn_exps, c_system2_exps, snn_metrics)
