import time

from torchvision import transforms

import models as nets

from visualization import plot_heatmaps, plot_heatmaps_compared, plot_segments, barplot

from architectures import HFL_System, CENTRAL_System

from similarity import compute_similarity

"""
CASE STUDY 1: Script for the simulation and analysis of horizontal federated learning (HFL) 
systems and centralized systems (CENTRAL) using CNN models on the Fashion-MNIST dataset.
"""

N_ROUNDS = 100

config = {"download": True, "dataset": "fashion_mnist", "transform": transforms.Compose([transforms.ToTensor()]),
          "config_seed": 42, "replacement": False, "nodes": 20, "n_classes": 10, "server_weight": 0.05,
          "balance_nodes": False, "nodes_weights": None, "balance_factor": 0.25, "balance_classes": True,
          "alpha_inf": 0.4, "alpha_sup": 0.6}

# ------------------------------------------------------------------------------------

# Generate the Client-Server federated system
hfl_cs_cnn_sys = HFL_System(name='hfl_cs_cnn', **config)
hfl_p2p_cnn_sys = HFL_System(name='hfl_p2p_cnn', **config)

# Show the data distribution on the nodes (node 0 = server) 
barplot(hfl_cs_cnn_sys.class_counter(), 'hfl_cs_cnn', ncols=5)
barplot(hfl_p2p_cnn_sys.class_counter(), 'hfl_p2p_cnn', ncols=5)

# Set CNN model defined in models module
hfl_cs_cnn_sys.set_model(arch = 'cs', build = nets.build_server_CNN_model)
hfl_p2p_cnn_sys.set_model(arch = 'p2p', build = nets.build_server_CNN_model)

# Set the federated model's explainers
hfl_cs_cnn_sys.set_explainers()
hfl_p2p_cnn_sys.set_explainers()

# Generate the centralized system from the federated system
c_system = CENTRAL_System(*hfl_cs_cnn_sys.to_centralized())

# Train client-server federated model
start_time = time.time()
hfl_cs_cnn_sys.train_n_rounds(n_rounds = N_ROUNDS)
end_time = time.time()
time_train_cnn = end_time - start_time
print(f"\n\nTrain time Client-Server (CNN): {time_train_cnn:.2f} seconds")

# Train peer-to-peer federated model
start_time = time.time()
hfl_p2p_cnn_sys.train_n_rounds(n_rounds = N_ROUNDS, clients_per_round=10)
end_time = time.time()
time_train_cnn = end_time - start_time
print(f"\n\nTrain time Peer-to-Peer (CNN): {time_train_cnn:.2f} seconds")

# Train centralized system
c_system.train(N_ROUNDS)

# Get centralized model's explanations
c_system.explain()
c_system_exps = c_system.get_explanations()

# Plot centralized explanations
plot_heatmaps(c_system_exps, 'central')

for hfl_system , node_id in zip([hfl_cs_cnn_sys, hfl_p2p_cnn_sys],[None, 0]):
    print(f'\n\n---- START {hfl_system._name} EXPLANTIONS ----\n')
    # Get federated model's explanations
    start_time = time.time()
    hfl_exps = hfl_system.get_explanations(node_id)
    end_time = time.time()
    time_exp_cnn = end_time - start_time
    print(f"\n\nExplanation time {hfl_system._name}(CNN): {time_exp_cnn:.2f} seconds")
    
    # Plot segments:
    plot_segments(*hfl_system.segments(node_id))
    
    # Plot explanations
    plot_heatmaps(*hfl_exps)
    plot_heatmaps(*hfl_system.label_explanations(node_id))

    # Compute similarity metrics and compare central and federated explanations
    print(f'\n\nComparing {hfl_system._name} explanations with those of the centralized model\n')
    cnn_metrics = compute_similarity(hfl_exps, c_system_exps)
    plot_heatmaps_compared(hfl_exps, c_system_exps, cnn_metrics)