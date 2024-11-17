import time

from torchvision import transforms

import models as nets

from visualization import plot_heatmaps, plot_segments, barplot

from architectures import HFL_System

device = 'cpu'

config = {"download": True, "dataset": "mnist", "transform": transforms.Compose([transforms.ToTensor()]),
          "config_seed": 42, "replacement": False, "nodes": 20, "n_classes": 10, "server_weight": 0.05,
          "balance_nodes": False, "nodes_weights": None, "balance_factor": 0.25, "balance_classes": True,
          "alpha_inf": 0.4, "alpha_sup": 0.6}

# ---- CNN case --------------------------------------------------------------------------------

# Generate the Client-Server federated system
hfl_cs_cnn_sys = HFL_System(name='hfl_cs_cnn', **config)

# Show the data distribution on the nodes (node 0 = server) 
barplot(hfl_cs_cnn_sys.class_counter(), 'hfl_cs_cnn', ncols=3)

# Set CNN model defined in models module
hfl_cs_cnn_sys.set_model(arch = 'cs', build = nets.build_server_CNN_model)

# Set the federated model's explainers
hfl_cs_cnn_sys.set_explainers()

# Tomar datos (NO HACER PORQUE SERÁN LOS DATOS DEL SERVIDOR)
data = hfl_cs_cnn_sys._flex_pool.servers._data.data[0]
data = data[0:50]

n_rounds = 5
# Train federated model
start_time = time.time()
exp_list = hfl_cs_cnn_sys.train_n_rounds(n_rounds = 30, data_to_explain=data)
end_time = time.time()
time_train_cnn = end_time - start_time
print(f"\n\nTrain time (CNN): {time_train_cnn:.2f} seconds")

for i, exp_round in enumerate(exp_list):
    explanations, label_explanations, segments = exp_round
    
    plot_segments(*segments, n_round = i+1)
    plot_heatmaps(*explanations, n_round = i+1)
    plot_heatmaps(*label_explanations, n_round = i+1)

