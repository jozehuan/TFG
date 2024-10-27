import torch
import torch.nn as nn

from flex.model import FlexModel
from flex.pool import init_server_model

#---- simple network ----

class Net(nn.Module):
    
    """
    Red neuronal básica
    """
    
    def __init__(self, num_classes=10):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=1)
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        return nn.functional.log_softmax(x, dim=1)

@init_server_model
def build_server_Net_model():
    
    """
    Inicializa el servidor
    """
    
    server_flex_model = FlexModel()

    server_flex_model["model"] = Net()
    # Required to store this for later stages of the FL training process
    server_flex_model["criterion"] = torch.nn.CrossEntropyLoss()
    server_flex_model["optimizer_func"] = torch.optim.Adam
    server_flex_model["optimizer_kwargs"] = {}

    return server_flex_model

#---- convolutional network ----

class Net_CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=5),
            nn.Dropout(),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(320, 50),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(50, 10),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, 320)
        x = self.fc_layers(x)
        return x
    
@init_server_model
def build_server_CNN_model():
    
    """
    Inicializa el servidor
    """
    
    server_flex_model = FlexModel()

    server_flex_model["model"] = Net_CNN()
    # Required to store this for later stages of the FL training process
    server_flex_model["criterion"] = torch.nn.CrossEntropyLoss()
    server_flex_model["optimizer_func"] = torch.optim.Adam
    server_flex_model["optimizer_kwargs"] = {}

    return server_flex_model