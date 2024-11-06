import torch
import torch.nn as nn

from flex.model import FlexModel
from flex.pool import init_server_model

#---- simple neuronal network ----

class Net(nn.Module):
    """ A simple neural network for image classification (e.g., MNIST dataset).
    
    This model consists of:
    - A flattening layer to convert the input image into a vector.
    - Two fully connected (linear) layers.
    - ReLU and Softmax activation functions to produce class probabilities.
    
    Parameters
    ----------
    num_classes : int, optional
        The number of output classes for classification. Default is 10.
    
    Returns
    -------
    torch.Tensor
        Probabilities for each class, with values between 0 and 1 for each output node.
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
        return nn.functional.softmax(x, dim=1)

@init_server_model
def build_server_Net_model():  
    """ Builds and initializes the server-side neural network model for FL, using the decorator @init_server_model.
    This function creates a FlexModel instance and sets up the simple neural network, loss criterion, and optimizer.

    Parameters
    ----------
    None

    Returns
    -------
    FlexModel
        An instance of FlexModel containing the configured neural network model,
        the loss function (CrossEntropyLoss), and the optimizer (Adam).
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
    """ A Convolutional Neural Network (CNN) for image classification.

    This model consists of:
    - Convolutional layers with ReLU activations and max pooling for feature extraction.
    - Dropout layers to prevent overfitting.
    - Fully connected layers followed by ReLU and Softmax for classification.

    Parameters
    ----------
    None

    Returns
    -------
    torch.Tensor
        Probabilities for each class, with values between 0 and 1 for each output node.
    """
    
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
    """ Builds and initializes the server-side neural network model for FL, using the decorator @init_server_model.
    This function creates a FlexModel instance and sets up the CNN, loss criterion, and optimizer.

    Parameters
    ----------
    None

    Returns
    -------
    FlexModel
        An instance of FlexModel containing the configured neural network model,
        the loss function (CrossEntropyLoss), and the optimizer (Adam).
    """
    
    server_flex_model = FlexModel()

    server_flex_model["model"] = Net_CNN()
    # Required to store this for later stages of the FL training process
    server_flex_model["criterion"] = torch.nn.CrossEntropyLoss()
    server_flex_model["optimizer_func"] = torch.optim.Adam
    server_flex_model["optimizer_kwargs"] = {}

    return server_flex_model