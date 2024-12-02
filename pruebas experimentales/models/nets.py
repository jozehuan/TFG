import torch
import torch.nn as nn

from flex.model import FlexModel
from flex.pool import init_server_model

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
        )
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, 320)
        x = self.fc_layers(x)
        return x
    

@init_server_model
def build_server_CNN_model(lr : float = 0.001):
    """ Builds and initializes the server-side neural network model for FL, using the decorator @init_server_model.
    This function creates a FlexModel instance and sets up the CNN, loss criterion, and optimizer.

    Parameters
    ----------
    lr : float, optional
        Adam optimizer's learining rate

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
    server_flex_model["optimizer_kwargs"] = {'lr' : lr}

    return server_flex_model