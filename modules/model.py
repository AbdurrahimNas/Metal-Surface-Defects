
import torch
from torch import nn 


class MetalSurfaceDefectsModel(nn.Module):
  def __init__(self,
               in_channels:int,
               out_channels:int,
               hidden_features:int):
    super().__init__()
    self.Layer_1 = nn.Sequential(
        nn.Conv2d(in_channels=in_channels,
                  out_channels=hidden_features,
                  kernel_size=2,
                  stride=1,
                  padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_features,
          out_channels=hidden_features,
          kernel_size=2,
          stride=1,
          padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
    )
    self.Layer_2 = nn.Sequential(
        nn.Conv2d(in_channels=hidden_features,
                  out_channels=hidden_features,
                  kernel_size=2,
                  stride=1,
                  padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_features,
          out_channels=hidden_features,
          kernel_size=2,
          stride=1,
          padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
    )
    self.out_layer = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Flatten(),
        nn.Linear(in_features=hidden_features*29*29,
                  out_features=out_channels)
    )
  def forward(self, x):
    #x = self.Layer_1(x)
    #print(x.shape)
    #x = self.Layer_2(x)
    #print(x.shape)
    #x = self.Layer_2(x)
    #print(x.shape)
    #x = self.out_layer(x)
    #print(x.shape)
    #return x
    return self.out_layer(self.Layer_2(self.Layer_2(self.Layer_1(x))))
def create_model(in_channels:int,
                out_channels:int,
                hidden_features:int,
                device:torch.device):

  """
  Creates a model.

  Keyword Arguments:
    :arg in_channels: in neurons
    :type in_channels:int 
    :arg out_channels: out neurons 
    :type out_channels: int 
    :arg hidden_features: hidden neurons
    :type hidden_features: int 
    :arg device: device to put model on "cuda" or "cpu"
    :type device: torch.device

  Example Usage:
    model = MetalSurfaceDefectsModel(in_channels=in_channels,
                                    out_channels=out_channels,
                                    hidden_features=hidden_features,
                                    device=device)
  """
  model = MetalSurfaceDefectsModel(in_channels=in_channels,
                                           out_channels=out_channels,
                                           hidden_features=hidden_features).to(device)

  return model
