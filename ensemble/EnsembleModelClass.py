import torch
import torch.nn as nn                                                                                                                                       
import torch.nn.functional as F

layer1_in = 5
layer1_out = 8 
layer2_out = 4 
layer3_out = 2

class EnsembleModel(nn.Module):
      def __init__(self):
          super().__init__()
          self.fc1 = nn.Linear(in_features = layer1_in, out_features = layer1_out, bias = True)
          self.fc2 = nn.Linear(in_features = layer1_out, out_features = layer2_out, bias = True)
          self.fc3 = nn.Linear(in_features = layer2_out, out_features = layer3_out, bias = True)
            
      def forward(self, X):
          X = F.relu(self.fc1(X))
          X = F.relu(self.fc2(X))
          #X = F.relu(self.fc3(X))
          X = F.softmax(self.fc3(X), dim = 1)
          return X


          

