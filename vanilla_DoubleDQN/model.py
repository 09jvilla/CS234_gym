import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class FwdModel(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_units=256):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(FwdModel, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size+action_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, state_size)

    def forward(self, state, act_onehot):
        """Build a network that maps state -> action values."""
        
        inp = torch.cat((state,act_onehot), dim=1)
        x = F.relu(self.fc1(inp))
        x = self.fc2(x) #no activation function since state can be anything
        return x

class InverseModel(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_units=256):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(InverseModel, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size*2, fc1_units)
        self.fc2 = nn.Linear(fc1_units, action_size)

    def forward(self, state, next_state):
        """Build a network that maps state -> action values."""
        inp = torch.cat((state,next_state), dim=1)
        x = F.relu(self.fc1(inp))
        x = self.fc2(x) #un-normalized. cross entropy loss will take care of applying softmax
        return x
