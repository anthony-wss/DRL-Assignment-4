import gymnasium
import numpy as np
from train import SACAgent
import torch

# Do not modify the input of the 'act' function and the '__init__' function. 
class Agent(SACAgent):
    """Agent that acts randomly."""
    def __init__(self):
        state_dim = 67
        action_dim = 21
        action_range = [-1, 1]
        super(Agent, self).__init__(state_dim, action_dim, action_range)
        
        self.load_ckpt("./ckpt_best.pt")

    def act(self, observation):
        state = torch.tensor(observation, dtype=torch.float, device=self.device)
        return self.select_action(state, deterministic=True)

