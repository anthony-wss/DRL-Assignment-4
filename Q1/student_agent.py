import gymnasium as gym
import numpy as np
from train import SACAgent
import torch

# Do not modify the input of the 'act' function and the '__init__' function. 
class Agent(SACAgent):
    """Agent that acts randomly."""
    def __init__(self):
        # Pendulum-v1 has a Box action space with shape (1,)
        # Actions are in the range [-2.0, 2.0]
        state_dim = 3
        action_dim = 1
        action_range = [-2, 2]
        super(Agent, self).__init__(state_dim, action_dim, action_range)
        
        self.load_ckpt("./outputs/ckpt_999.pt")

    def act(self, observation):
        state = torch.tensor(observation, device=self.device)
        return self.select_action(state, deterministic=True)