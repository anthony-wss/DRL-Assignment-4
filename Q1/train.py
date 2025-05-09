import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import gymnasium as gym
import random
import numpy as np
from torch.distributions import Normal
import time
import os

# Hyperparameters
LEARNING_RATE = 0.001
BATCH_SIZE = 64
GAMMA = 0.99
TAU = 0.05
ALPHA = 0.2
REPLAY_BUFFER_CAPACITY = 100000
EPISODES = 1000
RANDOM_START_STEPS = 10000
HIDDEN_DIM = 256
SAVE_STEPS = int(50000)

SEED = 197

class ReplayBuffer:
    """ Simple replay buffer with deque + uniform sampling.

    All the data format is stored in torch.Tensor.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.stack, zip(*batch))
        return states, actions, rewards, next_states, dones
    
    def size(self):
        return len(self.buffer)

class QNetwork(nn.Module):
    """ Q network for SAC algorithm.

    Q(s, a) = the value of the action a at state s.
    """
    def __init__(self, state_dim, action_dim, hidden_dim=HIDDEN_DIM):
        super(QNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.layers(x)

class PolicyNetwork(nn.Module):
    """ Policy network for SAC algorithm.

    pi(s) = the action distribution at state s.
    """
    def __init__(self, state_dim, action_dim, action_range, hidden_dim=HIDDEN_DIM):
        super(PolicyNetwork, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)

        self.register_buffer("action_scale", torch.tensor((action_range[1] - action_range[0]) / 2.0, dtype=torch.float32))
        self.register_buffer("action_bias", torch.tensor((action_range[1] + action_range[0]) / 2.0, dtype=torch.float32))

    
    def forward(self, state):
        """ Given a state, return the mean and std of the action distribution. """
        x = self.encoder(state)
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        std = torch.exp(log_std)
        return mean, std
    
    def sample(self, state, deterministic=False):
        """ Sample an action from the action distribution. 
        
        Will also return the log probability of the action.
        
        Args:
            state: torch.Tensor, current state array
            deterministic: False to add Gaussian noise to the action
        Returns:
            action: the sampled action
            log_prob: the log probability for entropy maximization
        """
        mean, std = self.forward(state)

        if deterministic:
            mean = torch.tanh(mean) * self.action_scale + self.action_bias
            return mean, None

        dist = Normal(mean, std)
        x_t = dist.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias

        # log probability for entropy maximization
        log_prob = dist.log_prob(x_t)
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob

class SACAgent:
    """ Agent that uses SAC algorithm for continuous action spaces """
    def __init__(self, state_dim, action_dim, action_range):
        
        self.replay_buffer = ReplayBuffer(capacity=REPLAY_BUFFER_CAPACITY)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        
        # Critic network
        self.critic = QNetwork(state_dim, action_dim).to(self.device)
        self.critic_target = QNetwork(state_dim, action_dim).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LEARNING_RATE)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Policy network
        self.policy_network = PolicyNetwork(state_dim, action_dim, action_range).to(self.device)
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=LEARNING_RATE)

    def select_action(self, state, deterministic=False):
        """ Select action from policy network

        Args:
            state: current state array
            deterministic: False to add Gaussian noise to the action
        Returns:
            action: the selected action
        """
        with torch.no_grad():
            action, _ = self.policy_network.sample(state, deterministic)
            return action.cpu().numpy()
    
    def learn_step(self):
        """ Perform one step of learning """
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(BATCH_SIZE)
        
        states = torch.tensor(states, device=self.device, dtype=torch.float32)
        actions = torch.tensor(actions, device=self.device, dtype=torch.float32)
        rewards = torch.tensor(rewards, device=self.device, dtype=torch.float32)
        next_states = torch.tensor(next_states, device=self.device, dtype=torch.float32)
        dones = torch.tensor(dones, device=self.device, dtype=torch.float32)
        
        with torch.no_grad():
            next_action, next_log_prob = self.policy_network.sample(next_states)
            next_q_value = self.critic_target(next_states, next_action)
            min_next_q_target = (next_q_value - ALPHA * next_log_prob).squeeze()
            next_q_target = rewards + GAMMA * (1 - dones) * min_next_q_target
        
        # Critic network
        q_value = self.critic(states, actions).squeeze()
        critic_loss = F.mse_loss(q_value, next_q_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Policy network
        action, log_prob = self.policy_network.sample(states)
        q_value = self.critic(states, action)
        policy_loss = -torch.mean(q_value - ALPHA * log_prob)
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Update target network
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
    
    def save_ckpt(self, save_dir, total_steps):
        """ Save all the components for evaluation, not resume training.
        
        Save critic_target network and policy network only.
        """
        save_path = os.path.join(save_dir, f"ckpt_{total_steps}.pt")
        os.makedirs(save_dir, exist_ok=True)
        models = {
            "critic_target": self.critic_target.state_dict(),
            "policy_network": self.policy_network.state_dict()
        }
        with open(save_path, "wb") as fp:
            torch.save(models, fp)
        print(f"Save ckpt at step: {total_steps}")

        # Move to the original device
        self.critic_target.to(self.device)
        self.policy_network.to(self.device)
    
    def load_ckpt(self, ckpt_path):
        """ Load from ckpt for evaluation only. """
        with open(ckpt_path, "rb") as fp:
            checkpoint = torch.load(fp, weights_only=False)
            self.critic_target.load_state_dict(checkpoint["critic_target"])
            self.policy_network.load_state_dict(checkpoint["policy_network"])

def make_env():
    # Create Pendulum-v1 environment
    env = gym.make("Pendulum-v1", render_mode=None)
    env.reset(seed=SEED)
    env.action_space.seed(SEED)
    return env

def train():
    env = make_env()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_range = [env.action_space.low, env.action_space.high]
    agent = SACAgent(state_dim, action_dim, action_range)

    total_steps = 0

    for episode in range(EPISODES):
        episode_reward = 0
        episode_steps = 0
        state, _ = env.reset()
        done = False

        time_elapsed = time.time()

        while not done:
            if total_steps < RANDOM_START_STEPS:
                action = env.action_space.sample()
            else:
                action = agent.select_action(torch.tensor(state, device=agent.device, dtype=torch.float32))
            
            if agent.replay_buffer.size() >= BATCH_SIZE:
                agent.learn_step()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            episode_steps += 1
            total_steps += 1

            if total_steps % SAVE_STEPS == 0:
                agent.save_ckpt("./outputs", total_steps)


        
        print(f"Episode {episode}, total steps: {total_steps}, episode reward: {episode_reward}")
        print(f"Avg time elapsed per step(ms): {(time.time() - time_elapsed) / episode_steps * 1000}")

if __name__ == "__main__":
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    train()