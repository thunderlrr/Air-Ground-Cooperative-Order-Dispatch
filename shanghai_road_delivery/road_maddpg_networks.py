import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
from typing import List, Dict, Tuple

class CentralizedActorNetwork(nn.Module):
    
    def __init__(self, global_state_dim: int, num_agents: int, action_dim: int, hidden_dim: int = 512):
        super(CentralizedActorNetwork, self).__init__()
        self.num_agents = num_agents
        self.action_dim = action_dim
        
        self.fc1 = nn.Linear(global_state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        
        self.action_heads = nn.ModuleList([
            nn.Linear(hidden_dim, action_dim) for _ in range(num_agents)
        ])
        
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.1)
        
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        for head in self.action_heads:
            nn.init.uniform_(head.weight, -0.003, 0.003)
        
    def forward(self, global_state):
        x = F.relu(self.fc1(global_state))
        x = self.dropout(x) if self.training else x
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        actions = []
        for i, head in enumerate(self.action_heads):
            action = self.tanh(head(x))
            actions.append(action)
        
        actions = torch.stack(actions, dim=1)
        return actions

class CentralizedCriticNetwork(nn.Module):
    
    def __init__(self, global_state_dim: int, num_agents: int, action_dim: int, hidden_dim: int = 512):
        super(CentralizedCriticNetwork, self).__init__()
        
        total_input_dim = global_state_dim + num_agents * action_dim
        
        self.fc1 = nn.Linear(total_input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1)
        
        self.dropout = nn.Dropout(0.2)
        
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.uniform_(self.fc4.weight, -0.003, 0.003)
        
    def forward(self, global_state, actions):
        actions_flat = actions.view(actions.shape[0], -1)
        x = torch.cat([global_state, actions_flat], dim=1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x) if self.training else x
        x = F.relu(self.fc2(x))
        x = self.dropout(x) if self.training else x
        x = F.relu(self.fc3(x))
        q_value = self.fc4(x)
        
        return q_value

class PrioritizedReplayBuffer:
    
    def __init__(self, maxlen=100000, alpha=0.6):
        self.buffer = deque(maxlen=maxlen)
        self.priorities = deque(maxlen=maxlen)
        self.alpha = alpha
        self.max_priority = 1.0
        
    def append(self, experience):
        self.buffer.append(experience)
        self.priorities.append(self.max_priority)
    
    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) < batch_size:
            return None
            
        priorities = np.array(self.priorities)
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        batch = [self.buffer[idx] for idx in indices]
        return batch, indices, weights
    
    def update_priorities(self, indices, td_errors):
        for idx, td_error in zip(indices, td_errors):
            priority = abs(td_error) + 1e-6
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self):
        return len(self.buffer)

class RoadMADDPG:
    
    def __init__(self, num_agents: int, state_dim: int, action_dim: int, 
                 lr_actor: float = 0.01, lr_critic: float = 0.02, 
                 gamma: float = 0.95, tau: float = 0.005, batch_size=32, hidden_dim: int = 256):
        
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        
        self.global_state_dim = state_dim * num_agents
        
        self.exploration_noise = 0.2
        self.noise_decay = 0.995
        self.min_noise = 0.05
        self.current_episode = 0
        
        self.actor = CentralizedActorNetwork(self.global_state_dim, num_agents, action_dim, hidden_dim)
        self.critic = CentralizedCriticNetwork(self.global_state_dim, num_agents, action_dim, hidden_dim)
        
        self.target_actor = CentralizedActorNetwork(self.global_state_dim, num_agents, action_dim, hidden_dim)
        self.target_critic = CentralizedCriticNetwork(self.global_state_dim, num_agents, action_dim, hidden_dim)
        
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor, weight_decay=1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic, weight_decay=1e-4)
        
        self.actor_scheduler = optim.lr_scheduler.ExponentialLR(self.actor_optimizer, gamma=0.995)
        self.critic_scheduler = optim.lr_scheduler.ExponentialLR(self.critic_optimizer, gamma=0.995)
        
        self.memory = PrioritizedReplayBuffer(maxlen=100000)
        self.batch_size = batch_size
        
        self.recent_actor_loss = 0.0
        self.recent_critic_loss = 0.0
        
        print(f"Centralized Road MADDPG initialized: {num_agents} agents")
        print(f"Global state dimension: {self.global_state_dim}, Action dimension: {action_dim}")
        print(f"Learning rates: Actor={lr_actor}, Critic={lr_critic}")
        print(f"Network: Centralized Actor+Critic, Hidden={hidden_dim}")
    
    def get_actions(self, states, add_noise=True, noise_scale=0.1):
        if isinstance(states, list):
            global_state = np.concatenate(states)
        else:
            global_state = states.flatten()
        
        global_state_tensor = torch.FloatTensor(global_state).unsqueeze(0)
        
        self.actor.eval()
        with torch.no_grad():
            actions_tensor = self.actor(global_state_tensor)
            actions_tensor = actions_tensor.squeeze(0)
        
        actions = []
        for i in range(self.num_agents):
            action = actions_tensor[i].numpy()
            
            if add_noise:
                current_noise = max(
                    self.exploration_noise * (self.noise_decay ** self.current_episode),
                    self.min_noise
                )
                
                if np.random.random() < 0.8:
                    noise = np.random.normal(0, current_noise, action.shape)
                else:
                    noise = np.random.uniform(-current_noise, current_noise, action.shape)
                
                action = np.clip(action + noise, -1, 1)
            
            actions.append(action)
        
        return actions
    
    def store_transition(self, states, actions, rewards, next_states, done):
        if isinstance(states, list):
            global_state = np.concatenate(states)
            global_next_state = np.concatenate(next_states)
        else:
            global_state = states.flatten()
            global_next_state = next_states.flatten()
        
        if isinstance(actions, list):
            actions_array = np.array(actions)
        else:
            actions_array = actions
        
        global_reward = sum(rewards) if isinstance(rewards, list) else rewards
        
        experience = (global_state, actions_array, global_reward, global_next_state, done)
        self.memory.append(experience)
    
    def soft_update(self, target_net, main_net, tau):
        for target_param, main_param in zip(target_net.parameters(), main_net.parameters()):
            target_param.data.copy_(tau * main_param.data + (1 - tau) * target_param.data)
    
    def update(self):
        if len(self.memory) < self.batch_size:
            return {"actor_loss": 0, "critic_loss": 0}
        
        self.actor.train()
        self.critic.train()
        
        batch_data = self.memory.sample(self.batch_size)
        if batch_data is None:
            return {"actor_loss": 0, "critic_loss": 0}
        
        batch, indices, importance_weights = batch_data
        
        global_states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.FloatTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch]).unsqueeze(1)
        global_next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.FloatTensor([e[4] for e in batch]).unsqueeze(1)
        importance_weights = torch.FloatTensor(importance_weights).unsqueeze(1)
        
        with torch.no_grad():
            next_actions = self.target_actor(global_next_states)
            target_q = self.target_critic(global_next_states, next_actions)
            target_q = rewards + self.gamma * target_q * (1 - dones)
        
        current_q = self.critic(global_states, actions)
        
        td_errors = (target_q - current_q).detach().numpy().flatten()
        
        critic_loss = (importance_weights * F.mse_loss(current_q, target_q, reduction='none')).mean()
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()
        
        predicted_actions = self.actor(global_states)
        actor_loss = -self.critic(global_states, predicted_actions).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()
        
        self.soft_update(self.target_actor, self.actor, self.tau)
        self.soft_update(self.target_critic, self.critic, self.tau)
        
        self.memory.update_priorities(indices, td_errors)
        
        if hasattr(self, 'update_count'):
            self.update_count += 1
        else:
            self.update_count = 1
            
        if self.update_count % 200 == 0:
            self.actor_scheduler.step()
            self.critic_scheduler.step()
        
        self.recent_actor_loss = actor_loss.item()
        self.recent_critic_loss = critic_loss.item()
        
        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item()
        }
    
    def save_models(self, path_prefix: str):
        torch.save(self.actor.state_dict(), f"{path_prefix}_centralized_actor.pth")
        torch.save(self.critic.state_dict(), f"{path_prefix}_centralized_critic.pth")
        print(f"Centralized models saved: {path_prefix}")
    
    def load_models(self, path_prefix: str):
        self.actor.load_state_dict(torch.load(f"{path_prefix}_centralized_actor.pth"))
        self.critic.load_state_dict(torch.load(f"{path_prefix}_centralized_critic.pth"))
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        print(f"Centralized models loaded: {path_prefix}")
    
    def set_episode(self, episode):
        self.current_episode = episode
