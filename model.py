import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import np
from collections import deque
import random

class ViperNetwork(nn.Module):
    """
    the brain of our snake ai 
    has multiple layers to help it understand complex patterns
    """
    def __init__(self, input_size, hidden_size, output_size, use_dueling=True):
        super().__init__()
        
        # layers that find interesting patterns in the game state
        self.feature_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # split thinking into "how good is the situation" and "which move is best"
        self.use_dueling = use_dueling
        if use_dueling:
            self.value_stream = nn.Linear(hidden_size, 1)
            self.advantage_stream = nn.Linear(hidden_size, output_size)
        else:
            self.output_layer = nn.Linear(hidden_size, output_size)
            
    def forward(self, x):
        features = self.feature_layer(x)
        
        if self.use_dueling:
            value = self.value_stream(features)
            advantages = self.advantage_stream(features)
            # combine overall situation assessment with specific move advantages
            q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        else:
            q_values = self.output_layer(features)
            
        return q_values

    def save(self, file_name='viper_model.pth'):
        """save the brain so we can use it again later"""
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_path = os.path.join(model_folder_path, file_name)
        
        # remember all the important bits about how this brain works
        torch.save({
            'state_dict': self.state_dict(),
            'input_size': self.feature_layer[0].in_features,
            'hidden_size': self.feature_layer[0].out_features,
            'output_size': self.advantage_stream.out_features if self.use_dueling else self.output_layer.out_features,
            'use_dueling': self.use_dueling
        }, file_path)
        
    @classmethod
    def load(cls, file_path='./model/viper_model.pth'):
        """reload a saved brain from a file"""
        if not os.path.exists(file_path):
            return None
            
        checkpoint = torch.load(file_path)
        model = cls(
            input_size=checkpoint['input_size'],
            hidden_size=checkpoint['hidden_size'],
            output_size=checkpoint['output_size'],
            use_dueling=checkpoint['use_dueling']
        )
        model.load_state_dict(checkpoint['state_dict'])
        return model


class PrioritizedReplayBuffer:
    """
    the snake's memory - remembers important experiences better than boring ones
    focuses on surprising or interesting moments for better learning
    """
    def __init__(self, capacity=100000, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
        
        # settings for how much to focus on surprising events
        self.alpha = alpha  # how much to prioritize surprising stuff
        self.beta = beta    # how much to correct for focusing too much on surprises
        self.beta_increment = beta_increment
        self.max_priority = 1.0
        
    def store(self, state, action, reward, next_state, done):
        """remember a new experience, assuming it might be important"""
        # save the experience
        experience = (state, action, reward, next_state, done)
        
        if self.size < self.capacity:
            self.buffer.append(experience)
            self.priorities[self.size] = self.max_priority
            self.size += 1
        else:
            self.buffer[self.position] = experience
            self.priorities[self.position] = self.max_priority
            
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        """pick out some memories to learn from, favoring surprising ones"""
        if self.size < batch_size:
            batch_size = self.size
            
        # gradually correct more for our focus on surprising events
        self.beta = min(1.0, self.beta + self.beta_increment)
            
        # figure out which memories to pick
        priorities = self.priorities[:self.size]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        # pick memories based on how surprising they were
        indices = np.random.choice(self.size, batch_size, replace=False, p=probabilities)
        
        # calculate how much to adjust for our bias toward surprising events
        weights = (self.size * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()
        weights = torch.tensor(weights, dtype=torch.float32)
        
        # grab the selected memories
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for i in indices:
            state, action, reward, next_state, done = self.buffer[i]
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
            
        # format everything for the learning process
        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(np.array(actions), dtype=torch.long)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        dones = torch.tensor(np.array(dones), dtype=torch.bool)
        
        return states, actions, rewards, next_states, dones, indices, weights
        
    def update_priorities(self, indices, priorities):
        """adjust how surprising we think certain memories are"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
            
    def __len__(self):
        return self.size


class ViperTrainer:
    """
    teaches the snake ai using its memories and two brains
    one brain makes decisions, the other gives more stable learning targets
    """
    def __init__(self, model, target_model, lr=0.001, gamma=0.95):
        self.lr = lr
        self.gamma = gamma
        self.model = model  # active brain for making decisions
        self.target_model = target_model  # stable brain for consistent learning
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss(reduction='none')  # lets us weight different memories
        self.target_update_counter = 0
        self.target_update_frequency = 10  # how often to sync the two brains
        
    def train_step(self, state, action, reward, next_state, done, weights=None):
        """
        teach the snake using one batch of memories
        uses two brains to avoid being too optimistic about future rewards
        """
        # handle single memory case
        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )
            if weights is not None:
                weights = torch.unsqueeze(weights, 0)
        
        # what our active brain thinks of these situations
        current_q_values = self.model(state)
        
        # use both brains together for better learning
        with torch.no_grad():
            # active brain picks the actions (what it thinks is best)
            online_next_q_values = self.model(next_state)
            best_actions = torch.argmax(online_next_q_values, dim=1, keepdim=True)
            
            # stable brain estimates how good those actions would be
            next_q_values = self.target_model(next_state)
            next_q_values = next_q_values.gather(1, best_actions).squeeze(1)
            
            # combine immediate rewards with future expectations
            target_q_values = reward + (~done) * self.gamma * next_q_values
        
        # get what our active brain thought about the actions we took
        action_indices = action.unsqueeze(1)
        predicted_q_values = current_q_values.gather(1, action_indices).squeeze(1)
        
        # calculate how wrong our brain was
        elementwise_loss = self.criterion(predicted_q_values, target_q_values)
        if weights is not None:
            # pay more attention to surprising memories
            loss = (weights * elementwise_loss).mean()
        else:
            loss = elementwise_loss.mean()
        
        # improve the brain based on what we learned
        self.optimizer.zero_grad()
        loss.backward()
        # prevent wild overreactions
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        # occasionally update our stable brain
        self.target_update_counter += 1
        if self.target_update_counter % self.target_update_frequency == 0:
            self.update_target()
            
        # tell the memory system how surprising each experience was
        return elementwise_loss.detach().numpy() + 1e-6  # tiny value to avoid zeros
    
    def update_target(self):
        """sync our stable brain with what the active brain has learned"""
        self.target_model.load_state_dict(self.model.state_dict())