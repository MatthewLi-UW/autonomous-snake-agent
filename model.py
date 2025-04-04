import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import numpy as np
from collections import deque
import random

class ViperNetwork(nn.Module):
    """
    Enhanced neural network architecture for viper AI
    Features deeper network with residual connections for better learning
    """
    def __init__(self, input_size, hidden_size, output_size, use_dueling=True):
        super().__init__()
        
        # Feature extraction layers
        self.feature_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Dueling architecture (separate value and advantage streams)
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
            # Combine value and advantages (with mean normalization)
            q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        else:
            q_values = self.output_layer(features)
            
        return q_values

    def save(self, file_name='viper_model.pth'):
        """Save model with metadata for better tracking"""
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_path = os.path.join(model_folder_path, file_name)
        
        # Save model state along with architecture info
        torch.save({
            'state_dict': self.state_dict(),
            'input_size': self.feature_layer[0].in_features,
            'hidden_size': self.feature_layer[0].out_features,
            'output_size': self.advantage_stream.out_features if self.use_dueling else self.output_layer.out_features,
            'use_dueling': self.use_dueling
        }, file_path)
        
    @classmethod
    def load(cls, file_path='./model/viper_model.pth'):
        """Load model with architecture reconstruction"""
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
    Enhanced memory buffer with prioritized experience replay
    Samples important transitions more frequently for better learning
    """
    def __init__(self, capacity=100000, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
        
        # PER hyperparameters
        self.alpha = alpha  # priority exponent
        self.beta = beta    # importance sampling weight
        self.beta_increment = beta_increment
        self.max_priority = 1.0
        
    def store(self, state, action, reward, next_state, done):
        """Store experience with maximum priority"""
        # Store experience
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
        """Sample batch with priority weights"""
        if self.size < batch_size:
            batch_size = self.size
            
        # Increase beta for importance sampling
        self.beta = min(1.0, self.beta + self.beta_increment)
            
        # Calculate sampling probabilities
        priorities = self.priorities[:self.size]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        # Sample indices based on priorities
        indices = np.random.choice(self.size, batch_size, replace=False, p=probabilities)
        
        # Calculate importance sampling weights
        weights = (self.size * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()
        weights = torch.tensor(weights, dtype=torch.float32)
        
        # Extract experiences
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for i in indices:
            state, action, reward, next_state, done = self.buffer[i]
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
            
        # Convert to tensors
        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(np.array(actions), dtype=torch.long)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        dones = torch.tensor(np.array(dones), dtype=torch.bool)
        
        return states, actions, rewards, next_states, dones, indices, weights
        
    def update_priorities(self, indices, priorities):
        """Update priorities based on TD errors"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
            
    def __len__(self):
        return self.size


class ViperTrainer:
    """
    Enhanced trainer using double Q-learning and prioritized experience replay
    """
    def __init__(self, model, target_model, lr=0.001, gamma=0.95):
        self.lr = lr
        self.gamma = gamma
        self.model = model  # Online network
        self.target_model = target_model  # Target network
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss(reduction='none')  # Element-wise loss for PER
        self.target_update_counter = 0
        self.target_update_frequency = 10  # Update target every N training steps
        
    def train_step(self, state, action, reward, next_state, done, weights=None):
        """
        Train the model on a batch of experiences
        With double Q-learning to reduce overestimation
        """
        # Handle single experience case
        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )
            if weights is not None:
                weights = torch.unsqueeze(weights, 0)
        
        # Current Q-values from online network
        current_q_values = self.model(state)
        
        # Double Q-learning: action selection from online network
        with torch.no_grad():
            online_next_q_values = self.model(next_state)
            best_actions = torch.argmax(online_next_q_values, dim=1, keepdim=True)
            
            # Value estimation from target network
            next_q_values = self.target_model(next_state)
            next_q_values = next_q_values.gather(1, best_actions).squeeze(1)
            
            # Calculate target Q-values
            target_q_values = reward + (~done) * self.gamma * next_q_values
        
        # Extract Q-values for taken actions - FIXED: action is already an index
        action_indices = action.unsqueeze(1)  # Changed from torch.argmax(action, dim=1)
        predicted_q_values = current_q_values.gather(1, action_indices).squeeze(1)
        
        # Calculate loss (with importance sampling weights if provided)
        elementwise_loss = self.criterion(predicted_q_values, target_q_values)
        if weights is not None:
            loss = (weights * elementwise_loss).mean()
        else:
            loss = elementwise_loss.mean()
        
        # Optimize model
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network periodically
        self.target_update_counter += 1
        if self.target_update_counter % self.target_update_frequency == 0:
            self.update_target()
            
        # Return TD errors for priority update
        return elementwise_loss.detach().numpy() + 1e-6  # Small constant for numerical stability
    
    def update_target(self):
        """Update target network with current model weights"""
        self.target_model.load_state_dict(self.model.state_dict())