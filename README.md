# 🐍 Autonomous Snake Agent

**Current bugs**:  
- Rendering bugs  
- Game number isn't updating  

---

This project implements a reinforcement learning system that enables a neural network-based agent to learn and play the classic Snake game. It combines modern RL strategies with neural network architectures, including Dueling DQNs and Prioritized Experience Replay.

---

## 📽️ Demo

> Add a demo GIF here (e.g. `docs/demo.gif`):

```
![Snake Agent Demo](docs/demo.gif)
```

---

## 📈 Training Loop Overview

![Training Flowchart](docs/training_flowchart.png)

1. **Observe State** →  
2. **Select Action** (via Epsilon-Greedy) →  
3. **Perform Action** in Game →  
4. **Get Next State and Reward** →  
5. **Store Transition** in Replay Buffer →  
6. **Sample Batch** →  
7. **Compute Target Q-values** →  
8. **Backpropagate Loss** →  
9. **Update Target Network Periodically**

---

## 📊 State, Actions, and Rewards

### Input State (Size = 11)
```python
[
  danger_straight, danger_right, danger_left,
  direction_left, direction_right, direction_up, direction_down,
  food_left, food_right, food_up, food_down
]
```

### Actions (Output = 3)
```python
[1, 0, 0]  # Straight  
[0, 1, 0]  # Right  
[0, 0, 1]  # Left  
```

### Rewards
| Event      | Reward |
|------------|--------|
| Eat        | +10    |
| Game Over  | -10    |
| Other      | 0      |

---

## 🧐 Neural Network Architecture

A simple feedforward network maps 11-dimensional input states to 3 possible actions.

### Model Diagram
![nn (1)](https://github.com/user-attachments/assets/1cf5d652-e4f8-49e6-b869-671bc9aeb2f9)

### Architecture
```python
class ViperNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, use_dueling=True):
        super().__init__()

        # Feature extraction layers
        self.feature_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

        # Dueling architecture
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
            q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        else:
            q_values = self.output_layer(features)
        return q_values
```

---

## ♻️ Reinforcement Learning Algorithms

### Deep Q-Learning (DQN)
Approximates the Q-function using a neural network:
```python
target_q_values = reward + (~done) * self.gamma * next_q_values
predicted_q_values = current_q_values.gather(1, action.unsqueeze(1)).squeeze(1)
```

### Double Q-Learning
Reduces overestimation by using two networks:
```python
with torch.no_grad():
    online_next_q_values = self.model(next_state)
    best_actions = torch.argmax(online_next_q_values, dim=1, keepdim=True)
    next_q_values = self.target_model(next_state).gather(1, best_actions).squeeze(1)
```

### Prioritized Experience Replay (PER)
Samples based on TD error:
```python
probabilities = priorities ** self.alpha
probabilities /= probabilities.sum()
indices = np.random.choice(self.size, batch_size, replace=False, p=probabilities)
```

### Importance Sampling
Corrects the bias introduced by PER:
```python
weights = (self.size * probabilities[indices]) ** (-self.beta)
weights /= weights.max()
loss = (weights * elementwise_loss).mean()
```

---

## ⚙️ Training Techniques

### Target Network
```python
def update_target(self):
    self.target_model.load_state_dict(self.model.state_dict())
```

### Gradient Clipping
```python
torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
```

### TD Error Updates
```python
return elementwise_loss.detach().numpy() + 1e-6

for idx, priority in zip(indices, priorities):
    self.priorities[idx] = priority
    self.max_priority = max(self.max_priority, priority)
```

---

## 🔧 Optimization

### Optimizer
```python
self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
```

### Batch Training
```python
states = torch.tensor(np.array(states), dtype=torch.float32)
actions = torch.tensor(np.array(actions), dtype=torch.long)
rewards = torch.tensor(np.array(rewards), dtype=torch.float32)
next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
dones = torch.tensor(np.array(dones), dtype=torch.bool)
```

---

## 📀 Model Persistence

### Checkpointing
```python
torch.save({
    'state_dict': self.state_dict(),
    'input_size': self.feature_layer[0].in_features,
    'hidden_size': self.feature_layer[0].out_features,
    'output_size': self.advantage_stream.out_features if self.use_dueling 
                  else self.output_layer.out_features,
    'use_dueling': self.use_dueling
}, file_path)
```

---

## 🛠️ Dev Notes

### Set up environment:
```bash
conda create -n pygame_env python=3.7
conda activate pygame_env
```

### Install dependencies:
```bash
pip install pygame
pip3 install torch torchvision
pip install matplotlib ipython
pip install pandas
```
