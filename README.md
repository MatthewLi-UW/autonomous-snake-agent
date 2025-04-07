# 🐍 Autonomous Snake Agent

**Current bugs**:  
- Rendering bugs  
- Game number isn't updating  

---

This project implements a sophisticated reinforcement learning system for a Snake game AI. Below is a summary of the key machine learning concepts used:

---

## 🧐 Neural Network Architecture

- **Deep Neural Networks**: Multi-layer network implemented with PyTorch to approximate the Q-function  
- **Dueling Architecture**: Separates state-value and action-advantage streams to better identify state value independent of specific actions  
- **Feature Extraction**: Dedicated layers that extract meaningful representations from raw input states  

---

## 🔹 Reinforcement Learning Algorithms

- **Deep Q-Learning (DQN)**: Neural network-based Q-learning to handle high-dimensional state spaces  
- **Double Q-Learning**: Uses separate networks (online and target) to reduce value overestimation bias  
- **Prioritized Experience Replay (PER)**: Memory buffer that prioritizes important experiences based on TD error  
- **Importance Sampling**: Corrects bias introduced by non-uniform sampling with appropriate weights  

---

## ⚙️ Advanced Training Techniques

- **Target Network**: Separate network with delayed updates to provide stable learning targets  
- **Periodic Target Updates**: Syncs target network with online network on a fixed schedule  
- **Gradient Clipping**: Prevents exploding gradients during backpropagation  
- **TD Error Calculation**: Computes temporal difference errors to measure prediction quality  

---

## 🔧 Optimization Methods

- **Adam Optimizer**: Advanced gradient-based optimizer with adaptive learning rates  
- **Experience Buffering**: Stores transitions in memory to break correlation between sequential samples  
- **Batch Learning**: Training on randomly sampled batches rather than single experiences  

---

## 📂 Model Persistence

- **Checkpointing**: Saves and loads model state with architecture metadata  
- **Architectural Flexibility**: Supports different network configurations via parameters  

---

## 🎯 Rewards

| Event      | Reward |
|------------|--------|
| Eat        | +10    |
| Game Over  | -10    |
| Other      | 0      |

---

## 🔹 Actions

```python
[1, 0, 0]  # Straight  
[0, 1, 0]  # Right  
[0, 0, 1]  # Left  
```

---

## 📊 States

```python
[
  danger_straight, danger_right, danger_left,
  direction_left, direction_right, direction_up, direction_down,
  food_left, food_right, food_up, food_down
]
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

