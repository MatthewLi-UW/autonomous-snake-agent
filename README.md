# autonomous-snake-agent
current bugs: rendering bugs, game number isnt updating

This project implements a sophisticated reinforcement learning system for a Snake game AI. Here's a summary of the key machine learning concepts used:\

Neural Network Architecture\
Deep Neural Networks: Multi-layer network implemented with PyTorch to approximate the Q-function\
Dueling Architecture: Separates state-value and action-advantage streams to better identify state value independent of specific actions
Feature Extraction: Dedicated layers that extract meaningful representations from raw input states
Reinforcement Learning Algorithms
Deep Q-Learning (DQN): Neural network-based Q-learning to handle high-dimensional state spaces
Double Q-Learning: Uses separate networks (online and target) to reduce value overestimation bias
Prioritized Experience Replay (PER): Memory buffer that prioritizes important experiences based on TD error
Importance Sampling: Corrects the bias introduced by non-uniform sampling with appropriate weights
Advanced Training Techniques
Target Network: Separate network with delayed updates to provide stable learning targets
Periodic Target Updates: Syncs target network with online network on a fixed schedule
Gradient Clipping: Prevents exploding gradients during backpropagation
TD Error Calculation: Computes temporal difference errors to measure prediction quality
Optimization Methods
Adam Optimizer: Advanced gradient-based optimizer with adaptive learning rates
Experience Buffering: Stores transitions in memory to break correlation between sequential samples
Batch Learning: Training on randomly sampled batches rather than single experiences
Model Persistence
Checkpointing: Saves and loads model state with architecture metadata
Architectural Flexibility: Support for different network configurations via parameters


Rewards:
Eat: +10
Game over: -10
Other: 0

Actions:
Straight: [1,0,0]
Right: [0,1,0]
Left: [0,0,1]

States:
[danger straight, danger right, danger left,
direction left, direction right, direction up, direction down,
food left, food right, food up, food down]

**Dev notes**

Set up environment:

conda create -n pygame_env python=3.7

conda activate pygame_env

Install dependencies:

pip install pygame

pip3 install torch torchvision

pip install matplotlib ipython

pip install pandas

