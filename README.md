# autonomous-snake-agent
current bugs: rendering bugs, game number isnt updating

![image](https://github.com/user-attachments/assets/7c703450-41f5-4e8e-b165-8a24226cbff5)
Part 1: RL Theory
Reinforcement learning is concerned with how we can teach software agents to behave in an environment, by maximizing the notion of cumulative reward.
Deep Q Learning: extends RL by using a deep neural network to predict the actions

Part 2: Game
Game loop -> play step -> gets action -> moves snake -> returns current reward, game over status, score -> agent

Part 3: Agent
Calculate state based off game -> calculate next action (call model predict) -> perform next action -> get new state -> remember states -> train model

Part 4: Model (Linear_QNet)
Feed forward neural net with a few linear layers
![image](https://github.com/user-attachments/assets/09e193ba-7ab2-4fdc-97bb-7f35d7abdfac)

Deep Q Learning:
Q value = quality of the action
Steps:
0. Initialize Q value
1. Choose action (predict, or random move when we have little info)
2. Perform action
3. Measure award
4. Update Q value and train model

Train the model using a loss function, which is used to gauge the deviation/error margin between model behaviour and actual target value
![image](https://github.com/user-attachments/assets/9aac0bb8-3962-4425-99cb-377125661702)
![image](https://github.com/user-attachments/assets/1c1cba4a-1312-4524-9f0b-b453b15a47f7)
![image](https://github.com/user-attachments/assets/971f95b1-f3be-4088-91a3-2ec549239afd)


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
