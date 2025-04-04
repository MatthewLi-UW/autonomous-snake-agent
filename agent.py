import torch
import random
import numpy as np
import os.path
from collections import namedtuple
from game import ViperSimulation, Vector, Orientation
from model import ViperNetwork, PrioritizedReplayBuffer, ViperTrainer
from plot import TrainingPlotter

# how big each square of the game is
CELL_SIZE = 20

class ViperAgent:
    """
    our snake ai that learns to play through trial and error
    """
    def __init__(self, state_size=11, action_size=3):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.95  # how much it values future rewards
        self.epsilon = 1.0  # how often it tries random moves
        self.epsilon_min = 0.01  # it'll always explore a little bit
        self.epsilon_decay = 0.995  # gradually gets less random
        self.learning_rate = 0.001  # how quickly it adapts
        self.batch_size = 64  # how many memories it learns from at once
        
        # create two brains - one for making decisions, one for stable learning
        self.online_network = ViperNetwork(
            input_size=state_size, 
            hidden_size=256, 
            output_size=action_size,
            use_dueling=True  # fancy technique that learns better
        )
        self.target_network = ViperNetwork(
            input_size=state_size, 
            hidden_size=256, 
            output_size=action_size,
            use_dueling=True
        )
        
        # start both brains with the same knowledge
        self.target_network.load_state_dict(self.online_network.state_dict())
        
        # create a memory bank that prioritizes important experiences
        self.memory = PrioritizedReplayBuffer(
            capacity=100000,  # remembers lots of games
            alpha=0.6,  # how much to prioritize surprising outcomes
            beta=0.4  # how much to correct for priority bias
        )
        
        # set up the learning system
        self.trainer = ViperTrainer(
            model=self.online_network,
            target_model=self.target_network,
            lr=self.learning_rate,
            gamma=self.gamma
        )
        
        # try to load a saved brain if one exists
        self.model_file = './model/viper_model.pth'
        if os.path.exists(self.model_file):
            self.load_model()
            
        # for tracking and visualizing progress
        self.plotter = None
    
    def load_model(self):
        """load a previously trained brain"""
        try:
            loaded_model = ViperNetwork.load(self.model_file)
            if loaded_model:
                self.online_network = loaded_model
                self.target_network.load_state_dict(self.online_network.state_dict())
                print(f"loaded previous training from {self.model_file}")
        except Exception as e:
            print(f"couldn't load previous training: {e}")
    
    def save_model(self):
        """save the brain for later"""
        self.online_network.save(file_name='viper_model.pth')
        print("brain saved - we can pick up where we left off later")
    
    def get_state(self, game_state):
        """
        figure out what's important in the game right now
        creates a simplified snapshot of the game situation
        """
        # get key information about the game
        viper = game_state['viper']
        head = viper.head
        sustenance = game_state['food']
        orientation = viper.orientation
        
        # check what's in the surrounding squares
        point_east = Vector(head.x + 1, head.y)
        point_west = Vector(head.x - 1, head.y)
        point_north = Vector(head.x, head.y - 1)
        point_south = Vector(head.x, head.y + 1)
        
        # get boundaries of the game field
        width = viper.width
        height = viper.height
        
        # figure out which direction we're facing
        is_east = orientation == Orientation.EAST
        is_west = orientation == Orientation.WEST
        is_north = orientation == Orientation.NORTH
        is_south = orientation == Orientation.SOUTH
        
        # helper to check if moving to a spot would be fatal
        def is_collision(point):
            # hit wall?
            if (point.x < 0 or point.x >= width or 
                point.y < 0 or point.y >= height):
                return True
            
            # hit self?
            if point in viper.segments[1:]:
                return True
                
            return False
        
        # create a simple representation of what matters right now
        state = [
            # danger straight ahead?
            (is_east and is_collision(point_east)) or 
            (is_west and is_collision(point_west)) or 
            (is_north and is_collision(point_north)) or 
            (is_south and is_collision(point_south)),
            
            # danger if we turn right?
            (is_east and is_collision(point_south)) or 
            (is_west and is_collision(point_north)) or 
            (is_north and is_collision(point_east)) or 
            (is_south and is_collision(point_west)),
            
            # danger if we turn left?
            (is_east and is_collision(point_north)) or 
            (is_west and is_collision(point_south)) or 
            (is_north and is_collision(point_west)) or 
            (is_south and is_collision(point_east)),
            
            # which way are we currently facing?
            is_west,
            is_east,
            is_north,
            is_south,
            
            # where is the food relative to us?
            sustenance.x < head.x,  # food is to the left
            sustenance.x > head.x,  # food is to the right
            sustenance.y < head.y,  # food is above us
            sustenance.y > head.y   # food is below us
        ]
        
        return np.array(state, dtype=int)
    
    def select_action(self, state, training=True):
        """decide whether to explore randomly or use our knowledge"""
        # sometimes try something random to discover new strategies
        if training and random.random() <= self.epsilon:
            return random.randint(0, self.action_size - 1)
        
        # otherwise use what we've learned to make the best move
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            q_values = self.online_network(state_tensor)
            return torch.argmax(q_values).item()
    
    def store_transition(self, state, action, reward, next_state, done):
        """remember what happened for learning later"""
        self.memory.store(state, action, reward, next_state, done)
    
    def train_step(self, batch_size=None):
        """learn a bit from our past experiences"""
        if batch_size is None:
            batch_size = self.batch_size
            
        # need enough memories to learn from
        if len(self.memory) < batch_size:
            return
            
        # get a batch of interesting memories
        states, actions, rewards, next_states, dones, indices, weights = self.memory.sample(batch_size)
        
        # learn from these experiences
        td_errors = self.trainer.train_step(states, actions, rewards, next_states, dones, weights)
        
        # update which memories we think are important
        self.memory.update_priorities(indices, td_errors)
        
        # gradually shift from exploration to exploitation
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def train(self, num_games=1000, render=True, plot=True):
        """play lots of games to get better"""
        game = ViperSimulation()
        record = 0
        total_score = 0
        
        # setup progress tracking
        if plot:
            self.plotter = TrainingPlotter(
                plot_metrics=['score', 'avg_score', 'record', 'epsilon'],
                moving_avg_window=50
            )
        
        try:
            for i in range(num_games):
                # start a fresh game
                game_state = game.reset()
                state = self.get_state(game_state)
                game_over = False
                game_score = 0
                
                # keep playing until game ends
                while not game_over:
                    # decide what to do
                    action_idx = self.select_action(state)
                    action = [0, 0, 0]
                    action[action_idx] = 1
                    
                    # do it and see what happens
                    reward, game_over, game_score = game.step(action)
                    next_state = self.get_state(game._get_state())
                    
                    # remember and learn
                    self.store_transition(state, action_idx, reward, next_state, game_over)
                    self.train_step()
                    
                    # move to next moment
                    state = next_state
                
                # track our progress
                total_score += game_score
                avg_score = total_score / (i + 1)
                
                # celebrate and save if we beat our record
                if game_score > record:
                    record = game_score
                    self.save_model()
                    
                    # capture this achievement
                    if plot and self.plotter:
                        self.plotter.save(f"record_score_{record}.png")
                
                # update our progress charts
                if plot and self.plotter:
                    self.plotter.update(
                        i, 
                        score=game_score,
                        avg_score=avg_score,
                        record=record,
                        epsilon=self.epsilon
                    )
                
                # show progress update every so often
                if i % 50 == 0:
                    print(f"game {i}, score: {game_score}, record: {record}, avg: {avg_score:.2f}, randomness: {self.epsilon:.2f}")
            
            # training complete - save our results
            if plot and self.plotter:
                self.plotter.save("final_results.png")
                self.plotter.save_data("training_data.csv")
                self.plotter.close()
                
        except KeyboardInterrupt:
            print("training stopped manually")
            if plot and self.plotter:
                self.plotter.save("training_interrupted.png")
                self.plotter.save_data("training_data.csv")
                self.plotter.close()


# for old code that still uses the previous name
SnakeAgent = ViperAgent

def main():
    agent = ViperAgent()
    agent.train(num_games=1000, plot=True)


if __name__ == "__main__":
    main()