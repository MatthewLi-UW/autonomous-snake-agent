import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import deque
import os
import time

class TrainingPlotter:
    """
    A class for real-time visualization and tracking of training metrics
    """
    def __init__(self, plot_metrics=['score', 'avg_score', 'record', 'epsilon'], 
                 moving_avg_window=100, save_dir='./plots'):
        self.metrics = {metric: [] for metric in plot_metrics}
        self.episodes = []
        self.episode_counter = 0
        self.moving_avg_window = moving_avg_window
        self.save_dir = save_dir
        self.last_update_time = time.time()
        self.update_interval = 2.0  # seconds between plot updates
        
        # Create save directory if it doesn't exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        # Initialize plot
        self.fig, self.axs = plt.subplots(2, 1, figsize=(10, 8))
        plt.ion()  # Turn on interactive mode
        self.fig.tight_layout(pad=3.0)
        
    def update(self, episode_num, **kwargs):
        """
        Update metrics with new values
        """
        self.episodes.append(episode_num)
        self.episode_counter = episode_num
        
        # Update all provided metrics
        for metric, value in kwargs.items():
            if metric in self.metrics:
                self.metrics[metric].append(value)
        
        # Update plot if enough time has passed
        current_time = time.time()
        if current_time - self.last_update_time > self.update_interval:
            self.plot()
            self.last_update_time = current_time
    
    def plot(self):
        """
        Plot current metrics
        """
        try:
            # Clear previous plots
            for ax in self.axs:
                ax.clear()
                
            # Plot scores in the first subplot
            ax1 = self.axs[0]
            if 'score' in self.metrics and len(self.metrics['score']) > 0:
                ax1.plot(self.episodes, self.metrics['score'], label='Score', color='blue', alpha=0.3)
                
                # Calculate and plot moving average if enough data
                if len(self.metrics['score']) >= self.moving_avg_window:
                    moving_avg = pd.Series(self.metrics['score']).rolling(self.moving_avg_window).mean()
                    ax1.plot(self.episodes, moving_avg, label=f'Moving Avg ({self.moving_avg_window})', 
                             color='blue', linewidth=2)
                
                # Plot record
                if 'record' in self.metrics and len(self.metrics['record']) > 0:
                    ax1.plot(self.episodes, self.metrics['record'], label='Record', 
                             color='green', linestyle='--')
                
                ax1.set_title('Training Progress')
                ax1.set_xlabel('Episodes')
                ax1.set_ylabel('Score')
                ax1.legend(loc='upper left')
                ax1.grid(True, linestyle='--', alpha=0.7)
            
            # Plot learning parameters in the second subplot
            ax2 = self.axs[1]
            if 'epsilon' in self.metrics and len(self.metrics['epsilon']) > 0:
                ax2.plot(self.episodes, self.metrics['epsilon'], label='Exploration Rate (ε)', 
                         color='red')
                
                # Plot average score if available
                if 'avg_score' in self.metrics and len(self.metrics['avg_score']) > 0:
                    ax2_twin = ax2.twinx()
                    ax2_twin.plot(self.episodes, self.metrics['avg_score'], label='Avg Score', 
                                 color='purple')
                    ax2_twin.set_ylabel('Average Score', color='purple')
                    ax2_twin.tick_params(axis='y', colors='purple')
                    ax2_twin.legend(loc='upper right')
                
                ax2.set_xlabel('Episodes')
                ax2.set_ylabel('Exploration Rate (ε)', color='red')
                ax2.tick_params(axis='y', colors='red')
                ax2.grid(True, linestyle='--', alpha=0.7)
                ax2.legend(loc='upper left')
                
            # Update display
            self.fig.tight_layout()
            plt.pause(0.01)
        except Exception as e:
            print(f"Warning: Plot update failed - {e}")
    
    def save(self, filename=None):
        """
        Save the current plot to a file
        """
        if filename is None:
            filename = f"training_plot_{self.episode_counter}.png"
        
        save_path = os.path.join(self.save_dir, filename)
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
        
    def save_data(self, filename=None):
        """
        Save the metrics data to a CSV file
        """
        if filename is None:
            filename = f"training_data_{self.episode_counter}.csv"
            
        save_path = os.path.join(self.save_dir, filename)
        
        # Create DataFrame from metrics
        data = {'episode': self.episodes}
        for metric, values in self.metrics.items():
            if len(values) == len(self.episodes):
                data[metric] = values
                
        df = pd.DataFrame(data)
        df.to_csv(save_path, index=False)
        print(f"Training data saved to {save_path}")
    
    def close(self):
        """
        Close the plot
        """
        plt.close(self.fig)
        plt.ioff()  # Turn off interactive mode