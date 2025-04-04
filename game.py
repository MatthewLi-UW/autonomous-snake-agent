import pygame
import random
import numpy as np
import math
import os
from enum import Enum
from collections import namedtuple
from dataclasses import dataclass

# Initialize core system
pygame.init()

# Game configuration
@dataclass
class GameConfig:
    CELL_DIMENSION: int = 20
    FRAME_RATE: int = 60  # Higher framerate for smoother animations
    STARVATION_FACTOR: int = 100
    ANIMATION_SPEED: float = 50  # Animation speed factor (lower is smoother but slower)
    
    # Visual configuration
    CELL_PADDING: int = 2
    EYE_RADIUS: int = 3
    FOOD_RADIUS_OUTER: int = 9
    FOOD_RADIUS_INNER: int = 7
    
    # Color palette - Enhanced colors
    PALETTE = {
        'background': (20, 70, 20),       # Darker green for base
        'grid': (30, 90, 30),             # Slightly lighter green for grid
        'text': (240, 240, 240),          # Almost white for text
        'text_shadow': (10, 40, 10),      # Dark green for text shadow
        'viper_head': (70, 210, 150),     # Teal for head
        'viper_body': (60, 180, 120),     # Slightly darker for body
        'viper_border': (30, 100, 70),    # Dark outline
        'apple_fill': (220, 60, 60),      # Brighter red for apple
        'apple_border': (180, 30, 30),    # Darker red for apple border
        'apple_stem': (100, 70, 30),      # Brown for apple stem
        'terrain_detail': (40, 100, 40),  # Light green for grass details
        'terrain_highlight': (50, 130, 50), # Highlight for grass
    }

# Vector handling for movement
class Vector(namedtuple('Vector', 'x, y')):
    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)
    
    def __eq__(self, other):
        if not isinstance(other, Vector):
            return False
        return self.x == other.x and self.y == other.y
    
    # Add linear interpolation for smooth animations
    def lerp(self, other, t):
        """Linear interpolation between this vector and another"""
        return Vector(
            self.x + (other.x - self.x) * t,
            self.y + (other.y - self.y) * t
        )

# Movement directions
class Orientation(Enum):
    EAST = Vector(1, 0)
    SOUTH = Vector(0, 1)
    WEST = Vector(-1, 0)
    NORTH = Vector(0, -1)
    
    @classmethod
    def sequence(cls):
        return [cls.EAST, cls.SOUTH, cls.WEST, cls.NORTH]

# Enhanced rendering engine
class ViperRenderer:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.config = GameConfig()
        self.display = pygame.display.set_mode((width, height))
        pygame.display.set_caption('Viper AI Simulation')
        self.font = pygame.font.SysFont('arial', 25, True)
        
        # Load or generate textures
        self.grass_texture = self._generate_grass_texture()
        self.terrain_base = self._generate_terrain()
        
    def _generate_grass_texture(self):
        """Generate a natural-looking grass texture"""
        # Try to load the texture from file if it exists
        if os.path.exists('grass_texture.png'):
            try:
                return pygame.image.load('grass_texture.png')
            except:
                pass  # Fall back to generated texture
                
        # Generate a tiled grass texture
        texture_size = 100
        texture = pygame.Surface((texture_size, texture_size), pygame.SRCALPHA)
        
        # Base color
        texture.fill(self.config.PALETTE['background'])
        
        # Generate random grass blades
        for _ in range(400):
            x = random.randint(0, texture_size-1)
            y = random.randint(0, texture_size-1)
            
            # Random grass blade height and width
            height = random.randint(2, 6)
            width = random.randint(1, 2)
            
            # Random grass color variation
            color_variation = random.randint(-15, 15)
            r = max(0, min(255, self.config.PALETTE['terrain_detail'][0] + color_variation))
            g = max(0, min(255, self.config.PALETTE['terrain_detail'][1] + color_variation))
            b = max(0, min(255, self.config.PALETTE['terrain_detail'][2] + color_variation))
            
            # Draw grass blade
            pygame.draw.line(
                texture, 
                (r, g, b), 
                (x, y + height), 
                (x, y), 
                width
            )
            
        # Add some highlights
        for _ in range(100):
            x = random.randint(0, texture_size-1)
            y = random.randint(0, texture_size-1)
            size = random.randint(1, 3)
            
            pygame.draw.rect(
                texture, 
                self.config.PALETTE['terrain_highlight'], 
                (x, y, size, size)
            )
            
        # Save for future use
        pygame.image.save(texture, 'grass_texture.png')
        return texture
        
    def _generate_terrain(self):
        """Generate a pre-rendered terrain surface with grass texture"""
        surface = pygame.Surface((self.width, self.height))
        
        # Tile the grass texture to fill the background
        grass_texture = self.grass_texture
        texture_width, texture_height = grass_texture.get_size()
        
        for x in range(0, self.width, texture_width):
            for y in range(0, self.height, texture_height):
                surface.blit(grass_texture, (x, y))
        
        # Add subtle grid overlay
        for x in range(0, self.width, self.config.CELL_DIMENSION):
            pygame.draw.line(
                surface, 
                self.config.PALETTE['grid'], 
                (x, 0), 
                (x, self.height),
                1  # Thinner grid lines
            )
        for y in range(0, self.height, self.config.CELL_DIMENSION):
            pygame.draw.line(
                surface, 
                self.config.PALETTE['grid'], 
                (0, y), 
                (self.width, y),
                1  # Thinner grid lines
            )
            
        return surface
        
    def render_frame(self, viper, sustenance, score):
        """Render a complete frame"""
        # Base terrain
        self.display.blit(self.terrain_base, (0, 0))
        
        # Render viper body with improved smoothness
        self._render_viper(viper)
        
        # Render food
        self._render_sustenance(sustenance)
        
        # Render HUD
        self._render_interface(score)
        
        # Update display
        pygame.display.flip()
    
    def _render_viper(self, viper):
        """Render the viper (snake) with all segments rounded"""
        cfg = self.config
        cell_dim = cfg.CELL_DIMENSION
        pad = cfg.CELL_PADDING
        
        # Get all segments with interpolation if animating
        segments = viper.get_interpolated_segments()
        
        # Track segment types (for proper rendering)
        for idx, segment in enumerate(segments):
            is_head = (idx == 0)
            is_tail = (idx == len(segments) - 1)
            
            # Convert grid position to pixel position
            position = (segment.x * cell_dim, segment.y * cell_dim)
            
            # Render based on segment type
            if is_head:
                # Render head with eyes
                self._render_head(position, cell_dim, viper.orientation, pad)
            elif is_tail:
                # Render tail (same as body but possibly different color)
                self._render_body_segment(position, cell_dim, pad, is_tail=True)
            else:
                # Render body segment - all rounded
                self._render_body_segment(position, cell_dim, pad)

    def _render_head(self, position, cell_dim, orientation, pad):
        """Render the snake's head with rounded shape and eyes"""
        # Border radius for all segments
        border_radius = 8
        
        # Outer shape (rounded rectangle)
        pygame.draw.rect(
            self.display,
            self.config.PALETTE['viper_border'],
            pygame.Rect(position, (cell_dim, cell_dim)),
            border_radius=border_radius
        )
        
        # Inner shape (rounded rectangle)
        inner_rect = pygame.Rect(
            position[0] + pad, 
            position[1] + pad,
            cell_dim - pad*2, 
            cell_dim - pad*2
        )
        pygame.draw.rect(
            self.display,
            self.config.PALETTE['viper_head'],
            inner_rect,
            border_radius=border_radius-1
        )
        
        # Eyes based on orientation
        eye_radius = self.config.EYE_RADIUS
        eye_positions = self._calculate_eye_positions(
            position[0], position[1], cell_dim, orientation
        )
        
        for eye_pos in eye_positions:
            pygame.draw.circle(
                self.display,
                self.config.PALETTE['text_shadow'],
                eye_pos,
                eye_radius
            )
            # Add glint to eyes for more life-like appearance
            glint_pos = (eye_pos[0] - 1, eye_pos[1] - 1)
            pygame.draw.circle(
                self.display,
                self.config.PALETTE['text'],
                glint_pos,
                1
            )

    def _render_body_segment(self, position, cell_dim, pad, is_tail=False):
        """Render a body segment with rounded corners"""
        # Border radius for all segments
        border_radius = 8
        
        # Outer shape (rounded rectangle)
        pygame.draw.rect(
            self.display,
            self.config.PALETTE['viper_border'],
            pygame.Rect(position, (cell_dim, cell_dim)),
            border_radius=border_radius
        )
        
        # Inner shape (rounded rectangle)
        inner_rect = pygame.Rect(
            position[0] + pad, 
            position[1] + pad,
            cell_dim - pad*2, 
            cell_dim - pad*2
        )
        pygame.draw.rect(
            self.display,
            self.config.PALETTE['viper_body'],
            inner_rect,
            border_radius=border_radius-1
        )
    
    def _render_corner(self, position, cell_dim, corner_type, pad):
        """Corner segments are now just normal rounded body segments"""
        self._render_body_segment(position, cell_dim, pad)
    
    def _render_tail(self, position, cell_dim, prev_segment, tail_segment, pad):
        """Tail is now rendered as a regular body segment"""
        self._render_body_segment(position, cell_dim, pad, is_tail=True)
    
    def _calculate_eye_positions(self, x, y, size, orientation):
        """Calculate eye positions based on direction"""
        if orientation == Orientation.EAST:
            return [(x + size - 6, y + 5), (x + size - 6, y + size - 9)]
        elif orientation == Orientation.WEST:
            return [(x + 5, y + 5), (x + 5, y + size - 9)]
        elif orientation == Orientation.SOUTH:
            return [(x + 5, y + size - 9), (x + size - 9, y + size - 9)]
        else:  # NORTH
            return [(x + 5, y + 5), (x + size - 9, y + 5)]
    
    def _render_sustenance(self, sustenance):
        """Render food item with enhanced appearance"""
        cfg = self.config
        cell_dim = cfg.CELL_DIMENSION
        pos_x = sustenance.x * cell_dim
        pos_y = sustenance.y * cell_dim
        center = (pos_x + cell_dim//2, pos_y + cell_dim//2)
        
        # Shadow for 3D effect
        shadow_offset = 2
        pygame.draw.circle(
            self.display,
            self.config.PALETTE['text_shadow'],
            (center[0] + shadow_offset, center[1] + shadow_offset),
            cfg.FOOD_RADIUS_OUTER - 1
        )
        
        # Outer circle (border)
        pygame.draw.circle(
            self.display,
            self.config.PALETTE['apple_border'],
            center,
            cfg.FOOD_RADIUS_OUTER
        )
        
        # Inner circle
        pygame.draw.circle(
            self.display,
            self.config.PALETTE['apple_fill'],
            center,
            cfg.FOOD_RADIUS_INNER
        )
        
        # Highlight for 3D effect
        highlight_pos = (center[0] - 3, center[1] - 3)
        pygame.draw.circle(
            self.display,
            (255, 180, 180),  # Light red
            highlight_pos,
            3
        )
        
        # Stem
        pygame.draw.rect(
            self.display,
            self.config.PALETTE['apple_stem'],
            (pos_x + cell_dim//2 - 1, pos_y + 2, 2, 4)
        )
        
        # Leaf
        leaf_points = [
            (pos_x + cell_dim//2 + 2, pos_y + 3),
            (pos_x + cell_dim//2 + 5, pos_y + 1),
            (pos_x + cell_dim//2 + 3, pos_y + 4)
        ]
        pygame.draw.polygon(
            self.display,
            (60, 160, 60),  # Leaf green
            leaf_points
        )
    
    def _render_interface(self, score):
        """Render score and other UI elements with enhanced appearance"""
        score_text = f"Score: {score}"
        
        # Background for text
        text_width, text_height = self.font.size(score_text)
        bg_rect = pygame.Rect(5, 5, text_width + 10, text_height + 10)
        
        # Semi-transparent background
        bg_surface = pygame.Surface((bg_rect.width, bg_rect.height), pygame.SRCALPHA)
        bg_surface.fill((0, 0, 0, 128))  # RGBA with alpha for transparency
        self.display.blit(bg_surface, bg_rect)
        
        # Shadow for better visibility
        shadow = self.font.render(score_text, True, self.config.PALETTE['text_shadow'])
        self.display.blit(shadow, [12, 12])
        
        # Main text
        text = self.font.render(score_text, True, self.config.PALETTE['text'])
        self.display.blit(text, [10, 10])

# Viper (Snake) state and logic with animation support
class ViperState:
    def __init__(self, width, height, cell_size):
        self.width = width // cell_size
        self.height = height // cell_size
        self.cell_size = cell_size
        self.animation_progress = 1.0  # 1.0 means no animation in progress
        self.prev_segments = []  # Previous segment positions for animation
        self.reset()
    
    def reset(self):
        """Reset viper to initial state"""
        self.orientation = Orientation.EAST
        
        # Initial position (middle of screen)
        mid_x = self.width // 2
        mid_y = self.height // 2
        
        # Create initial segments
        self.segments = [
            Vector(mid_x, mid_y),
            Vector(mid_x - 1, mid_y),
            Vector(mid_x - 2, mid_y)
        ]
        self.prev_segments = self.segments.copy()
        self.animation_progress = 1.0
    
    def update(self, action):
        """Update viper position based on action"""
        # Store previous segments for animation
        self.prev_segments = self.segments.copy()
        self.animation_progress = 0.0  # Start animation
        
        # Determine new orientation based on action
        # action = [straight, right, left]
        orientations = Orientation.sequence()
        curr_idx = orientations.index(self.orientation)
        
        if np.array_equal(action, [1, 0, 0]):
            # Continue straight
            new_orientation = orientations[curr_idx]
        elif np.array_equal(action, [0, 1, 0]):
            # Turn right
            new_orientation = orientations[(curr_idx + 1) % 4]
        else:  # [0, 0, 1]
            # Turn left
            new_orientation = orientations[(curr_idx - 1) % 4]
        
        self.orientation = new_orientation
        
        # Calculate new head position
        direction = self.orientation.value
        new_head = Vector(
            self.segments[0].x + direction.x,
            self.segments[0].y + direction.y
        )
        
        # Insert new head
        self.segments.insert(0, new_head)
        
    def truncate(self):
        """Remove tail segment"""
        self.segments.pop()
    
    @property
    def head(self):
        """Get current head position"""
        return self.segments[0]
    
    def check_collision(self):
        """Check for collision with walls or self"""
        head = self.head
        
        # Wall collision
        if (head.x < 0 or head.x >= self.width or 
            head.y < 0 or head.y >= self.height):
            return True
            
        # Self collision (check if head collides with any other segment)
        if head in self.segments[1:]:
            return True
            
        return False
    
    def get_interpolated_segments(self):
        """Get segments with animation interpolation"""
        if self.animation_progress >= 1.0:
            return self.segments
        
        # If segments lengths don't match, animation isn't possible
        if len(self.segments) != len(self.prev_segments):
            return self.segments
            
        # Interpolate between previous and current positions
        interpolated = []
        for i in range(len(self.segments)):
            if i < len(self.prev_segments):
                interpolated.append(
                    self.prev_segments[i].lerp(
                        self.segments[i],
                        self.animation_progress
                    )
                )
            else:
                interpolated.append(self.segments[i])
                
        return interpolated
    
    def advance_animation(self, dt, speed_factor):
        """Progress the movement animation"""
        if self.animation_progress < 1.0:
            self.animation_progress = min(1.0, self.animation_progress + dt * speed_factor)
        return self.animation_progress >= 1.0  # Return True when animation is complete

# Main game controller
class ViperSimulation:
    def __init__(self, width=640, height=480):
        self.config = GameConfig()
        self.width = width
        self.height = height
        self.cell_size = self.config.CELL_DIMENSION
        
        # Initialize components
        self.renderer = ViperRenderer(width, height)
        self.viper = ViperState(width, height, self.cell_size)
        self.clock = pygame.time.Clock()
        
        # Game state
        self.score = 0
        self.sustenance = None
        self.frame_count = 0
        self.animation_in_progress = False
        self.queued_action = None
        
        # Initialize food
        self._spawn_sustenance()
    
    def reset(self):
        """Reset the game to initial state"""
        self.viper.reset()
        self.score = 0
        self.frame_count = 0
        self.animation_in_progress = False
        self.queued_action = None
        self._spawn_sustenance()
        return self._get_state()
    
    def _spawn_sustenance(self):
        """Place food at a random valid location"""
        while True:
            # Generate random position
            x = random.randint(0, self.viper.width - 1)
            y = random.randint(0, self.viper.height - 1)
            
            # Check if position is valid (not overlapping with snake)
            new_pos = Vector(x, y)
            if new_pos not in self.viper.segments:
                self.sustenance = new_pos
                break
    
    def _get_state(self):
        """Get current game state for AI"""
        return {
            'viper': self.viper,
            'food': self.sustenance,
            'score': self.score
        }
    
    def step(self, action):
        """Execute one game step and return results"""
        self.frame_count += 1
        
        # Handle quit events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
                
        # If animation is in progress, continue it
        if self.animation_in_progress:
            # Calculate time since last frame for smooth animation
            dt = 1.0 / self.config.FRAME_RATE
            
            # Update animation progress
            animation_complete = self.viper.advance_animation(
                dt, 
                self.config.ANIMATION_SPEED
            )
            
            # Render the current animation frame
            self.renderer.render_frame(self.viper, self.sustenance, self.score)
            self.clock.tick(self.config.FRAME_RATE)
            
            # If animation is complete, process the next action if queued
            if animation_complete:
                self.animation_in_progress = False
                if self.queued_action is not None:
                    # Process the queued action
                    action = self.queued_action
                    self.queued_action = None
                    return self._process_action(action)
                    
            # Animation still in progress, no state change
            return 0, False, self.score
            
        # Start processing a new action
        return self._process_action(action)
    
    def _process_action(self, action):
        """Process game logic for an action"""
        # Update viper position
        self.viper.update(action)
        self.animation_in_progress = True
        
        # Check for collisions or timeout
        reward = 0
        game_over = False
        
        if self.viper.check_collision() or self.frame_count > self.config.STARVATION_FACTOR * len(self.viper.segments):
            game_over = True
            reward = -10
            self.animation_in_progress = False
            
            # Render final frame
            self.renderer.render_frame(self.viper, self.sustenance, self.score)
            self.clock.tick(self.config.FRAME_RATE)
            
            return reward, game_over, self.score
        
        # Check for food consumption
        if self.viper.head == self.sustenance:
            self.score += 1
            reward = 10
            self._spawn_sustenance()
        else:
            # Remove tail if no food eaten
            self.viper.truncate()
        
        # Render frame
        self.renderer.render_frame(self.viper, self.sustenance, self.score)
        self.clock.tick(self.config.FRAME_RATE)
        
        return reward, game_over, self.score

# For backward compatibility
SnakeGameAI = ViperSimulation

# Example usage for testing:
if __name__ == "__main__":
    game = ViperSimulation()
    running = True
    while running:
        # Always go straight for testing
        reward, game_over, score = game.step([1, 0, 0])
        if game_over:
            running = False
    pygame.quit()