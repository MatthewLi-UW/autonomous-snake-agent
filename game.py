import pygame
import random
import numpy as np
import math
import os
from enum import Enum
from collections import namedtuple
from dataclasses import dataclass

# get pygame ready to use
pygame.init()

# game settings that control how everything looks and behaves
@dataclass
class GameConfig:
    CELL_DIMENSION: int = 20
    FRAME_RATE: int = 60
    STARVATION_FACTOR: int = 200
    ANIMATION_SPEED: float = 10  # balance between smooth and responsive
    ENABLE_ANIMATIONS: bool = True  # toggle for performance
    
    # visual settings
    CELL_PADDING: int = 2
    EYE_RADIUS: int = 3
    FOOD_RADIUS_OUTER: int = 8
    FOOD_RADIUS_INNER: int = 6
    
    # performance options
    USE_HARDWARE_ACCELERATION: bool = True
    PRERENDER_SNAKE_SEGMENTS: bool = True
    
    # color palette - simplified
    PALETTE = {
        'background': (20, 70, 20),      # base green
        'grid': (30, 90, 30),            # lighter green
        'text': (240, 240, 240),         # off-white
        'text_shadow': (10, 40, 10),     # dark green
        'viper_head': (70, 210, 150),    # teal
        'viper_body': (60, 180, 120),    # slightly darker teal
        'viper_border': (30, 100, 70),   # dark border
        'apple_fill': (220, 60, 60),     # bright red
        'apple_border': (180, 30, 30),   # darker red
        'apple_stem': (100, 70, 30),     # brown
    }

# vector class for handling positions and movement
class Vector(namedtuple('Vector', 'x, y')):
    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)
    
    def __eq__(self, other):
        if not isinstance(other, Vector):
            return False
        return self.x == other.x and self.y == other.y
    
    # smooth movement between two positions
    def lerp(self, other, t):
        """smooth transition between positions"""
        return Vector(
            self.x + (other.x - self.x) * t,
            self.y + (other.y - self.y) * t
        )

# the four directions the snake can move
class Orientation(Enum):
    EAST = Vector(1, 0)
    SOUTH = Vector(0, 1)
    WEST = Vector(-1, 0)
    NORTH = Vector(0, -1)
    
    @classmethod
    def sequence(cls):
        return [cls.EAST, cls.SOUTH, cls.WEST, cls.NORTH]

# handles all the drawing to make the game look pretty
class ViperRenderer:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.config = GameConfig()
        self.display = pygame.display.set_mode((width, height))
        pygame.display.set_caption('snake ai')
        self.font = pygame.font.SysFont('arial', 25, True)
        
        # prepare graphics for faster rendering
        self.grass_texture = self._generate_grass_texture()
        self.terrain_base = self._generate_terrain()
        
    def _generate_grass_texture(self):
        """creates a natural-looking grass pattern"""
        # try loading existing texture to save time
        if os.path.exists('grass_texture.png'):
            try:
                return pygame.image.load('grass_texture.png')
            except:
                pass  # if loading fails, we'll create a new one
                
        # make a new grass texture from scratch
        texture_size = 100
        texture = pygame.Surface((texture_size, texture_size), pygame.SRCALPHA)
        
        # start with a solid background
        texture.fill(self.config.PALETTE['background'])
        
        # add lots of little grass blades
        for _ in range(400):
            x = random.randint(0, texture_size-1)
            y = random.randint(0, texture_size-1)
            
            # vary the size of grass blades
            height = random.randint(2, 6)
            width = random.randint(1, 2)
            
            # slightly vary the color for realism
            color_variation = random.randint(-15, 15)
            r = max(0, min(255, self.config.PALETTE['terrain_detail'][0] + color_variation))
            g = max(0, min(255, self.config.PALETTE['terrain_detail'][1] + color_variation))
            b = max(0, min(255, self.config.PALETTE['terrain_detail'][2] + color_variation))
            
            # draw each blade
            pygame.draw.line(
                texture, 
                (r, g, b), 
                (x, y + height), 
                (x, y), 
                width
            )
            
        # add some light spots to make it look more natural
        for _ in range(100):
            x = random.randint(0, texture_size-1)
            y = random.randint(0, texture_size-1)
            size = random.randint(1, 3)
            
            pygame.draw.rect(
                texture, 
                self.config.PALETTE['terrain_highlight'], 
                (x, y, size, size)
            )
            
        # save for next time so we don't have to recreate it
        pygame.image.save(texture, 'grass_texture.png')
        return texture
        
    def _generate_terrain(self):
        """creates the game background with grass and grid lines"""
        surface = pygame.Surface((self.width, self.height))
        
        # cover the whole area with grass texture
        grass_texture = self.grass_texture
        texture_width, texture_height = grass_texture.get_size()
        
        for x in range(0, self.width, texture_width):
            for y in range(0, self.height, texture_height):
                surface.blit(grass_texture, (x, y))
        
        # add faint grid lines so players can see the cells
        for x in range(0, self.width, self.config.CELL_DIMENSION):
            pygame.draw.line(
                surface, 
                self.config.PALETTE['grid'], 
                (x, 0), 
                (x, self.height),
                1  # thin lines
            )
        for y in range(0, self.height, self.config.CELL_DIMENSION):
            pygame.draw.line(
                surface, 
                self.config.PALETTE['grid'], 
                (0, y), 
                (self.width, y),
                1  # thin lines
            )
            
        return surface
        
    def render_frame(self, viper, sustenance, score):
        """draws everything for one frame of the game"""
        # put down the background first
        self.display.blit(self.terrain_base, (0, 0))
        
        # then the snake
        self._render_viper(viper)
        
        # then the food
        self._render_sustenance(sustenance)
        
        # finally the score at the top
        self._render_interface(score)
        
        # show it all on screen
        pygame.display.flip()
    
    def _render_viper(self, viper):
        """draws the snake with all its segments"""
        cfg = self.config
        cell_dim = cfg.CELL_DIMENSION
        pad = cfg.CELL_PADDING
        
        # get positions with animation if enabled
        segments = viper.get_interpolated_segments()
        
        # draw each part of the snake
        for idx, segment in enumerate(segments):
            is_head = (idx == 0)
            is_tail = (idx == len(segments) - 1)
            
            # convert grid position to screen position
            position = (segment.x * cell_dim, segment.y * cell_dim)
            
            # draw each segment differently
            if is_head:
                # head gets eyes
                self._render_head(position, cell_dim, viper.orientation, pad)
            elif is_tail:
                # tail gets a slightly different look
                self._render_body_segment(position, cell_dim, pad, is_tail=True)
            else:
                # regular body segment
                self._render_body_segment(position, cell_dim, pad)

    def _render_head(self, position, cell_dim, orientation, pad):
        """draws the snake's head with eyes looking in the right direction"""
        # rounded corners for smoother look
        border_radius = 8
        
        # outer border of the head
        pygame.draw.rect(
            self.display,
            self.config.PALETTE['viper_border'],
            pygame.Rect(position, (cell_dim, cell_dim)),
            border_radius=border_radius
        )
        
        # inner colored part
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
        
        # add eyes based on which way the snake is facing
        eye_radius = self.config.EYE_RADIUS
        eye_positions = self._calculate_eye_positions(
            position[0], position[1], cell_dim, orientation
        )
        
        for eye_pos in eye_positions:
            # main eye
            pygame.draw.circle(
                self.display,
                self.config.PALETTE['text_shadow'],
                eye_pos,
                eye_radius
            )
            # small white glint to make eyes look alive
            glint_pos = (eye_pos[0] - 1, eye_pos[1] - 1)
            pygame.draw.circle(
                self.display,
                self.config.PALETTE['text'],
                glint_pos,
                1
            )

    def _render_body_segment(self, position, cell_dim, pad, is_tail=False):
        """draws a body segment with rounded corners"""
        # all segments have rounded corners
        border_radius = 8
        
        # outer border
        pygame.draw.rect(
            self.display,
            self.config.PALETTE['viper_border'],
            pygame.Rect(position, (cell_dim, cell_dim)),
            border_radius=border_radius
        )
        
        # inner colored part
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
        """corners now look just like regular body segments"""
        self._render_body_segment(position, cell_dim, pad)
    
    def _render_tail(self, position, cell_dim, prev_segment, tail_segment, pad):
        """tail is just a regular body segment"""
        self._render_body_segment(position, cell_dim, pad, is_tail=True)
    
    def _calculate_eye_positions(self, x, y, size, orientation):
        """figures out where to put the eyes based on which way the snake is facing"""
        if orientation == Orientation.EAST:
            return [(x + size - 6, y + 5), (x + size - 6, y + size - 9)]
        elif orientation == Orientation.WEST:
            return [(x + 5, y + 5), (x + 5, y + size - 9)]
        elif orientation == Orientation.SOUTH:
            return [(x + 5, y + size - 9), (x + size - 9, y + size - 9)]
        else:  # NORTH
            return [(x + 5, y + 5), (x + size - 9, y + 5)]
    
    def _render_sustenance(self, sustenance):
        """draws the food for the snake to eat"""
        cfg = self.config
        cell_dim = cfg.CELL_DIMENSION
        pos_x = sustenance.x * cell_dim
        pos_y = sustenance.y * cell_dim
        center = (pos_x + cell_dim//2, pos_y + cell_dim//2)
        
        # shadow underneath for 3d effect
        shadow_offset = 2
        pygame.draw.circle(
            self.display,
            self.config.PALETTE['text_shadow'],
            (center[0] + shadow_offset, center[1] + shadow_offset),
            cfg.FOOD_RADIUS_OUTER - 1
        )
        
        # outer circle (border)
        pygame.draw.circle(
            self.display,
            self.config.PALETTE['apple_border'],
            center,
            cfg.FOOD_RADIUS_OUTER
        )
        
        # inner circle (main color)
        pygame.draw.circle(
            self.display,
            self.config.PALETTE['apple_fill'],
            center,
            cfg.FOOD_RADIUS_INNER
        )
        
        # highlight to make it look shiny
        highlight_pos = (center[0] - 3, center[1] - 3)
        pygame.draw.circle(
            self.display,
            (255, 180, 180),  # light red
            highlight_pos,
            3
        )
        
        # stem at the top
        pygame.draw.rect(
            self.display,
            self.config.PALETTE['apple_stem'],
            (pos_x + cell_dim//2 - 1, pos_y + 2, 2, 4)
        )
        
        # little leaf on the side
        leaf_points = [
            (pos_x + cell_dim//2 + 2, pos_y + 3),
            (pos_x + cell_dim//2 + 5, pos_y + 1),
            (pos_x + cell_dim//2 + 3, pos_y + 4)
        ]
        pygame.draw.polygon(
            self.display,
            (60, 160, 60),  # leaf green
            leaf_points
        )
    
    def _render_interface(self, score):
        """shows the player's score in a nice box"""
        score_text = f"score: {score}"
        
        # dark background so text is easy to read
        text_width, text_height = self.font.size(score_text)
        bg_rect = pygame.Rect(5, 5, text_width + 10, text_height + 10)
        
        # semi-transparent background
        bg_surface = pygame.Surface((bg_rect.width, bg_rect.height), pygame.SRCALPHA)
        bg_surface.fill((0, 0, 0, 128))  # black with 50% transparency
        self.display.blit(bg_surface, bg_rect)
        
        # shadow behind text for better contrast
        shadow = self.font.render(score_text, True, self.config.PALETTE['text_shadow'])
        self.display.blit(shadow, [12, 12])
        
        # main text
        text = self.font.render(score_text, True, self.config.PALETTE['text'])
        self.display.blit(text, [10, 10])

# keeps track of the snake's position and handles movement
class ViperState:
    def __init__(self, width, height, cell_size):
        self.width = width // cell_size
        self.height = height // cell_size
        self.cell_size = cell_size
        self.animation_progress = 1.0  # 1.0 means animation is complete
        self.prev_segments = []  # needed for smooth animations
        self.reset()
    
    def reset(self):
        """puts the snake back to starting position"""
        self.orientation = Orientation.EAST
        
        # start in middle of screen
        mid_x = self.width // 2
        mid_y = self.height // 2
        
        # create a small snake to start
        self.segments = [
            Vector(mid_x, mid_y),       # head
            Vector(mid_x - 1, mid_y),   # middle
            Vector(mid_x - 2, mid_y)    # tail
        ]
        self.prev_segments = self.segments.copy()
        self.animation_progress = 1.0
    
    def update(self, action):
        """moves the snake based on the chosen direction"""
        # remember old position for smooth animation
        self.prev_segments = self.segments.copy()
        self.animation_progress = 0.0  # start new animation
        
        # figure out which way to turn based on action
        # action = [straight, right, left]
        orientations = Orientation.sequence()
        curr_idx = orientations.index(self.orientation)
        
        if np.array_equal(action, [1, 0, 0]):
            # keep going straight
            new_orientation = orientations[curr_idx]
        elif np.array_equal(action, [0, 1, 0]):
            # turn right
            new_orientation = orientations[(curr_idx + 1) % 4]
        else:  # [0, 0, 1]
            # turn left
            new_orientation = orientations[(curr_idx - 1) % 4]
        
        self.orientation = new_orientation
        
        # move the head forward
        direction = self.orientation.value
        new_head = Vector(
            self.segments[0].x + direction.x,
            self.segments[0].y + direction.y
        )
        
        # add new head at front of snake
        self.segments.insert(0, new_head)
        
    def truncate(self):
        """removes the tail (happens when not eating food)"""
        self.segments.pop()
    
    @property
    def head(self):
        """quick way to get the head position"""
        return self.segments[0]
    
    def check_collision(self):
        """sees if the snake hit a wall or itself"""
        head = self.head
        
        # hit wall?
        if (head.x < 0 or head.x >= self.width or 
            head.y < 0 or head.y >= self.height):
            return True
            
        # hit self?
        if head in self.segments[1:]:
            return True
            
        return False
    
    def get_interpolated_segments(self):
        """creates smooth movement between positions"""
        if self.animation_progress >= 1.0:
            return self.segments
        
        # can't animate if snake length changed
        if len(self.segments) != len(self.prev_segments):
            return self.segments
            
        # blend old and new positions
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
        """updates how far along the animation is"""
        if self.animation_progress < 1.0:
            self.animation_progress = min(1.0, self.animation_progress + dt * speed_factor)
        return self.animation_progress >= 1.0  # true when animation is done

# main game logic that ties everything together
class ViperSimulation:
    def __init__(self, width=640, height=480):
        self.config = GameConfig()
        self.width = width
        self.height = height
        self.cell_size = self.config.CELL_DIMENSION
        
        # create all the parts we need
        self.renderer = ViperRenderer(width, height)
        self.viper = ViperState(width, height, self.cell_size)
        self.clock = pygame.time.Clock()
        
        # track game state
        self.score = 0
        self.sustenance = None
        self.frame_count = 0
        self.animation_in_progress = False
        self.queued_action = None
        
        # place first food
        self._spawn_sustenance()
    
    def reset(self):
        """starts a fresh game"""
        self.viper.reset()
        self.score = 0
        self.frame_count = 0
        self.animation_in_progress = False
        self.queued_action = None
        self._spawn_sustenance()
        return self._get_state()
    
    def _spawn_sustenance(self):
        """places food somewhere the snake isn't"""
        while True:
            # pick a random spot
            x = random.randint(0, self.viper.width - 1)
            y = random.randint(0, self.viper.height - 1)
            
            # make sure it's not where the snake is
            new_pos = Vector(x, y)
            if new_pos not in self.viper.segments:
                self.sustenance = new_pos
                break
    
    def _get_state(self):
        """packages up the game state for the ai to use"""
        return {
            'viper': self.viper,
            'food': self.sustenance,
            'score': self.score
        }
    
    def step(self, action):
        """moves the game forward one step"""
        self.frame_count += 1
        
        # check if user wants to quit
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
                
        # if we're in the middle of animating a move
        if self.animation_in_progress:
            # figure out how much time has passed
            dt = 1.0 / self.config.FRAME_RATE
            
            # continue the animation
            animation_complete = self.viper.advance_animation(
                dt, 
                self.config.ANIMATION_SPEED
            )
            
            # show current frame
            self.renderer.render_frame(self.viper, self.sustenance, self.score)
            self.clock.tick(self.config.FRAME_RATE)
            
            # if animation just finished
            if animation_complete:
                self.animation_in_progress = False
                # if we have another action waiting, do it now
                if self.queued_action is not None:
                    action = self.queued_action
                    self.queued_action = None
                    return self._process_action(action)
                    
            # still animating, nothing new happened
            return 0, False, self.score
            
        # process a new move
        return self._process_action(action)
    
    def _process_action(self, action):
        """handles what happens when the snake moves"""
        # move the snake
        self.viper.update(action)
        self.animation_in_progress = True
        
        # check if something interesting happened
        reward = 0
        game_over = False
        
        # did we die?
        if self.viper.check_collision() or self.frame_count > self.config.STARVATION_FACTOR * len(self.viper.segments):
            game_over = True
            reward = -10
            self.animation_in_progress = False
            
            # show final state
            self.renderer.render_frame(self.viper, self.sustenance, self.score)
            self.clock.tick(self.config.FRAME_RATE)
            
            return reward, game_over, self.score
        
        # did we eat?
        if self.viper.head == self.sustenance:
            self.score += 1
            reward = 10
            self._spawn_sustenance()
        else:
            # if we didn't eat, remove the tail
            self.viper.truncate()
        
        # show what happened
        self.renderer.render_frame(self.viper, self.sustenance, self.score)
        self.clock.tick(self.config.FRAME_RATE)
        
        return reward, game_over, self.score

# for compatibility with older code
SnakeGameAI = ViperSimulation

# test the game by itself
if __name__ == "__main__":
    game = ViperSimulation()
    running = True
    while running:
        # just go straight for testing
        reward, game_over, score = game.step([1, 0, 0])
        if game_over:
            running = False
    pygame.quit()