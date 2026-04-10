import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
import Formation_control_A as base
WIDTH = 1980
HEIGHT = 1080

class FormationEnv(gym.env):
    def __init__(self, screen):
        self.robot = base.Robot(0, (
        np.random.uniform(10, WIDTH -100),   # random x within (10, 100)
        np.random.uniform(10, HEIGHT - 100)   # random y within bounds
    ))
        self.action_space = spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32)
        self.screen = screen

    def step(self, action):
        steering, throttle = action
        # apply action
        self.robot.speed += throttle + self.robot.speed
        self.robot.angle += steering

        done = self.robot.is_in_formation()
        reward = self.robot.get_reward()

        obs = self._get_observation()
        info = {"ongoing"}
        return obs, reward, done, info
    
    def reset(self):
        x,y = 0
        self.robot = base.Robot(0,x,y)
        obs = self._get_observation()
        return obs

    def _get_observation(self):
        speed = [self.robot.x_speed, self.robot.y_speed]
        angle = self.robot.angle
        
        return np.concatenate((speed, angle))
    
    def close(self):
        pygame.quit()