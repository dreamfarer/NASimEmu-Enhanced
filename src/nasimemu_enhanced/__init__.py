import gymnasium as gym
from .env import NASimEmuEnv

gym.register(id='NASimEmu-v0', entry_point='nasimemu_enhanced.env:NASimEmuEnv')
