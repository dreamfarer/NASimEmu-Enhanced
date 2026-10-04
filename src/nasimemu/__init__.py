import gymnasium as gym
from .env import NASimEmuEnv

# the observation grows as hosts are discovered, so the env defines no fixed spaces for the env checker to validate
gym.register(id='NASimEmu-v0', entry_point='nasimemu.env:NASimEmuEnv', disable_env_checker=True)
