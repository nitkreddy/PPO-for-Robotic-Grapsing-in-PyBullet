# visualize.py

import gym
from gym.envs import registration

if isinstance(registration.registry, dict) and not hasattr(registration.registry, "env_specs"):
    class _Registry(dict):
        @property
        def env_specs(self):
            return self

    registration.registry = _Registry(registration.registry)


import time
import pybullet as p
import numpy as np
from pybullet_envs.bullet.kuka_diverse_object_gym_env import KukaDiverseObjectEnv

import torch
import collections
from environment_setup import _preprocess, STACK_FRAMES
from ppo_setup import PPOAgent, PPOConfig

env = KukaDiverseObjectEnv(renders=True, isDiscrete=True, removeHeightHack=False, maxSteps=20, numObjects=1)

device = torch.device("mps" if torch.backends.mps.is_available()
                      else ("cuda" if torch.cuda.is_available() else "cpu"))

n_actions = env.action_space.n
cfg = PPOConfig(device=str(device))
agent = PPOAgent(STACK_FRAMES, n_actions, cfg)

agent.net.load_state_dict(torch.load("kuka_ppo_policy.pt", map_location=device))
agent.net.eval()
print("Loaded trained PPO model")

env.reset()
screen = env._get_observation().transpose((2, 0, 1))  
screen = np.ascontiguousarray(screen, dtype=np.float32) / 255.0  
screen = torch.from_numpy(screen)
frame = _preprocess(screen).to(device)  

stack = collections.deque([frame for _ in range(STACK_FRAMES)], maxlen=STACK_FRAMES)

for step in range(200):
    obs_t = torch.cat(list(stack), dim=0).unsqueeze(0).to(device)

    with torch.no_grad():
        logits, _ = agent.net(obs_t)
        probs = torch.softmax(logits, dim=-1)
        action = torch.distributions.Categorical(probs).sample().item()

    _, _, done, _ = env.step(action)

    screen = env._get_observation().transpose((2, 0, 1)) 
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255.0  
    screen = torch.from_numpy(screen)
    frame = _preprocess(screen).to(device) 
    stack.append(frame)

    time.sleep(0.05)  

    if done:
        env.reset()
        screen = env._get_observation().transpose((2, 0, 1))  
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255.0  
        screen = torch.from_numpy(screen)
        frame = _preprocess(screen).to(device)
        stack = collections.deque([frame for _ in range(STACK_FRAMES)], maxlen=STACK_FRAMES)

env.close()