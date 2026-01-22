# environment_setup.py

import gym
from gym.envs import registration

if isinstance(registration.registry, dict) and not hasattr(registration.registry, "env_specs"):
    class _Registry(dict):
        @property
        def env_specs(self):
            return self

    registration.registry = _Registry(registration.registry)

import collections
import numpy as np
import torch
import torchvision.transforms as T

import pybullet as p
from pybullet_envs.bullet.kuka_diverse_object_gym_env import KukaDiverseObjectEnv

STACK_FRAMES = 3

_preprocess = T.Compose([
    T.ToPILImage(),
    T.Grayscale(num_output_channels=1),
    T.Resize(40),           
    T.ToTensor()
])


class KukaVisionEnv:
    
    def __init__(self, renders=False, max_steps=20, test_mode=False, device=None):
        self.env = KukaDiverseObjectEnv(
            renders=renders,
            isDiscrete=True,
            removeHeightHack=False,
            maxSteps=max_steps,
            numObjects=1,
            isTest=test_mode
        )
        self.env.cid = p.connect(p.DIRECT)
        self.device = device or torch.device("cpu")
        self.stack = collections.deque(maxlen=STACK_FRAMES)
        self.max_steps = max_steps

    @property
    def action_space_n(self):
        return self.env.action_space.n

    def _get_single_frame(self):
        screen = self.env._get_observation().transpose((2, 0, 1))
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255.0
        screen = torch.from_numpy(screen)
        frame = _preprocess(screen).to(self.device)
        return frame  

    def reset(self):
        self.env.reset()
        self.stack.clear()
        f = self._get_single_frame()
        for _ in range(self.stack.maxlen):
            self.stack.append(f)
        return self._stacked()

    def step(self, action: int):
        _, reward, done, info = self.env.step(action)
        frame = self._get_single_frame()
        self.stack.append(frame)
        obs = self._stacked()
        return obs, float(reward), bool(done), info

    def _stacked(self):
        return torch.cat(list(self.stack), dim=0).unsqueeze(0) 

    def close(self):
        self.env.close()
