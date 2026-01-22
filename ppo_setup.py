# ppo_setup.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

class CNNBackbone(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 8, 4)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)

        def out(size, k, s):
            return (size - k) // s + 1

        c1 = out(40, 8, 4)   
        c2 = out(c1, 4, 2)   
        c3 = out(c2, 3, 1)   
        self.flat_dim = 64 * c3 * c3  

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x.view(x.size(0), -1)


class ActorCritic(nn.Module):
    def __init__(self, in_channels: int, n_actions: int):
        super().__init__()
        self.backbone = CNNBackbone(in_channels)
        self.pi = nn.Sequential(
            nn.Linear(self.backbone.flat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
        )
        self.v = nn.Sequential(
            nn.Linear(self.backbone.flat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        z = self.backbone(x)
        logits = self.pi(z)
        value = self.v(z)
        return logits, value


@dataclass
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    learning_rate: float = 3e-4
    rollout_steps: int = 1024
    minibatch_size: int = 128
    ppo_epochs: int = 4
    max_grad_norm: float = 0.5
    device: str = "cpu"


class PPOAgent:
    def __init__(self, obs_channels: int, n_actions: int, cfg: PPOConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.net = ActorCritic(obs_channels, n_actions).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.learning_rate)

    def act(self, obs):
        with torch.no_grad():
            logits, value = self.net(obs)
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            logp = dist.log_prob(action)
        return action.item(), logp, value.squeeze(0)

    def evaluate_actions(self, obs, actions):
        logits, values = self.net(obs)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        logp = dist.log_prob(actions)
        entropy = dist.entropy()
        values = values.squeeze(-1)  
        if values.dim() > 1:
            values = values.flatten()
        return logp, entropy, values

    def compute_gae(self, rewards, values, dones, last_value):
        
        T = rewards.size(0)
        advantages = torch.zeros(T, device=self.device)
        last_gae = 0.0
        for t in reversed(range(T)):
            nonterminal = 1.0 - dones[t]
            next_value = last_value if t == T - 1 else values[t + 1]
            delta = rewards[t] + self.cfg.gamma * next_value * nonterminal - values[t]
            last_gae = delta + self.cfg.gamma * self.cfg.gae_lambda * nonterminal * last_gae
            advantages[t] = last_gae
        returns = advantages + values
        return advantages, returns

    def update(self, batch):
        cfg = self.cfg

        obs = batch["obs"].to(self.device)            
        actions = batch["actions"].to(self.device)     
        old_logp = batch["logp"].to(self.device)       
        returns = batch["returns"].to(self.device).flatten()     
        advantages = batch["advantages"].to(self.device).flatten()  

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_steps = obs.size(0)   
        idxs = torch.arange(total_steps, device=self.device)

        for _ in range(cfg.ppo_epochs):
            perm = idxs[torch.randperm(total_steps)]
            for start in range(0, total_steps, cfg.minibatch_size):
                end = start + cfg.minibatch_size
                mb_idx = perm[start:end]

                mb_obs = obs[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_logp = old_logp[mb_idx]
                mb_returns = returns[mb_idx]
                mb_adv = advantages[mb_idx]

                new_logp, entropy, values = self.evaluate_actions(mb_obs, mb_actions)

                ratio = torch.exp(new_logp - mb_old_logp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(
                    ratio,
                    1.0 - cfg.clip_range,
                    1.0 + cfg.clip_range
                ) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                values = values.flatten()          
                mb_returns = mb_returns.flatten()  
                value_loss = F.mse_loss(values, mb_returns)

                entropy_loss = entropy.mean()

                loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy_loss

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), cfg.max_grad_norm)
                self.opt.step()
