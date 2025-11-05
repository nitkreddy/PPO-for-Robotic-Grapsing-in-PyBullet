# ppo_train.py

import gym
from gym.envs import registration

if isinstance(registration.registry, dict) and not hasattr(registration.registry, "env_specs"):
    class _Registry(dict):
        @property
        def env_specs(self):
            return self

    registration.registry = _Registry(registration.registry)

import time
import torch
import numpy as np

from environment_setup import KukaVisionEnv, STACK_FRAMES
from ppo_setup import PPOAgent, PPOConfig


def collect_rollout(env, agent: PPOAgent, steps: int, device):
    obs_buf = []
    act_buf = []
    logp_buf = []
    rew_buf = []
    val_buf = []
    done_buf = []

    obs = env.reset()  
    for _ in range(steps):
        action, logp, value = agent.act(obs)
        next_obs, reward, done, _ = env.step(action)

        obs_buf.append(obs.squeeze(0))  
        act_buf.append(action)
        logp_buf.append(logp)
        rew_buf.append(reward)
        val_buf.append(value)
        done_buf.append(float(done))

        obs = next_obs
        if done:
            obs = env.reset()

    with torch.no_grad():
        _, last_value = agent.net(obs)
        last_value = last_value.squeeze()
        if last_value.dim() > 0:
            last_value = last_value[0] if last_value.numel() == 1 else last_value.flatten()[0]

    obs_tensor = torch.stack(obs_buf).to(device)         
    act_tensor = torch.tensor(act_buf, device=device, dtype=torch.long)
    logp_tensor = torch.stack(logp_buf).to(device)
    rew_tensor = torch.tensor(rew_buf, device=device, dtype=torch.float32)
    val_tensor = torch.stack(val_buf).to(device).flatten()  
    done_tensor = torch.tensor(done_buf, device=device, dtype=torch.float32)

    adv, ret = agent.compute_gae(rew_tensor, val_tensor, done_tensor, last_value)
    data = {
        "obs": obs_tensor,
        "actions": act_tensor,
        "logp": logp_tensor,
        "returns": ret,
        "advantages": adv
    }
    return data, np.sum(rew_buf)


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    env = KukaVisionEnv(renders=False, max_steps=20, test_mode=False, device=device)
    n_actions = env.action_space_n
    obs_channels = STACK_FRAMES  

    cfg = PPOConfig(device=str(device), rollout_steps=256, minibatch_size=64, ppo_epochs=3)
    
    agent = PPOAgent(obs_channels, n_actions, cfg)

    save_path = "kuka_ppo_policy.pt"

    moving_return = None
    for iteration in range(1, 2001):
        t0 = time.time()
        batch, rollout_return = collect_rollout(env, agent, cfg.rollout_steps, device)
        agent.update(batch)

        if moving_return is None:
            moving_return = rollout_return
        else:
            moving_return = 0.9 * moving_return + 0.1 * rollout_return

        dt = time.time() - t0
        print(f"[{iteration:04d}] rollout_return={rollout_return:.2f} avg={moving_return:.2f} time={dt:.2f}s")

        if iteration % 100 == 0:
            torch.save(agent.net.state_dict(), save_path)
            print(f"model saved to {save_path}")

    env.close()


if __name__ == "__main__":
    main()