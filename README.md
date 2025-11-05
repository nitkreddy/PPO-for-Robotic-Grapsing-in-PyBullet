# PPO Kuka Grasping

This project trains a Kuka robot arm in PyBullet to pick up objects using reinforcement learning. The agent learns a grasping policy with PPO and then runs it inside a simulated environment.

## Features
- Custom PyBullet robot setup and environment  
- PPO reinforcement learning algorithm  
- Training script with logging and checkpoint saving  
- Policy visualization to watch the robot grasp in real time  
- Configurable observation and action spaces  
- TensorBoard support for monitoring training  
- Pre-trained policy file included

## Installation

Using pip:
```
pip install -r requirements.txt
```

Or using Conda:
```
conda env create -f environment.yml
conda activate ppo_grasping
```

## Run

Train a policy:
```
python ppo_train.py
```

Visualize a trained policy:
```
python visualize.py
```

## How It Works
- A Kuka arm interacts with a PyBullet simulation environment
- The agent receives rewards based on grasp success and stable movements
- PPO updates the policy gradually over many training rollouts
- Checkpoints are stored during training
- The visualize script loads a saved model and runs the robot in simulation

## Notes
- Python 3.11 recommended  
- Training time depends on your CPU  
- The provided model works but still has room for improvement  
- You can tune rewards and hyperparameters to get better results  
- Object spawn settings and camera configurations can be modified in the environment file
