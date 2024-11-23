import torch
import gymnasium as gym
import highway_env
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp
import random

import env_exec

env = gym.make(
    "highway-fast-v0",
    render_mode="rgb_array",
    config={
        "action": {
            "type": "DiscreteMetaAction",
        },
        "observation": {
            "type": "LidarObservation",
            "cells": 128,
            "maximum_range": 64,
            "normalise": True,
        },
        "vehicles_count": 50,
    },
)

epochs = 20
episodes = 1000
epsilon = 0.2
episilon_decay = 0.99
hidden_size = 512
learning_rate = 0.05
momentum = 0.9
test_episodes = 100

obs, info = env.reset()

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.mps.is_available() else "cpu"
)
device

flattened_observation_size = np.prod(obs.shape)
net = torch.nn.Sequential(
    torch.nn.Linear(flattened_observation_size, hidden_size),
    torch.nn.LeakyReLU(),
    torch.nn.Dropout(0.2),
    torch.nn.Linear(hidden_size, hidden_size),
    torch.nn.LeakyReLU(),
    torch.nn.Dropout(0.2),
    torch.nn.Linear(hidden_size, env.action_space.n),
).to(device)

net.load_state_dict(torch.load("model.pth"))

for epoch in range(epochs):
    # Test the network
    net.eval()
    for episode in range(test_episodes):
        obs, info = env.reset(seed=random.randint(0, 1000))
        done, truncated = False, False
        total_reward = 0
        while not done and not truncated:
            x = torch.tensor(obs, dtype=torch.float32).flatten().to(device)
            y_pred = net(x)
            action = max(enumerate(y_pred), key=lambda x: x[1])[0]
            nobs, reward, done, truncated, info = env.step(action)

            y = y_pred.clone()
            y[action] = torch.tensor(reward, dtype=torch.float32).to(device)

            total_reward += reward

            obs = nobs

            env.render()
