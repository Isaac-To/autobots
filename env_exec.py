import gymnasium as gym
import highway_env

def run_episode(seed):
    lenv = gym.make(
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
            "vehicles_count": 20,
        },
    )
            
    obs, info = lenv.reset(seed=seed)
    done, truncated = False, False
    observations = []
    action_rewards = []
    reward = 0
    while not done and not truncated:
        action = lenv.action_space.sample()
        nobs, reward, done, truncated, info = lenv.step(action)
        observations.append(obs)
        action_reward = [0.5] * lenv.action_space.n
        action_reward[action] = reward
        action_rewards.append(action_reward)
        obs = nobs
    return observations, action_rewards