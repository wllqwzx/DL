import gym
import numpy as np

env = gym.make("CartPole-v0")

def basic_policy(obs):
    angle = obs[2]
    return 0 if angle<0 else 1

totals = []
for episode in range(500):
    episode_rewards = 0
    obs = env.reset()
    for step in range(1000):
        action = basic_policy(obs)
        obs, reward, done, info = env.step(action)
        episode_rewards += reward
        if done:
            break
    totals.append(episode_rewards)

print("mean:",np.mean(totals), "std:", np.std(totals), "max:", np.max(totals), "min:", np.min(totals))
