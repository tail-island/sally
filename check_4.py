import os

from funcy import last
from glob import glob
from self_driving import SelfDriving
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy


env = SelfDriving()

model_path = last(sorted(glob('log-4/*.zip'), key=lambda f: os.stat(f).st_mtime))
model = SAC.load(model_path, env)

print(model_path)

reward_mean, _ = evaluate_policy(model, env, n_eval_episodes=1, render=True, warn=False)

print(f'reward: {reward_mean:.02f}')
