from game import FPS, GAME_PERIOD_SEC
from self_driving_4 import SelfDriving
from stable_baselines3 import SAC
from PIL import Image
from stable_baselines3.common.callbacks import CheckpointCallback


env = SelfDriving()
model = SAC('MlpPolicy', env, verbose=1)

model.learn(total_timesteps=1_000_000, log_interval=10, callback=CheckpointCallback(save_freq=GAME_PERIOD_SEC * FPS * 10, save_path='log-4', name_prefix='self-driving'))
model.save('self-driving-train-4')

# 学習結果を確認するために、動かしてみます。

images = []

observation = env.reset()
done = False

while not done:
    images.append(env.render(mode='rgb_array'))

    action, _ = model.predict(observation, deterministic=True)
    observation, reward, done, _ = env.step(action)

images.append(env.render(mode='rgb_array'))

images = tuple(map(lambda image: Image.fromarray(image), images))
images[0].save('self-driving-train-4.gif', save_all=True, append_images=images[1:], duration=1 / 30 * 1000)
