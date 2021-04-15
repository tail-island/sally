from self_driving_1 import SelfDriving
from stable_baselines3 import SAC
from PIL import Image


env = SelfDriving()
model = SAC('MlpPolicy', env, verbose=1)

model.learn(total_timesteps=10000, log_interval=1)
model.save('self-driving-train-1')

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
images[0].save('self-driving-train-1.gif', save_all=True, append_images=images[1:], duration=1 / 30 * 1000)
