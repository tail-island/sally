import os

from funcy import last
from glob import glob
from self_driving import SelfDriving
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy


env = SelfDriving()

model_path = last(sorted(glob('log/*.zip'), key=lambda f: os.stat(f).st_mtime))
model = SAC.load(model_path, env)

print(model_path)

reward_mean, _ = evaluate_policy(model, env, n_eval_episodes=1, render=True, warn=False)

print(f'reward: {reward_mean:.02f}')

for _ in range(10):
    env.seed(None)  # 乱数シードをNone（現在時刻を使う）に設定します。

    observation = env.reset()
    done = False

    while not done:
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, done, _ = env.step(action)
        env.render()

# import cv2
# import numpy as np

# from game import OBSTACLE_COUNT, STAR_COUNT

# def plot_observation(observation):
#     def plot_body(body, color):
#         angle, length = body[0:2] * np.array((np.pi, 320), dtype=np.float32)

#         cv2.circle(image, (320 + int(np.cos(angle) * length), 320 - int(np.sin(angle) * length)), 5, color)

#     image = np.zeros((841, 641, 3), dtype=np.uint8)

#     cv2.circle(image, (320, 320), 320, (128, 128, 128))
#     cv2.line(image, (0, 320), (640, 320), (128, 128, 128))
#     cv2.line(image, (320, 0), (320, 640), (128, 128, 128))

#     for body in np.reshape(observation[9: 9 + 7 * 8], (7, 8)):
#         plot_body(body, (255, 128, 128))

#     for body in np.reshape(observation[9 + 7 * 8: 9 + 7 * 8 + OBSTACLE_COUNT * 2], (OBSTACLE_COUNT, 2)):
#         plot_body(body, (128, 128, 128))

#     for body in np.reshape(observation[9 + 7 * 8 + OBSTACLE_COUNT * 2: 9 + 7 * 8 + OBSTACLE_COUNT * 2 + STAR_COUNT * 2], (STAR_COUNT, 2)):
#         plot_body(body, (128, 255, 128))

#     cv2.line(image, (0, 740), (640, 740), (255, 255, 255))

#     for i, v in enumerate(observation):
#         cv2.line(image, (i * 3 + 0, 740), (i * 3 + 0, 740 - int(100 * v)), (255, 255, 255))
#         cv2.line(image, (i * 3 + 1, 740), (i * 3 + 1, 740 - int(100 * v)), (255, 255, 255))
#         cv2.line(image, (i * 3 + 2, 740), (i * 3 + 2, 740 - int(100 * v)), (255, 255, 255))

#     return image

# cv2.namedWindow('observation')

# for _ in range(10):
#     # env.seed(None)

#     observation = env.reset()
#     done = False

#     while not done:
#         action, _ = model.predict(observation, deterministic=True)
#         observation, reward, done, _ = env.step(action)
#         env.render()

#         cv2.imshow('observation', plot_observation(observation))
#         cv2.waitKey(1)

# cv2.destroyAllWindows()


# from PIL import Image

# images = []

# env.seed(None)
# observation = env.reset()
# done = False

# while not done:
#     action, _ = model.predict(observation)
#     observation, reward, done, _ = env.step(action)

#     images.append(env.render(mode='rgb_array'))

# images = tuple(map(lambda image: Image.fromarray(image), images))
# images[0].save('game.gif', save_all=True, append_images=images[1:], duration=1 / 30 * 1000)
