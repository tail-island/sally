import gym
import numpy as np
import pygame
import pymunk

from funcy import concat, flatten, mapcat
from game import FPS, Game, OBSTACLE_COUNT, STAR_COUNT
from operator import itemgetter, methodcaller
from simulator import MAX_SPEED


class SelfDriving(gym.Env):
    def __init__(self):
        self._seed = None
        self.name = 'SelfDriving'

        self.action_space = gym.spaces.Box(np.array((-1, -1, -1), dtype=np.float32), np.array((1, 1, 1), dtype=np.float32), dtype=np.float32)

        self.observation_space = gym.spaces.Box(
            np.array(tuple(concat(
                (
                    -1,  # my_car.position.x
                    -1,  # my_car.position.y
                    -1,  # my_car.angle
                    -1,  # my_car.velocity_angle
                    0,   # my_car.velocity_length
                    -1,  # my_car.steering_angle
                    -1,  # my_car.steering_torque
                    0,   # my_car.score
                    0,   # my_car.crash_energy
                ),
                mapcat(lambda _: (
                    -1,  # other_car.position_angle
                    0,   # other_car.position_length
                    -1,  # other_car.angle
                    -1,  # other_car.velocity_angle
                    0,   # other_car.velocity_length
                    -1,  # other_car.steering_angle
                    0,   # other_car.score
                    0,   # other_car.crash_energy
                ), range(7)),
                mapcat(lambda _: (
                    -1,  # obstacle.position_angle
                    0    # obstacle.position_length
                ), range(OBSTACLE_COUNT)),
                mapcat(lambda _: (
                    -1,  # star.position_angle
                    0    # star.position_length
                ), range(STAR_COUNT)),
            )), dtype=np.float32),
            np.array(tuple(concat(
                (
                    1,   # my_car.position.x
                    1,   # my_car.position.y
                    1,   # my_car.angle
                    1,   # my_car.velocity_angle
                    1,   # my_car.velocity_length
                    1,   # my_car.steering_angle
                    1,   # my_car.steering_torque
                    1,   # my_car.score
                    1,   # my_car.crash_energy
                ),
                mapcat(lambda _: (
                    1,   # other_car.position_angle
                    1,   # other_car.position_length
                    1,   # other_car.angle
                    1,   # other_car.velocity_angle
                    1,   # other_car.velocity_length
                    1,   # other_car.steering_angle
                    1,   # other_car.score
                    1,   # other_car.crash_energy
                ), range(7)),
                mapcat(lambda _: (
                    1,   # obstacle.position_angle
                    1    # obstacle.position_length
                ), range(OBSTACLE_COUNT)),
                mapcat(lambda _: (
                    1,   # star.position_angle
                    1    # star.position_length
                ), range(STAR_COUNT)),
            )), dtype=np.float32),
            dtype=np.float32
        )

        self.screen = None

        self.reset()

    @classmethod
    def _create_observation(cls, game):
        def get_values(observation):
            return concat(
                flatten(observation['my_car'].values()),  # Vec2dにはxとyの2要素があるので、flattenしておきます。
                mapcat(methodcaller('values'), sorted(observation['other_cars'], key=itemgetter('position_length'))),  # 距離が近い順にソートします。前後も分けたほうが良い？
                mapcat(methodcaller('values'), sorted(observation['obstacles' ], key=itemgetter('position_length'))),  # noqa: E202
                mapcat(methodcaller('values'), sorted(observation['stars'     ], key=itemgetter('position_length')))   # noqa: E202
            )

        observation = (
            np.array(tuple(get_values(game.create_observation(game.cars[0]))), np.float32) /  # noqa: W504
            np.array(tuple(concat(
                (
                    1000,                   # my_car.position.x
                    1000,                   # my_car.position.y
                    np.pi,                  # my_car.angle
                    np.pi,                  # my_car.velocity_angle
                    MAX_SPEED / FPS,        # my_car.velocity_length
                    np.pi,                  # my_car.steering_angle
                    10,                     # my_car.steering_torque
                    30,                     # my_car.score
                    10 * FPS,               # my_car.crash_energy
                ),
                mapcat(lambda _: (
                    np.pi,                  # other_car.position_angle
                    1000,                   # other_car.position_length
                    np.pi,                  # other_car.angle
                    np.pi,                  # other_car.velocity_angle
                    MAX_SPEED / FPS * 2,    # other_car.velocity_length
                    np.pi,                  # other_car.steering_angle
                    30,                     # other_car.score
                    10 * FPS,               # other_car.crash_energy
                ), range(7)),
                mapcat(lambda _: (
                    np.pi,                  # obstacle.position_angle
                    1000                    # obstacle.position_length
                ), range(OBSTACLE_COUNT)),
                mapcat(lambda _: (
                    np.pi,                  # star.position_angle
                    1000                    # star.position_length
                ), range(STAR_COUNT)),
            )), dtype=np.float32)
        )

        observation[observation < -1] = -1
        observation[observation >  1] =  1  # noqa: E222

        return observation

    def reset(self):
        self.game = Game((self,), self._seed)

        return self._create_observation(self.game)

    @classmethod
    def _calc_car_and_star_distance(cls, game, car):
        return min(map(lambda star: (star.position - car.position).length, game.stars))

    @classmethod
    def _calc_reward(cls, game, car, last_score, last_distance):
        if car.score > last_score:
            return 100

        if car.crash_energy:
            return -1

        return delta if (delta := last_distance - cls._calc_car_and_star_distance(game, car)) > 1 else 0

    def step(self, action):
        last_score = self.game.cars[0].score
        last_distance = self._calc_car_and_star_distance(self.game, self.game.cars[0])

        self.action = action
        done = self.game.step()

        return self._create_observation(self.game), self._calc_reward(self.game, self.game.cars[0], last_score, last_distance), done, {}

    def render(self, mode='human'):
        pygame.init()
        pymunk.pygame_util.positive_y_is_up = True

        surface = self.game.create_surface()

        if mode == 'rgb_array':
            return np.reshape(np.frombuffer(pygame.image.tostring(surface, 'RGB'), dtype=np.uint8), (surface.get_height(), surface.get_width(), 3))

        if mode == 'human':
            if self.screen is None and mode == 'human':
                pygame.display.set_caption('self driving')
                self.screen = pygame.display.set_mode((800, 640))

            self.screen.blit(surface, (0, 0))
            pygame.display.flip()

    def seed(self, val):
        self._seed = val

    def get_action(self, _):
        return self.action[0], self.action[1] / 2 + 0.5, self.action[2]
