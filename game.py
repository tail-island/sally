import pygame
import pymunk
import sys

from funcy import concat, distinct, first, mapcat, repeat, repeatedly
from math import pi
from operator import attrgetter
from player_proxy import PlayerProxy
from random import Random
from simulator import Car, Obstacle, Star
from ui import create_surface


FPS = 30
OBSTACLE_COUNT = 40
STAR_COUNT = 2
GAME_PERIOD_SEC = 60


class Game:
    @classmethod
    def _crash(cls, arbiter, space, data):
        for car in filter(lambda body: isinstance(body, Car), distinct(map(lambda body: body.car if hasattr(body, 'car') else body, map(lambda shape: shape.body, arbiter.shapes)))):
            car.crash_energy = min(car.crash_energy + arbiter.total_ke / 2, 10 * FPS * 100000)

    @classmethod
    def _catch(cls, arbiter, space, data):
        car = first(filter(lambda body: isinstance(body, Car), distinct(map(lambda body: body.car if hasattr(body, 'car') else body, map(lambda shape: shape.body, arbiter.shapes)))))

        if not car:
            return True

        car.score += 1

        star = first(filter(lambda body: isinstance(body, Star), distinct(map(lambda shape: shape.body, arbiter.shapes))))
        star.is_catched = True

        return False

    def _random_position(self, sigma):
        return first(filter(lambda p: all(map(lambda b: (b.position - p).length >= 50, concat(self.cars, self.obstacles, self.stars))), filter(lambda p: 100 < p.length < 950, repeatedly(lambda: pymunk.Vec2d(self.game_random.gauss(0, sigma), 0).rotated(self.game_random.uniform(0, pi * 2))))))

    def _reset_star_position(self, star):
        star.set_position_and_angle(self._random_position(300), self.game_random.uniform(0, pi * 2))

    def _append_wall(self, a, b):
        wall = pymunk.Body(body_type=pymunk.Body.STATIC)

        shape = pymunk.Segment(wall, a, b, 10)
        shape.elasticity = 1

        self.space.add(wall, shape)

    def _append_car(self, position, angle):
        car = Car(self.space)
        car.set_position_and_angle(position, angle)
        car.crash_energy = 0
        car.score = 0

        for shape in concat(car.shapes, mapcat(lambda tire: tire.shapes, (car.tire_lf, car.tire_rf, car.tire_lr, car.tire_rr))):
            shape.collision_type = 1

        self.cars.append(car)

    def _append_obstacle(self):
        obstacle = Obstacle(self.space)
        obstacle.set_position_and_angle(self._random_position(500), self.game_random.uniform(0, pi * 2))

        self.obstacles.append(obstacle)

    def _append_star(self):
        star = Star(self.space)

        for shape in star.shapes:
            shape.collision_type = 2

        self._reset_star_position(star)
        star.is_catched = False

        self.stars.append(star)

    def __init__(self, players, seed=None):
        self.game_random = Random(seed)
        self.control_random = Random(seed)

        self.players = players

        self.elapse = 0
        self.actions = repeat((0, 0, 0), len(players))

        self.space = pymunk.Space()

        self.space.add_wildcard_collision_handler(1).post_solve = self._crash
        self.space.add_wildcard_collision_handler(2).begin = self._catch

        self.cars = []
        self.obstacles = []
        self.stars = []

        for a, b in ((-1000, 1000), (1000, 1000)), ((1000, 1000), (1000, -1000)), ((1000, -1000), (-1000, -1000)), ((-1000, -1000), (-1000, 1000)):
            self._append_wall(a, b)

        for position, angle in map(lambda i: (pymunk.Vec2d(80, 0).rotated(pi * 2 / 8 * i), pi * 2 / 8 * i), range(8)):
            self._append_car(position, angle)

        for _ in range(OBSTACLE_COUNT):
            self._append_obstacle()

        for _ in range(STAR_COUNT):
            self._append_star()

    @classmethod
    def _normalize_angle(cls, angle):
        return (angle + pi * 2) % (pi * 2)

    @classmethod
    def _normalize_relative_angle(cls, angle):
        result = cls._normalize_angle(angle)

        return result if result <= pi else result - pi * 2

    @classmethod
    def _get_my_car_observation(cls, my_car):
        return {
            'position': my_car.position,
            'angle': cls._normalize_relative_angle(my_car.angle),
            'velocity_angle': cls._normalize_relative_angle(my_car.velocity.rotated(-my_car.angle).angle),
            'velocity_length': my_car.velocity.rotated(-my_car.angle).length / FPS,
            'steering_angle': cls._normalize_relative_angle(my_car.tire_lf.angle - my_car.angle),
            'steering_torque': (my_car.tire_lf.moment * my_car.tire_lf.angular_velocity + my_car.tire_rf.moment * my_car.tire_rf.angular_velocity) / 20000,
            'score': my_car.score,
            'crash_energy': my_car.crash_energy / 100000
        }

    @classmethod
    def _get_other_car_observation(cls, other_car, my_car):
        return {
            'position_angle': cls._normalize_relative_angle((other_car.position - my_car.position).rotated(-my_car.angle).angle),
            'position_length': (other_car.position - my_car.position).length,
            'angle': cls._normalize_relative_angle(other_car.angle - my_car.angle),
            'velocity_angle': cls._normalize_relative_angle((other_car.velocity - my_car.velocity).rotated(-my_car.angle).angle),
            'velocity_length': (other_car.velocity - my_car.velocity).rotated(-my_car.angle).length / FPS,
            'steering_angle': cls._normalize_relative_angle(other_car.tire_lf.angle - other_car.angle),
            'score': other_car.score,
            'crash_energy': other_car.crash_energy / 100000
        }

    @classmethod
    def _get_obstacle_or_star_observation(cls, obstacle_or_star, my_car):
        return {
            'position_angle': cls._normalize_relative_angle((obstacle_or_star.position - my_car.position).rotated(-my_car.angle).angle),
            'position_length': (obstacle_or_star.position - my_car.position).length,
        }

    def create_observation(self, my_car):
        return {
            'my_car': self._get_my_car_observation(my_car),
            'other_cars': tuple(map(lambda other_car: self._get_other_car_observation(other_car, my_car), filter(lambda car: car != my_car, self.cars))),
            'obstacles': tuple(map(lambda obstacle: self._get_obstacle_or_star_observation(obstacle, my_car), self.obstacles)),
            'stars': tuple(map(lambda star: self._get_obstacle_or_star_observation(star, my_car), self.stars))
        }

    @classmethod
    def _clip(cls, value, min_value, max_value):
        return min(max(value, min_value), max_value)

    def step(self):
        self.elapse += 1
        self.actions = []

        for car, player in zip(self.cars, concat(self.players, repeat(None))):
            # アクションを取得します。
            acceleration, braking, steering = player.get_action(self.create_observation(car)) if player else (0, 0, 0)

            # アクションを正規化します。
            acceleration = self._clip(acceleration, -1, 1)
            braking      = self._clip(braking,       0, 1)  # noqa: E221, E241
            steering     = self._clip(steering,     -1, 1)  # noqa: E221, E241

            # 正規化したアクションを記録します。
            self.actions.append((acceleration, braking, steering))

            # 衝突して故障した車は、修理が終わるまでは行動できません。
            if car.crash_energy > 0:
                car.crash_energy = max(car.crash_energy - 100000, 0)
                continue

            # ゆらぎを出すために、アクションに小さな正規乱数を加えます。スターの次の出現位置が変わると強化学習が難しくなりそうなので、別のRandomインスタンスを使用します。
            acceleration = self._clip(acceleration + self.control_random.gauss(0, 0.05), -1, 1)
            braking      = self._clip(braking      + self.control_random.gauss(0, 0.05),  0, 1)  # noqa: E221, E241
            steering     = self._clip(steering     + self.control_random.gauss(0, 0.05), -1, 1)  # noqa: E221

            # アクションを実行します。
            car.accelerate(acceleration * 20000)
            car.brake(braking * 200000)
            car.steer(steering * 20000)

        self.space.step(1 / FPS)

        for star in filter(lambda star: star.is_catched, self.stars):
            self._reset_star_position(star)
            star.is_catched = False

        return self.elapse >= GAME_PERIOD_SEC * FPS  # ゲームはGAME_PERIOD_SECで終了します。

    def create_surface(self):
        return create_surface(self.space, self.cars, self.players, self.actions)


def play(program_names, seed, screen):
    images = []

    game = Game(tuple(map(PlayerProxy, program_names)), seed=seed)
    done = False

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit(0)

        done = game.step()
        surface = game.create_surface()

        screen.blit(surface, (0, 0))
        pygame.display.flip()

        images.append(pygame.image.tostring(surface, 'RGB'))

    return map(attrgetter('name'), game.players), map(attrgetter('score'), game.cars), images


if __name__ == '__main__':
    from argparse import ArgumentParser
    from PIL import Image

    parser = ArgumentParser()
    parser.add_argument('program_names', metavar='PROGRAM-NAME', nargs='+', help='player program\'s name')
    parser.add_argument('--seed', nargs='?', help='random seed')
    parser.add_argument('--animation', action='store_true', help='generate GIF animation')

    args = parser.parse_args()

    pygame.init()
    pymunk.pygame_util.positive_y_is_up = True

    pygame.display.set_caption('self driving')
    screen = pygame.display.set_mode((800, 640))

    names, scores, images = play(args.program_names, args.seed, screen)

    for name, score in zip(names, scores):
        print(f'{name}\t{score}')

    if args.animation:
        images = tuple(map(lambda image: Image.frombuffer('RGB', (800, 640), image), images))
        images[0].save('game.gif', save_all=True, append_images=images[1:], duration=1 / 30 * 1000)
