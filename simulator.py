import pymunk
import pymunk.pygame_util

from math import pi


MAX_SPEED = 300


class Body(pymunk.Body):
    def __init__(self):
        super().__init__()

    def set_position_and_angle(self, position, angle):
        self.position = position
        self.angle = angle
        self.space.reindex_shapes_for_body(self)


class Tire(Body):
    def __init__(self, space, car, grip):
        super().__init__()

        self.velocity_func = self._update_velocity
        self.car = car
        self.grip = grip

        shape = pymunk.Poly(self, ((-5, 2.5), (5, 2.5), (5, -2.5), (-5, -2.5)))
        shape.collision_type = 1
        shape.density = 1
        shape.elasticity = 0.2

        space.add(self, shape)

    def _update_surge_velocity(self):
        self.apply_force_at_local_point(-1 * self.mass * (self.velocity.rotated(-self.angle).dot((1, 0)) * pymunk.Vec2d(1, 0)).normalized() * 5)

    def _update_sway_velocity(self):
        p = -1 * self.mass * self.velocity.rotated(-self.angle).dot((0, 1)) * pymunk.Vec2d(0, 1)

        if p.length > self.grip:  # 現実では滑り出すと止まらないので、閾値は一定ではないのですけど……。
            p *= self.grip / p.length

        self.apply_impulse_at_local_point(p)

    def _update_angular_velocity(self, dt):
        self.torque -= self.moment * self.angular_velocity * (1 / dt) * 0.1  # 現実では速度と関係がありそう……。据え切りは重いけど、動いていれば軽いですもんね。

    def _limit_velocity(self):
        if self.velocity.length > MAX_SPEED:
            self.velocity *= MAX_SPEED / self.velocity.length

    def _update_velocity(self, body, gravity, damping, dt):
        pymunk.Body.update_velocity(body, gravity, damping, dt)

        self._update_surge_velocity()
        self._update_sway_velocity()
        self._update_angular_velocity(dt)

        self._limit_velocity()

    def accelerate(self, force):
        if force < 0:  # 後退は前進より遅くします。そうしないとひたすらバックで進む人がでちゃう。
            force *= 0.5

        self.apply_force_at_local_point((force, 0))

    def brake(self, force):
        p = -1 * self.mass * self.velocity.rotated(-self.angle).dot((1, 0)) * pymunk.Vec2d(1, 0)

        if p.length > force:
            p *= force / p.length

        self.apply_force_at_local_point(p)

    def steer(self, torque):
        self.torque += torque


class Car(Body):
    def __init__(self, space):
        super().__init__()

        shape = pymunk.Poly(self, ((-20, 7.5), (-5, 10), (20, 5), (20, -5), (-5, -10), (-20, -7.5)))
        shape.collision_type = 1
        shape.density = 0.01  # タイヤ側の動作しか計算していないので、車両側は軽くして影響を減らしました……。
        shape.elasticity = 0.2

        self.tire_lf = Tire(space, self, 500)
        self.tire_rf = Tire(space, self, 500)
        self.tire_lr = Tire(space, self, 300)  # ドリフトするように、リア・タイヤのグリップを落としました。
        self.tire_rr = Tire(space, self, 300)

        self.tire_lf.position = ( 12.5,  12.5)  # noqa: E201, E241
        self.tire_rf.position = ( 12.5, -12.5)  # noqa: E201
        self.tire_lr.position = (-12.5,  15.0)  # noqa: E241
        self.tire_rr.position = (-12.5, -15.0)

        pyvot_tire_lf = pymunk.PivotJoint(self, self.tire_lf, self.tire_lf.position, (0, 0))
        pyvot_tire_rf = pymunk.PivotJoint(self, self.tire_rf, self.tire_rf.position, (0, 0))
        pyvot_tire_lr = pymunk.PivotJoint(self, self.tire_lr, self.tire_lr.position, (0, 0))
        pyvot_tire_rr = pymunk.PivotJoint(self, self.tire_rr, self.tire_rr.position, (0, 0))

        rotation_tire_lf = pymunk.RotaryLimitJoint(self, self.tire_lf, -pi / 6, pi / 6)
        rotation_tire_rf = pymunk.RotaryLimitJoint(self.tire_lf, self.tire_rf, 0, 0)
        rotation_tire_lr = pymunk.RotaryLimitJoint(self, self.tire_lr, 0, 0)
        rotation_tire_rr = pymunk.RotaryLimitJoint(self, self.tire_rr, 0, 0)

        space.add(self, shape, pyvot_tire_lf, pyvot_tire_rf, pyvot_tire_lr, pyvot_tire_rr, rotation_tire_lf, rotation_tire_rf, rotation_tire_lr, rotation_tire_rr)

    def set_position_and_angle(self, position, angle):
        super().set_position_and_angle(position, angle)

        self.tire_lf.set_position_and_angle(self.tire_lf.position.rotated(angle) + position, angle)
        self.tire_rf.set_position_and_angle(self.tire_rf.position.rotated(angle) + position, angle)
        self.tire_lr.set_position_and_angle(self.tire_lr.position.rotated(angle) + position, angle)
        self.tire_rr.set_position_and_angle(self.tire_rr.position.rotated(angle) + position, angle)

    def accelerate(self, force):
        self.tire_lr.accelerate(force / 2)
        self.tire_rr.accelerate(force / 2)

    def brake(self, force):
        self.tire_lf.brake(force / 4)
        self.tire_rf.brake(force / 4)
        self.tire_lr.brake(force / 4)
        self.tire_rr.brake(force / 4)

    def steer(self, torque):
        self.tire_lf.steer(torque / 2)
        self.tire_rf.steer(torque / 2)


class Star(Body):
    def __init__(self, space):
        super().__init__()

        self.velocity_func = self._update_velocity

        outers = tuple(map(lambda i: pymunk.Vec2d(20, 0).rotated(                 pi * 2 / 5 * i), range(5)))  # noqa: E201
        inners = tuple(map(lambda i: pymunk.Vec2d(10, 0).rotated(pi * 2 / 5 / 2 + pi * 2 / 5 * i), range(5)))

        def create_triangle(i):
            shape = pymunk.Poly(self, (outers[i], inners[i], inners[i - 1]))
            shape.density = 1
            shape.elasticity = 0.2

            return shape

        triangles = map(create_triangle, range(5))

        pentagon = pymunk.Poly(self, inners)
        pentagon.density = 1
        pentagon.elasticity = 0.2

        space.add(self, *triangles, pentagon)

    def _update_velocity(self, body, gravity, damping, dt):
        pymunk.Body.update_velocity(body, gravity, damping, dt)

        self.force  += -1 * self.mass   * self.velocity         * 0.5  # noqa: E221
        self.torque += -1 * self.moment * self.angular_velocity * 0.5


class Obstacle(Body):
    def __init__(self, space):
        super().__init__()

        self.velocity_func = self._update_velocity

        shape = pymunk.Circle(self, 10)
        shape.density = 2
        shape.elasticity = 0.2

        space.add(self, shape)

    def _update_velocity(self, body, gravity, damping, dt):
        pymunk.Body.update_velocity(body, gravity, damping, dt)

        self.force  += -1 * self.mass   * self.velocity         * 0.5  # noqa: E221
        self.torque += -1 * self.moment * self.angular_velocity * 0.5


if __name__ == '__main__':
    import pygame
    import sys

    pygame.init()
    pygame.display.set_caption('manual driving')

    screen = pygame.display.set_mode((800, 800))
    clock = pygame.time.Clock()

    pymunk.pygame_util.positive_y_is_up = True
    draw_options = pymunk.pygame_util.DrawOptions(screen)
    draw_options.flags = pymunk.SpaceDebugDrawOptions.DRAW_SHAPES | pymunk.SpaceDebugDrawOptions.DRAW_COLLISION_POINTS

    space = pymunk.Space()

    car = Car(space)
    car.set_position_and_angle((400, 200), pi / 2)

    other_car = Car(space)
    other_car.set_position_and_angle((400, 600), -pi / 2)

    star = Star(space)
    star.set_position_and_angle((400, 400), pi / 2)

    for obstacle_position in ((200, 200), (600, 200), (200, 600), (600, 600)):
        obstacle = Obstacle(space)
        obstacle.set_position_and_angle(obstacle_position, 0)

    for wall_start, wall_end in ((5, 5), (794, 5)), ((794, 5), (794, 794)), ((794, 794), (5, 794)), ((5, 794), (5, 5)):
        wall = pymunk.Segment(space.static_body, wall_start, wall_end, 10)
        wall.elasticity = 0.8
        space.add(wall)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit(0)

        keys = pygame.key.get_pressed()

        if keys[pygame.K_UP]:
            car.accelerate(20000)

        if keys[pygame.K_DOWN]:
            car.accelerate(-10000)

        if keys[pygame.K_SPACE]:
            car.brake(200000)

        if keys[pygame.K_LEFT]:
            car.steer( 20000)  # noqa: E201

        if keys[pygame.K_RIGHT]:
            car.steer(-20000)

        space.step(1 / 60)
        clock.tick(60)

        screen.fill((255, 255, 255))
        space.debug_draw(draw_options)
        pygame.display.flip()
