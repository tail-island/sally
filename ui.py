import pygame
import pymunk.pygame_util
import pymunk.space_debug_draw_options

from funcy import count, juxt, mapcat
from simulator import Car, Star, Tire


def _draw_circle(options, left_bottom, shape, outline_color, fill_color):
    options.draw_circle(
        shape.offset + shape.body.position - left_bottom,
        shape.body.angle,
        shape.radius,
        outline_color,
        fill_color
    )


def _draw_segment(options, left_bottom, shape, outline_color, fill_color):
    options.draw_fat_segment(
        shape.a + shape.body.position - left_bottom,
        shape.b + shape.body.position - left_bottom,
        shape.radius,
        outline_color,
        fill_color
    )


def _draw_poly(options, left_bottom, shape, outline_color, fill_color):
    options.draw_polygon(
        tuple(map(lambda v: v.rotated(shape.body.angle) + shape.body.position - left_bottom, shape.get_vertices())),
        shape.radius,
        outline_color,
        fill_color
    )


def _create_space_surface(space):
    def get_min_max_coordinate():
        coordinates = tuple(mapcat(lambda shape: (shape.bb.left, shape.bb.top, shape.bb.right, shape.bb.bottom), filter(lambda shape: shape.body.body_type == pymunk.Body.DYNAMIC, space.shapes)))

        return juxt(max, min)(coordinates)

    def get_car_from_shape(shape):
        if isinstance(shape.body, Tire):
            return shape.body.car

        if isinstance(shape.body, Car):
            return shape.body

        return None

    def get_color(r, g, b):
        return pymunk.space_debug_draw_options.SpaceDebugColor(r, g, b, 255)

    def get_fill_color(shape):
        if isinstance(shape.body, Star):
            return get_color(128, 255, 128)

        if get_car_from_shape(shape):
            if get_car_from_shape(shape).crash_energy == 0:
                return get_color(128, 128, 255)
            else:
                return get_color(255, 128, 128)

        return get_color(192, 192, 192)

    max_coordinate, min_coordinate = get_min_max_coordinate()

    size = (max_coordinate - min_coordinate, max_coordinate - min_coordinate)
    left_bottom = (min_coordinate, min_coordinate)

    surface = pygame.surface.Surface(size)
    options = pymunk.pygame_util.DrawOptions(surface)

    for shape in space.shapes:
        fill_color = get_fill_color(shape)

        if isinstance(shape, pymunk.shapes.Circle):
            _draw_circle(options, left_bottom, shape, fill_color, fill_color)
        elif isinstance(shape, pymunk.shapes.Segment):
            _draw_segment(options, left_bottom, shape, fill_color, fill_color)
        elif isinstance(shape, pymunk.shapes.Poly):
            _draw_poly(options, left_bottom, shape, fill_color, fill_color)

    return surface, *left_bottom, *size


def create_surface(space, cars, players, actions):
    space_surface, space_left, space_bottom, space_width, space_height = _create_space_surface(space)

    surface = pygame.surface.Surface((800, 640))
    font = pygame.font.Font(None, 24)

    surface.blit(pygame.transform.smoothscale(space_surface, (640, 640)), (0, 0))

    pygame.draw.rect(surface, (255, 255, 255), (640, 0, 800, 640))

    for i, car, player, action in zip(count(), cars, players, actions):
        surface.blit(font.render(player.name, True, (0, 0, 0)), (640 + 4, 80 * i + 4))

        pygame.draw.line(surface, (64, 64, 64), ((640, 80 * i + 12)), ((car.position.x - space_left) * (640 / space_width), 640 - (car.position.y - space_bottom) * (640 / space_height)))

        pygame.draw.line(surface, (192, 192, 192), (640 + 24 - 2, 80 * i + 48), (640 + 24 + 130 + 2, 80 * i + 48))

        for j, action_value in zip(count(), action):
            pygame.draw.rect(surface, (128, 128, 128), (640 + 24 + 50 * j, min(80 * i + 48, 80 * i + 48 - int(action_value * 20)), 30, abs(int(action_value * 20))))

    return surface
