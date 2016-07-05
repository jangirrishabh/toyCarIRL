import random
import math
import numpy as np

import pygame
from pygame.color import THECOLORS

import pymunk
from pymunk.vec2d import Vec2d
from pymunk.pygame_util import draw

# PyGame init
width = 1000
height = 700
pygame.init()
screen = pygame.display.set_mode((width, height))
clock = pygame.time.Clock()

# Turn off alpha since we don't use it.
screen.set_alpha(None)

# Showing sensors and redrawing slows things down.
flag = 1
show_sensors = flag
draw_screen = flag


class GameState:
    def __init__(self, weights):
        # Global-ish.
        self.crashed = False

        # Physics stuff.
        self.space = pymunk.Space()
        self.space.gravity = pymunk.Vec2d(0., 0.)
        self.W = weights #weights for the reward function

        # Create the car.
        self.create_car(150, 20, 15)

        # Record steps.
        self.num_steps = 0
        self.num_obstacles_type = 4

        # Create walls.
        static = [
            pymunk.Segment(
                self.space.static_body,
                (0, 1), (0, height), 1),
            pymunk.Segment(
                self.space.static_body,
                (1, height), (width, height), 1),
            pymunk.Segment(
                self.space.static_body,
                (width-1, height), (width-1, 1), 1),
            pymunk.Segment(
                self.space.static_body,
                (1, 1), (width, 1), 1)
        ]
        for s in static:
            s.friction = 1.
            s.group = 1
            s.collision_type = 1
            s.color = THECOLORS['red']
        self.space.add(static)

        # Create some obstacles, semi-randomly.
        # We'll create three and they'll move around to prevent over-fitting.
        self.obstacles = []
        #self.obstacles.append(self.create_obstacle(380, 220, 70, "yellow"))
        #self.obstacles.append(self.create_obstacle(250, 500, 70, "yellow"))
        #self.obstacles.append(self.create_obstacle(780, 330, 70, "brown"))
        #self.obstacles.append(self.create_obstacle(530, 500, 70, "brown"))

        self.obstacles.append(self.create_obstacle([100, 100], [100, 585] , 7, "yellow"))
        self.obstacles.append(self.create_obstacle([450, 600], [100, 600] , 7, "yellow"))
        self.obstacles.append(self.create_obstacle([900, 100], [900, 585] , 7, "yellow"))
        self.obstacles.append(self.create_obstacle([900, 600], [550, 600] , 7, "yellow"))

        self.obstacles.append(self.create_obstacle([200, 100], [200, 480] , 7, "yellow"))
        self.obstacles.append(self.create_obstacle([450, 500], [200, 500] , 7, "yellow"))
        self.obstacles.append(self.create_obstacle([800, 100], [800, 480] , 7, "yellow"))
        self.obstacles.append(self.create_obstacle([800, 500], [550, 500] , 7, "yellow"))

        self.obstacles.append(self.create_obstacle([300, 100], [300, 350] , 7, "yellow"))
        self.obstacles.append(self.create_obstacle([700, 380], [300, 380] , 7, "yellow"))
        self.obstacles.append(self.create_obstacle([700, 100], [700, 350] , 7, "yellow"))

        self.obstacles.append(self.create_obstacle([400, 100], [600, 100] , 7, "brown"))
        self.obstacles.append(self.create_obstacle([400, 200], [600, 200] , 7, "brown"))
        self.obstacles.append(self.create_obstacle([400, 300], [600, 300] , 7, "brown"))

        self.obstacles.append(self.create_obstacle([700, 370], [300, 370] , 7, "brown"))
        self.obstacles.append(self.create_obstacle([310, 100], [310, 350] , 7, "brown"))
        self.obstacles.append(self.create_obstacle([690, 100], [690, 350] , 7, "brown"))




        # Create a cat.
        #self.create_cat()

    def create_obstacle(self, xy1, xy2, r, color):
        c_body = pymunk.Body(pymunk.inf, pymunk.inf)
        #c_shape = pymunk.Circle(c_body, r)
        c_shape = pymunk.Segment(c_body, xy1, xy2, r)
        #c_shape.elasticity = 1.0
        #c_body.position = x, y
        c_shape.friction = 1.
        c_shape.group = 1
        c_shape.collision_type = 1
        c_shape.color = THECOLORS[color]
        self.space.add(c_body, c_shape)
        return c_body

    def create_cat(self):
        inertia = pymunk.moment_for_circle(1, 0, 14, (0, 0))
        self.cat_body = pymunk.Body(1, inertia)
        self.cat_body.position = 50, height - 100
        self.cat_shape = pymunk.Circle(self.cat_body, 30)
        self.cat_shape.color = THECOLORS["orange"]
        self.cat_shape.elasticity = 1.0
        self.cat_shape.angle = 0.5
        direction = Vec2d(1, 0).rotated(self.cat_body.angle)
        self.space.add(self.cat_body, self.cat_shape)

    def create_car(self, x, y, r):
        inertia = pymunk.moment_for_circle(1, 0, 14, (0, 0))
        self.car_body = pymunk.Body(1, inertia)
        self.car_body.position = x, y
        self.car_shape = pymunk.Circle(self.car_body, r)
        self.car_shape.color = THECOLORS["green"]
        self.car_shape.elasticity = 1.0
        self.car_body.angle = 1.4
        driving_direction = Vec2d(1, 0).rotated(self.car_body.angle)
        self.car_body.apply_impulse(driving_direction)
        self.space.add(self.car_body, self.car_shape)

    def frame_step(self, action):
        if action == 0:  # Turn left.
            self.car_body.angle -= .3
        elif action == 1:  # Turn right.
            self.car_body.angle += .3

        #Move obstacles.
        #if self.num_steps % 100 == 0:
            #self.move_obstacles()

        # Move cat.
        #if self.num_steps % 5 == 0:
            #self.move_cat()

        driving_direction = Vec2d(1, 0).rotated(self.car_body.angle)
        self.car_body.velocity = 100 * driving_direction

        # Update the screen and stuff.
        screen.fill(THECOLORS["black"])
        draw(screen, self.space)
        self.space.step(1./10)
        if draw_screen:
            pygame.display.flip()
        clock.tick()

        # Get the current location and the readings there.
        x, y = self.car_body.position
        readings = self.get_sonar_readings(x, y, self.car_body.angle)

        # Set the reward.
        # Car crashed when any reading == 1
        if self.car_is_crashed(readings):
            self.crashed = True
            readings.append(1)
            self.recover_from_crash(driving_direction)
        else:
            readings.append(0)
            
            
        reward = np.dot(self.W, readings)
        state = np.array([readings])

        self.num_steps += 1

        return reward, state, readings

    def move_obstacles(self):
        # Randomly move obstacles around.
        for obstacle in self.obstacles:
            speed = random.randint(1, 5)
            direction = Vec2d(1, 0).rotated(self.car_body.angle + random.randint(-2, 2))
            obstacle.velocity = speed * direction

    def move_cat(self):
        speed = random.randint(20, 200)
        self.cat_body.angle -= random.randint(-1, 1)
        direction = Vec2d(1, 0).rotated(self.cat_body.angle)
        self.cat_body.velocity = speed * direction

    def car_is_crashed(self, readings):
        if readings[0] >= 0.96 or readings[1] >= 0.96 or readings[2] >= 0.96:
            return True
        else:
            return False

    def recover_from_crash(self, driving_direction):
        """
        We hit something, so recover.
        """
        while self.crashed:
            # Go backwards.
            self.car_body.velocity = -100 * driving_direction
            self.crashed = False
            for i in range(10):
                self.car_body.angle += .2  # Turn a little.
                #screen.fill(THECOLORS["red"])  # Red is scary!
                draw(screen, self.space)
                self.space.step(1./10)
                if draw_screen:
                    pygame.display.flip()
                clock.tick()

    # def sum_readings(self, readings):
    #     """Sum the number of non-zero readings."""
    #     tot = 0
    #     print ("readings  :: ", readings)
    #     for i in readings:
    #         tot += i
    #     return tot

    def get_sonar_readings(self, x, y, angle):
        readings = []
        """
        Instead of using a grid of boolean(ish) sensors, sonar readings
        simply return N "distance" readings, one for each sonar
        we're simulating. The distance is a count of the first non-zero
        reading starting at the object. For instance, if the fifth sensor
        in a sonar "arm" is non-zero, then that arm returns a distance of 5.
        """
        # Make our arms.
        arm_left = self.make_sonar_arm(x, y)
        arm_middle = arm_left
        arm_right = arm_left
        
        obstacleType = []
        obstacleType.append(self.get_arm_distance(arm_left, x, y, angle, 0.75)[1])
        obstacleType.append(self.get_arm_distance(arm_middle, x, y, angle, 0)[1])
        obstacleType.append(self.get_arm_distance(arm_right, x, y, angle, -0.75)[1])

        ObstacleNumber = np.zeros(self.num_obstacles_type)

        for i in obstacleType:
            if i == 0:
                ObstacleNumber[0] += 1
            elif i == 1:
                ObstacleNumber[1] += 1
            elif i == 2:
                ObstacleNumber[2] += 1
            elif i == 3:
                ObstacleNumber[3] += 1
           

        # Rotate them and get readings.
        readings.append(1.0 - float(self.get_arm_distance(arm_left, x, y, angle, 0.75)[0]/39.0)) # 39 = max distance
        readings.append(1.0 - float(self.get_arm_distance(arm_middle, x, y, angle, 0)[0]/39.0))
        readings.append(1.0 - float(self.get_arm_distance(arm_right, x, y, angle, -0.75)[0]/39.0))
        readings.append(float(ObstacleNumber[0]/3.0))
        readings.append(float(ObstacleNumber[1]/3.0))
        readings.append(float(ObstacleNumber[2]/3.0))
        readings.append(float(ObstacleNumber[3]/3.0))

        if show_sensors:
            pygame.display.update()
        
        return readings

    def get_arm_distance(self, arm, x, y, angle, offset):
        # Used to count the distance.
        i = 0

        # Look at each point and see if we've hit something.
        for point in arm:
            i += 1

            # Move the point to the right spot.
            rotated_p = self.get_rotated_point(
                x, y, point[0], point[1], angle + offset
            )

            # Check if we've hit something. Return the current i (distance)
            # if we did.
            if rotated_p[0] <= 0 or rotated_p[1] <= 0 \
                    or rotated_p[0] >= width or rotated_p[1] >= height:
                return [i, 3]  # Sensor is off the screen, return 3 for wall obstacle
            else:
                obs = screen.get_at(rotated_p)
                temp = self.get_track_or_not(obs)
                if temp != 0:
                    return [i, temp] #sensor hit a round obstacle, return the type of obstacle

            if show_sensors:
                pygame.draw.circle(screen, (255, 255, 255), (rotated_p), 2)

        # Return the distance for the arm.
        return [i, 0] #sensor did not hit anything return 0 for black space

    def make_sonar_arm(self, x, y):
        spread = 8  # Default spread.
        distance = 7  # Gap before first sensor.
        arm_points = []
        # Make an arm. We build it flat because we'll rotate it about the
        # center later.
        for i in range(1, 40):
            arm_points.append((distance + x + (spread * i), y))

        return arm_points

    def get_rotated_point(self, x_1, y_1, x_2, y_2, radians):
        # Rotate x_2, y_2 around x_1, y_1 by angle.
        x_change = (x_2 - x_1) * math.cos(radians) + \
            (y_2 - y_1) * math.sin(radians)
        y_change = (y_1 - y_2) * math.cos(radians) - \
            (x_1 - x_2) * math.sin(radians)
        new_x = x_change + x_1
        new_y = height - (y_change + y_1)
        return int(new_x), int(new_y)

    # def get_track_or_not(self, reading):
    #     if reading == THECOLORS['black']:
    #         return 0
    #     else:
    #         return 1

    def get_track_or_not(self, reading): # basically differentiate b/w the objects the car views.
        if reading == THECOLORS['yellow']:
            return 1  # Sensor is on a yellow obstacle
        elif reading == THECOLORS['brown']:
            return 2  # Sensor is on brown obstacle
        else:
            return 0 #for black


if __name__ == "__main__":
    weights = [1, 1, 1, 1, 1, 1, 1, 1]
    game_state = GameState()
    while True:
        game_state.frame_step((random.randint(0, 2)))
