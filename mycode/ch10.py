## Import modules and boilerplate
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from utils import decorate, savefig, underride, three_frame, set_palette
from scipy.signal import correlate2d
from Cell2D import Cell2D, draw_array

## Class and function definition
class Driver:
    """Represents a driver in a traffic simulation."""

    def __init__(self, loc, speed=4):
        """Initialize the attributes.

        loc: position on track, in miles
        speed: speed in miles per hour
        """
        self.start = loc
        self.loc = loc
        self.speed = speed

    def choose_acceleration(self, dist):
        """Chooses acceleration based on distance.

        dist: distance from car to car in front of it

        returns: acceleration
        """
        return 1

    def set_odometer(self):
        self.start = self.loc

    def read_odometer(self):
        return self.loc - self.start

class BetterDriver(Driver):

    def choose_acceleration(self, dist):
        if not hasattr(Driver, 'last_dist'):
            self.last_dist = 0
        if self.last_dist > dist:
            # gap is closing
            self.last_dist = dist
            # distance of no deceleration
            d0 = 200
            return (10 * dist)/d0 - 10
        else:
            # gap is increasing
            self.last_dist = dist
            # distance of max acceleration
            d1 = 100
            return dist/d1

class Highway(Cell2D):

    max_acc = 1
    min_acc = -10
    speed_limit = 40

    def __init__(self, n=10, length=1000, eps=0, constructor=Driver):
        """Initializes the attributes.

        n: number of drivers
        length: length of the track
        eps: variability in speed
        constructor: function used to instantiate drivers
        """
        self.length = length
        self.eps = eps
        self.crashes = 0

        # create the drivers
        locs = np.linspace(0, length, n, endpoint=False)
        self.drivers = [constructor(loc) for loc in locs]

        # and link them up
        for i in range(n):
            j = (i+1) % n
            self.drivers[i].next = self.drivers[j]

    def step(self):
        """Performs one time step."""
        for driver in self.drivers:
            self.move(driver)

    def move(self, driver):
        """Updates 'driver'.

        driver: Driver object
        """
        # get the distance to the next driver
        dist = self.distance(driver)

        # let the driver choose acceleration
        acc = driver.choose_acceleration(dist)
        acc = min(acc, self.max_acc)
        acc = max(acc, self.min_acc)
        speed = driver.speed + acc

        # add random noise to speed
        speed *= np.random.uniform(1-self.eps, 1+self.eps)

        # keep speed nonnegative and under the speed limit
        speed = max(speed, 0)
        speed = min(speed, self.speed_limit)

        # if current speed would collide with next driver, stop
        if speed > dist:
            speed = 0
            self.crashes += 1

        # update speed and loc
        driver.speed = speed
        driver.loc += speed

    def distance(self, driver):
        """Distance from 'driver' to next driver.

        driver: Driver object
        """
        dist = driver.next.loc - driver.loc
        # fix wraparound
        if dist < 0:
            dist += self.length
        return dist

    def set_odometers(self):
        return [driver.set_odometer() for driver in self.drivers]

    def read_odometers(self):
        return np.mean([driver.read_odometer() for driver in self.drivers])

    def draw(self):
        """Draw drivers and show collisions.
        """
        self.init_fig()
        return self.fig

    def init_fig(self):
        """Initialize Sugarscape figure for animation and drawing."""
        drivers = self.drivers

        self.fig, self.ax = plt.subplots()
        self.ax.set_axis_off()
        self.ax.aspect='equal'
        self.ax.set_xlim(-1.05, 1.05)
        self.ax.set_ylim(-1.05, 1.05)

        # draw the drivers
        xs, ys = self.get_coords(drivers)
        self.driver_points = self.ax.plot(xs, ys, 'bs', markersize=10, alpha=0.7)[0]

        # draw the collisions
        stopped = [driver for driver in drivers if driver.speed==0]
        xs, ys = self.get_coords(stopped, r=0.8)
        self.collision_points = self.ax.plot(xs, ys,
                                            'r^', markersize=12, alpha=0.7)[0]

    def update_anim(frame, self):
        """Update frame of animation"""
        drivers = self.drivers

        self.step()

        # draw the drivers
        xs, ys = self.get_coords(drivers)
        self.driver_points.set_data(xs, ys)

        # draw the collisions
        stopped = [driver for driver in drivers if driver.speed==0]
        xs, ys = self.get_coords(stopped, r=0.8)
        self.collision_points.set_data(xs, ys)

    def get_coords(self, drivers, r=1):
        """Get the coordinates of the drivers.

        Transforms from (row, col) into (x, y)

        drivers: sequence of Driver
        r: radius of the circle

        returns: tuple of sequences, (xs, ys)
        """
        locs = np.array([driver.loc for driver in drivers])
        locs *= 2 * np.pi / self.length
        xs = r * np.cos(locs)
        ys = r * np.sin(locs)
        return xs, ys

def run_simulation(eps, constructor=Driver, iters=100):
    """Measure average speed of all 'constructor' over 'iters' steps.

    eps: float, random speed error proportion
    constructor: Driver
    iters: int, steps to measure speed over
    """
    res = []
    for n in range(5, 100, 5):
        highway = Highway(n, eps=eps, constructor=constructor)
        highway.loop(iters)

        highway.set_odometers()
        highway.loop(iters)

        avg_speed = highway.read_odometers() / iters
        res.append((n, avg_speed))

    return np.transpose(res)

def speed_v_eps():
    """Plot speed versus epsilon for range of drivers.
    """
    np.random.seed(20)
    set_palette('Blues', 4, reverse=True)

    for eps in [0.0, 0.001, 0.01]:
        xs, ys = run_simulation(eps)
        plt.plot(xs, ys, label='eps=%g' %eps)

    decorate(xlabel='Number of cars',
            ylabel='Average speed',
            xlim = [0, 100], ylim=[0, 42])

    savefig('myfigs/chap10-2')
    plt.show(block=True)

def compare_drivers(eps=0.0):
    """Compare different driver constructors.
    """
    for constructor in [Driver, BetterDriver]:
        xs, ys = run_simulation(eps=eps, constructor=constructor)
        plt.plot(xs, ys, label=constructor.__name__)

    decorate(xlabel='Number of cars',
            ylabel='Average speed',
            xlim=[0,100], ylim=[0, 42])

    plt.show(block=True)

if __name__ == '__main__':
    """Scripts for execution."""
    ## Compare drivers
    compare_drivers(eps=0.001)

    ## Prototype the BetterDriver
    highway = Highway(40, eps=0.001, constructor=BetterDriver)
    ani = highway.animate()
    plt.show(block=True)
