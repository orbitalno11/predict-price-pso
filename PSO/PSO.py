import numpy as np
import random
from time import time


class PSO:
    def __init__(self, population: int, dimension: int, n_run: int, upper_bound: float, lower_bound: float, lower_w: float, upper_w: float, c_0: float, c_1: float):
        self.population = population
        self.dimension = dimension
        self.swarm = Swarm(self.dimension, self.population)
        self.n_run = n_run
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.v_max = (self.upper_bound - self.lower_bound) / 2
        self.g_best_value = 0
        self.upper_w = upper_w
        self.lower_w = lower_w
        self.c_0 = c_0
        self.c_1 = c_1
        random.seed(int(time() * 1000))
        self.nfc = None
        self.maxnfc = 100000

    def evolution(self):
        iw = np.random.uniform(self.lower_w, self.upper_w, 1)[0]
        w = self.lower_w
        self.initialization()

    def initialization(self):
        self.nfc = 0

        for index, particle in enumerate(self.swarm.population):
            for dim in range(self.dimension):
                particle.x[dim] = self.lower_bound + \
                    (self.upper_bound - self.lower_bound) * random.random()
                particle.v[dim] = random.random()

            particle.setFitness(-(random.random()))
            particle.p_best = particle.x.copy()
            particle.p_best_value = particle.finess

            if particle.finess < self.swarm.g_best_value:
                self.swarm.g_best = index
                self.swarm.g_best_value = particle.finess

    def evaluateSwarm(self):
        for index, particle in enumerate(self.swarm.population):
            particle.setFitness(-(random.random()))
            self.nfc = self.nfc+1

        for index, particle in enumerate(self.swarm.population):
            if particle.finess < particle.p_best_value:
                particle.p_best = particle.x.copy()
                particle.p_best_value = particle.finess

                if particle.finess < self.swarm.g_best_value:
                    self.swarm.g_best = index
                    self.swarm.g_best_value = particle.finess


class Swarm:
    def __init__(self, dimension: int, population: int):
        self.population = [Particle(dimension) for index in range(population)]
        self.g_best_value = 0
        self.g_best = 0


class Particle:
    finess: float
    p_best_value: float

    def __init__(self, dimension: int):
        self.x = [None] * dimension
        self.v = [None] * dimension
        self.p_best = [None] * dimension

    def setFitness(self, fit: float):
        self.finess = fit


pso = PSO(5, 2, 1, 1, -1, 0.9, 0.4, 0.5, 0.5)
pso.evolution()
pso.evaluateSwarm()
