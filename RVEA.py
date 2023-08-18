#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/8/7 23:01
# @Author  : Xavier Ma
# @Email   : xavier_mayiming@163.com
# @File    : RVEA.py
# @Statement : Reference vector-guided evolutionary algorithm (RVEA)
# @Reference : Cheng R, Jin Y, Olhofer M, et al. A reference vector guided evolutionary algorithm for many-objective optimization[J]. IEEE Transactions on Evolutionary Computation, 2016, 20(5): 773-791.
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.spatial.distance import cdist, pdist, squareform


def cal_obj(pop, nobj):
    # DTLZ1
    g = 100 * (pop.shape[1] - nobj + 1 + np.sum((pop[:, nobj - 1:] - 0.5) ** 2 - np.cos(20 * np.pi * (pop[:, nobj - 1:] - 0.5)), axis=1))
    objs = np.zeros((pop.shape[0], nobj))
    temp_pop = pop[:, : nobj - 1]
    for i in range(nobj):
        f = 0.5 * (1 + g)
        f *= np.prod(temp_pop[:, : temp_pop.shape[1] - i], axis=1)
        if i > 0:
            f *= 1 - temp_pop[:, temp_pop.shape[1] - i]
        objs[:, i] = f
    return objs


def factorial(n):
    # calculate n!
    if n == 0 or n == 1:
        return 1
    else:
        return n * factorial(n - 1)


def combination(n, m):
    # choose m elements from a n-length set
    if m == 0 or m == n:
        return 1
    elif m > n:
        return 0
    else:
        return factorial(n) // (factorial(m) * factorial(n - m))


def reference_points(npop, nvar):
    # calculate approximately npop uniformly distributed reference points on nvar dimensions
    h1 = 0
    while combination(h1 + nvar, nvar - 1) <= npop:
        h1 += 1
    points = np.array(list(combinations(np.arange(1, h1 + nvar), nvar - 1))) - np.arange(nvar - 1) - 1
    points = (np.concatenate((points, np.zeros((points.shape[0], 1)) + h1), axis=1) - np.concatenate((np.zeros((points.shape[0], 1)), points), axis=1)) / h1
    if h1 < nvar:
        h2 = 0
        while combination(h1 + nvar - 1, nvar - 1) + combination(h2 + nvar, nvar - 1) <= npop:
            h2 += 1
        if h2 > 0:
            temp_points = np.array(list(combinations(np.arange(1, h2 + nvar), nvar - 1))) - np.arange(nvar - 1) - 1
            temp_points = (np.concatenate((temp_points, np.zeros((temp_points.shape[0], 1)) + h2), axis=1) - np.concatenate((np.zeros((temp_points.shape[0], 1)), temp_points), axis=1)) / h2
            temp_points = temp_points / 2 + 1 / (2 * nvar)
            points = np.concatenate((points, temp_points), axis=0)
    return points


def selection(pop, npop, nvar):
    # select the mating pool
    ind = np.random.randint(0, pop.shape[0], npop)
    mating_pool = pop[ind]
    if npop % 2 == 1:
        mating_pool = np.concatenate((mating_pool, mating_pool[0].reshape(1, nvar)), axis=0)
    return mating_pool


def crossover(mating_pool, lb, ub, eta_c):
    # simulated binary crossover (SBX)
    (noff, dim) = mating_pool.shape
    nm = int(noff / 2)
    parent1 = mating_pool[:nm]
    parent2 = mating_pool[nm:]
    beta = np.zeros((nm, dim))
    mu = np.random.random((nm, dim))
    flag1 = mu <= 0.5
    flag2 = ~flag1
    beta[flag1] = (2 * mu[flag1]) ** (1 / (eta_c + 1))
    beta[flag2] = (2 - 2 * mu[flag2]) ** (-1 / (eta_c + 1))
    offspring1 = (parent1 + parent2) / 2 + beta * (parent1 - parent2) / 2
    offspring2 = (parent1 + parent2) / 2 - beta * (parent1 - parent2) / 2
    offspring = np.concatenate((offspring1, offspring2), axis=0)
    offspring = np.min((offspring, np.tile(ub, (noff, 1))), axis=0)
    offspring = np.max((offspring, np.tile(lb, (noff, 1))), axis=0)
    return offspring


def mutation(pop, lb, ub, pm, eta_m):
    # polynomial mutation
    (npop, nvar) = pop.shape
    lb = np.tile(lb, (npop, 1))
    ub = np.tile(ub, (npop, 1))
    site = np.random.random((npop, nvar)) < pm / nvar
    mu = np.random.random((npop, nvar))
    delta1 = (pop - lb) / (ub - lb)
    delta2 = (ub - pop) / (ub - lb)
    temp = np.logical_and(site, mu <= 0.5)
    pop[temp] += (ub[temp] - lb[temp]) * ((2 * mu[temp] + (1 - 2 * mu[temp]) * (1 - delta1[temp]) ** (eta_m + 1)) ** (1 / (eta_m + 1)) - 1)
    temp = np.logical_and(site, mu > 0.5)
    pop[temp] += (ub[temp] - lb[temp]) * (1 - (2 * (1 - mu[temp]) + 2 * (mu[temp] - 0.5) * (1 - delta2[temp]) ** (eta_m + 1)) ** (1 / (eta_m + 1)))
    pop = np.min((pop, ub), axis=0)
    pop = np.max((pop, lb), axis=0)
    return pop


def GAoperation(pop, objs, npop, nvar, nobj, lb, ub, pm, eta_c, eta_m):
    # genetic algorithm (GA) operation
    mating_pool = selection(pop, npop, nvar)
    offspring = crossover(mating_pool, lb, ub, eta_c)
    offspring = mutation(offspring, lb, ub, pm, eta_m)
    new_objs = cal_obj(offspring, nobj)
    return np.concatenate((pop, offspring), axis=0), np.concatenate((objs, new_objs), axis=0)


def environmental_selection(pop, objs, V, t, iter, alpha, nvar, nobj, theta):
    # environmental selection
    # Step 1. Objective translation
    t_objs = objs - np.min(objs, axis=0)  # translated objectives

    # Step 2. Population partition
    angle = np.arccos(1 - cdist(t_objs, V, 'cosine'))
    association = np.argmin(angle, axis=1)

    # Step 3. Angle-penalized distance calculation
    theta0 = (t / iter) ** alpha
    points = np.unique(association)
    next_pop = np.zeros((points.shape[0], nvar))
    next_objs = np.zeros((points.shape[0], nobj))
    for i in range(points.shape[0]):
        ind = np.where(association == points[i])[0]
        APD = (1 + nobj * theta0 * angle[ind, points[i]] / theta[points[i]]) * np.sqrt(np.sum(t_objs[ind] ** 2, axis=1))
        best = ind[np.argmin(APD)]
        next_pop[i] = pop[best]
        next_objs[i] = objs[best]
    return next_pop, next_objs


def dominates(obj1, obj2):
    # determine whether obj1 dominates obj2
    sum_less = 0
    for i in range(len(obj1)):
        if obj1[i] > obj2[i]:
            return False
        elif obj1[i] != obj2[i]:
            sum_less += 1
    return sum_less > 0


def main(npop, iter, lb, ub, nobj, eta_c=30, eta_m=20, alpha=2, fr=0.1):
    """
    The main function
    :param npop: population size
    :param iter: iteration number
    :param lb: lower bound
    :param ub: upper bound
    :param nobj: the dimension of objective space
    :param eta_c: spread factor distribution index (default = 30)
    :param eta_m: perturbance factor distribution index (default = 20)
    :param alpha: the parameter to control the change rate of APD (default = 2)
    :param fr: reference vector adaption parameter (default = 0.1)
    :return:
    """
    # Step 1. Initialization
    nvar = len(lb)  # the dimension of decision space
    pop = np.random.uniform(lb, ub, (npop, nvar))  # population
    objs = cal_obj(pop, nobj)  # objectives
    V0 = reference_points(npop, nobj)  # original reference vectors
    V = V0  # reference vectors
    sigma = 1 - squareform(pdist(V, metric='cosine'), force='no', checks=True)
    sigma = np.sort(sigma)
    theta = np.arccos(sigma[:, -2])  # the smallest angle value of each reference vector to the others

    # Step 2. The main loop
    for t in range(iter):

        if (t + 1) % 200 == 0:
            print('Iteration: ' + str(t + 1) + ' completed.')

        # Step 2.1. GA operation
        pop, objs = GAoperation(pop, objs, npop, nvar, nobj, lb, ub, 1, eta_c, eta_m)

        # Step 2.2. Environmental selection
        pop, objs = environmental_selection(pop, objs, V, t, iter, alpha, nvar, nobj, theta)

        # Step 2.3. Reference vector adaption
        if t % (iter * fr) == 0:
            V = V0 * (np.max(objs, axis=0) - np.min(objs, axis=0))
            sigma = 1 - squareform(pdist(V, metric='cosine'), force='no', checks=True)
            sigma = np.sort(sigma)
            theta = np.arccos(sigma[:, -2])

    # Step 3. Sort the results
    npop = pop.shape[0]
    dom = np.full(npop, False)
    for i in range(npop - 1):
        for j in range(i, npop):
            if not dom[i] and dominates(objs[j], objs[i]):
                dom[i] = True
            if not dom[j] and dominates(objs[i], objs[j]):
                dom[j] = True
    pf = objs[~dom]
    ax = plt.figure().add_subplot(111, projection='3d')
    ax.view_init(45, 45)
    x = [o[0] for o in pf]
    y = [o[1] for o in pf]
    z = [o[2] for o in pf]
    ax.scatter(x, y, z, color='red')
    ax.set_xlabel('objective 1')
    ax.set_ylabel('objective 2')
    ax.set_zlabel('objective 3')
    plt.title('The Pareto front of DTLZ1')
    plt.savefig('Pareto front')
    plt.show()


if __name__ == '__main__':
    main(105, 5000, np.array([0] * 7), np.array([1] * 7), 3)
