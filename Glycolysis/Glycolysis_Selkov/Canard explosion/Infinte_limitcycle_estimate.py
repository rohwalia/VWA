from multiprocessing import Pool, Manager, cpu_count
import numpy as np
from scipy.integrate import ode
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from statistics import median
from sklearn.linear_model import LinearRegression
import time
import pandas as pd
import pandas as pd

initial = np.random.rand(3, 2)*np.array([2.01,2.01])
t_max = 1000
dt = 0.001
tol = 0.001
min = 10

stability = []

def trajectory(t, var, alpha, gamma):
    x, y = var
    dxdt = 1 - x * y ** gamma
    dydt = alpha * y * (x * y ** (gamma - 1) - 1)
    return [dxdt, dydt]

def stability_calc(alpha, gamma):
    global initial
    global t_max
    global dt
    global tol
    global min_count
    stab = []
    for ex in initial:
        sect = []
        r = ode(trajectory)
        r.set_integrator('lsoda')
        r.set_initial_value(ex, 0)
        r.set_f_params(alpha, gamma)
        while r.successful() and r.t < t_max:
            r.integrate(r.t + dt)
            x_cor = r.y[0]
            y_cor = r.y[1]
            if abs((x_cor - y_cor) / np.sqrt(2)) < tol:
                sect.append(np.sqrt(x_cor ** 2 + y_cor ** 2))
        stab.append(len(sect))

    if sum(stab)/len(stab) > 2*min:
        return [alpha, gamma, 1]
    else:
        return [alpha, gamma, 0]


if __name__ == '__main__':
    tic = time.perf_counter()
    parameter = []
    div = 100
    for gamma in np.linspace(1, 5, div):
        for alpha in np.linspace(0, 1.5 / (gamma - 1), div):
            if alpha > 1 / (gamma - 1) and gamma > 1:
                parameter.append([alpha, gamma])
    print(len(parameter))
    with Pool(processes=cpu_count()) as p:
        for i in p.starmap(stability_calc, parameter, div):
            if i is not None:
                stability.append(i)
    stability = np.array(stability)
    print(stability)
    output = pd.DataFrame(stability, columns=["alpha", "gamma", "oscillations"])
    with pd.ExcelWriter('Infinite_cycle.xlsx') as writer:
        sheetname = "Estimate"
        output.to_excel(writer, sheet_name=sheetname, index=False)
    toc = time.perf_counter()
    print(toc-tic)