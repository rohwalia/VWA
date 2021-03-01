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

initial =[[1.01,1.01]] # np.random.rand(5, 2)*np.array([2.01,2.01])
t_max = 300
dt = 0.001
tol = 0.001
min_count = 10

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
        r.set_integrator('vode', method='bdf')
        r.set_initial_value(ex, 0)
        r.set_f_params(alpha, gamma)
        while r.successful() and r.t < t_max:
            r.integrate(r.t + dt)
            x_cor = r.y[0]
            y_cor = r.y[1]
            if abs((x_cor - y_cor) / np.sqrt(2)) < tol:
                sect.append(np.sqrt(x_cor ** 2 + y_cor ** 2))
        # print("Intersections found")
        sect_filt = []
        if len(sect) > 2 * min_count:
            for el in sect:
                count = 1
                temp = [el]
                while count < 3:
                    try:
                        if abs(el - sect[sect.index(el) + count]) < 10 * tol:
                            temp.append(sect[sect.index(el) + count])
                            sect.remove(sect[sect.index(el) + count])
                            count = count + 1
                        else:
                            break
                    except IndexError:
                        count = count + 1
                sect_filt.append(median(temp))
            # print("Intersections filtered")
            average = sum(sect_filt) / len(sect_filt)
            sect_filt_up = np.delete(sect_filt, np.where(sect_filt < average))
            points = list(zip(sect_filt_up[:-1], sect_filt_up[1:]))
            i = 0
            while i < len(points):
                if abs(np.diff(points[i]) / np.sqrt(2)) > tol / 10:
                    points.remove(points[i])
                else:
                    i = i + 1
            # print("Omitting stray points")
            if len(points) > min_count:
                points = np.array(points)
                model = LinearRegression().fit(points[:, 0].reshape(-1, 1), points[:, 1])
                stab.append(model.coef_[0])
    if len(stab) > 0:
        return [alpha, gamma, sum(stab) / len(stab)]


if __name__ == '__main__':
    tic = time.perf_counter()
    """parameter = []
    div = 10
    for alpha in np.linspace(0, 5, div):
        for gamma in np.linspace(0, 5, div):
            if alpha > 1 / (gamma - 1) and gamma > 1:
                parameter.append([alpha, gamma])
    """
    excel_file = 'Infinite_cycle.xlsx'
    df = pd.read_excel(excel_file, sheet_name='Estimate', dtpye=float, engine="openpyxl")
    alpha = df['alpha'].tolist()
    gamma = df['gamma'].tolist()
    oscillations = df['oscillations'].tolist()
    alpha = np.array(alpha)
    gamma = np.array(gamma)
    oscillations = np.array(oscillations)
    alpha_1 = alpha[np.where(oscillations == 1)]
    gamma_1 = gamma[np.where(oscillations == 1)]
    data_1 = list(zip(alpha_1, gamma_1))
    data_1_fill = []
    one_prev = data_1[0][1]
    i = 0
    row = []
    while i < len(data_1):
        if data_1[i][1] == one_prev:
            row.append(data_1[i])
        else:
            row = np.array(row)
            for j in np.linspace(row[np.argmin(row[:, 0])][0], row[np.argmax(row[:, 0])][0], 100):
                data_1_fill.append([j, one_prev])
            one_prev = data_1[i][1]
            row = []
            row.append(data_1[i])
        if i == len(data_1) - 1:
            row = np.array(row)
            for j in np.linspace(row[np.argmin(row[:, 0])][0], row[np.argmax(row[:, 0])][0], 100):
                data_1_fill.append([j, one_prev])
        i = i + 1
    data_1_fill = np.array(data_1_fill)
    print(data_1_fill)
    print(len(data_1_fill))
    with Pool(processes=cpu_count()-2) as p:
        for i in p.starmap(stability_calc, data_1_fill, 100):
            if i is not None:
                stability.append(i)
    stability = np.array(stability)
    print(stability)
    output = pd.DataFrame(stability, columns=["alpha", "gamma", "stability"])
    with pd.ExcelWriter('Limit_cycle_1.xlsx') as writer:
        sheetname = "Poincare"
        output.to_excel(writer, sheet_name=sheetname, index=False)
    toc = time.perf_counter()
    print(toc-tic)