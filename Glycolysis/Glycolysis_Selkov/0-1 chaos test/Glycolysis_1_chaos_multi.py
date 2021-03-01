from multiprocessing import Pool, Manager, cpu_count
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from statistics import median
from sklearn.linear_model import LinearRegression
import time
import pandas as pd

factor = np.random.rand(25)*2*np.pi
t_max = 100
dt = 0.1
t = np.linspace(0, t_max, int(t_max/dt))
initial = [1.01, 1.01]

chaos = []

def trajectory(var, t, alpha, gamma):
    x, y = var
    dxdt = 1 - x * y ** gamma
    dydt = alpha * y * (x * y ** (gamma - 1) - 1)
    return [dxdt, dydt]

def chaos_calc(alpha, gamma):
    global initial
    global t_max
    global dt
    global factor
    c_val = []
    for c in factor:
        r = odeint(trajectory, initial, t, (alpha, gamma))
        p_c = []
        q_c = []
        p_now = 0
        q_now = 0
        for i in range(len(r)):
            amp = np.sqrt(r[i][0] ** 2 + r[i][1] ** 2)
            p_now = p_now + amp * np.cos((i + 1) * c)
            q_now = q_now + amp * np.sin((i + 1) * c)
            p_c.append(p_now)
            q_c.append(q_now)
        #plt.plot(p_c, q_c)
        #plt.show()
        M = []
        for i in range(round(len(p_c) / 10)):
            M_value = 0
            for el in range(len(p_c) - round(len(p_c) / 10)):
                M_value = M_value + ((p_c[i + el] - p_c[el]) ** 2 + (q_c[i + el] - q_c[el]) ** 2) * dt ** 2
            M.append(M_value / (len(p_c) - round(len(p_c) / 10)))
        M = np.array(M)
        """plt.plot(np.log(np.array(range(len(M)))[1:]), np.log(M[1:]))
        plt.show()"""
        model = LinearRegression().fit(np.log(np.array(range(len(M)))[1:]).reshape(-1, 1), np.log(M[1:]))
        c_val.append(model.coef_[0])
    return [alpha, gamma, median(c_val)]


if __name__ == '__main__':
    tic = time.perf_counter()
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
        chaos = p.starmap(chaos_calc, data_1_fill, 100)
    print(chaos)
    output = pd.DataFrame(chaos, columns=["alpha", "gamma", "chaos"])
    with pd.ExcelWriter('Chaos.xlsx') as writer:
        sheetname = "0-1 chaos test"
        output.to_excel(writer, sheet_name=sheetname, index=False)
    toc = time.perf_counter()
    print(toc-tic)