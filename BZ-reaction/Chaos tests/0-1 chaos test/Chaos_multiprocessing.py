import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from statistics import median
from sklearn.linear_model import LinearRegression
import time
from multiprocessing import Pool, cpu_count
import pandas as pd

excel_file = 'Hopf.xlsx'
df = pd.read_excel(excel_file, sheet_name='bifurcation', dtpye=float, engine="openpyxl")
f_bif =  df['f'].tolist()
A_bif = df['[A]'].tolist()
k_5_bif = df['k_5'].tolist()
bifurcation = list(zip(k_5_bif, A_bif, f_bif))

k_1 = 2
k_2 = 10**6
k_3 = 10
k_4 = 2000

factor = np.random.rand(100)*2*np.pi #[[0.3,40,0.2]]
t_max = 100
dt = 0.1
t = np.linspace(0, t_max, int(t_max/dt))
initial = [0.3,40,0.2]

def trajectory(var, t, k_5, A, f):
    global k_1
    global k_2
    global k_3
    global k_4
    x, y, z = var
    kappa = 2 * k_4 * k_5 / (A * k_2 * k_3)
    epsilon = k_5 / (k_3 * A)
    phi = 2 * k_4 * k_1 / (k_2 * k_3)
    dxdt = (phi*y-x*y+x-x**2)/epsilon
    dydt = (-phi*y-x*y+2*f*z)/kappa
    dzdt = x-z
    return [dxdt, dydt, dzdt]

def chaos_calc(k_5, A, f):
    global initial
    global factor
    global t
    global dt
    c_val = []
    for c in factor:
        r = odeint(trajectory, initial, t, (k_5, A, f))
        p_c=[]
        q_c=[]
        p_now = 0
        q_now = 0
        for i in range(len(r)):
            amp = np.sqrt(r[i][0] ** 2 + r[i][1] ** 2 + r[i][2] ** 2)
            p_now = p_now + amp * np.cos((i+1)*c)
            q_now = q_now + amp * np.sin((i+1)*c)
            p_c.append(p_now)
            q_c.append(q_now)
        """plt.plot(p_c, q_c)
        plt.show()"""
        M = []
        for i in range(round(len(p_c)/10)):
            M_value = 0
            for el in range(len(p_c)-round(len(p_c)/10)):
                M_value = M_value + ((p_c[i+el]-p_c[el])**2 + (q_c[i+el]-q_c[el])**2)*dt**2
            M.append(M_value/(len(p_c)-round(len(p_c)/10)))
        M = np.array(M)
        """plt.plot(np.log(np.array(range(len(M)))[1:]), np.log(M[1:]))
        plt.show()"""
        model = LinearRegression().fit(np.log(np.array(range(len(M)))[1:]).reshape(-1, 1), np.log(M[1:]))
        c_val.append(model.coef_[0])
    return [f, A, k_5, median(c_val)]

if __name__ == '__main__':
    tic = time.perf_counter()
    concentration_samples = 50
    parameter = []
    for A in np.linspace(0.01, 1.2, concentration_samples):
        for k_5 in np.linspace(1, 20, 50):
            for f in np.linspace(0, 2.5, 50):
                for j in bifurcation:
                    if abs(A - j[1]) < 1.2 / 100 and abs(f - j[2]) < 2.5 / 300 and k_5 < j[0]:
                        parameter.append([k_5, A, f])
    with Pool(processes=cpu_count()) as p:
        chaos = p.starmap(chaos_calc, parameter, concentration_samples)
    print(chaos)
    output = pd.DataFrame(chaos, columns=["f", "[A]", "k_5", "chaos"])
    with pd.ExcelWriter('Chaos.xlsx') as writer:
        sheetname = "0-1 chaos test"
        output.to_excel(writer, sheet_name=sheetname, index=False)
    toc = time.perf_counter()
    print(toc-tic)