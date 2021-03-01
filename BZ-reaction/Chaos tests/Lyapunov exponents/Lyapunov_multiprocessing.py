import numpy as np
from scipy.integrate import ode
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

initial = np.random.rand(20, 3) * np.array([1, 50, 1])  # [[0.3,40,0.2]]
t_max = 100
dt = 0.01
z_init = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1])

def trajectory(t, var, k_5, A, f):
    global k_1
    global k_2
    global k_3
    global k_4
    x, y, z, v_11, v_12, v_13, v_21, v_22, v_23, v_31, v_32, v_33 = var
    kappa = 2 * k_4 * k_5 / (A * k_2 * k_3)
    epsilon = k_5 / (k_3 * A)
    phi = 2 * k_4 * k_1 / (k_2 * k_3)
    dxdt = (phi*y-x*y+x-x**2)/epsilon
    dydt = (-phi*y-x*y+2*f*z)/kappa
    dzdt = x-z
    dv_11dt = (v_11*(1-y-2*x)+v_12*(phi-x))/epsilon
    dv_12dt = (-v_11*y-v_12*(phi+x)+v_13*2*f)/kappa
    dv_13dt = v_11-v_13
    dv_21dt = (v_21*(1-y-2*x)+v_22*(phi-x))/epsilon
    dv_22dt = (-v_21*y-v_22*(phi+x)+v_23*2*f)/kappa
    dv_23dt = v_21-v_23
    dv_31dt = (v_31*(1-y-2*x)+v_32*(phi-x))/epsilon
    dv_32dt = (-v_31*y-v_32*(phi+x)+v_33*2*f)/kappa
    dv_33dt = v_31-v_33
    return [dxdt, dydt, dzdt, dv_11dt, dv_12dt, dv_13dt, dv_21dt, dv_22dt, dv_23dt, dv_31dt, dv_32dt, dv_33dt]

def exponent_calc(k_5, A, f):
    global initial
    global z_init
    global t_max
    global dt
    l_val = []
    for ex in initial:
        N_1 = []
        N_2 = []
        N_3 = []
        t = []
        r = ode(trajectory)
        r.set_integrator('vode', method='bdf')
        r.set_f_params(k_5, A, f)
        t_now = 0
        N_1_value = 0
        N_2_value = 0
        N_3_value = 0
        while t_now < t_max:
            r.set_initial_value(np.append(ex, z_init), 0)
            while r.successful() and r.t < dt:
                r.integrate(r.t + dt)
                vector_1 = np.array([r.y[3], r.y[4], r.y[5]])
                vector_2 = np.array([r.y[6], r.y[7], r.y[8]])
                vector_3 = np.array([r.y[9], r.y[10], r.y[11]])
                N_1_value = N_1_value + np.log(np.linalg.norm(vector_1))
                N_1.append(N_1_value)
                vector_2_n = np.subtract(vector_2, vector_1*(np.dot(vector_1, vector_2)/(np.sqrt(sum(vector_1**2)))**2))
                N_2_value = N_2_value + np.log(np.linalg.norm(vector_2_n))
                N_2.append(N_2_value)
                vector_3_n = np.subtract(vector_3, np.sum([vector_1*(np.dot(vector_1, vector_3)/(np.sqrt(sum(vector_1**2)))**2),
                                               vector_2_n*(np.dot(vector_2_n, vector_3)/(np.sqrt(sum(vector_2_n**2)))**2)], axis=0))
                N_3_value = N_3_value + np.log(np.linalg.norm(vector_3_n))
                N_3.append(N_3_value)
                t_now = t_now + dt
                t.append(t_now)
                z_init = np.concatenate((vector_1/np.linalg.norm(vector_1), vector_2_n/np.linalg.norm(vector_2_n), vector_3_n/np.linalg.norm(vector_3_n)), axis=None)
        """plt.plot(t, N_1)
        plt.show()
        plt.plot(t, N_2)
        plt.show()
        plt.plot(t, N_3)
        plt.show()"""
        model_1 = LinearRegression().fit(np.array(t).reshape(-1, 1), np.array(N_1))
        model_2 = LinearRegression().fit(np.array(t).reshape(-1, 1), np.array(N_2))
        model_3 = LinearRegression().fit(np.array(t).reshape(-1, 1), np.array(N_3))
        l_val.append([model_1.coef_[0], model_2.coef_[0], model_3.coef_[0]])
    l_val = np.array(l_val)
    print("ok")
    return [f, A, k_5, sum(l_val[:,0])/len(l_val[:,0]), sum(l_val[:,1])/len(l_val[:,1]), sum(l_val[:,2])/len(l_val[:,2])]
if __name__ == '__main__':
    tic = time.perf_counter()
    concentration_samples = 14
    parameter = []
    for A in np.linspace(0.01, 1.2, concentration_samples):
        for k_5 in np.linspace(1, 20, 14):
            for f in np.linspace(0, 2.5, 14):
                for j in bifurcation:
                    if abs(A - j[1]) < 1.2 / 100 and abs(f - j[2]) < 2.5 / 300 and k_5 < j[0]:
                        parameter.append([k_5, A, f])
    print(len(parameter))
    with Pool(processes=int(cpu_count()-2)) as p:
        lyapunov = p.starmap(exponent_calc, parameter, concentration_samples)
    print(lyapunov)
    output = pd.DataFrame(lyapunov, columns=["f", "[A]", "k_5", "L_1", "L_2", "L_3"])
    with pd.ExcelWriter('Lyapunov.xlsx') as writer:
        sheetname = "Exponents"
        output.to_excel(writer, sheet_name=sheetname, index=False)
    toc = time.perf_counter()
    print(toc-tic)