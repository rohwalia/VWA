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

excel_file = 'Hopf.xlsx'
df = pd.read_excel(excel_file, sheet_name='bifurcation', dtpye=float, engine="openpyxl")
f_bif =  df['f'].tolist()
A_bif = df['[A]'].tolist()
k_5_bif = df['k_5'].tolist()
bifurcation = list(zip(k_5_bif, A_bif, f_bif))
stability = []

k_1 = 2
k_2 = 10 ** 6
k_3 = 10
k_4 = 2000

initial = np.random.rand(20, 3) * np.array([1, 50, 1])  # [[0.3,40,0.2]]
t_max = 100
dt = 0.001
tol = 0.001
min_count = 2

def trajectory(t, var, k_5, A, f):
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

def stability_calc(k_5, A, f):
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
        r.set_f_params(k_5, A, f)
        while r.successful() and r.t < t_max:
            r.integrate(r.t + dt)
            x_cor = r.y[0]
            y_cor = r.y[1]
            z_cor = r.y[2]
            if abs((x_cor - y_cor) / np.sqrt(2)) < tol:
                sect.append(np.sqrt(x_cor ** 2 + y_cor ** 2 + z_cor ** 2))
        sect_filt = []
        if len(sect)>2*min_count:
            for el in sect:
                count = 1
                temp = [el]
                while count < 3:
                    try:
                        if abs(el - sect[sect.index(el) + count]) < 10*tol:
                            temp.append(sect[sect.index(el) + count])
                            sect.remove(sect[sect.index(el) + count])
                            count = count + 1
                        else:
                            break
                    except IndexError:
                        count = count + 1
                sect_filt.append(median(temp))
            average = sum(sect_filt)/len(sect_filt)
            sect_filt_up = np.delete(sect_filt, np.where(sect_filt < average))
            points = list(zip(sect_filt_up[:-1], sect_filt_up[1:]))
            i = 0
            while i < len(points):
                if abs(np.diff(points[i])/ np.sqrt(2)) > tol/10:
                    points.remove(points[i])
                else:
                    i = i + 1
            if len(points)>min_count:
                points = np.array(points)
                model = LinearRegression().fit(points[:,0].reshape(-1, 1), points[:,1])
                stab.append(model.coef_[0])
    if len(stab)>0:
        return [f, A, k_5, sum(stab)/len(stab)]
    else:
        return [f, A, k_5, sum(stab)]


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
        for i in p.starmap(stability_calc, parameter, concentration_samples):
            if i is not None:
                stability.append(i)
    stability = np.array(stability)
    print(stability)
    output = pd.DataFrame(stability, columns=["f", "[A]", "k_5", "stability"])
    with pd.ExcelWriter('Limit_cycle.xlsx') as writer:
        sheetname = "Poincare"
        output.to_excel(writer, sheet_name=sheetname, index=False)
    toc = time.perf_counter()
    print(toc-tic)