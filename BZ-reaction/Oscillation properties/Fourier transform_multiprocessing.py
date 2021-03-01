import numpy as np
from scipy.integrate import odeint
from scipy import fftpack
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from statistics import median
from sklearn.linear_model import LinearRegression
import time
from multiprocessing import Pool, cpu_count
import plotly.graph_objects as go
import pandas as pd

excel_file = 'Hopf.xlsx'
df = pd.read_excel(excel_file, sheet_name='bifurcation', dtpye=float, engine="openpyxl")
f_bif =  df['f'].tolist()
A_bif = df['[A]'].tolist()
k_5_bif = df['k_5'].tolist()
bifurcation = list(zip(k_5_bif, A_bif, f_bif))

k_1 = 2
k_2 = 10 ** 6
k_3 = 10
k_4 = 2000

initial = np.random.rand(20, 3) * np.array([1, 50, 1])  # [[0.3,40,0.2]]
t_max = 100
dt = 0.001
t = np.linspace(0, t_max, int(t_max/dt))
up = 0.01

def trajectory(var, t, k_5, A, f):
    global k_1
    global k_2
    global k_3
    global k_4
    x, y, z = var
    kappa = 2 * k_4 * k_5 / (A * k_2 * k_3)
    epsilon = k_5 / (k_3 * A)
    phi = 2 * k_4 * k_1 / (k_2 * k_3)
    dxdt = (phi * y - x * y + x - x ** 2) / epsilon
    dydt = (-phi * y - x * y + 2 * f * z) / kappa
    dzdt = x - z
    return [dxdt, dydt, dzdt]

def fourier_transform(k_5, A, f):
    global initial
    global t
    global up
    global dt
    amp_main = []
    freq_main = []
    for ex in initial:
        r = odeint(trajectory, ex, t, (k_5, A, f))
        fourier = []
        for i in range(len(r)):
            amp = np.sqrt(r[i][0] ** 2 + r[i][1] ** 2 + r[i][2] ** 2)
            fourier.append(amp)
        fourier = np.array(fourier)
        fft = fftpack.fft(fourier)
        freqs = fftpack.fftfreq(fourier.size, d=dt)
        power = np.abs(fft)[np.where(freqs > up)]
        freqs = freqs[np.where(freqs > up)]
        freq_main.append(freqs[power.argmax()])
        amp_main.append(power[power.argmax()])
    return [[f, A, k_5, sum(freq_main)/len(freq_main)], [f, A, k_5, sum(amp_main) / len(amp_main)]]

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
    print(len(parameter))
    with Pool(processes=cpu_count()) as p:
        data = p.starmap(fourier_transform, parameter, concentration_samples)
    data = np.array(data)
    frequency = data[:,0]
    amplitude = data[:,1]
    print(frequency)
    output = pd.DataFrame(frequency, columns=["f", "[A]", "k_5", "frequency"])
    output2 = pd.DataFrame(amplitude, columns=["f", "[A]", "k_5", "amplitude"])
    with pd.ExcelWriter('FFT.xlsx') as writer:
        sheetname = "FreqAmp"
        output.to_excel(writer, sheet_name=sheetname, index=False)
        output2.to_excel(writer, sheet_name=sheetname, index=False, startrow = 0, startcol = 5)
    toc = time.perf_counter()
    print(toc-tic)