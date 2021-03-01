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

initial = [[1.01, 1.01]] #np.random.rand(20, 2) * np.array([2.01, 2.01])
t_max = 300
dt = 0.001
t = np.linspace(0, t_max, int(t_max/dt))
up = 0.01

def trajectory(var, t, alpha, gamma):
    x, y = var
    dxdt = 1 - x * y ** gamma
    dydt = alpha * y * (x * y ** (gamma - 1) - 1)
    return [dxdt, dydt]

def fourier_transform(alpha, gamma):
    global initial
    global t
    global up
    global dt
    amp_main = []
    freq_main = []
    for ex in initial:
        r = odeint(trajectory, ex, t, (alpha, gamma))
        fourier = []
        for i in range(len(r)):
            amp = np.sqrt(r[i][0] ** 2 + r[i][1] ** 2)
            fourier.append(amp)
        fourier = np.array(fourier)
        fft = fftpack.fft(fourier)
        freqs = fftpack.fftfreq(fourier.size, d=dt)
        power = np.abs(fft)[np.where(freqs > up)]
        freqs = freqs[np.where(freqs > up)]
        freq_main.append(freqs[power.argmax()])
        amp_main.append(power[power.argmax()])
    return [[alpha, gamma, sum(freq_main) / len(freq_main)], [alpha, gamma, sum(amp_main) / len(amp_main)]]

if __name__ == '__main__':
    tic = time.perf_counter()
    """parameter = []
    div = 50
    for alpha in np.linspace(0, 25, div):
        for gamma in np.linspace(0, 25, div):
            if alpha > 1 / (gamma - 1) and gamma > 1:
                parameter.append([alpha, gamma])
    print(len(parameter))"""
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
        data = p.starmap(fourier_transform, data_1_fill, 100)
    data = np.array(data)
    frequency = data[:,0]
    amplitude = data[:,1]
    print(frequency)
    output = pd.DataFrame(frequency, columns=["alpha", "gamma", "frequency"])
    output2 = pd.DataFrame(amplitude, columns=["alpha", "gamma", "amplitude"])
    with pd.ExcelWriter('FFT_1.xlsx') as writer:
        sheetname = "FreqAmp"
        output.to_excel(writer, sheet_name=sheetname, index=False)
        output2.to_excel(writer, sheet_name=sheetname, index=False, startrow = 0, startcol = 4)
    toc = time.perf_counter()
    print(toc-tic)