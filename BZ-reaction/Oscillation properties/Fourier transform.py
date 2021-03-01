import numpy as np
from scipy.integrate import odeint
from scipy import fftpack
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from statistics import median
from sklearn.linear_model import LinearRegression
import time

tic = time.perf_counter()

k_1 = 2
k_2 = 10 ** 6
k_3 = 10
k_4 = 2000


def trajectory(var, t, k_5, A, f):
    x, y, z = var
    kappa = 2 * k_4 * k_5 / (A * k_2 * k_3)
    epsilon = k_5 / (k_3 * A)
    phi = 2 * k_4 * k_1 / (k_2 * k_3)
    dxdt = (phi * y - x * y + x - x ** 2) / epsilon
    dydt = (-phi * y - x * y + 2 * f * z) / kappa
    dzdt = x - z
    return [dxdt, dydt, dzdt]


initial = np.random.rand(20, 3) * np.array([1, 50, 1])  # [[0.3,40,0.2]]
t_max = 100
dt = 0.001
t = np.linspace(0, t_max, int(t_max/dt))
k_5 = 1
A = 1
f = 0.5
up = 0.01

amplitude = []
frequency = []
concentration_samples = 1
for A in np.linspace(0, 1.2, concentration_samples):
    for k_5 in np.linspace(0.01, 20, 1):
        for f in np.linspace(0, 2.5, 1):
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
            frequency.append([f, A, k_5, sum(freq_main)/len(freq_main)])
            amplitude.append([f, A, k_5, sum(amp_main) / len(amp_main)])

print(frequency)
toc = time.perf_counter()
print(toc-tic)