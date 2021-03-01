import numpy as np
from scipy.integrate import odeint
from scipy import fftpack
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from statistics import median
from sklearn.linear_model import LinearRegression
import time
tic = time.perf_counter()

def trajectory(var, t, alpha, gamma):
    x, y = var
    dxdt = 1 - x * y ** gamma
    dydt = alpha * y * (x * y ** (gamma - 1) - 1)
    return [dxdt, dydt]


initial = np.random.rand(20, 2) * np.array([2.01, 2.01])
t_max = 100
dt = 0.001
t = np.linspace(0, t_max, int(t_max/dt))
up = 0.01

amplitude = []
frequency = []
for alpha in np.linspace(0, 5, 15):
    for gamma in np.linspace(0, 5, 15):
        if alpha > (1 / (gamma - 1)) and gamma > 1:
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
            frequency.append([alpha, gamma, sum(freq_main)/len(freq_main)])
            amplitude.append([alpha, gamma, sum(amp_main) / len(amp_main)])

toc = time.perf_counter()
print(toc-tic)

print(frequency)
print(amplitude)
print(len(frequency))
print(len(amplitude))
