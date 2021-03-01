import numpy as np
from scipy.integrate import odeint
import pandas as pd
from scipy import fftpack
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

k_1 = 2
k_2 = 10**6
k_3 = 10
k_4 = 2000
k_5= 1.077
f= 0.566
A_list = []
excel_file2 = 'Bifurcation results.xlsx'
df2 = pd.read_excel(excel_file2, sheet_name='Experiment', dtpye=float, engine="openpyxl")
mass = df2['Mass'].tolist()
oscillation = df2['Oscillation'].tolist()
oscillation = np.array(oscillation)
for i in mass:
    c_trans = (i/150.89)/(69/1000)
    c_end = c_trans*(6/8.5)
    A_list.append(c_end)
A_list = np.array(A_list)
A_list = A_list[np.where(oscillation==1)]
print(A_list)
A = A_list[0]
print(A)

def trajectory(var, t):
    x, y, z = var
    kappa = 2 * k_4 * k_5 / (A * k_2 * k_3)
    epsilon = k_5 / (k_3 * A)
    phi = 2 * k_4 * k_1 / (k_2 * k_3)
    dxdt = (phi*y-x*y+x-x**2)/epsilon
    dydt = (-phi*y-x*y+2*f*z)/kappa
    dzdt = x-z
    return [dxdt, dydt, dzdt]

t= np.linspace(0,600,300000)
up = 0
dt = 600/300000
initial = [0, 3, 0.01]
r= odeint(trajectory, initial, t)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(r[:,0], r[:,1], r[:,2])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()

plt.plot(t, r[:,2])
plt.show()

ferroin = np.array(r[:,2])
fft = fftpack.fft(ferroin)
freqs = fftpack.fftfreq(ferroin.size, d=dt)
power = np.abs(fft)[np.where(freqs > up)]
freqs = freqs[np.where(freqs > up)]
plt.plot(freqs, power)
plt.show()
print(freqs[power.argmax()])


