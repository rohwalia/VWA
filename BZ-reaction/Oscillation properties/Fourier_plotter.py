import plotly.graph_objects as go
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

excel_file = 'FFT.xlsx'
df = pd.read_excel(excel_file, sheet_name='FreqAmp', dtpye=float, engine="openpyxl")
f =  df['f'].tolist()
A = df['[A]'].tolist()
k_5 = df['k_5'].tolist()
frequency = df['frequency'].tolist()
amplitude = df['amplitude'].tolist()
good_data = list(zip(f, A, k_5, frequency, amplitude))
concentration_samples = 50
data_set = []
for A_val in np.linspace(0.01, 1.2, concentration_samples):
        for k_5_val in np.linspace(1, 20, 50):
                for f_val in np.linspace(0, 2.5, 50):
                        ok = 0
                        for j in good_data:
                                if abs(f_val-j[0])<0.01 and abs(k_5_val-j[2])<0.01 and abs(A_val-j[1])<0.01:
                                        data_set.append([f_val, A_val, k_5_val, j[3], j[4]])
                                        ok = 1
                                        break
                        if ok ==0:
                            data_set.append([f_val, A_val, k_5_val, min(frequency)-1, min(amplitude)-1])
data_set = np.array(data_set)
"""fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlabel('f')
ax.set_ylabel('[A]')
ax.set_zlabel('k_5')
ax.scatter(data_set[:,0], data_set[:,1], data_set[:,2])
plt.show()"""
fig = go.Figure(data=go.Volume(
        x=data_set[:,0],
        y=data_set[:,1],
        z=data_set[:,2],
        value=data_set[:,3],
        isomin= min(frequency),
        isomax= max(frequency),
        opacity=0.1,
        surface = dict(count = 80),
    ))
fig.update_layout(
    title='Frequency (Hz)',
    scene = dict(
        xaxis = dict(range=[0.2,1.5],),
        xaxis_title='f',
        yaxis_title='[A]',
        zaxis_title='k_5',
        yaxis = dict(range=[0,1.2],),
        zaxis = dict(range=[0,12],),))
fig.show()
fig2 = go.Figure(data=go.Volume(
    x=data_set[:, 0],
    y=data_set[:, 1],
    z=data_set[:, 2],
    value=data_set[:,4],
    isomin= min(amplitude),
    isomax= max(amplitude),
    opacity=0.1,
    surface = dict(count = 80),
))
fig2.update_layout(
    title='Amplitude',
    scene = dict(
        xaxis = dict(range=[0.2,1.5],),
        xaxis_title='f',
        yaxis_title='[A]',
        zaxis_title='k_5',
        yaxis = dict(range=[0,1.2],),
        zaxis = dict(range=[0,12],),))
fig2.show()