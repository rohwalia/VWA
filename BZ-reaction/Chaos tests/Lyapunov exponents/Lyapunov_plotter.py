import plotly.graph_objects as go
import pandas as pd
import numpy as np
from scipy import interpolate
excel_file = 'Lyapunov.xlsx'
df = pd.read_excel(excel_file, sheet_name='Exponents', dtpye=float, engine="openpyxl")
f =  df['f'].tolist()
A = df['[A]'].tolist()
k_5 = df['k_5'].tolist()
L_1 = df['L_1'].tolist()
good_data = list(zip(f, A, k_5, L_1))
concentration_samples = 14
data_set = []
for A_val in np.linspace(0.01, 1.2, concentration_samples):
        for k_5_val in np.linspace(1, 20, 14):
                for f_val in np.linspace(0, 2.5, 14):
                        ok = 0
                        for j in good_data:
                                if abs(f_val-j[0])<0.01 and abs(k_5_val-j[2])<0.01 and abs(A_val-j[1])<0.01:
                                        data_set.append([f_val, A_val, k_5_val, j[3]])
                                        ok = 1
                                        break
                        if ok ==0:
                                data_set.append([f_val, A_val, k_5_val, min(L_1)-1])
data_set = np.array(data_set)
fig = go.Figure(data=go.Volume(
        x=data_set[:, 0],
        y=data_set[:, 1],
        z=data_set[:, 2],
        value=data_set[:,3],
        isomin= min(L_1),
        isomax= max(L_1),
        opacity=0.1,
        surface = dict(count = 40),
    ))
fig.update_layout(
    scene = dict(
        xaxis = dict(range=[0,1.5],),
        xaxis_title='f',
        yaxis_title='[A]',
        zaxis_title='k_5',
        yaxis = dict(range=[0,1.2],),
        zaxis = dict(range=[0,11],),))
fig.show()