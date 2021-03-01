import plotly.graph_objects as go
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

excel_file = 'Infinite_cycle.xlsx'
df = pd.read_excel(excel_file, sheet_name='Estimate', dtpye=float, engine="openpyxl")
alpha =  df['alpha'].tolist()
gamma = df['gamma'].tolist()
oscillations = df['oscillations'].tolist()

alpha = np.array(alpha)
gamma = np.array(gamma)
oscillations = np.array(oscillations)

oscillations_1 = oscillations[np.where(oscillations == 1)]
alpha_1 = alpha[np.where(oscillations == 1)]
gamma_1 = gamma[np.where(oscillations == 1)]
print(len(oscillations_1))

oscillations_0 = oscillations[np.where(oscillations == 0)]
alpha_0 = alpha[np.where(oscillations == 0)]
gamma_0 = gamma[np.where(oscillations == 0)]

data_1 = list(zip(alpha_1, gamma_1))
data_1_fill=[]
one_prev = data_1[0][1]
i=0
row = []
while i < len(data_1):
    if data_1[i][1]==one_prev:
        row.append(data_1[i])
    else:
        row = np.array(row)
        data_1_fill.append([row[np.argmin(row[:,0])], row[np.argmax(row[:,0])]])
        one_prev = data_1[i][1]
        row = []
        row.append(data_1[i])
    if i == len(data_1)-1:
        row = np.array(row)
        data_1_fill.append([row[np.argmin(row[:, 0])], row[np.argmax(row[:, 0])]])
    i=i+1
data_1_fill = np.array(data_1_fill)

data_0 = list(zip(alpha_0, gamma_0))
data_0_fill=[]
one_prev = data_0[0][1]
i=0
row = []
while i < len(data_0):
    if data_0[i][1]==one_prev:
        row.append(data_0[i])
    else:
        row = np.array(row)
        data_0_fill.append([row[np.argmin(row[:,0])], row[np.argmax(row[:,0])]])
        one_prev = data_0[i][1]
        row = []
        row.append(data_0[i])
    if i == len(data_0)-1:
        row = np.array(row)
        data_0_fill.append([row[np.argmin(row[:, 0])], row[np.argmax(row[:, 0])]])
    i=i+1
data_0_fill = np.array(data_0_fill)

"""plt.scatter(data_1_fill[:,0][:,0], data_1_fill[:,0][:,1])
plt.scatter(data_1_fill[:,1][:,0], data_1_fill[:,1][:,1])
plt.scatter(data_0_fill[:,0][:,0], data_0_fill[:,0][:,1])
plt.scatter(data_0_fill[:,1][:,0], data_0_fill[:,1][:,1])
plt.show()"""

gamma_plot= np.linspace(1,5,10000000)
alpha_function = lambda x: 1/(x-1)
alpha_plot = alpha_function(gamma_plot)
alpha_plot = alpha_plot[np.where(alpha_plot<40)]
gamma_plot = gamma_plot[np.where(alpha_plot<40)]

#plt.scatter(alpha_1, gamma_1, label="Finite limit cycles")
#plt.scatter(alpha_0, gamma_0, label = "Infinite limit cycles")
plt.fill(np.concatenate([data_1_fill[:,0][:,0], data_1_fill[:,1][:,0][::-1]]),
         np.concatenate([data_1_fill[:,0][:,1], data_1_fill[:,1][:,1][::-1]]), label="Finite limit cycles")
plt.fill(np.concatenate([data_0_fill[:,0][:,0], data_0_fill[:,1][:,0][::-1]]),
         np.concatenate([data_0_fill[:,0][:,1], data_0_fill[:,1][:,1][::-1]]), label = "Infinite limit cycles")
plt.plot(alpha_plot, gamma_plot, label="Hopf bifurcation", color="k")
plt.xlabel('alpha')
plt.ylabel('gamma')
plt.legend(loc=1)
plt.show()