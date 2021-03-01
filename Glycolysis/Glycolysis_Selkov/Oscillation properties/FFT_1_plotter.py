import plotly.graph_objects as go
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

excel_file = 'FFT_1.xlsx'
df = pd.read_excel(excel_file, sheet_name='FreqAmp', dtpye=float, engine="openpyxl")
alpha =  df['alpha'].tolist()
gamma = df['gamma'].tolist()
frequency = df['frequency'].tolist()
amplitude = df['amplitude'].tolist()

alpha = np.array(alpha)
gamma = np.array(gamma)
frequency = np.array(frequency)
amplitude = np.array(amplitude)/1000

print(len(frequency))
l= 2500
#plt.scatter(alpha, gamma)
#plt.show()
tri = Triangulation(gamma, frequency)
mask = []
i =0
while i < len(tri.triangles):
    if abs(max(tri.triangles[i])-min(tri.triangles[i]))>860:
        mask.append(False)
    else:
        mask.append(True)
    i=i+1
mask = np.array(mask)
print(tri.triangles[mask])

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlabel('alpha')
ax.set_ylabel('gamma')
ax.set_zlabel('Frequency')
p = ax.scatter(alpha[l:], gamma[l:], frequency[l:], c=frequency[l:], cmap=plt.cm.jet, linewidth=0.2)
#ax.plot_trisurf(alpha, gamma, frequency, triangles = tri.triangles[mask], cmap=plt.cm.jet, linewidth=0.2)
cbar = fig.colorbar(p)
cbar.ax.set_ylabel('Frequency')
plt.show()

"""fig2, ax2 = plt.subplots()
division = 20
ax2.tricontour(alpha, gamma, frequency, levels=division, linewidths = 0.8, linestyles = "dotted", colors="k")
ax2.set_xlabel('alpha')
ax2.set_ylabel('gamma')
cbar = fig2.colorbar(ax2.tricontourf(alpha, gamma, frequency, levels=division, cmap=plt.cm.jet), ax=ax2)
cbar.ax.set_ylabel('Frequency')
plt.show()"""

"""nan_list = np.logical_not(np.isnan(amplitude))
alpha = alpha[nan_list]
gamma = gamma[nan_list]
amplitude = amplitude[nan_list]"""

tri2 = Triangulation(amplitude, gamma)
mask = []
i =0
while i < len(tri2.triangles):
    if abs(max(tri2.triangles[i])-min(tri2.triangles[i]))>8000:
        mask.append(False)
    else:
        mask.append(True)
    i=i+1
mask = np.array(mask)
print(tri2.triangles[mask])

fig3 = plt.figure()
ax3 = fig3.gca(projection='3d')
ax3.set_xlabel('alpha')
ax3.set_ylabel('gamma')
ax3.set_zlabel('Amplitude')
p2 = ax3.scatter(alpha[l:], gamma[l:], amplitude[l:], c=amplitude[l:], cmap=plt.cm.jet, linewidth=0.2)
cbar2 = fig3.colorbar(p2)
cbar2.ax.set_ylabel('Amplitude')
#ax3.plot_trisurf(alpha, gamma, amplitude, triangles = tri2.triangles[mask], cmap=plt.cm.jet, linewidth=0.2) #[mask]
plt.show()

"""fig4, ax4 = plt.subplots()
division = 20
ax4.tricontour(alpha, gamma, amplitude, levels=division, linewidths = 0.8, linestyles = "dotted", colors="k")
ax4.set_xlabel('alpha')
ax4.set_ylabel('gamma')
cbar = fig2.colorbar(ax2.tricontourf(alpha, gamma, amplitude, levels=division, cmap=plt.cm.jet), ax=ax2)
cbar.ax.set_ylabel('Amplitude')
plt.show()"""