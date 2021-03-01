import plotly.graph_objects as go
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

excel_file = 'Limit_cycle_1.xlsx'
df = pd.read_excel(excel_file, sheet_name='Poincare', dtpye=float, engine="openpyxl")
alpha =  df['alpha'].tolist()
gamma = df['gamma'].tolist()
stability = df['stability'].tolist()

alpha = np.array(alpha)
gamma = np.array(gamma)
stability = np.array(stability)
print(len(stability))
l = 3000

tri = Triangulation(gamma, stability)
mask = []
i =0
while i < len(tri.triangles):
    if abs(max(tri.triangles[i])-min(tri.triangles[i]))>10000000:
        mask.append(False)
    else:
        mask.append(True)
    i=i+1
mask = np.array(mask)
#print(tri.triangles[mask])

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlabel('alpha')
ax.set_ylabel('gamma')
ax.set_zlabel('Stability')
p = ax.scatter(alpha[l:], gamma[l:], stability[l:], c=stability[l:], cmap=plt.cm.jet, linewidth=0.2)
#ax.plot_trisurf(alpha, gamma, stability, triangles = tri.triangles[mask], cmap=plt.cm.jet, linewidth=0.2)
cbar = fig.colorbar(p)
cbar.ax.set_ylabel('Stability')
plt.show()