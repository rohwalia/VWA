import plotly.graph_objects as go
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

excel_file = 'Chaos.xlsx'
df = pd.read_excel(excel_file, sheet_name='0-1 chaos test', dtpye=float, engine="openpyxl")
alpha =  df['alpha'].tolist()
gamma = df['gamma'].tolist()
chaos = df['chaos'].tolist()

alpha = np.array(alpha)
gamma = np.array(gamma)
chaos = np.array(chaos)
print(len(chaos))
l=2000

tri = Triangulation(gamma, chaos)
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
ax.set_zlabel('Growth rate K')
p = ax.scatter(alpha[l:], gamma[l:], chaos[l:], c=chaos[l:], cmap=plt.cm.jet, linewidth=0.2)
#ax.plot_trisurf(alpha, gamma, chaos, triangles = tri.triangles[mask], cmap=plt.cm.jet, linewidth=0.2)
cbar = fig.colorbar(p)
cbar.ax.set_ylabel('Growth rate K')
plt.show()