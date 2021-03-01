import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
excel_file = 'Hopf.xlsx'
df = pd.read_excel(excel_file, sheet_name='bifurcation', dtpye=float, engine="openpyxl")
f = df['f'].tolist()
A = df['[A]'].tolist()
k_5 = df['k_5'].tolist()
result = list(zip(f, A, k_5))
result_tot = np.array(result)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlabel('f')
ax.set_ylabel('[A]')
ax.set_zlabel('k_5')
ax.plot_trisurf(result_tot[:,0], result_tot[:,1], result_tot[:,2], cmap=plt.cm.jet, linewidth=0.2)
plt.show()

fig2, ax2 = plt.subplots()
division = 20
ax2.tricontour(result_tot[:,0], result_tot[:,1], result_tot[:,2], levels=division, linewidths = 0.8, linestyles = "dotted", colors="k")
ax2.set_xlabel('f')
ax2.set_ylabel('[A]')
cbar = fig2.colorbar(ax2.tricontourf(result_tot[:,0], result_tot[:,1], result_tot[:,2], levels=division, cmap=plt.cm.jet), ax=ax2)
cbar.ax.set_ylabel('k_5')
plt.show()

