import plotly.graph_objects as go
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

excel_file = 'Hopf.xlsx'
df = pd.read_excel(excel_file, sheet_name='bifurcation', dtpye=float, engine="openpyxl")
f_bif =  df['f'].tolist()
A_bif = df['[A]'].tolist()
k_5_bif = df['k_5'].tolist()
f_bif = np.array(f_bif)
A_bif = np.array(A_bif)
k_5_bif = np.array(k_5_bif)
bifurcation = list(zip(f_bif, A_bif, k_5_bif))
bifurcation = np.array(bifurcation)[np.where(A_bif<0.2)]

excel_file2 = 'Bifurcation results.xlsx'
df2 = pd.read_excel(excel_file2, sheet_name='Experiment', dtpye=float, engine="openpyxl")
mass = df2['Mass'].tolist()
concentration = []
for i in mass:
    c_trans = (i/150.89)/(69/1000)
    c_end = c_trans*(6/8.5)
    concentration.append(c_end)
oscillation = df2['Oscillation'].tolist()
concentration = np.array(concentration)
oscillation = np.array(oscillation)

print(concentration)
print(oscillation)

min_border = np.max(concentration[np.where(oscillation==0)])
max_border = np.min(concentration[np.where(oscillation==1)])
bifurcation_point = (min_border+max_border)/2
print(bifurcation_point)

k_1 = 2
k_2 = 10**6
k_3 = 10
k_4 = 2000

def eigenvalue_calc(k_5, f, A):
    global eigenvalues
    kappa = 2*k_4*k_5/(A*k_2*k_3)
    epsilon = k_5/(k_3*A)
    phi = 2*k_4*k_1/(k_2*k_3)
    x_star = 0.5*(np.sqrt(4*f**2 + 4*f*(3*phi-1) + (phi+1)**2) - 2*f - phi + 1)
    y_star = 0.25*(-np.sqrt(4*f**2 + 4*f*(3*phi-1) + (phi+1)**2) + 6*f + phi + 1)
    a = ((2*x_star+y_star-1)/epsilon) + ((phi+x_star)/kappa)
    b = (2*phi*(x_star+y_star)-phi+2*x_star**2-x_star)/(epsilon*kappa)
    coeff_3 = 1
    coeff_2 = a+1
    coeff_1 = b+a
    coeff_0 = b-(2*f*(phi-x_star)/(epsilon*kappa))
    coeff = [coeff_3, coeff_2, coeff_1, coeff_0]
    eigenvalues = np.roots(coeff)

previous = 0
result_tot=[]
tol = 0.001
tol_A = 0.02
concentration_samples = 500
for A in np.linspace(0.1, 0.2, concentration_samples):
    for k_5 in np.linspace(0.2, 3, 100):
        for f in np.linspace(0, 1, 100):
            eigenvalue_calc(k_5, f, A)
            for i in eigenvalues:
                if i.imag!=0 and abs(i.real)<tol and previous==0 and abs(A)>tol_A:
                    result_tot.append([f, A, k_5])
                    previous = 1
                    break
                else:
                    previous = 0
result_tot = np.array(result_tot)

bifurc_sel_pre = result_tot[np.where(abs(result_tot[:,1]-bifurcation_point)<0.0002)]
bifurc_sel = bifurc_sel_pre[np.where(bifurc_sel_pre[:,1]>bifurcation_point)]
print(bifurc_sel)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlabel('f')
ax.set_ylabel('[A]')
ax.set_zlabel('k_5')
bif = ax.plot_trisurf(bifurcation[:,0], bifurcation[:,1],bifurcation[:,2], linewidth=0.2, alpha=0.3, label="Bifurcation plane")
bif._facecolors2d = bif._facecolors3d
bif._edgecolors2d = bif._edgecolors3d

ax.scatter(bifurc_sel[:,0][-2], bifurc_sel[:,1][-2], bifurc_sel[:,2][-2], marker="x", s=50, color="r", label="Selected bifurcation point")
ax.scatter(np.concatenate([bifurc_sel[:,0][:-2], bifurc_sel[:,0][-1:]]), np.concatenate([bifurc_sel[:,1][:-2], bifurc_sel[:,1][-1:]])
           ,np.concatenate([bifurc_sel[:,2][:-2], bifurc_sel[:,2][-1:]]), marker="x", s=50, color="k", label="Potential bifurcation points")

yes = concentration[np.where(concentration>bifurcation_point)][2:]
no = concentration[np.where(concentration<bifurcation_point)]
ax.scatter( [bifurc_sel[:,0][-2]]*len(yes), yes, [bifurc_sel[:,2][-2]]*len(yes), marker = "o", s=50, color="lightcoral", label="Oscillatory")
ax.scatter([bifurc_sel[:,0][-2]]*len(no), no, [bifurc_sel[:,2][-2]]*len(no), marker = "o", s=50, color="mediumseagreen", label="Non-oscillatory")
ax.scatter(bifurc_sel[:,0][-2], bifurcation_point, bifurc_sel[:,2][-2], marker = "o", s=50, color="orchid", label="Approximate bifurcation point")
ax.legend(loc=1)
plt.show()
