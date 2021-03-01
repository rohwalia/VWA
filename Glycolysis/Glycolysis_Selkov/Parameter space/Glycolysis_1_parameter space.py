import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patch
import seaborn as sns
import cmath
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

def eigenvalue_calc(alpha, gamma):
    global eigenvalues
    lambda_1 = 0.5* (alpha* (gamma - 1) - 1 - cmath.sqrt((alpha* (gamma - 1) - 1)**2 - 4 *alpha))
    lambda_2 = 0.5* (alpha* (gamma - 1) - 1 + cmath.sqrt((alpha* (gamma - 1) - 1)**2 - 4 *alpha))
    eigenvalues = [lambda_1, lambda_2]

max_param = 5
previous = 0
result_tot=[]
tol = 0.001
focus_tol = 0.0001
for alpha in np.linspace(0, max_param, 1500):
    focus = 1
    for gamma in np.linspace(0, max_param, 1500):
        eigenvalue_calc(alpha, gamma)
        for i in eigenvalues:
            if alpha > tol:
                if abs(i.imag)<focus_tol and abs(i.real)>focus_tol:
                    focus_new = 1
                else:
                    focus_new = 0
                if focus_new!= focus:
                    if i.real>0:
                        result_tot.append([alpha, gamma, "Focus-positiv"])
                    else:
                        result_tot.append([alpha, gamma, "Focus-negativ"])
                focus = focus_new

            if abs(i.imag)<tol and abs(i.real)<tol and previous==0:
                result_tot.append([alpha, gamma, "Zero-eigenvalue"])
                previous = 1
                break
            if abs(i.imag)>tol and abs(i.real)<tol and previous==0:
                result_tot.append([alpha, gamma, "Hopf"])
                previous = 1
                break
            else:
                previous = 0


result_tot = np.array(result_tot)

result_zero = np.array([l[:2].astype(float) for l in result_tot[np.where(result_tot[:,2] == "Zero-eigenvalue")]])
result_hopf = np.array([l[:2].astype(float) for l in result_tot[np.where(result_tot[:,2] == "Hopf")]])
result_positiv = np.array([l[:2].astype(float).tolist() for l in result_tot[np.where(result_tot[:,2] == "Focus-positiv")]])
result_negativ = np.array([l[:2].astype(float).tolist() for l in result_tot[np.where(result_tot[:,2] == "Focus-negativ")]])

min_value = min([max(result_zero[:,1]), max(result_hopf[:,1]), max(result_positiv[:,1]), max(result_negativ[:,1])])
zero = np.array(result_zero[np.where(result_zero[:,1] <= min_value)])
hopf = np.array(result_hopf[np.where(result_hopf[:,1] <= min_value)])
positiv = np.array(result_positiv[np.where(result_positiv[:,1] <= min_value)])
negativ = np.array(result_negativ[np.where(result_negativ[:,1] <= min_value)])

zero_plot_x = np.concatenate([zero[:,0], negativ[:,0]], axis = None)
zero_plot_y = np.concatenate([zero[:,1], negativ[:,1]], axis = None)
negativ_plot_x = np.concatenate([negativ[:,0], hopf[:,0][::-1]], axis = None)
negativ_plot_y = np.concatenate([negativ[:,1], hopf[:,1][::-1]], axis = None)
hopf_plot_x = np.concatenate([hopf[:,0], positiv[:,0][::-1]], axis = None)
hopf_plot_y = np.concatenate([hopf[:,1], positiv[:,1][::-1]], axis = None)
positiv_plot_x = np.concatenate([positiv[:,0], [max_param]], axis = None)
positiv_plot_y = np.concatenate([positiv[:,1], [min_value]], axis = None)

"""plt.plot(result_zero[:,0].astype(float), result_zero[:,1].astype(float), label="Zero-eigenvalue")
plt.plot(result_hopf[:,0].astype(float), result_hopf[:,1].astype(float), label="Hopf")
plt.plot(result_positiv[:,0].astype(float), result_positiv[:,1].astype(float), label="Focus-positiv", linestyle= "--")
plt.plot(result_negativ[:,0].astype(float), result_negativ[:,1].astype(float), label="Focus-negativ", linestyle= "--")
plt.xlabel('alpha')
plt.ylabel('gamma')
plt.legend()
plt.show()"""

plt.plot(hopf[:,0], hopf[:,1], label="Hopf bifurcation", color = "black")
plt.fill_between(zero_plot_x, zero_plot_y, label="Stable node", facecolor = "orchid", edgecolor = "purple")
plt.fill(negativ_plot_x, negativ_plot_y, label="Stable focus", facecolor = "mediumseagreen", edgecolor = "seagreen")
plt.fill(hopf_plot_x, hopf_plot_y, label="Limit cycle/unstable focus", facecolor = "skyblue", edgecolor = "dodgerblue")
plt.fill(positiv_plot_x, positiv_plot_y, label="Limit cycle/unstable node", facecolor="lightcoral", edgecolor="red")
plt.xlabel('alpha')
plt.ylabel('gamma')
plt.legend(loc=1)
plt.show()