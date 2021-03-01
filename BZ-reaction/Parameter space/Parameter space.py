import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math as m
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

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
result_f=[]
for f in np.linspace(0, 2.5, 100):
    eigenvalue_calc(1, f, 1)
    for i in eigenvalues:
        result_f.append([f, i.real, i.imag])
"""fig1 = plt.figure()
ax = fig1.gca(projection='3d')
ax.set_xlabel('real')
ax.set_ylabel('imag')
ax.set_zlabel('f')
result_f = np.array(result_f)
#print(result_f)
ax.scatter(result_f[:,1], result_f[:,2], result_f[:,0])
plt.show()"""

result_k=[]
for k_5 in np.linspace(0.01, 25, 100):
    eigenvalue_calc(k_5, 0.5, 1)
    for i in eigenvalues:
        result_k.append([k_5, i.real, i.imag])
"""fig1 = plt.figure()
ax = fig1.gca(projection='3d')
ax.set_xlabel('real')
ax.set_ylabel('imag')
ax.set_zlabel('k_5')
result_k = np.array(result_k)
#print(result_k)
ax.scatter(result_k[:,1], result_k[:,2], result_k[:,0])
plt.show()"""

result_A=[]
for A in np.linspace(0, 1.2, 100):
    eigenvalue_calc(1, 0.5, A)
    for i in eigenvalues:
        result_A.append([A, i.real, i.imag])
"""fig1 = plt.figure()
ax = fig1.gca(projection='3d')
ax.set_xlabel('real')
ax.set_ylabel('imag')
ax.set_zlabel('A')
result_A = np.array(result_A)
#print(result_A)
ax.scatter(result_A[:,1], result_A[:,2], result_A[:,0])
plt.show()"""

result_tot=[]
tol = 0.001
tol_A = 0.02
concentration_samples = 100
for A in np.linspace(0, 1.2, concentration_samples):
    for k_5 in np.linspace(0.01, 20, 300):
        for f in np.linspace(0, 2.5, 300):
            eigenvalue_calc(k_5, f, A)
            for i in eigenvalues:
                if abs(i.imag)<tol and abs(i.real)<tol and previous==0 and abs(A)>tol_A:
                    #result_tot.append([f, A, k_5, "Zero-eigenvalue"])
                    previous = 1
                    break
                if i.imag!=0 and abs(i.real)<tol and previous==0 and abs(A)>tol_A:
                    result_tot.append([f, A, k_5, "Hopf"])
                    previous = 1
                    break
                else:
                    previous = 0
result_tot = np.array(result_tot)
output = pd.DataFrame(result_tot, columns=["f", "[A]", "k_5", "type"])
with pd.ExcelWriter('Hopf.xlsx') as writer:
    sheetname = "bifurcation"
    output.to_excel(writer, sheet_name=sheetname, index=False)
#result_split = [result_tot[i:i + concentration_samples] for i in range(0, len(result_tot), concentration_samples)]
print(result_tot)
print(len(result_tot))

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlabel('f')
ax.set_ylabel('[A]')
ax.set_zlabel('k_5')
ax.scatter(result_tot[:,0].astype(float), result_tot[:,1].astype(float), result_tot[:,2].astype(float))
plt.show()

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlabel('f')
ax.set_ylabel('[A]')
ax.set_zlabel('k_5')
ax.plot_trisurf(result_tot[:,0].astype(float), result_tot[:,1].astype(float), result_tot[:,2].astype(float), cmap=plt.cm.jet, linewidth=0.2)
plt.show()
