import numpy as np
from scipy.integrate import ode
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from statistics import median
from sklearn.linear_model import LinearRegression
import time

k_1 = 2
k_2 = 10**6
k_3 = 10
k_4 = 2000

def trajectory(t, var, k_5, A, f):
    x, y, z = var
    kappa = 2 * k_4 * k_5 / (A * k_2 * k_3)
    epsilon = k_5 / (k_3 * A)
    phi = 2 * k_4 * k_1 / (k_2 * k_3)
    dxdt = (phi*y-x*y+x-x**2)/epsilon
    dydt = (-phi*y-x*y+2*f*z)/kappa
    dzdt = x-z
    return [dxdt, dydt, dzdt]


initial = np.random.rand(50, 3)*np.array([1,50,1]) #[[0.3,40,0.2]]
t_max= 100
dt = 0.001
concentration_samples = 5
tol = 0.001
min = 10
tic = time.perf_counter()

k_5= 1
A= 1
f= 0.5
stability = []
"""for A in np.linspace(0.01, 1.2, concentration_samples):
    for k_5 in np.linspace(1, 20, 5):
        for f in np.linspace(0, 2.5, 5):"""
stab = []
for ex in initial:
    sect = []
    r = ode(trajectory)
    r.set_integrator('vode', method='bdf')
    r.set_initial_value(ex, 0)
    r.set_f_params(k_5, A, f)
    while r.successful() and r.t < t_max:
        r.integrate(r.t + dt)
        x_cor = r.y[0]
        y_cor = r.y[1]
        z_cor = r.y[2]
        if abs((x_cor - y_cor) / np.sqrt(2)) < tol:
            sect.append(np.sqrt(x_cor ** 2 + y_cor ** 2 + z_cor ** 2))
    #print("Intersections found")
    sect_filt = []
    if len(sect)>2*min:
        for el in sect:
            count = 1
            temp = [el]
            while count < 3:
                try:
                    if abs(el - sect[sect.index(el) + count]) < 10*tol:
                        temp.append(sect[sect.index(el) + count])
                        sect.remove(sect[sect.index(el) + count])
                        count = count + 1
                    else:
                        break
                except IndexError:
                    count = count + 1
            sect_filt.append(median(temp))
        #print("Intersections filtered")
        average = sum(sect_filt)/len(sect_filt)
        sect_filt_up = np.delete(sect_filt, np.where(sect_filt < average))
        points = list(zip(sect_filt_up[:-1], sect_filt_up[1:]))
        i = 0
        while i < len(points):
            if abs(np.diff(points[i])/ np.sqrt(2)) > tol/10:
                points.remove(points[i])
            else:
                i = i + 1
        #print("Omitting stray points")
        if len(points)>min:
            points = np.array(points)
            model = LinearRegression().fit(points[:,0].reshape(-1, 1), points[:,1])
            intersection = plt.scatter(points[:, 0], points[:, 1], label="Intersection points")
            stable_point = plt.scatter(model.intercept_/(1-model.coef_[0]), model.intercept_/(1-model.coef_[0]), marker="x", color="red", label="Fixed point")
            first_median = plt.plot([0, 1], [0, 1], "k", linestyle = "--", label="y=x")
            fit = plt.plot([0, 1], [model.intercept_, model.coef_[0]+model.intercept_], "orange", label = "Linear regression")
            plt.legend()
            plt.show()
            stab.append(model.coef_[0])
if len(stab)>0:
    stability.append([f, A, k_5, sum(stab)/len(stab)])
print("ok")

print(stability)
toc = time.perf_counter()
print(toc-tic)

"""fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlabel('f')
ax.set_ylabel('[A]')
ax.set_zlabel('k_5')
ax.plot_trisurf(stability[:,0], stability[:,1], stability[:,2], cmap=plt.cm.jet, linewidth=0.2)
plt.show()"""


