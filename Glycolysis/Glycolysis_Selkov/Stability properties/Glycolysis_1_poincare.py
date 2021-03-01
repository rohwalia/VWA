import numpy as np
from scipy.integrate import ode
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from statistics import median
from sklearn.linear_model import LinearRegression
import time


def trajectory(t, var, alpha, gamma):
    x, y = var
    dxdt = 1 - x * y ** gamma
    dydt = alpha * y * (x * y ** (gamma - 1) - 1)
    return [dxdt, dydt]


initial =[[1.01,1.01]] # np.random.rand(5, 2)*np.array([2.01,2.01])
t_max = 100
dt = 0.001
tol = 0.001
min = 10

stability = []
for alpha in np.linspace(0, 5, 10):
    for gamma in np.linspace(1, 2, 10):
        if alpha > 1/(gamma-1) and gamma > 1:
            stab = []
            for ex in initial:
                sect = []
                r = ode(trajectory)
                r.set_integrator('vode', method='bdf')
                r.set_initial_value(ex, 0)
                r.set_f_params(alpha, gamma)
                while r.successful() and r.t < t_max:
                    r.integrate(r.t + dt)
                    x_cor = r.y[0]
                    y_cor = r.y[1]
                    if abs((x_cor - y_cor) / np.sqrt(2)) < tol:
                        sect.append(np.sqrt(x_cor ** 2 + y_cor ** 2 ))
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
                        stab.append(model.coef_[0])
            if len(stab)>0:
                stability.append([alpha, gamma, sum(stab)/len(stab)])
        print("ok")

print(stability)
stability = np.array(stability)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlabel('alpha')
ax.set_ylabel('gamma')
ax.set_zlabel('Stability')
ax.plot_trisurf(stability[:,0], stability[:,1], stability[:,2], cmap=plt.cm.jet, linewidth=0.2)
plt.show()


