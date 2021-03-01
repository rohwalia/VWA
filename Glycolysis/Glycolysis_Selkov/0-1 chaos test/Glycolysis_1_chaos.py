import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from statistics import median
from sklearn.linear_model import LinearRegression
import time
from multiprocessing import Pool

def trajectory(var, t, alpha, gamma):
    x, y = var
    dxdt = 1 - x * y ** gamma
    dydt = alpha * y * (x * y ** (gamma - 1) - 1)
    return [dxdt, dydt]

factor = np.random.rand(100)*2*np.pi
t_max = 100
dt = 0.1
t = np.linspace(0, t_max, int(t_max/dt))
initial = [1.01, 1.01]

chaos = []

for alpha in np.linspace(0, 5, 5):
    for gamma in np.linspace(0, 5, 5):
        if alpha > 1 / (gamma - 1):
            c_val = []
            for c in factor:
                r = odeint(trajectory, initial, t, (alpha, gamma))
                p_c=[]
                q_c=[]
                p_now = 0
                q_now = 0
                for i in range(len(r)):
                    amp = np.sqrt(r[i][0] ** 2 + r[i][1] ** 2)
                    p_now = p_now + amp * np.cos((i+1)*c)
                    q_now = q_now + amp * np.sin((i+1)*c)
                    p_c.append(p_now)
                    q_c.append(q_now)
                plt.plot(p_c, q_c)
                plt.show()
                M = []
                for i in range(round(len(p_c)/10)):
                    M_value = 0
                    for el in range(len(p_c)-round(len(p_c)/10)):
                        M_value = M_value + ((p_c[i+el]-p_c[el])**2 + (q_c[i+el]-q_c[el])**2)*dt**2
                    M.append(M_value/(len(p_c)-round(len(p_c)/10)))
                M = np.array(M)
                """plt.plot(np.log(np.array(range(len(M)))[1:]), np.log(M[1:]))
                plt.show()"""
                model = LinearRegression().fit(np.log(np.array(range(len(M)))[1:]).reshape(-1, 1), np.log(M[1:]))
                c_val.append(model.coef_[0])
            chaos.append([alpha, gamma, median(c_val)])
print(chaos)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlabel('alpha')
ax.set_ylabel('gamma')
ax.set_zlabel('Stability')
ax.plot_trisurf(chaos[:,0], chaos[:,1], chaos[:,2], cmap=plt.cm.jet, linewidth=0.2)
plt.show()
