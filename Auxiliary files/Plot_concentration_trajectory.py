import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
k_1=0.5
k_2=0.05
k_3=0.4
k_4=0.3
def model(r,t):
    a=r[0]
    b=r[1]
    c=r[2]
    dadt=-(k_1*a-k_2*b)
    dbdt=(k_1*a-k_2*b)-(k_3*b-k_4*c)
    dcdt=(k_3*b-k_4*c)
    return[dadt, dbdt, dcdt]
z0=[10,0,0]
t= np.linspace(0,10,100)
r= odeint(model, z0, t)
a = r[:,0]
b = r[:,1]
c= r[:,2]

plt.plot(t,a, label="[A]")
plt.plot(t,b, label="[B]")
plt.plot(t,c, label="[C]")
plt.xlabel('time')
plt.ylabel('Concentrations')
plt.legend()
plt.show()

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(a,b,c)
ax.set_xlabel('[A]')
ax.set_ylabel('[B]')
ax.set_zlabel('[C]')
plt.show()